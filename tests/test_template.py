"""Tests on the synthesised CloudFormation template.

These cover the handful of properties whose absence does not fail synth but does fail the deployment,
or fails it slowly and confusingly. Each one has been got wrong at least once.
"""

from __future__ import annotations

import json
import os
import sys

import aws_cdk as cdk
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "infra"))
from hardware import ConfigError  # noqa: E402
from serving_stack import ServingStack  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# The zero-config shape: what a fresh account gets, so it is what most tests run against.
BASE = {
    "region": "us-west-2",
    "instanceType": "g7e.2xlarge",
    "instanceCount": 1,
    "modelId": "some-org/some-model",
    "image": "example/vllm:test",
    "quantization": "fp8",
    # Required, and not generated inside the stack: synth runs on every deploy, so a
    # value minted here would rotate the key each time. app.py generates one once and persists it.
    "apiKey": "test-key-not-a-real-secret",
}

AZ_CONTEXT = {
    # CDK looks zones up from the account at synth time and falls back to three dummy zones when it
    # cannot. Seeding four makes the VPC's zone spread observable in the template.
    "availability-zones:account=111122223333:region=us-west-2":
        ["us-west-2a", "us-west-2b", "us-west-2c", "us-west-2d"],
    # us-east-2 is the second region these tests synthesise for. Three zones, and g7e in only two of
    # them - the case that makes `availabilityZones` necessary.
    "availability-zones:account=111122223333:region=us-east-2":
        ["us-east-2a", "us-east-2b", "us-east-2c"],
    # Seeded prefix-list lookup for the CloudFront origin-facing list, which the ALB security group
    # admits. PrefixList.from_lookup goes through the Cloud Control API context provider, hence the
    # shape of the key. Without this, synth performs an SDK call and falls back to pl-xxxxxxxx.
    **{("cc-api-provider:account=111122223333:expectedMatchCount=exactly-one"
        ":propertiesToReturn.0=PrefixListId"
        ":propertyMatch.PrefixListName=com.amazonaws.global.cloudfront.origin-facing"
        f":region={region}:typeName=AWS$:$:EC2$:$:PrefixList"): [{"PrefixListId": "pl-0cf00ffee"}]
       for region in ("us-west-2", "us-east-2")},
}


def synth(**overrides) -> dict:
    cfg = {**BASE, **overrides}
    app = cdk.App(context=dict(AZ_CONTEXT))
    stack = ServingStack(app, "T", cfg=cfg,
                         env=cdk.Environment(account="111122223333", region=cfg["region"]))
    return app.synth().get_stack_artifact(stack.artifact_id).template


def only(template: dict, resource_type: str) -> dict:
    matches = [r["Properties"] for r in template["Resources"].values()
               if r["Type"] == resource_type]
    assert len(matches) == 1, f"expected one {resource_type}, found {len(matches)}"
    return matches[0]


def test_weights_are_cached_in_a_shared_host_directory():
    """Without this, a crash-looping task writes a fresh copy of the weights per attempt and fills
    the root volume, which presents as a load that never completes rather than as a disk error."""
    template = synth()
    task_def = only(template, "AWS::ECS::TaskDefinition")

    volume = task_def["Volumes"][0]
    assert volume["Host"]["SourcePath"], "volume must be backed by a host path, not be empty"

    mount = task_def["ContainerDefinitions"][0]["MountPoints"][0]
    assert mount["SourceVolume"] == volume["Name"]
    assert mount["ReadOnly"] is False, "the container stages weights into this path"


def _entrypoint() -> str:
    """The container entrypoint as text. Several properties of the template are only meaningful if the
    entrypoint agrees with them, and it is a shell script rather than something importable."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "container", "serve")
    with open(path) as fh:
        return fh.read()


def test_container_mount_matches_the_path_the_entrypoint_uses():
    """The mount is inert if it lands somewhere the entrypoint does not write to."""
    template = synth()
    mount = vllm_container(template)["MountPoints"][0]
    body = _entrypoint()
    assert f'SHARED_CACHE="${{SHARED_CACHE:-{mount["ContainerPath"]}}}"' in body


def test_the_hugging_face_cache_also_lands_on_the_shared_volume():
    """Regression test for a real bug: the cache export was once conditional, so a default deployment
    let vLLM download into ~/.cache/huggingface inside the container. Every task
    start then wrote its own copy of the weights into the container's writable layer, a stopped
    container kept it, and a crash-looping task filled the root volume - the exact disk exhaustion the
    shared volume exists to prevent.
    """
    mount = vllm_container(synth())["MountPoints"][0]
    body = _entrypoint()
    assert "export HF_HOME=" in body, "the hub cache must be redirected, not left at its default"
    assert "HUGGINGFACE_HUB_CACHE" in body
    assert body.index("export HF_HOME=") < body.index("ARGS=("), "set before the engine starts"
    assert mount["ContainerPath"] in body.split("export HF_HOME=")[1].splitlines()[0] or \
        "SHARED_CACHE" in body.split("export HF_HOME=")[1].splitlines()[0], \
        "the hub cache must point inside the shared host volume"


def test_spot_uses_a_mixed_instances_policy_and_respects_the_requested_type():
    """A launch configuration cannot carry MixedInstancesPolicy, so `useSpot` silently did nothing
    until an explicit launch template was introduced. `capacity-optimized` without `-prioritized`
    ignores the overrides and launches a different instance type."""
    asg = only(synth(useSpot=True), "AWS::AutoScaling::AutoScalingGroup")

    policy = asg["MixedInstancesPolicy"]
    assert policy["InstancesDistribution"]["SpotAllocationStrategy"] == \
        "capacity-optimized-prioritized"
    assert policy["InstancesDistribution"]["OnDemandPercentageAboveBaseCapacity"] == 0
    assert [o["InstanceType"] for o in policy["LaunchTemplate"]["Overrides"]] == ["g7e.2xlarge"]
    # Mutually exclusive with MixedInstancesPolicy; CloudFormation rejects a template with both.
    assert "LaunchTemplate" not in asg


def test_on_demand_does_not_use_a_mixed_instances_policy():
    asg = only(synth(useSpot=False), "AWS::AutoScaling::AutoScalingGroup")
    assert "MixedInstancesPolicy" not in asg
    assert "LaunchTemplate" in asg


@pytest.mark.parametrize("region", ["us-west-2", "us-east-2"])
def test_gpu_ami_is_the_al2023_gpu_image_resolved_at_deploy_time(region):
    """The hardest failure in this project, and the one with no error message anywhere.

    The Amazon Linux 2 ECS GPU AMI tops out at NVIDIA driver 550, which cannot drive this GPU
    generation (sm_120). The instance boots, passes every EC2 and ASG health check, joins the cluster,
    and the ECS agent reports `agentConnected: true` - registering an EMPTY GPU resource list. Every
    task then sits in PROVISIONING forever with containerInstanceArn=null. Nothing logs a cause, and
    the symptom is indistinguishable from a capacity shortage.

    Two properties therefore have to hold, in every region: the AL2023 GPU image specifically, and
    resolved through SSM at deploy time rather than baked in - a literal ami-* id is region-scoped, so
    a template carrying one works only where it was generated."""
    template = synth(region=region)
    defaults = " ".join(str(v.get("Default", "")) for v in template.get("Parameters", {}).values())
    assert ("/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id"
            in defaults), f"expected the AL2023 GPU AMI parameter, got: {defaults}"
    assert "amazon-linux-2/" not in defaults, "Amazon Linux 2 cannot drive this GPU generation"

    assert "ami-" not in json.dumps(template), "AMI must not be baked into the template"


def test_recommended_eight_gpu_shape_places_all_eight_engines():
    """8 replicas x TP=1 - one engine per GPU - is the recommended starting point for an 8-GPU
    instance, because it is the topology that measured best on two GPUs, just wider.

    ECS reserves the full memory limit per task, so eight tasks each claiming the instance's whole
    share would leave seven unplaceable - and you would get one working engine at an eighth of the
    throughput you pay for, with no error anywhere. The limit must therefore be divided by the
    replica count, and `replicas` x GPUs-per-replica must not exceed the instance's GPUs.
    """
    template = synth(instanceType="g7e.48xlarge", tuning={"tensorParallel": 1, "replicas": 8})
    container = vllm_container(template)
    gpu = [r for r in container["ResourceRequirements"] if r["Type"] == "GPU"][0]

    assert int(gpu["Value"]) == 1, "one GPU per engine"
    assert only(template, "AWS::ECS::Service")["DesiredCount"] == 8, "one task per replica"

    single = synth(instanceType="g7e.48xlarge", tuning={"tensorParallel": 1, "replicas": 1})
    whole = vllm_container(single)["Memory"]
    assert container["Memory"] == whole // 8, "the host share is divided between the replicas"
    assert container["Memory"] * 8 <= whole, "eight tasks must fit in what one task could claim"


def test_documented_high_cache_hit_alternative_also_places():
    """4 x TP=2 is documented as the alternative for workloads with shared prompt prefixes, so it
    has to work too."""
    template = synth(instanceType="g7e.48xlarge", tuning={"tensorParallel": 2, "replicas": 4})
    container = vllm_container(template)
    gpu = [r for r in container["ResourceRequirements"] if r["Type"] == "GPU"][0]

    assert int(gpu["Value"]) == 2
    assert only(template, "AWS::ECS::Service")["DesiredCount"] == 4


def test_measured_engine_settings_reach_the_container():
    """Each of these is a measured default. A knob that is validated but never passed to the engine
    is worse than no knob, because the config file then documents behaviour that does not happen."""
    env = {e["Name"]: e["Value"]
           for e in vllm_container(synth())["Environment"]}

    assert env["KV_CACHE_DTYPE"] == "fp8"            # +12% decode
    assert env["MAX_NUM_SEQS"] == "256"              # 128 -> 256 measured +1.4%
    assert env["GPU_MEMORY_UTILIZATION"] == "0.95"   # 0.97 crashes dense models
    assert env["ENABLE_PREFIX_CACHING"] == "true"    # load-bearing at high concurrency
    assert env["ENABLE_EXPERT_PARALLEL"] == "false"  # -13% in FP8
    # 0 tells the entrypoint to omit the flag: both measured no effect across the useful range.
    assert env["MAX_MODEL_LEN"] == "0"
    assert env["MAX_NUM_BATCHED_TOKENS"] == "0"


def test_fully_unquantised_deployment_is_supported():
    """Weight precision and KV cache precision are independent decisions, and a deployment that
    refuses both must still produce a working stack rather than an error or a silent fp8 cache."""
    template = synth(quantization="", tuning={"kvCacheDtype": "auto"})
    env = {e["Name"]: e["Value"]
           for e in vllm_container(template)["Environment"]}

    assert env["KV_CACHE_DTYPE"] == "auto"
    assert "QUANTIZATION" not in env, "an empty quantization must not reach the engine as a flag"


def test_shipped_config_does_not_warn(capsys):
    """The default configuration must be internally consistent. A warning that fires on an untouched
    config is worse than no warning, because it teaches people to ignore the next one."""
    import yaml
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    with open(os.path.join(root, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh)

    synth(quantization=cfg.get("quantization"), tuning=cfg["tuning"])
    assert "Warning" not in capsys.readouterr().err


def test_risky_combination_warns_without_polluting_the_template(capsys):
    """`cdk synth > template.yaml` is normal, so the warning must go to stderr. A warning on stdout
    lands in the middle of the template and produces invalid YAML."""
    synth(quantization="", tuning={"kvCacheDtype": "fp8"})
    captured = capsys.readouterr()

    assert "gpuMemoryUtilization" in captured.err
    assert "Warning" not in captured.out


def test_unauthenticated_requests_never_reach_the_model():
    template = synth()
    listener = only(template, "AWS::ElasticLoadBalancingV2::Listener")
    default = listener["DefaultActions"][0]
    assert default["Type"] == "fixed-response"
    assert default["FixedResponseConfig"]["StatusCode"] == "403"

    rule = only(template, "AWS::ElasticLoadBalancingV2::ListenerRule")
    header = rule["Conditions"][0]["HttpHeaderConfig"]
    assert header["HttpHeaderName"] == "Authorization"
    assert header["Values"] == ["Bearer test-key-not-a-real-secret"], \
        "the header every OpenAI client and gateway already sends"


def vllm_container(template: dict) -> dict:
    return only(template, "AWS::ECS::TaskDefinition")["ContainerDefinitions"][0]


def test_no_hugging_face_token_unless_a_secret_is_named():
    """Public models need none, and a token is a real credential, so it is opt-in."""
    assert "Secrets" not in vllm_container(synth())


def test_the_model_name_output_is_the_model_id_clients_must_send():
    assert synth(modelId="org/some-model")["Outputs"]["ModelName"]["Value"] == "org/some-model"


def test_a_named_secret_reaches_the_engine_as_hf_token_and_only_the_engine():
    """Gated models then take the same path as public ones. Passed as an ECS secret, so it is never in
    the template as a value; the sidecar does not get it."""
    template = synth(hfTokenSecretName="gpu-llm-serving/hf-token")
    secrets = vllm_container(template)["Secrets"]
    assert [s["Name"] for s in secrets] == ["HF_TOKEN"]
    assert "gpu-llm-serving/hf-token" in str(secrets[0]["ValueFrom"])
    sidecar = [c for c in only(template, "AWS::ECS::TaskDefinition")["ContainerDefinitions"]
               if c["Name"] == "metrics"][0]
    assert "Secrets" not in sidecar
    assert "hf_" not in str(template), "no token value anywhere in the template"


@pytest.mark.parametrize("instance_type,tp", [
    ("g7e.2xlarge", 1),      # TP=1 never touches /dev/shm, so the setting must not be conditional
    ("g7e.24xlarge", 4),     # dies in seconds on Docker's 64 MiB default
])
def test_shared_memory_is_always_raised(instance_type, tp):
    """`RuntimeError: Insufficient space in /dev/shm` was the hardest failure in the project to
    diagnose. Raising it only when TP>1 would reintroduce it for anyone using EXTRA_ARGS."""
    container = only(synth(instanceType=instance_type, tuning={"tensorParallel": tp}),
                     "AWS::ECS::TaskDefinition")["ContainerDefinitions"][0]
    assert container["LinuxParameters"]["SharedMemorySize"] >= 1024


def test_circuit_breaker_is_off():
    """On purpose: it rolls back to a revision that is frequently also broken, and the two then
    contend for the same GPU indefinitely. Documented in serving_stack.py."""
    service = only(synth(), "AWS::ECS::Service")
    assert "DeploymentCircuitBreaker" not in service.get("DeploymentConfiguration", {})


def test_four_availability_zones():
    """A capacity decision, not a resilience one - GPU capacity is allocated per zone."""
    template = synth()
    subnets = [r for r in template["Resources"].values() if r["Type"] == "AWS::EC2::Subnet"]
    zones = {str(s["Properties"]["AvailabilityZone"]) for s in subnets}
    assert len(zones) == 4



def test_extra_args_reaches_the_container():
    """The docs point at extraArgs as the way to try an unmodelled engine flag (data parallelism) and
    to diagnose a reluctant engine. It was documented in three places before it was plumbed, so a
    reader following that advice had nowhere to put it."""
    template = synth(extraArgs="--data-parallel-size 2")
    env = {e["Name"]: e.get("Value") for e in vllm_container(template)["Environment"]}
    assert env.get("EXTRA_ARGS") == "--data-parallel-size 2"


def test_empty_extra_args_is_omitted_entirely():
    """An empty string must not reach the container as a variable at all."""
    env = {e["Name"]: e.get("Value") for e in vllm_container(synth(extraArgs=""))["Environment"]}
    assert "EXTRA_ARGS" not in env

    env = {e["Name"]: e.get("Value") for e in vllm_container(synth())["Environment"]}
    assert "EXTRA_ARGS" not in env


def test_tool_call_and_reasoning_parsers_reach_the_container():
    """Agent frameworks send `tools` and read tool_calls back; without the parser the engine returns the
    call as plain text and every agent stalls on its first step. Empty means no variable at all."""
    env = {e["Name"]: e.get("Value") for e in vllm_container(synth(toolCallParser="hermes", reasoningParser="qwen3"))["Environment"]}
    assert env["TOOL_CALL_PARSER"] == "hermes" and env["REASONING_PARSER"] == "qwen3"
    env = {e["Name"]: e.get("Value") for e in vllm_container(synth())["Environment"]}
    assert "TOOL_CALL_PARSER" not in env and "REASONING_PARSER" not in env
    with pytest.raises(ConfigError, match="toolCallParser"):
        synth(toolCallParser="--tool-call-parser hermes")


def _scaling(template: dict) -> tuple[list, list]:
    targets = [r["Properties"] for r in template["Resources"].values()
               if r["Type"] == "AWS::ApplicationAutoScaling::ScalableTarget"]
    policies = [r["Properties"] for r in template["Resources"].values()
                if r["Type"] == "AWS::ApplicationAutoScaling::ScalingPolicy"]
    return targets, policies


def test_a_fixed_fleet_creates_no_scaling_policy():
    """Autoscaling must be opt-in. It cannot protect a seconds-scale latency budget (a new task is
    minutes from serving), so a fleet that silently resized itself would be worse than one that does
    not: the operator would size the minimum assuming scaling covers surges."""
    targets, policies = _scaling(synth(instanceCount=5))
    assert targets == [] and policies == []


def test_the_asg_cannot_grow_past_the_configured_maximum():
    """The capacity provider launches instances for unplaceable tasks, so the ASG ceiling is the real
    limit on a runaway scale-out - not the task count."""
    asg = only(synth(instanceCount=5), "AWS::AutoScaling::AutoScalingGroup")
    assert int(asg["MaxSize"]) == 5, "a fixed fleet must not be able to grow"

    asg = only(synth(instanceCount=5, maxInstanceCount=12), "AWS::AutoScaling::AutoScalingGroup")
    assert int(asg["MaxSize"]) == 12
    # MinSize IS instanceCount, and that is what makes a first deploy come up at the right size: an
    # ASG created without DesiredCapacity takes MinSize as its initial desired capacity. With MinSize 0
    # a fresh deploy started at zero instances and waited for something to scale it.
    assert int(asg["MinSize"]) == 5, "instanceCount is documented as the floor, so it must be one"


def test_an_autoscaling_fleet_has_a_declared_initial_size_but_no_resettable_instance_count():
    """The two halves of this are a split, and both directions have bitten.

    Declaring AutoScalingGroup.DesiredCapacity meant any later `cdk deploy` pulled a scaled-out fleet
    back to the configured minimum, TERMINATING instances whose tasks the policy still wanted and
    discarding tens of GiB of loaded weights. So it is omitted, and MinSize carries the floor - which
    also gives a new ASG its initial size, because an ASG created without DesiredCapacity adopts
    MinSize.

    The service is the opposite: omitting DesiredCount makes ECS default a NEW service to one task, and
    registering a scalable target with a higher MinCapacity is documented to correct capacity only when
    UPDATING an existing target, not when creating one. So a first deploy would serve from a single
    task. It is declared, and the cost is a deploy dipping the task count, which the policy restores in
    about 11 minutes. That is the milder failure, though not free: managed termination protection is
    disabled, so the capacity provider scales in the surplus instances roughly 15 minutes later and
    their warm host caches are lost - the deploy just does not terminate anything mid-request.
    """
    scaled = synth(instanceCount=6, maxInstanceCount=8)
    asg = only(scaled, "AWS::AutoScaling::AutoScalingGroup")
    assert "DesiredCapacity" not in asg, "a deploy must not reset a scaled-out fleet"
    assert int(asg["MinSize"]) == 6, "MinSize is both the floor and the initial size"
    assert int(only(scaled, "AWS::ECS::Service")["DesiredCount"]) == 6, \
        "a first deploy must not come up with one task"
    target = [r for r in scaled["Resources"].values()
              if r["Type"] == "AWS::ApplicationAutoScaling::ScalableTarget"][0]["Properties"]
    assert int(target["MinCapacity"]) == 6

    # A FIXED fleet is the opposite case: nothing else owns the counts, so declaring them is what
    # makes the size assertable from config.
    fixed = synth(instanceCount=5)
    assert int(only(fixed, "AWS::AutoScaling::AutoScalingGroup")["DesiredCapacity"]) == 5
    assert int(only(fixed, "AWS::ECS::Service")["DesiredCount"]) == 5


def test_autoscaling_tracks_request_count_per_target():
    """GPU utilisation sits high while latency is fine, and response time only rises once requests are
    already slow. Request count per target is the leading signal, so pin the metric type."""
    targets, policies = _scaling(synth(instanceCount=5, maxInstanceCount=12))
    assert len(targets) == 1 and len(policies) == 1

    cfg = policies[0]["TargetTrackingScalingPolicyConfiguration"]
    assert cfg["PredefinedMetricSpecification"]["PredefinedMetricType"] == "ALBRequestCountPerTarget"
    assert cfg["TargetValue"] == 925
    # Asymmetric on purpose: a discarded engine costs minutes to replace.
    assert cfg["ScaleInCooldown"] > cfg["ScaleOutCooldown"]


def test_scaling_bounds_are_tasks_not_instances():
    """The scalable dimension is ECS task count, so both bounds must be multiplied by replicas. Using
    instance counts directly would cap a 2-replica fleet at half the tasks it is configured for."""
    targets, _ = _scaling(synth(instanceType="g7e.12xlarge", instanceCount=3, maxInstanceCount=6,
                                tuning={"tensorParallel": 1, "replicas": 2}))
    assert int(targets[0]["MinCapacity"]) == 6      # 3 instances x 2 replicas
    assert int(targets[0]["MaxCapacity"]) == 12     # 6 instances x 2 replicas


def test_the_scaling_target_is_configurable():
    _, policies = _scaling(synth(instanceCount=1, maxInstanceCount=4,
                                 scalingRequestsPerTarget=400))
    assert policies[0]["TargetTrackingScalingPolicyConfiguration"]["TargetValue"] == 400


# --------------------------------------------------------------------------------------------
# Access: HTTPS out of the box through CloudFront, with the load balancer never facing the internet.
# --------------------------------------------------------------------------------------------

def cdn(template: dict) -> dict:
    return only(template, "AWS::CloudFront::Distribution")["DistributionConfig"]


def test_the_endpoint_is_https_with_nothing_to_own():
    """No domain, no certificate, no hosted zone: CloudFront's own name and certificate. The
    Endpoint output is the distribution's domain, so scripts and readers get the right URL."""
    template = synth()
    assert cdn(template)["Enabled"] is True
    assert not [r for r in template["Resources"].values()
                if r["Type"] in ("AWS::CertificateManager::Certificate", "AWS::Route53::RecordSet")]
    endpoint = template["Outputs"]["Endpoint"]["Value"]
    assert endpoint["Fn::Join"][1][0] == "https://"
    assert endpoint["Fn::Join"][1][1]["Fn::GetAtt"][1] == "DomainName"


def test_the_load_balancer_is_internal_and_reached_only_through_a_vpc_origin():
    """The API key is a header, so the hop carrying it must never cross the internet in plaintext.
    Internal scheme plus a VPC origin keeps CloudFront-to-ALB on the AWS network."""
    template = synth()
    alb = only(template, "AWS::ElasticLoadBalancingV2::LoadBalancer")
    assert alb["Scheme"] == "internal"
    vpc_origin = only(template, "AWS::CloudFront::VpcOrigin")["VpcOriginEndpointConfig"]
    assert vpc_origin["Name"] == "T-us-west-2", "VPC origin names are account-wide; the region disambiguates"
    assert vpc_origin["OriginProtocolPolicy"] == "http-only"
    assert vpc_origin["HTTPPort"] == 80
    origin = cdn(template)["Origins"][0]
    assert "VpcOriginConfig" in origin
    assert origin["VpcOriginConfig"]["OriginReadTimeout"] == 120, "the quota ceiling, no request needed"
    assert origin["VpcOriginConfig"]["OriginKeepaliveTimeout"] < 300, "below the ALB idle timeout"


def test_the_load_balancer_admits_only_cloudfront():
    """The one rule that took a live deploy to get right. A VPC origin delivers packets through an
    interface in the private subnets, but with CloudFront's own source addresses - so a rule on the VPC
    CIDR never matches and CloudFront times out on connect. The managed prefix list is the answer,
    looked up by name so the id is right in every region."""
    template = synth()
    alb_sg = next(rid for rid, r in template["Resources"].items()
                  if r["Type"] == "AWS::EC2::SecurityGroup" and "Alb" in rid)
    # A prefix-list peer renders as a standalone ingress resource, not an inline rule.
    inline = template["Resources"][alb_sg]["Properties"].get("SecurityGroupIngress") or []
    standalone = [r["Properties"] for r in template["Resources"].values()
                  if r["Type"] == "AWS::EC2::SecurityGroupIngress"
                  and r["Properties"]["GroupId"] == {"Fn::GetAtt": [alb_sg, "GroupId"]}]
    rules = inline + standalone
    assert len(rules) == 1
    assert rules[0]["SourcePrefixListId"] == "pl-0cf00ffee", "the seeded lookup result, not the dummy"
    assert "CidrIp" not in rules[0], "the VPC CIDR does not match VPC-origin traffic"
    assert rules[0]["FromPort"] == 80
    listeners = [r["Properties"] for r in template["Resources"].values()
                 if r["Type"] == "AWS::ElasticLoadBalancingV2::Listener"]
    assert [(lis["Protocol"], lis["Port"]) for lis in listeners] == [("HTTP", 80)]


def test_cloudfront_is_a_pass_through_that_can_stream():
    """Every one of these makes CloudFront get out of the way of an API that streams. Caching off
    (the managed CachingDisabled policy), every viewer header forwarded except Host (so Authorization
    reaches the ALB rule), all methods (POST), no compression (a compressed stream is a buffered
    one), HTTPS only (a redirect would turn a POST into a GET)."""
    b = cdn(synth())["DefaultCacheBehavior"]
    assert b["CachePolicyId"] == "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"          # CachingDisabled
    assert b["OriginRequestPolicyId"] == "b689b0a8-53d0-40ab-baf2-68738e2966ac"  # AllViewerExceptHost
    assert set(b["AllowedMethods"]) >= {"POST", "GET", "OPTIONS"}
    assert b["Compress"] is False
    assert b["ViewerProtocolPolicy"] == "https-only"


def test_cloudfront_does_not_replay_a_transient_error_for_ten_seconds():
    """CloudFront caches error responses for 10 s by default. A 503 during a task restart would be
    served to every caller for those 10 s, including ones the fleet could have handled."""
    errors = {e["ErrorCode"]: e["ErrorCachingMinTTL"] for e in cdn(synth())["CustomErrorResponses"]}
    assert errors == {500: 0, 502: 0, 503: 0, 504: 0}


def test_the_dashboard_watches_cloudfront_in_us_east_1_whatever_the_stack_region():
    """CloudFront metrics live in us-east-1. A widget that read them from the stack's region would
    render an empty graph that looks exactly like zero errors."""
    w = widget(synth(), "Is CloudFront timing out?")
    metric = w["metrics"][0]
    assert metric[0] == "AWS/CloudFront" and metric[1] == "5xxErrorRate"
    assert metric[-1]["region"] == "us-east-1"


# --------------------------------------------------------------------------------------------
# Region and availability zones
# --------------------------------------------------------------------------------------------

def test_availability_zones_can_be_restricted_to_where_the_instance_is_offered():
    """us-east-2 has three zones and offers g7e in two. An ASG that picks the third fails with
    `Unsupported`, which is permanent - so the fleet must be confinable to the offering zones."""
    template = synth(region="us-east-2",
                     availabilityZones=["us-east-2a", "us-east-2b"])
    zones = {r["Properties"]["AvailabilityZone"] for r in template["Resources"].values()
             if r["Type"] == "AWS::EC2::Subnet"}
    assert zones == {"us-east-2a", "us-east-2b"}


def test_a_single_availability_zone_is_rejected():
    with pytest.raises(ConfigError, match="at least two"):
        synth(availabilityZones=["us-west-2a"])


# --------------------------------------------------------------------------------------------
# The two fleet shapes that must work from config alone
# --------------------------------------------------------------------------------------------

def test_a_six_to_eight_fleet_synthesises_one_task_per_gpu_with_autoscaling_bounds():
    """The shape used for testing: 6 x g7e.2xlarge scaling to 8, one engine per GPU. Bounds on the
    scalable target are task counts, the ASG has no DesiredCapacity while autoscaling owns it."""
    template = synth(instanceType="g7e.2xlarge", instanceCount=6, maxInstanceCount=8)
    asg = only(template, "AWS::AutoScaling::AutoScalingGroup")
    assert asg["MaxSize"] == "8" and "DesiredCapacity" not in asg
    gpu = [r for r in vllm_container(template)["ResourceRequirements"]
           if r["Type"] == "GPU"]
    assert gpu[0]["Value"] == "1"
    target = only(template, "AWS::ApplicationAutoScaling::ScalableTarget")
    assert (target["MinCapacity"], target["MaxCapacity"]) == (6, 8)


def test_the_largest_instance_runs_one_engine_per_gpu_without_being_told():
    """A g7e.48xlarge has 8 GPUs, and the whole point of deriving replicas is that the user does not
    have to work out that they want 8 engines. Two instances is 16 tasks."""
    template = synth(instanceType="g7e.48xlarge", instanceCount=2)

    service = only(template, "AWS::ECS::Service")
    assert service["DesiredCount"] == 16

    container = vllm_container(template)
    gpu = [r for r in container["ResourceRequirements"] if r["Type"] == "GPU"]
    assert gpu[0]["Value"] == "1"
    # Eight reservations have to fit the host, or ECS places one task and leaves seven GPUs idle.
    assert container["Memory"] * 8 <= 2048 * 1024 * 0.60 + 1


def test_replicas_default_to_the_instance_gpu_count():
    from hardware import get_instance, resolve_topology, model_bytes
    for name, expected in [("g7e.2xlarge", 1), ("g7e.12xlarge", 2), ("g7e.48xlarge", 8)]:
        inst = get_instance(name)
        t = resolve_topology(inst, {}, model_bytes(30, 1.0))
        assert t["replicas"] == expected, name
        assert t["tensorParallel"] == 1, name


# --------------------------------------------------------------------------------------------
# The repo is going to be shared, so nothing real may be committed in the tracked config.
# --------------------------------------------------------------------------------------------

def test_shipped_config_contains_no_account_id():
    """The tracked config is generic; anything deployment-specific belongs in config.local.yaml."""
    import re
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
    with open(path) as fh:
        lines = fh.readlines()
    for n, line in enumerate(lines, 1):
        code = line.split("#", 1)[0]          # comments may carry example.com placeholders
        assert not re.search(r"\b\d{12}\b", code), f"config.yaml:{n} looks like an account id"


# --------------------------------------------------------------------------------------------
# Pulling the image, and reading the model's precision correctly. Both pass synth and fail later.
# --------------------------------------------------------------------------------------------

def test_execution_role_can_actually_pull_from_ecr():
    """`ecr:GetAuthorizationToken` alone authenticates but cannot fetch a manifest or layers, so the
    task fails at start with `CannotPullContainerError: ... denied`.

    Referencing the image as an opaque URI string leaves CDK unable to tell it names an ECR repository
    in this account, so it grants nothing - and warns about exactly that, in wording easy to dismiss as
    boilerplate. Recognising our own registry and using from_ecr_repository makes CDK issue the grant."""
    template = synth(image="111122223333.dkr.ecr.us-west-2.amazonaws.com/gpu-llm-serving:vllm-0.28.0")

    granted = set()
    for res in template["Resources"].values():
        if res["Type"] != "AWS::IAM::Policy":
            continue
        roles = json.dumps(res["Properties"].get("Roles", []))
        if "ExecutionRole" not in roles:
            continue
        for stm in res["Properties"]["PolicyDocument"]["Statement"]:
            actions = stm["Action"] if isinstance(stm["Action"], list) else [stm["Action"]]
            granted.update(a for a in actions if a.startswith("ecr:"))

    for needed in ("ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer",
                   "ecr:BatchCheckLayerAvailability"):
        assert needed in granted, (
            f"the task execution role is missing {needed}; the image cannot be pulled. "
            f"granted: {sorted(granted)}")


def test_an_image_from_another_registry_is_still_accepted():
    """The ECR path must not become the only path - a public or third-party image has to keep
    working, it simply gets no grant because none is needed."""
    template = synth(image="vllm/vllm-openai:v0.28.0")
    assert vllm_container(template)["Image"] == "vllm/vllm-openai:v0.28.0"


def test_an_official_quantised_checkpoint_is_not_mistaken_for_full_precision(capsys):
    """A publisher's FP8 build is already quantised on disk, so `quantization` is correctly EMPTY.

    Judging precision from `quantization` alone misreads that as bf16, which doubles the estimated
    weight size and makes the memory warning advise clearing kvCacheDtype - giving up a measured 12%
    of decode to fix a problem that does not exist."""
    synth(modelId="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", quantization="")
    assert "unquantised weights" not in capsys.readouterr().err


@pytest.mark.parametrize("model_id", [
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "some-org/model-AWQ-4bit",
    "some-org/model-gptq-int4",
    "some-org/Model-W8A8",
])
def test_quantisation_markers_in_the_model_id_are_recognised(model_id):
    from hardware import bytes_per_param_for, weights_are_quantised
    assert weights_are_quantised(model_id, "")
    assert bytes_per_param_for(model_id, "") < 2.0, "quantised: 1 byte for 8-bit, 0.5 for 4-bit"


def test_an_unquantised_model_with_an_fp8_cache_still_warns(capsys):
    """The counterpart: the check must not be so permissive that it stops catching the real case."""
    synth(modelId="Qwen/Qwen3-30B-A3B-Instruct-2507", quantization="",
          tuning={"kvCacheDtype": "fp8"})
    assert "unquantised weights" in capsys.readouterr().err


def test_the_stack_refuses_to_invent_a_key():
    """Generating one here would rotate it on every synth, so the stack requires it and points at
    where a stable one comes from."""
    cfg = {k: v for k, v in BASE.items() if k != "apiKey"}
    app = cdk.App(context=dict(AZ_CONTEXT))
    with pytest.raises(ConfigError, match="apiKey"):
        ServingStack(app, "T", cfg=cfg,
                     env=cdk.Environment(account="111122223333", region="us-west-2"))


def test_app_generates_a_key_once_and_reuses_it(tmp_path):
    """Driven through the real entry point, because that is where generation lives - and the value
    must survive a second invocation rather than being regenerated."""
    import subprocess
    import textwrap
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent("""
        region: us-west-2
        instanceType: g7e.2xlarge
        modelId: some-org/some-model-FP8
        image: example/vllm:test
    """))
    env = {**os.environ, "CONFIG": str(cfg), "CDK_DEFAULT_ACCOUNT": "111122223333"}
    env.pop("SERVING_IMAGE", None)

    def run():
        return subprocess.run([sys.executable, os.path.join(root, "infra", "app.py")],
                              capture_output=True, text=True, env=env)

    assert run().returncode == 0
    local = tmp_path / "config.local.yaml"
    assert local.exists(), "the generated key must be persisted next to the config in use"
    import yaml
    first = yaml.safe_load(local.read_text())["apiKey"]
    assert first

    assert run().returncode == 0
    again = yaml.safe_load(local.read_text())["apiKey"]
    assert again == first, "a second deploy must not rotate the key"


# --------------------------------------------------------------------------------------------
# Settings whose absence only shows up under real traffic
# --------------------------------------------------------------------------------------------

def test_an_untagged_ecr_reference_still_gets_a_pull_grant():
    """`account.dkr.ecr.region.amazonaws.com/repo` with no tag is a legal reference meaning `:latest`.

    The regex required a tag or a digest, so this form did not match and fell through to
    `from_registry` - no pull grant, and the `CannotPullContainerError` the ECR branch exists to
    prevent, for a URI that looks entirely reasonable."""
    blob = json.dumps(synth(image="111122223333.dkr.ecr.us-west-2.amazonaws.com/gpu-llm-serving"))
    assert '"ecr:BatchGetImage"' in blob, "an untagged ECR URI is still an ECR URI"
    assert "gpu-llm-serving:latest" in blob, "an untagged reference means :latest"


def test_the_ecr_grant_does_not_depend_on_the_resolved_account():
    """The grant is built from the ARN implied by the image URI, not by comparing it to this stack's
    account - because the CDK CLI overwrites CDK_DEFAULT_ACCOUNT from the ambient credentials, so the
    comparison failed exactly when it mattered and silently fell back to no permissions."""
    blob = json.dumps(synth(image="444455556666.dkr.ecr.us-west-2.amazonaws.com/other:tag"))
    assert '"ecr:BatchGetImage"' in blob, "an ECR URI must be recognised whatever the stack account is"
    assert "444455556666" in blob, "the grant should name the repository's own account"


def test_a_non_ecr_image_is_left_alone():
    """Docker Hub and public galleries must still work; only ECR URIs get the grant."""
    blob = json.dumps(synth(image="vllm/vllm-openai:v0.28.0"))
    assert '"ecr:BatchGetImage"' not in blob


def test_the_load_balancer_waits_longer_than_a_generation_takes():
    """A non-streaming completion sends nothing until it is finished, so the ALB's idle timer covers
    the whole generation. At the 60 s default a long output returns 504 while the engine is healthy -
    measured: a 760-token completion took ~17 s and p99 exceeded 13 s under load."""
    attrs = {a["Key"]: a["Value"]
             for a in only(synth(), "AWS::ElasticLoadBalancingV2::LoadBalancer")
             .get("LoadBalancerAttributes", [])}
    assert int(attrs["idle_timeout.timeout_seconds"]) >= 120


def test_draining_gives_an_in_flight_generation_time_to_finish():
    """At the 30 s defaults, any request still generating was killed on every scale-in and every
    deployment, and `min_healthy_percent=0` lets ECS stop several tasks at once."""
    template = synth()
    attrs = {a["Key"]: a["Value"]
             for a in only(template, "AWS::ElasticLoadBalancingV2::TargetGroup")
             .get("TargetGroupAttributes", [])}
    assert int(attrs["deregistration_delay.timeout_seconds"]) >= 120
    container = vllm_container(template)
    assert int(container["StopTimeout"]) >= 120


def test_spot_fleets_rebalance_and_rely_on_managed_draining_not_the_agent_flag():
    """ECS managed draining handles the spot reclaim notice itself, so the agent's
    ECS_ENABLE_SPOT_INSTANCE_DRAINING is redundant and no longer set. Capacity rebalance stays on."""
    spot = synth(useSpot=True)
    lt = json.dumps(only(spot, "AWS::EC2::LaunchTemplate"))
    assert "ECS_ENABLE_SPOT_INSTANCE_DRAINING" not in lt
    assert only(spot, "AWS::AutoScaling::AutoScalingGroup").get("CapacityRebalance") is True
    assert only(spot, "AWS::ECS::CapacityProvider")["AutoScalingGroupProvider"]["ManagedDraining"] == "ENABLED"


def test_max_below_min_is_a_config_error_not_an_internal_one():
    """CDK otherwise raises a bare jsii RuntimeError about ASG bounds, which reads as an internal
    fault rather than the config mistake it is."""
    with pytest.raises(ConfigError):
        synth(instanceCount=6, maxInstanceCount=2)


def test_a_scalar_written_where_a_list_belongs_says_so():
    """YAML makes this easy to write - `availabilityZones: us-west-2a` looks like a value, not a list -
    and a bare string is iterable, so it was consumed CHARACTER BY CHARACTER: ten one-letter zone
    names. That surfaced as a jsii internal error deep in CDK rather than as the one-line config
    mistake it is."""
    with pytest.raises(ConfigError) as err:
        synth(availabilityZones="us-west-2a")
    assert "list" in str(err.value).lower()


def test_an_instance_count_of_zero_synthesises_an_empty_fleet():
    """The documented way to park a deployment without destroying it: the ALB, the ECR image and the
    cached weights survive, and only the GPU instances - the entire cost - go away. It has to reach the
    template as a real zero rather than being floored to one somewhere on the way."""
    template = synth(instanceCount=0, maxInstanceCount=0)
    asg = only(template, "AWS::AutoScaling::AutoScalingGroup")
    assert (asg["MinSize"], asg["MaxSize"]) == ("0", "0")
    assert only(template, "AWS::ECS::Service")["DesiredCount"] == 0


def test_parking_the_fleet_without_lowering_the_ceiling_is_refused():
    """`instanceCount: 0` alone, with the shipped `maxInstanceCount: 8`, is broken in both directions.

    It cannot come back: with no tasks the target group publishes no ALBRequestCountPerTarget
    datapoints at all, and target tracking does not scale out on a missing metric - nor on a metric
    below its target. And it does not park either, because MaxSize stays at 8 and DesiredCapacity is
    absent, so the deploy never lowers ASG capacity and the shutdown falls back on the ~15-minute
    managed scale-in. Both halves pass synth, so this is the only place to catch it.
    """
    with pytest.raises(ConfigError) as e:
        synth(instanceCount=0, maxInstanceCount=8)
    msg = str(e.value)
    assert "maxInstanceCount: 0" in msg, f"the message must say what to do, got: {msg}"


def test_the_same_zone_written_twice_is_not_two_zones():
    """Counting entries rather than distinct zones let this through, and CloudFormation then rejected
    two subnets in one AZ several minutes into the deploy - which is precisely what a synth-time zone
    check exists to prevent."""
    with pytest.raises(ConfigError):
        synth(availabilityZones=["us-west-2a", "us-west-2a"])


def test_a_padded_image_uri_does_not_reach_the_task_definition():
    """The URI was stripped for the ECR match and then the RAW value handed to from_registry, so
    `image: "  vllm/vllm-openai:v0.11  "` put the padding into the task definition and failed at task
    start."""
    template = synth(image="  vllm/vllm-openai:v0.11  ")
    image = vllm_container(template)["Image"]
    assert image == "vllm/vllm-openai:v0.11", repr(image)


def test_a_trailing_slash_does_not_reach_the_repository_arn():
    """Newly reachable once the tag became optional: the name group would swallow the slash and produce
    an ARN ending in `repository/name/`, which IAM accepts and nothing matches."""
    blob = json.dumps(synth(image="111122223333.dkr.ecr.us-west-2.amazonaws.com/gpu-llm-serving/"))
    assert "repository/gpu-llm-serving/" not in blob.replace("repository/gpu-llm-serving/\"", "")
    assert '"ecr:BatchGetImage"' in blob


def test_a_scalar_where_a_settings_block_belongs_is_a_config_error():
    """`tuning: fast` otherwise raised a bare AttributeError from .get() on a str, printing a
    traceback that names the code instead of the line of YAML at fault."""
    with pytest.raises(ConfigError):
        synth(tuning="fast")


@pytest.mark.parametrize("overrides", [
    {"instanceCount": 0.5},
    {"maxInstanceCount": 8.7},
    # Only read when autoscaling is on, so the fleet has to allow scale-out for this one.
    {"instanceCount": 1, "maxInstanceCount": 4, "scalingRequestsPerTarget": 2.7},
])
def test_a_fractional_count_is_refused_at_synth(overrides):
    """Truncation is silent, and one of these is now dangerous rather than merely wrong: 0.5 instances
    truncates to 0, which is a legal parked fleet, so the typo deploys a stack that comes up healthy
    with no instances and 503s forever."""
    with pytest.raises(ConfigError):
        synth(**overrides)


def test_a_zero_scaling_target_is_rejected_rather_than_defaulted():
    """`or 925` treated it as absent, so it silently became 925 - while the quoted "0" was rejected,
    which meant the same value behaved differently depending on how it was written."""
    with pytest.raises(ConfigError):
        synth(instanceCount=1, maxInstanceCount=4, scalingRequestsPerTarget=0)


def test_a_zero_ceiling_below_a_nonzero_floor_is_rejected():
    """`or instance_count` swallowed it, making the "below instanceCount" error unreachable
    for exactly the value someone parking a fleet reaches for first."""
    with pytest.raises(ConfigError):
        synth(instanceCount=2, maxInstanceCount=0)


# --------------------------------------------------------------------------------------------
# Observability
#
# The failure this section exists to catch is a dashboard that synthesises perfectly and then draws
# nothing, because the metric behind a widget is not published unless something enables it.
# --------------------------------------------------------------------------------------------

def dashboard_widgets(template: dict) -> list:
    """The dashboard body, parsed. It is a Fn::Join of literals and resource references, so the
    references are substituted with a placeholder to make the result parseable JSON."""
    body = only(template, "AWS::CloudWatch::Dashboard")["DashboardBody"]
    parts = body["Fn::Join"][1] if isinstance(body, dict) else [body]
    return json.loads("".join(p if isinstance(p, str) else "TOKEN" for p in parts))["widgets"]


def widget(template: dict, title_fragment: str) -> dict:
    matches = [w["properties"] for w in dashboard_widgets(template)
               if title_fragment in (w["properties"].get("title") or "")]
    assert len(matches) == 1, f"expected one widget titled ~{title_fragment}, found {len(matches)}"
    return matches[0]


def alarms(template: dict) -> dict:
    return {r["Properties"]["AlarmName"]: r["Properties"] for r in template["Resources"].values()
            if r["Type"] == "AWS::CloudWatch::Alarm"}


def test_the_fleet_size_widget_has_metrics_to_draw():
    """An ASG publishes NOTHING about its own size unless group metrics are enabled - the metrics do
    not exist, so the widget renders an empty graph that reads as a dead fleet rather than a missing
    setting. This is the only widget on the dashboard whose data source has to be turned on."""
    asg = only(synth(), "AWS::AutoScaling::AutoScalingGroup")
    collected = asg["MetricsCollection"][0]
    assert collected["Granularity"] == "1Minute"
    assert set(collected["Metrics"]) == {"GroupInServiceInstances", "GroupDesiredCapacity"}


def test_every_deployment_gets_a_dashboard_and_the_two_universal_alarms():
    """No flag to find and no flag to forget - it is free, so it is unconditional. The two alarms here
    are true for any deployment ("an engine is down", "the load balancer is erroring"); the latency one
    needs a number only the operator has, so it is opt-in (tested separately)."""
    template = synth()
    assert only(template, "AWS::CloudWatch::Dashboard")["DashboardName"] == "T-us-west-2", \
        "dashboard names are account-wide, so the region has to be in the name"
    assert set(alarms(template)) == {"T-engines-unhealthy", "T-load-balancer-erroring"}


def test_the_dashboard_url_is_an_output_and_points_at_the_dashboard_created():
    """A dashboard nobody can find is not observability, and the console path for a named dashboard is
    not something anyone guesses. So the URL is an output, not just the name."""
    template = synth()
    url = template["Outputs"]["DashboardUrl"]["Value"]
    assert url == ("https://us-west-2.console.aws.amazon.com/cloudwatch/home"
                   "?region=us-west-2#dashboards:name=T-us-west-2")
    assert url.endswith(only(template, "AWS::CloudWatch::Dashboard")["DashboardName"])


def test_the_widgets_are_titled_as_questions_not_as_metric_names():
    """The reader is someone woken by an alarm who has never seen this stack. A widget called
    "TargetResponseTime p95" tells them nothing they can act on."""
    titles = [w["properties"].get("title") for w in dashboard_widgets(synth())
              if w["type"] == "metric"]
    assert all(titles), titles
    assert not any("TargetResponseTime" in t or "HTTPCode" in t or "HostCount" in t
                   for t in titles), titles


def test_the_latency_alarm_reaches_both_the_graph_and_the_alarm():
    """One number, two places. Hardcoding either would let the graph and the alarm disagree about what
    "too slow" means, which is worse than having neither."""
    template = synth(latencyAlarmSeconds=3.5)
    assert widget(template, "Is it slow?")["annotations"]["horizontal"][0]["value"] == 3.5
    assert alarms(template)["T-too-slow"]["Threshold"] == 3.5


def test_no_latency_alarm_and_no_budget_line_unless_asked():
    """A latency target is the caller's, not the model's. Shipping one would put an arbitrary red line
    on every deployment's graph and page whoever crossed it. Off by default; the graph stays."""
    template = synth()
    assert "T-too-slow" not in alarms(template)
    w = widget(template, "Is it slow?")
    assert "annotations" not in w or not w["annotations"].get("horizontal")


def test_the_latency_alarm_is_a_plain_percentile_alarm():
    """A metric carrying a `label` cannot be expressed in CloudFormation's simple alarm form, so CDK
    silently renders the alarm as a metric-math query with no MetricName - valid, but unreadable in the
    console and in describe-alarms. Labels belong on widgets, not on alarm metrics."""
    alarm = alarms(synth(latencyAlarmSeconds=8))["T-too-slow"]
    assert alarm["MetricName"] == "TargetResponseTime"
    assert alarm["ExtendedStatistic"] == "p95"
    assert "Metrics" not in alarm


def test_the_scaling_threshold_is_drawn_on_the_graph_it_governs():
    """The threshold is the one config value most likely to be wrong for a given workload, so it is
    annotated on the metric it is compared against rather than left in a scaling policy nobody opens."""
    template = synth(instanceCount=1, maxInstanceCount=4, scalingRequestsPerTarget=225)
    annotation = widget(template, "requests a minute per task")["annotations"]["horizontal"][0]
    assert annotation["value"] == 225


def test_a_fixed_size_fleet_draws_no_scaling_threshold():
    """With no scaling policy there is no threshold, and a line labelled "scale out at ..." on a fleet
    that cannot scale is a lie about the deployment."""
    template = synth(instanceCount=2, maxInstanceCount=2)
    assert "annotations" not in widget(template, "requests a minute per task")


def test_no_alarm_fires_on_an_idle_fleet():
    """An idle load balancer publishes no response times and no error counts at all, so the default
    (INSUFFICIENT_DATA, or breaching) makes every alarm meaningless the moment traffic stops."""
    for name, alarm in alarms(synth()).items():
        assert alarm["TreatMissingData"] == "notBreaching", name


def test_alarms_without_a_topic_still_exist_but_notify_nobody():
    """Absent must mean "alarms with no actions", not a synth error - the alarms are still worth having
    for the console and for describe-alarms."""
    for name, alarm in alarms(synth()).items():
        assert "AlarmActions" not in alarm, name


def test_a_supplied_topic_is_wired_to_every_alarm_and_no_topic_is_created():
    """Every alarm, not just the first - a half-wired set is the shape where the one alarm you needed is
    the silent one. And the topic is referenced, never created: a topic minted here would have no
    subscription, so it would look like a working notification while paging nobody.

   """
    arn = "arn:aws:sns:us-west-2:111122223333:llm-alerts"
    template = synth(alarmTopicArn=arn)
    for name, alarm in alarms(template).items():
        assert alarm["AlarmActions"] == [arn], name
    assert not [r for r in template["Resources"].values() if r["Type"] == "AWS::SNS::Topic"]


@pytest.mark.parametrize("bad", ["llm-alerts", "arn:aws:sqs:us-west-2:111122223333:q",
                                 "arn:aws:sns:us-west-2:111122223333:"])
def test_a_topic_arn_that_is_not_one_is_refused_at_synth(bad):
    """Otherwise it is accepted here and rejected by CloudFormation after the VPC, ALB and ASG exist -
    and a notification that was never wired up fails silently, which is the whole risk."""
    with pytest.raises(ConfigError):
        synth(alarmTopicArn=bad)


# ------------------------------------------------------------------ engine metrics sidecar

def sidecar(template: dict) -> dict:
    defs = only(template, "AWS::ECS::TaskDefinition")["ContainerDefinitions"]
    assert defs[0]["Name"] == "vllm", "the engine stays first; everything else indexes it as [0]"
    return next(d for d in defs if d["Name"] == "metrics")


def collector_config(template: dict) -> str:
    """The config embeds the log group name, a Ref at synth time, so CloudFormation renders the
    whole string as an Fn::Join. Flatten it, rendering each Ref/GetAtt as a placeholder."""
    value = next(e["Value"] for e in sidecar(template)["Environment"]
                 if e["Name"] == "OTEL_CONFIG")
    if isinstance(value, str):
        return value
    return "".join(part if isinstance(part, str) else "<token>" for part in value["Fn::Join"][1])


def test_every_task_carries_a_metrics_sidecar_that_cannot_take_the_engine_down():
    """The engine says WHY the fleet is slow; the load balancer only says THAT it is. But a metrics
    bug must never stop inference, so the collector is a non-essential container."""
    c = sidecar(synth())
    assert c["Image"].startswith("ghcr.io/open-telemetry/opentelemetry-collector-releases/opentelemetry-collector-contrib:0.")
    assert c["Command"] == ["--config=env:OTEL_CONFIG"], "the upstream image reads its config from this flag"
    assert c["Essential"] is False
    assert c["Memory"] == 256


def test_the_collector_is_configured_inline_with_exactly_the_shortlist():
    """Inline config means no custom image, no mounted file and no parameter store to keep in step.
    The shortlist is the bill: custom metrics are charged per name."""
    cfg = collector_config(synth())
    assert "targets: [\"localhost:8080\"]" in cfg, "awsvpc: both containers share localhost"
    assert "match_type: strict" in cfg
    shortlist = cfg.split("filter/shortlist:")[1].split("transform/bands:")[0]
    for name in ("vllm:num_requests_waiting", "vllm:num_requests_running", "vllm:kv_cache_usage_perc",
                 "vllm:num_preemptions_total", "vllm:request_prompt_tokens_le"):
        assert name in shortlist
    assert "namespace: T/Engine" in cfg
    assert "dimensions: [[]]" in cfg, "no per-task dimension: Maximum/Average across the fleet"


def test_the_collector_writes_into_the_stack_log_group_not_one_of_its_own():
    """So retention and teardown are the stack's, and there is one place to look for logs."""
    template = synth()
    assert "log_group_name: <token>" in collector_config(template), \
        "a Ref to the stack's log group, not a literal name of a second one"
    assert len([r for r in template["Resources"].values()
                if r["Type"] == "AWS::Logs::LogGroup"]) == 1


def test_the_task_role_writes_logs_only_and_has_no_putmetricdata():
    """EMF records become metrics inside CloudWatch Logs; PutMetricData is never called. A grant for it
    was dead permission and a misleading comment."""
    template = synth()
    statements = [st for r in template["Resources"].values() if r["Type"] == "AWS::IAM::Policy"
                  for st in r["Properties"]["PolicyDocument"]["Statement"]]
    assert not [st for st in statements if "cloudwatch:PutMetricData" in str(st["Action"])]
    task_role = next(lid for lid, r in template["Resources"].items()
                     if r["Type"] == "AWS::IAM::Role" and "TaskRole" in lid)
    log_writes = [st for r in template["Resources"].values() if r["Type"] == "AWS::IAM::Policy"
                  and {"Ref": task_role} in r["Properties"]["Roles"]
                  for st in r["Properties"]["PolicyDocument"]["Statement"]
                  if "logs:PutLogEvents" in str(st["Action"])]
    assert log_writes and all(st["Resource"] != "*" for st in log_writes)


def test_the_dashboard_shows_the_engine_row_from_the_same_namespace_the_collector_writes_to():
    """Namespace mismatch between exporter and widget is a silent empty graph. One constant, two
    places, checked here."""
    template = synth()
    for fragment in ("queueing inside the engines", "KV cache filling", "redoing work"):
        w = widget(template, fragment)
        namespaces = {m[0] for m in w["metrics"] if isinstance(m[0], str) and not m[0].startswith(".")}
        assert namespaces == {"T/Engine"}, (fragment, namespaces)


def test_security_group_rule_descriptions_use_only_characters_ec2_accepts():
    """EC2 rejects an apostrophe in a rule description with a 400 at CREATE time, after synth has
    passed and CloudFormation has started - which cost a full deploy-rollback cycle here. Structurally
    valid template, wrong value; exactly the class of defect synth cannot see."""
    import re
    allowed = re.compile(r"^[a-zA-Z0-9. _\-:/()#,@\[\]+=&;{}!$*]{0,255}$")
    template = synth()
    descriptions = []
    for r in template["Resources"].values():
        if r["Type"] == "AWS::EC2::SecurityGroup":
            for rule in (r["Properties"].get("SecurityGroupIngress") or []) + \
                        (r["Properties"].get("SecurityGroupEgress") or []):
                descriptions.append(rule.get("Description", ""))
        elif r["Type"] in ("AWS::EC2::SecurityGroupIngress", "AWS::EC2::SecurityGroupEgress"):
            descriptions.append(r["Properties"].get("Description", ""))
    assert descriptions
    bad = [d for d in descriptions if not allowed.match(d)]
    assert not bad, bad


def test_the_root_volume_is_encrypted():
    """Accounts that enforce EBS encryption with an SCP refuse the launch otherwise, and the failure
    surfaces as an ASG activity error rather than at synth."""
    lt = only(synth(), "AWS::EC2::LaunchTemplate")["LaunchTemplateData"]
    assert lt["BlockDeviceMappings"][0]["Ebs"]["Encrypted"] is True


def test_a_repeated_zone_is_rejected_even_when_two_distinct_ones_exist():
    """[a, a, b] passed the distinct-count check and failed at subnet creation minutes later."""
    with pytest.raises(ConfigError, match="repeats"):
        synth(availabilityZones=["us-west-2a", "us-west-2a", "us-west-2b"])


def test_image_layers_reach_s3_through_a_free_gateway_endpoint_not_the_nat():
    """ECR keeps layers in S3. Eight GB per new instance through the NAT gateway is paid data
    processing for nothing."""
    template = synth()
    endpoints = [r["Properties"] for r in template["Resources"].values()
                 if r["Type"] == "AWS::EC2::VPCEndpoint"]
    assert len(endpoints) == 1
    assert endpoints[0]["VpcEndpointType"] == "Gateway"
    assert "s3" in str(endpoints[0]["ServiceName"])


def test_tasks_cannot_reach_the_instance_metadata_service():
    """Task credentials come from the ECS endpoint. Instance credentials must not be reachable from a
    container, and the setting has to be explicit rather than a side effect of the IMDSv2 hop limit."""
    lt = only(synth(), "AWS::EC2::LaunchTemplate")["LaunchTemplateData"]
    assert "ECS_AWSVPC_BLOCK_IMDS=true" in str(lt["UserData"])


@pytest.mark.parametrize("bad", ["short", "has*wildcard-in-it-1234", "has?query-mark-1234567",
                                 "x" * 122, "has space in it 12345"])
def test_an_api_key_the_load_balancer_would_mismatch_is_rejected_at_synth(bad):
    """ALB header conditions treat * and ? as wildcards, cap values at 128 characters and match
    case-insensitively. A key containing * became a prefix match; a long one failed at deploy."""
    with pytest.raises(ConfigError, match="apiKey"):
        synth(apiKey=bad)


def _run_entrypoint(tmp_path, **env):
    """Run container/serve with a fake `vllm` on PATH that records its argv, NUL-separated."""
    import subprocess
    fake = tmp_path / "bin"; fake.mkdir(exist_ok=True)
    (fake / "vllm").write_text('#!/bin/bash\nprintf "%s\\0" "$@" > "$ARGV_OUT"\n')
    (fake / "vllm").chmod(0o755)
    out = tmp_path / "argv"
    full = {"PATH": f"{fake}:{os.environ['PATH']}", "ARGV_OUT": str(out),
            "SHARED_CACHE": str(tmp_path / "cache"), "MODEL_ID": "org/model", **env}
    r = subprocess.run(["bash", os.path.join(HERE, "..", "container", "serve")],
                       env=full, capture_output=True, text=True)
    argv = out.read_text().split("\0")[:-1] if out.exists() else []
    return r.returncode, argv, r.stderr


def test_extra_args_reach_the_engine_verbatim_with_shell_quoting(tmp_path):
    """`ARGS+=(${EXTRA_ARGS})` word-split and globbed. shlex keeps a quoted JSON value whole, including
    a newline inside it, and a * is not expanded."""
    code, argv, _ = _run_entrypoint(
        tmp_path, EXTRA_ARGS="""--speculative-config '{"method":"eagle3",\n"n":3}' --pattern '*'""")
    assert code == 0
    i = argv.index("--speculative-config")
    assert argv[i + 1] == '{"method":"eagle3",\n"n":3}'
    assert argv[argv.index("--pattern") + 1] == "*"


def test_unparseable_extra_args_stop_the_start_instead_of_being_dropped(tmp_path):
    """The old mapfile-from-process-substitution swallowed the shlex error and started the engine
    without the flags, so a fix looked applied and was not."""
    code, argv, err = _run_entrypoint(tmp_path, EXTRA_ARGS="--unterminated 'quote")
    assert code != 0 and argv == [] and "not shell-parseable" in err


def test_whitespace_only_extra_args_add_nothing(tmp_path):
    code, argv, _ = _run_entrypoint(tmp_path, EXTRA_ARGS="   ")
    assert code == 0 and "" not in argv


def test_the_entrypoint_starts_when_extra_args_is_unset(tmp_path):
    """The stack sets EXTRA_ARGS only when extraArgs is configured, so this is the default deployment.
    `${EXTRA_ARGS// /}` under `set -u` aborted with 'unbound variable' in bash 4 and 5; the three
    tests above all set the variable, and the local bash 3.2 tolerated it, so nothing caught it."""
    code, argv, err = _run_entrypoint(tmp_path)   # no EXTRA_ARGS at all
    assert code == 0, err
    assert argv[:2] == ["serve", "--model"] or "--model" in argv


def test_an_empty_last_extra_arg_is_kept(tmp_path):
    """"\\0".join() leaves the last element unterminated, so mapfile dropped a trailing empty
    argument: --served-model-name "" reached the engine as --served-model-name alone."""
    code, argv, _ = _run_entrypoint(tmp_path, EXTRA_ARGS='--served-model-name ""')
    assert code == 0 and argv[-2:] == ["--served-model-name", ""]


def test_config_edge_cases_are_config_errors_not_tracebacks():
    """Each of these reached a traceback or a broken task definition from the correctness review:
    an ARN where a secret name belongs, a parameter count that overflows, padding in quantization."""
    with pytest.raises(ConfigError, match="NAME, not its ARN"):
        synth(hfTokenSecretName="arn:aws:secretsmanager:us-west-2:111122223333:secret:hf-token-AbCdEf")
    with pytest.raises(ConfigError, match="at most"):
        synth(estimatedParamsBillions=1e300)
    env = {e["Name"]: e["Value"] for e in
           vllm_container(synth(quantization="  "))["Environment"]}
    assert "QUANTIZATION" not in env, "blank quantization means none, not a flag with spaces in it"
    # A YAML list is the natural spelling for extraArgs and str(list) sent vLLM `['--a',` and `1]`;
    # `quantization: false` shipped `--quantization False`.
    with pytest.raises(ConfigError, match="extraArgs must be a string"):
        synth(extraArgs=["--data-parallel-size", "2"])
    with pytest.raises(ConfigError, match="quantization must be a string"):
        synth(quantization=False)


def test_instance_draining_is_ecs_managed_with_no_lambda_hook():
    """CDK's default adds a Lambda-backed lifecycle hook that ECS managed draining makes redundant,
    and whose log group outlives cdk destroy."""
    template = synth()
    assert not [r for r in template["Resources"].values() if r["Type"] == "AWS::Lambda::Function"]
    provider = only(template, "AWS::ECS::CapacityProvider")["AutoScalingGroupProvider"]
    assert provider["ManagedDraining"] == "ENABLED"


def test_the_engine_never_blocks_on_logging_and_the_sidecar_restarts():
    defs = only(synth(), "AWS::ECS::TaskDefinition")["ContainerDefinitions"]
    vllm = next(c for c in defs if c["Name"] == "vllm")
    assert vllm["LogConfiguration"]["Options"]["mode"] == "non-blocking"
    metrics = next(c for c in defs if c["Name"] == "metrics")
    assert metrics["RestartPolicy"]["Enabled"] is True


def test_a_redeploy_may_take_every_engine_down_rather_than_hang():
    """At the default MinimumHealthyPercent 100 a fixed fleet has no free GPU for the new engine, so a
    deployment never placed a task and hung for hours. 0 lets ECS stop engines to make room; the README
    documents the outage this costs and the headroom that avoids it."""
    service = only(synth(), "AWS::ECS::Service")
    assert service["DeploymentConfiguration"]["MinimumHealthyPercent"] == 0
    assert "DeploymentCircuitBreaker" not in service["DeploymentConfiguration"]


def test_cumulative_engine_metrics_become_per_minute_deltas():
    """The engine's histograms and counters are cumulative since start. Exported as-is, "preemptions per
    minute" summed lifetime totals and the request averages were lifetime averages."""
    cfg = collector_config(synth())
    assert "cumulativetodelta" in cfg and "[filter/shortlist, transform/bands, cumulativetodelta]" in cfg
    deltas = cfg.split("cumulativetodelta:")[1].split("exporters:")[0]
    for name in ("vllm:num_preemptions_total", "vllm:time_to_first_token_seconds",
                 "vllm:e2e_request_latency_seconds", "vllm:request_prompt_tokens",
                 "vllm:request_generation_tokens"):
        assert name in deltas, name
    # The gauges are derived by position from ENGINE_METRICS; a gauge routed through the delta step would
    # read as "change in queue depth per minute" and look like an idle fleet.
    for gauge in ("vllm:num_requests_waiting", "vllm:num_requests_running", "vllm:kv_cache_usage_perc"):
        assert gauge not in deltas, gauge


def test_the_latency_widget_says_what_the_load_balancer_times():
    """A tester asked whether the graph was time to first token or to the last. It is both, depending
    on whether the call streams, and the title has to say so."""
    titles = [w["properties"].get("title", "") for w in dashboard_widgets(synth()) if w["type"] == "metric"]
    assert any("first token if streaming" in t for t in titles), titles
    assert any(t.startswith("How long does a request take inside the engine?") for t in titles)
    assert any(t.startswith("What shape are the requests being served?") for t in titles)


def test_request_size_bands_and_prefix_cache_hit_rate_reach_the_dashboard():
    """The engine's token histograms lose their buckets on the way to CloudWatch, so the collector
    scrapes them a second time as plain per-bucket counters with an `le` dimension. The widget stacks
    adjacent-bucket differences; the prefix cache widget divides hits by queries."""
    cfg = collector_config(synth())
    assert "job_name: bands" in cfg and "transform/bands" in cfg
    assert 'replacement: "vllm:request_prompt_tokens_le"' in cfg
    assert "500.0|1000.0|2000.0|5000.0|10000.0|20000.0|\\\\+Inf" in cfg, "the shipped prompt bands, and +Inf escaped for the regex"
    assert '- dimensions: [["le"]]' in cfg and "${" not in cfg, "no ${...}: CloudFormation and the collector both expand it"
    widgets = {w["properties"].get("title", ""): w["properties"] for w in dashboard_widgets(synth()) if w["type"] == "metric"}
    prompts = next(p for t, p in widgets.items() if t.startswith("What size are the prompts?"))
    assert prompts["stacked"] is True
    expressions = [m[0]["expression"] for m in prompts["metrics"] if isinstance(m[0], dict) and "expression" in m[0]]
    assert expressions == ["b0", "b1 - b0", "b2 - b1", "b3 - b2", "b4 - b3", "b5 - b4", "b6 - b5"]
    cache = next(p for t, p in widgets.items() if t.startswith("Is the prefix cache paying off?"))
    assert any(isinstance(m[0], dict) and m[0].get("expression") == "100 * hits / queries" for m in cache["metrics"])


def test_configured_token_bands_shape_both_the_collector_and_the_widget():
    """The same edges must reach the scrape filter (which buckets are kept) and the widget (which
    differences are drawn); a mismatch would draw bands from buckets that never arrive."""
    template = synth(promptTokenBands=[1000, 10000], outputTokenBands=[200])
    cfg = collector_config(template)
    assert "vllm:request_prompt_tokens_bucket;(1000.0|10000.0|\\\\+Inf)" in cfg
    assert "vllm:request_generation_tokens_bucket;(200.0|\\\\+Inf)" in cfg
    widgets = {w["properties"].get("title", ""): w["properties"] for w in dashboard_widgets(template) if w["type"] == "metric"}
    prompts = next(p for t, p in widgets.items() if t.startswith("What size are the prompts?"))
    labels = [m[0]["label"] for m in prompts["metrics"] if isinstance(m[0], dict) and "expression" in m[0]]
    assert labels == ["up to 1,000", "1,000 to 10,000", "10,000 and more"]
    answers = next(p for t, p in widgets.items() if t.startswith("How long are the answers?"))
    assert [m[0]["label"] for m in answers["metrics"] if isinstance(m[0], dict) and "expression" in m[0]] == ["up to 200", "200 and more"]
    with pytest.raises(ConfigError, match="promptTokenBands"):
        synth(promptTokenBands=[8000])


def test_extra_env_reaches_the_engine_and_cannot_shadow_the_stack_variables():
    """A model card's recipe set VLLM_USE_FLASHINFER_MOE_FP4=0 to dodge a faulting kernel; before this
    the only way to set an engine variable was to rebuild the image."""
    env = {e["Name"]: e["Value"] for e in vllm_container(synth(extraEnv={"VLLM_USE_FLASHINFER_MOE_FP4": "0",
                                                                          "VLLM_LOGGING_LEVEL": "DEBUG",
                                                                          "COUNT": 3}))["Environment"]}
    assert env["VLLM_USE_FLASHINFER_MOE_FP4"] == "0" and env["VLLM_LOGGING_LEVEL"] == "DEBUG" and env["COUNT"] == "3"
    assert "MODEL_ID" in env, "the stack's own variables are still there"
    with pytest.raises(ConfigError, match="already sets"):
        synth(extraEnv={"MODEL_ID": "x"})
    with pytest.raises(ConfigError, match="not a valid variable name"):
        synth(extraEnv={"BAD-NAME": "x"})
    with pytest.raises(ConfigError, match="extraEnv must be a"):
        synth(extraEnv=["A=B"])
    assert "EXTRA_ENV" not in {e["Name"] for e in vllm_container(synth())["Environment"]}, "empty adds nothing"


def test_data_parallel_reaches_the_container_as_a_gpu_reservation_and_a_flag():
    """dp attention groups need tp x dp GPUs in one container, and the entrypoint passes the size on."""
    c = vllm_container(synth(instanceType="p5.48xlarge", estimatedParamsBillions=235, quantization="fp8",
                             tuning={"tensorParallel": 4, "dataParallel": 2, "enableExpertParallel": True}))
    env = {e["Name"]: e["Value"] for e in c["Environment"]}
    assert env["DATA_PARALLEL"] == "2" and env["TENSOR_PARALLEL"] == "4"
    gpu = [r for r in c["ResourceRequirements"] if r["Type"] == "GPU"][0]
    assert gpu["Value"] == "8"


def test_the_entrypoint_passes_data_parallel_only_when_above_one(tmp_path):
    code, argv, _ = _run_entrypoint(tmp_path, DATA_PARALLEL="2", TENSOR_PARALLEL="4")
    assert code == 0 and argv[argv.index("--data-parallel-size") + 1] == "2"
    code, argv, _ = _run_entrypoint(tmp_path, DATA_PARALLEL="1")
    assert code == 0 and "--data-parallel-size" not in argv


def test_sticky_sessions_is_a_cookie_on_the_target_group_and_off_by_default():
    """Prefix caching is per engine; a load balancer cookie keeps a conversation on the engine that holds
    its prefix. Off by default: one upstream client would pin everything to one engine."""
    tg = only(synth(stickySessions=True), "AWS::ElasticLoadBalancingV2::TargetGroup")
    attrs = {a["Key"]: a["Value"] for a in tg["TargetGroupAttributes"]}
    assert attrs["stickiness.enabled"] == "true" and attrs["stickiness.type"] == "lb_cookie"
    assert attrs["stickiness.lb_cookie.duration_seconds"] == "3600"
    tg = only(synth(), "AWS::ElasticLoadBalancingV2::TargetGroup")
    attrs = {a["Key"]: a["Value"] for a in tg.get("TargetGroupAttributes", [])}
    assert attrs.get("stickiness.enabled", "false") == "false"
    with pytest.raises(ConfigError):
        synth(stickySessions="yes please")


def test_the_measured_thresholds_and_timeouts_are_the_ones_deployed():
    """Every value here was set after a failure: the health check that marked a loading engine failed,
    the alarm window that cried wolf during deploys, the volume throughput that made loading take
    eight minutes, the termination protection that held instances for an hour. A template change that
    nudges one of them must fail a test, not a deployment."""
    t = synth(latencyAlarmSeconds=8)
    tg = only(t, "AWS::ElasticLoadBalancingV2::TargetGroup")
    assert (tg["HealthCheckIntervalSeconds"], tg["HealthCheckTimeoutSeconds"],
            tg["HealthyThresholdCount"], tg["UnhealthyThresholdCount"]) == (30, 10, 2, 5)
    svc = only(t, "AWS::ECS::Service")
    assert svc["HealthCheckGracePeriodSeconds"] == 1800
    assert only(t, "AWS::Logs::LogGroup")["RetentionInDays"] == 7
    lt = only(t, "AWS::EC2::LaunchTemplate")["LaunchTemplateData"]
    ebs = lt["BlockDeviceMappings"][0]["Ebs"]
    assert (ebs["VolumeSize"], ebs["Throughput"], ebs["Iops"], ebs["Encrypted"]) == (500, 500, 6000, True)
    assert lt["MetadataOptions"]["HttpTokens"] == "required"
    cp = only(t, "AWS::ECS::CapacityProvider")["AutoScalingGroupProvider"]
    assert cp["ManagedTerminationProtection"] == "DISABLED"
    alarms = {a["Properties"]["AlarmName"]: a["Properties"] for a in t["Resources"].values()
              if a["Type"] == "AWS::CloudWatch::Alarm"}
    by_suffix = {k.split("-", 1)[1] if "-" in k else k: v for k, v in alarms.items()}
    unhealthy = next(v for k, v in alarms.items() if k.endswith("engines-unhealthy"))
    erroring = next(v for k, v in alarms.items() if k.endswith("load-balancer-erroring"))
    slow = next(v for k, v in alarms.items() if k.endswith("too-slow"))
    assert (unhealthy["Threshold"], unhealthy["EvaluationPeriods"]) == (0, 15)
    assert (erroring["Threshold"], erroring["EvaluationPeriods"]) == (10, 2)
    assert (slow["Threshold"], slow["EvaluationPeriods"]) == (8, 3)
    roles = [r["Properties"] for r in t["Resources"].values() if r["Type"] == "AWS::IAM::Role"]
    assert any("AmazonSSMManagedInstanceCore" in json.dumps(r.get("ManagedPolicyArns", [])) for r in roles)


def test_dev_shm_is_half_the_container_memory_with_an_8_gib_floor():
    """A fixed 8 GiB /dev/shm refused to start the engine with a CPU KV offload buffer configured:
    `Insufficient space in /dev/shm: 32768 MiB required, 8192 MiB free`. vLLM puts both the
    tensor-parallel broadcast buffers and the offload region there, so it follows the container's memory."""
    lp = [r["Properties"] for r in synth(instanceType="g7e.4xlarge")["Resources"].values()
          if r["Type"] == "AWS::ECS::TaskDefinition"][0]["ContainerDefinitions"]
    vllm = next(c for c in lp if c["Name"] == "vllm")
    assert vllm["Memory"] == 78643 and vllm["LinuxParameters"]["SharedMemorySize"] == 39321
    small = [r["Properties"] for r in synth(instanceType="g7e.2xlarge")["Resources"].values()
             if r["Type"] == "AWS::ECS::TaskDefinition"][0]["ContainerDefinitions"]
    assert next(c for c in small if c["Name"] == "vllm")["LinuxParameters"]["SharedMemorySize"] == 19660
