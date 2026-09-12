"""One GPU serving cluster: VPC, ECS with a GPU capacity provider, an internal ALB behind CloudFront,
and a vLLM service.

One stack. It is small enough that splitting it would add navigation cost without
buying anything, and a single `cdk destroy` then removes everything. `__init__` builds it in one pass;
the banner comments mark the sections.

WHY EC2 AND NOT FARGATE
    Fargate has no GPU support. GPU containers on ECS must run on EC2, so the cluster needs an Auto
    Scaling Group and an ECS capacity provider.

HOW REQUESTS ARE AUTHENTICATED
    An ALB cannot validate API keys natively, so the listener's default action is a hard 403 and
    traffic reaches the target group only if the request carries `Authorization: Bearer <key>`, the
    header every OpenAI-compatible client and gateway already sends. What
    that key is and is not is explained where it is built, and in README.md.

HOW THE ENDPOINT GETS HTTPS WITHOUT A DOMAIN
    CloudFront, with its own *.cloudfront.net name and certificate, reaching the load balancer through
    a VPC origin. The load balancer is internal and never sees the internet; CloudFront is the only way
    in. Nothing to own, nothing to validate, nothing to renew.
"""

from __future__ import annotations

import re
import sys

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Size, Stack
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_sns as sns
from constructs import Construct

from hardware import (DEFAULT_OUTPUT_TOKEN_BANDS, DEFAULT_PROMPT_TOKEN_BANDS, ROOT_VOLUME_GIB,
                      ROOT_VOLUME_IOPS, ROOT_VOLUME_THROUGHPUT_MBPS, ConfigError, _flag, _given, _list,
                      _mapping, _num, bytes_per_param_for, get_instance, memory_pressure_warning, model_bytes,
                      resolve_topology, validate_token_bands)

CONTAINER_PORT = 8080

# How long CloudFront waits for the first byte of a response, and then between bytes. 120 s is the
# most the account quota allows without a support request. A STREAMED answer never gets near it -
# tokens arrive continuously. A non-streamed answer has to finish inside it, whole.
CLOUDFRONT_READ_TIMEOUT_S = 120

# Where the container keeps downloaded weights, and the host directory backing it. Must match the
# path used by container/serve.
CONTAINER_MODEL_PATH = "/opt/model"
HOST_MODEL_CACHE = "/opt/modelcache"

# The metrics sidecar: the upstream OpenTelemetry collector (contrib build), run unmodified from the
# public image with its configuration passed inline. Upstream rather than the AWS distribution because
# the request-size bands below need its `transform` processor, which the AWS build does not ship (tested
# against v0.43.3 and latest). Pinned, because a floating tag would change what the dashboard reads
# without any change in this repository.
METRICS_SIDECAR_IMAGE = ("ghcr.io/open-telemetry/opentelemetry-collector-releases/"
                         "opentelemetry-collector-contrib:0.135.0")
METRICS_SIDECAR_MEMORY_MIB = 256
# The engine metrics the dashboard reads. Ten of the ~86 families the engine exposes; the rest are
# either derivable from these or duplicated by the load balancer. Custom metrics are billed per name,
# so the shortlist is also the bill.
ENGINE_METRICS = (
    "vllm:num_requests_waiting",   # queued but not running - saturation, before latency shows it
    "vllm:num_requests_running",   # sequences in the batch, against maxNumSeqs
    "vllm:kv_cache_usage_perc",    # 0..1; near 1 means preemption is next
    "vllm:num_preemptions_total",  # counter; anything above zero is the engine going backwards
    # Histograms. The collector hands CloudWatch their sum and count, not their buckets (tested: the
    # EMF record is {Sum, Count} even with detailed_metrics on), so these give averages per minute and
    # no percentiles. Latency percentiles are the load balancer's response-time widget.
    "vllm:time_to_first_token_seconds",   # queue wait plus prefill
    "vllm:e2e_request_latency_seconds",   # whole request, as the engine saw it
    "vllm:request_prompt_tokens",         # the input shape actually being served
    "vllm:request_generation_tokens",     # the output shape
    "vllm:prefix_cache_queries_total",    # prompt tokens looked up in the prefix cache
    "vllm:prefix_cache_hits_total",       # of those, found; hits / queries is the hit rate
)
# Cumulative since engine start. Converted to per-scrape deltas in the collector, so a minute on the
# dashboard means that minute: without it "preemptions per minute" summed lifetime totals.
ENGINE_CUMULATIVE = ENGINE_METRICS[3:]


class ServingStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, cfg: dict, **kw) -> None:
        super().__init__(scope, cid, **kw)

        inst = get_instance(cfg["instanceType"])
        # Resolved ONCE, before anything reads it. As a bare cfg.get("useSpot") a quoted "false" from
        # YAML was truthy everywhere it was read - silently turning spot ON for someone who had
        # written it off.
        use_spot = _flag(cfg.get("useSpot"), "useSpot")
        sticky_sessions = _flag(cfg.get("stickySessions"), "stickySessions")

        # The model's parameter count is not knowable at synth time without a network call, so it is
        # an ESTIMATE, defaulting to a size that suits a single 96 GiB GPU. `estimatedParamsBillions`
        # in config.yaml overrides it, and getting it wrong matters: it is what the tensor-parallel
        # degree and the memory-pressure check are derived from, so a 70B model left at the default
        # derives TP=1, synthesises cleanly, deploys, and then runs out of VRAM while loading.
        # minimum, because a finite-but-nonsensical size passed silently: a negative parameter count
        # produced negative weight bytes and derived a happy TP=1.
        # maximum: 1e300 passed the finiteness check and overflowed to infinity in model_bytes, which
        # escaped as an OverflowError traceback instead of a ConfigError.
        est_params_b = _num(_given(cfg.get("estimatedParamsBillions"), 30),
                            "estimatedParamsBillions", float, minimum=0.001, maximum=100_000)
        # Considers the MODEL ID as well as `quantization`: a publisher's official fp8 build is
        # already quantised on disk and correctly leaves `quantization` empty, so keying off that
        # alone doubles the estimated weight size and misreads a correct configuration.
        # Stripped once, here, so a padded or non-string value cannot reach the task definition as-is:
        # `quantization: "  "` shipped `--quantization "  "` while the size estimate treated it as bf16.
        for key in ("modelId", "quantization", "extraArgs", "toolCallParser", "reasoningParser"):
            if cfg.get(key) is not None and not isinstance(cfg[key], str):
                raise ConfigError(
                    f"{key} must be a string (got {type(cfg[key]).__name__}: {cfg[key]!r})."
                    + ("\n  extraArgs is one line, quoted like a shell command, not a YAML list." if key == "extraArgs" else "")
                )
        cfg["modelId"] = cfg["modelId"].strip()
        cfg["quantization"] = _given(cfg.get("quantization"), "").strip()
        bytes_per_param = bytes_per_param_for(cfg["modelId"], cfg["quantization"])
        est_weight_bytes = model_bytes(est_params_b, bytes_per_param)

        # Resolve tensorParallel and replicas together - they are one decision about how the
        # instance's GPUs are divided into engines, and the default is one engine per GPU so that a
        # multi-GPU instance uses all of them without the user computing anything.
        tuning = resolve_topology(inst, cfg.get("tuning"),
                                  est_weight_bytes)

        # Print the assumption, on stderr, because a silent wrong estimate is the failure above.
        print(f"assuming ~{est_params_b:g}B parameters at {bytes_per_param:g} byte(s) each = "
              f"{est_weight_bytes / 1024 ** 3:.1f} GiB of weights "
              f"-> tensorParallel={tuning['tensorParallel']}, replicas={tuning['replicas']}.\n"
              f"  Set estimatedParamsBillions in config.yaml if that is not your model's size.",
              file=sys.stderr)

        warning = memory_pressure_warning(est_weight_bytes // tuning["tensorParallel"], tuning,
                                          quantised=bytes_per_param < 2.0, gpu_vram_gib=inst.gpu_vram_gib)
        if warning:
            # stderr, because `cdk synth > template.yaml` is normal and stdout carries the template.
            print(f"\nWarning: {warning}\n", file=sys.stderr)

        # ------------------------------------------------------------------ networking
        # Always its own VPC. Not every region has a default VPC - us-east-2 does not - so relying on
        # one makes the stack undeployable in exactly the regions most likely to have spare GPU
        # capacity.
        #
        # By default it spans up to FOUR AZs, and that is a capacity decision rather than a resilience
        # one: GPU capacity is allocated per availability zone and the scarce shapes are unavailable
        # in most of them at any moment, so more zones means more places the ASG can look.
        #
        # But an AZ that does not OFFER the instance type is worse than useless - the ASG will pick it
        # and the launch fails with `Unsupported`, which is permanent rather than transient, so it does
        # not resolve by retrying. Not hypothetical: us-east-2 has three AZs and offers g7e in two.
        # Set `availabilityZones` to the offering zones (config.yaml has the command to find them) and
        # the VPC, and therefore the ASG, is confined to them.
        configured_azs = [z.strip() for z in _list(cfg.get("availabilityZones"), "availabilityZones")
                          if z.strip()]
        # DISTINCT zones, not entries. `["us-west-2a", "us-west-2a"]` is an easy thing to end up with
        # while editing a list, and counting entries passed it - then CloudFormation rejected two
        # subnets in one AZ minutes into the deploy, which is exactly what this check exists to
        # forestall.
        if configured_azs and (len(set(configured_azs)) < 2
                               or len(set(configured_azs)) != len(configured_azs)):
            raise ConfigError(
                f"availabilityZones needs at least two zones with no repeats (got {configured_azs}).\n"
                "  A load balancer requires subnets in two, even when the GPU instances only ever\n"
                "  land in one of them."
            )
        vpc_kwargs = dict(
            ip_addresses=ec2.IpAddresses.cidr("10.30.0.0/16"),
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC,
                                        cidr_mask=20),
                ec2.SubnetConfiguration(name="private", cidr_mask=20,
                                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            ],
        )
        if configured_azs:
            vpc_kwargs["availability_zones"] = configured_azs
        else:
            vpc_kwargs["max_azs"] = 4
        vpc = ec2.Vpc(self, "Vpc", **vpc_kwargs)
        # Image layers come from S3 (ECR stores them there). Without this, every pull of the ~8 GB
        # image on every new instance crosses the NAT gateway at data-processing rates. The gateway
        # endpoint is free. Model weights come from Hugging Face and still use the NAT.
        vpc.add_gateway_endpoint("S3", service=ec2.GatewayVpcEndpointAwsService.S3)

        cluster = ecs.Cluster(self, "Cluster", vpc=vpc)

        # ------------------------------------------------------- GPU capacity provider
        instance_role = iam.Role(
            self, "InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonEC2ContainerServiceforEC2Role"),
                # Lets you open a shell on the host without SSH or a bastion, which is how you
                # inspect the NVIDIA driver or the ECS agent log when something is wrong.
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
            ],
        )

        # An EXPLICIT launch template rather than letting the ASG construct create one.
        #
        # Two reasons. First, CDK's AutoScalingGroup creates an AWS::AutoScaling::LaunchConfiguration
        # by default, and launch configurations do not support MixedInstancesPolicy - which is the
        # only way to request spot. Second, an explicit template means the ASG's exact configuration
        # is visible here instead of depending on the construct's internal child naming, which has
        # changed between CDK versions.
        gpu_user_data = ec2.UserData.for_linux()
        # No ECS_ENABLE_SPOT_INSTANCE_DRAINING here: with managed draining on the capacity provider
        # below, ECS drains the instance itself on the two-minute spot reclaim notice, and AWS documents
        # the agent setting as redundant.
        # Tasks get credentials from the ECS credential endpoint, never from the instance's IMDS. Block
        # IMDS for awsvpc tasks explicitly rather than relying on the IMDSv2 hop limit to do it by
        # accident (the disableEcsImdsBlocking flag in cdk.json turns off the construct's own blocking).
        gpu_user_data.add_commands("echo ECS_AWSVPC_BLOCK_IMDS=true >> /etc/ecs/ecs.config")

        launch_template = ec2.LaunchTemplate(
            self, "GpuLaunchTemplate",
            instance_type=ec2.InstanceType(inst.name),
            # Amazon Linux 2023, NOT Amazon Linux 2 - a hard requirement. The AL2 GPU AMI's NVIDIA
            # driver is too old for this GPU generation, and it fails silently: the instance boots,
            # passes every health check, and registers with an EMPTY GPU list, so tasks sit in
            # PROVISIONING forever with no error anywhere. Resolved from an SSM parameter, so no
            # custom AMI and no per-region id. See docs/troubleshooting.md for the diagnostic.
            machine_image=ecs.EcsOptimizedImage.amazon_linux2023(ecs.AmiHardwareType.GPU),
            role=instance_role,
            security_group=ec2.SecurityGroup(self, "InstanceSg", vpc=vpc,
                                             description="GPU serving instances"),
            user_data=gpu_user_data,
            require_imdsv2=True,
            block_devices=[ec2.BlockDevice(
                device_name="/dev/xvda",
                volume=ec2.BlockDeviceVolume.ebs(
                    # Sizing and throughput are explained on the constants in hardware.py.
                    ROOT_VOLUME_GIB,
                    volume_type=ec2.EbsDeviceVolumeType.GP3,
                    throughput=ROOT_VOLUME_THROUGHPUT_MBPS,
                    iops=ROOT_VOLUME_IOPS,
                    # Encrypted with the account's default EBS key. Many accounts enforce this with
                    # an SCP, and an unencrypted volume then fails the launch with no obvious reason.
                    encrypted=True,
                    delete_on_termination=True),
            )],
        )

        # Fleet size, resolved once. `maxInstanceCount` doubles as the autoscaling on/off switch:
        # anything above `instanceCount` creates a scaling policy, equal or unset pins the fleet.
        # `_given`, not `or`: a key written as `instanceCount:` with nothing after it parses to None,
        # which satisfies .get() and then fails on int(None), so the default has to cover a blank
        # value - but 0 is falsy and must NOT be swallowed by it.
        # ZERO is allowed and meaningful: it is the durable way to stop paying while keeping the
        # stack, endpoint and configuration. Scaling to zero with the CLI is drift that the next
        # `cdk deploy` reverts - MinSize, DesiredCount and the scalable floor are all re-established
        # from the template - so config is the only place a zero fleet survives a deploy.
        instance_count = _num(_given(cfg.get("instanceCount"), 1), "instanceCount", minimum=0)
        configured_max = _num(_given(cfg.get("maxInstanceCount"), instance_count),
                              "maxInstanceCount", minimum=0)
        if instance_count == 0 and configured_max > 0:
            # A parked fleet has to be a FIXED-size fleet of zero, and this combination is neither
            # parked nor autoscaling. It leaves MaxSize above zero and DesiredCapacity absent, so the
            # deploy never lowers ASG capacity and the shutdown falls back on the ~15-minute managed
            # scale-in below. Worse, it cannot come back: with no tasks the target group publishes no
            # ALBRequestCountPerTarget datapoints at all, and target tracking does not scale out on a
            # missing metric - nor on a metric below its target, which is the other half of the trap.
            raise ConfigError(
                f"instanceCount is 0 but maxInstanceCount is {configured_max}.\n"
                "  Parking the deployment means a fixed-size fleet of zero, so set BOTH to 0:\n"
                "      instanceCount: 0\n"
                "      maxInstanceCount: 0\n"
                "  Left as it is, autoscaling owns a fleet it can never grow: with no tasks running\n"
                "  there is no request-rate metric to scale on, and the empty fleet would not be\n"
                "  durable either - MaxSize stays above zero and the deploy never lowers capacity."
            )
        if configured_max < instance_count:
            # Otherwise CDK raises a bare jsii RuntimeError about ASG bounds, which reads as an
            # internal fault rather than as the config mistake it is.
            raise ConfigError(
                f"maxInstanceCount ({configured_max}) is below instanceCount ({instance_count}).\n"
                "  maxInstanceCount is the autoscaling CEILING, so it has to be at least the floor.\n"
                "  Set them equal for a fixed-size fleet, or raise it to allow scale-out."
            )
        autoscaling_enabled = configured_max > instance_count

        asg = autoscaling.AutoScalingGroup(
            self, "GpuAsg",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            launch_template=launch_template,
            # MinSize IS the floor, and that solves two problems at once.
            #
            # An ASG created without DesiredCapacity takes MinSize as its initial desired capacity. So
            # with min_capacity=instanceCount a FIRST deploy comes up at the right size with nothing
            # else needing to intervene - where min_capacity=0 plus an omitted DesiredCapacity would
            # come up with zero instances and wait for something to scale it, which for a brand-new
            # scalable target is not documented to happen (AWS documents min/max enforcement for
            # UPDATING an existing scalable target, not for registering one).
            #
            # And because MinSize is a floor rather than a target, a later `cdk deploy` does NOT pull a
            # scaled-out fleet back down: CloudFormation leaves DesiredCapacity alone when it is absent
            # from the template. That is the important half - resetting desired capacity terminates
            # instances that still hold tasks and tens of GiB of loaded weights.
            #
            # `instanceCount` is documented as the minimum, so a floor is also what it should mean.
            min_capacity=instance_count,
            max_capacity=configured_max,
            # Declared only when nothing else manages the fleet, so a hand-scaled ASG returns to the
            # configured size on the next deploy. (CDK warns about that reset; it is the intent.)
            **({} if autoscaling_enabled else {"desired_capacity": instance_count}),
            # An ASG publishes NOTHING about its own size unless group metrics are enabled - the
            # `GroupInServiceInstances` / `GroupDesiredCapacity` metrics simply do not exist until
            # then, so a dashboard widget built on them renders an empty graph and looks like a broken
            # fleet rather than a missing setting. Enabling them is free, and unconditional because
            # they are the only historical record of how big the fleet was at a given moment; that is
            # the first thing anyone asks when reading a latency spike after the fact.
            group_metrics=[autoscaling.GroupMetrics(
                autoscaling.GroupMetric.IN_SERVICE_INSTANCES,
                autoscaling.GroupMetric.DESIRED_CAPACITY,
            )],
        )

        if use_spot:
            # Spot is a separate capacity pool from on-demand and often has availability when
            # on-demand does not. It also has a SEPARATE, much smaller quota - see README.md.
            #
            # `capacity-optimized-prioritized` rather than `capacity-optimized`: the latter ignores
            # the override order entirely and launches from whichever pool has the most spare
            # capacity, which silently gives you a different instance type than you asked for.
            cfn_asg = asg.node.default_child
            cfn_asg.add_property_override("MixedInstancesPolicy", {
                "LaunchTemplate": {
                    "LaunchTemplateSpecification": {
                        "LaunchTemplateId": launch_template.launch_template_id,
                        "Version": launch_template.latest_version_number,
                    },
                    "Overrides": [{"InstanceType": inst.name}],
                },
                "InstancesDistribution": {
                    "OnDemandAllocationStrategy": "prioritized",
                    "OnDemandBaseCapacity": 0,
                    "OnDemandPercentageAboveBaseCapacity": 0,
                    "SpotAllocationStrategy": "capacity-optimized-prioritized",
                },
            })
            # The plain LaunchTemplate property and MixedInstancesPolicy are mutually exclusive;
            # CloudFormation rejects a template carrying both.
            cfn_asg.add_property_deletion_override("LaunchTemplate")
            # Start replacing an instance on the rebalance recommendation, which arrives BEFORE the
            # two-minute reclaim notice, so the replacement has a head start on loading weights.
            cfn_asg.add_property_override("CapacityRebalance", True)

        capacity_provider = ecs.AsgCapacityProvider(
            self, "GpuCapacity", auto_scaling_group=asg,
            # Left OFF. Enabled, it stops the ASG terminating an instance that still has
            # tasks - which also means a scale to zero waits on it. The measured ~5 minute
            # teardown depends on this being off.
            enable_managed_termination_protection=False,
            # ECS drains the instance itself when the ASG terminates it. Without this CDK adds its own
            # Lambda-backed lifecycle hook that does the same job, plus a Lambda and a log group that
            # cdk destroy leaves behind.
            enable_managed_draining=True,
        )
        cluster.add_asg_capacity_provider(capacity_provider)

        # ------------------------------------------------------------------ API key
        #
        # It must come from config rather than being generated here, because this runs at SYNTH time:
        # a fresh value would be a NEW key on every `cdk deploy`. app.py generates one once and
        # persists it to config.local.yaml so redeploys are stable.
        #
        # BE CLEAR ABOUT WHAT THIS IS. An ALB listener rule is evaluated by the load balancer, which
        # cannot resolve a Secrets Manager reference, so the condition needs a LITERAL. The key
        # therefore appears in the template and the stack outputs - it is not confidential from anyone
        # who can read the stack. A lightweight gate that keeps unauthenticated
        # traffic off the model, and NOT an authorization layer. README.md has the upgrade path.
        key_value = str(cfg.get("apiKey") or "").strip()
        if not key_value:
            raise ConfigError(
                "apiKey is not set.\n"
                "  Deploy through `cdk deploy` from the infra/ directory and one is generated and\n"
                "  saved to config.local.yaml automatically. If you are calling the stack directly,\n"
                "  pass apiKey in the config - it cannot be generated here, because synth runs on\n"
                "  every deploy and a fresh value would rotate the key each time."
            )
        # The value lands in an ALB header condition, which has rules of its own: 128 characters at most
        # (minus "Bearer "), `*` and `?` are wildcards, and matching is case-insensitive. A key with a
        # `*` would become a prefix match; a long one fails at deploy time after the VPC exists.
        if not re.fullmatch(r"[A-Za-z0-9._~+/=-]{16,121}", key_value):
            raise ConfigError(
                "apiKey must be 16 to 121 characters from A-Z a-z 0-9 . _ ~ + / = -\n"
                "  No * or ? (the load balancer treats them as wildcards) and no spaces. The generated\n"
                "  key satisfies this; if you set your own, keep to that alphabet.")

        # ------------------------------------------------------------------ load balancer
        #
        # INTERNAL, always. Nothing here faces the internet: CloudFront (below) is the only way in, and
        # it reaches this load balancer through a VPC origin - a network interface CloudFront creates
        # inside the private subnets. The hop from CloudFront to here therefore never leaves the AWS
        # network, so a plain HTTP listener is acceptable when a *public* HTTP listener would
        # not be: the API key is a request header, and a header is exactly as private as the transport
        # under it.
        alb = elbv2.ApplicationLoadBalancer(
            self, "Alb", vpc=vpc, internet_facing=False,
            # The ALB's default idle timeout is 60 seconds, and for a NON-streaming generation the
            # engine sends nothing until the whole response is finished - so the idle timer covers the
            # entire generation, not the gaps within it. 300 s here so this hop is never the limit;
            # CloudFront's read timeout (CLOUDFRONT_READ_TIMEOUT_S) is the one that binds.
            idle_timeout=Duration.seconds(300),
        )

        listener = alb.add_listener(
            "Endpoint", port=80, protocol=elbv2.ApplicationProtocol.HTTP,
            # `open=False` so CDK does not add an unconditional 0.0.0.0/0 ingress rule.
            open=False,
            # Anything without a valid key gets a flat 403 and never reaches the model.
            default_action=elbv2.ListenerAction.fixed_response(
                403, content_type="application/json",
                message_body='{"error":"missing or invalid Authorization: Bearer <key> header"}'),
        )
        # Who may reach the load balancer: CloudFront, by its managed prefix list of origin-facing
        # addresses. NOT the VPC CIDR - that was tried first and CloudFront timed out on every connect.
        # A VPC origin's traffic enters through an interface CloudFront places in the private subnets,
        # but the packets carry CloudFront's own source addresses, not the interface's private IP, so
        # a rule on the VPC range never matches. The prefix list is looked up by name so this works in
        # any region (its id differs per region). The result is cached in cdk.context.json, which is
        # gitignored, so each clone performs the lookup once on its first synth.
        # No apostrophes in rule descriptions: EC2 rejects them at deploy time, and synth does not check.
        cloudfront_origins = ec2.PrefixList.from_lookup(
            self, "CloudFrontOriginFacing",
            prefix_list_name="com.amazonaws.global.cloudfront.origin-facing")
        alb.connections.allow_from(ec2.Peer.prefix_list(cloudfront_origins.prefix_list_id),
                                   ec2.Port.tcp(80), "CloudFront origin-facing addresses")

        target_group = elbv2.ApplicationTargetGroup(
            self, "Targets",
            vpc=vpc, port=CONTAINER_PORT, protocol=elbv2.ApplicationProtocol.HTTP,
            # `ip` is required for awsvpc networking, where each task has its own ENI and address.
            target_type=elbv2.TargetType.IP,
            health_check=elbv2.HealthCheck(
                path="/health",
                # Loading a large model takes minutes, during which the container is up but not yet
                # answering. A generous threshold stops the deployment being marked failed while it
                # is legitimately still starting.
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
                healthy_threshold_count=2,
                unhealthy_threshold_count=5,
            ),
            # Long enough for most in-flight generations to finish. The default 30 s (and the container's
            # 30 s stop timeout below) killed any request still generating when a task was replaced -
            # on every scale-in and every deployment. Draining holds the target open for this long; a
            # streamed answer that runs longer than 180 s still dies when its task is replaced.
            deregistration_delay=Duration.seconds(180),
        )
        # Prefix caching is per engine. Round robin sends the next turn of a conversation to a random
        # engine, so on N engines a follow-up hits its cached prefix about 1/N of the time. A load
        # balancer cookie pins a client's session to one engine: measured on eight engines, multi-turn
        # traffic went from a 21% to a 75% hit rate and +14% throughput. Off by default because one
        # upstream client with one cookie jar would pin all of its traffic to a single engine.
        # docs/tuning.md, "Prefix caching is a routing decision".
        if sticky_sessions:
            target_group.enable_cookie_stickiness(Duration.hours(1))

        listener.add_action(
            "AuthedForward", priority=10,
            conditions=[elbv2.ListenerCondition.http_header("Authorization",
                                                            [f"Bearer {key_value}"])],
            action=elbv2.ListenerAction.forward([target_group]),
        )

        # ------------------------------------------------------------------ CloudFront
        #
        # The public HTTPS endpoint, with nothing to own: CloudFront's *.cloudfront.net name and
        # certificate. It is a pass-through, not a cache - every setting below exists to make it get
        # out of the way of an API that streams:
        #
        #   caching disabled            responses are never cached, and POST is never cacheable anyway
        #   all viewer headers forwarded  Authorization must reach the load balancer's rule (Host is
        #                               dropped, because the origin has its own)
        #   all methods                 POST
        #   compression off             a compressed stream is a buffered stream
        #   HTTPS only                  a plain-http call gets a 403 rather than a silent redirect
        #                               that would turn a POST into a GET
        #
        # Streaming works because CloudFront forwards bytes as the origin sends them, and its read
        # timeout resets on every byte. The one limit it imposes is on answers that do NOT stream: the
        # whole response must arrive within CLOUDFRONT_READ_TIMEOUT_S or the caller gets a 504 while
        # the engine is still happily generating. docs/tuning.md covers what that means for sizing.
        distribution = cloudfront.Distribution(
            self, "Cdn",
            comment=f"{self.stack_name}: HTTPS front door for the inference endpoint",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.VpcOrigin.with_application_load_balancer(
                    alb,
                    # Account-wide name, like the dashboard: a second region with CDK's default name
                    # failed with "another vpc origin with the same name already exists". CloudFront
                    # will not rename an origin a distribution uses, so this name must not change again.
                    vpc_origin_name=f"{self.stack_name}-{self.region}",
                    protocol_policy=cloudfront.OriginProtocolPolicy.HTTP_ONLY,
                    http_port=80,
                    read_timeout=Duration.seconds(CLOUDFRONT_READ_TIMEOUT_S),
                    # Reuse connections to the load balancer under load. Must stay below the ALB's
                    # 300 s idle timeout, or CloudFront reuses a connection the ALB has closed.
                    keepalive_timeout=Duration.seconds(60),
                ),
                allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
                compress=False,
            ),
            http_version=cloudfront.HttpVersion.HTTP2_AND_3,
            # CloudFront otherwise caches error responses for 10 s. A 503 while a task restarts would
            # be replayed to every caller for those 10 s, including ones the fleet could have served.
            error_responses=[cloudfront.ErrorResponse(http_status=code, ttl=Duration.seconds(0))
                             for code in (500, 502, 503, 504)],
        )
        endpoint = f"https://{distribution.distribution_domain_name}"

        # ------------------------------------------------------------------ the service
        log_group = logs.LogGroup(self, "Logs", retention=logs.RetentionDays.ONE_WEEK,
                                 removal_policy=RemovalPolicy.DESTROY)

        task_def = ecs.Ec2TaskDefinition(self, "TaskDef", network_mode=ecs.NetworkMode.AWS_VPC)

        # A shared HOST directory for the weights cache. This is required, not an optimisation.
        #
        # Without it, each container start writes its own copy of the weights into the container's
        # writable layer, and a stopped container keeps that layer. A task that crash-loops therefore
        # accumulates a full copy of the model per attempt: a dozen restarts of a ~57 GiB model
        # consumed close to 500 GB of disk, at which point nothing can start and the failure presents
        # as a model load that never completes rather than as a disk error.
        #
        # Mounting a host path instead means one copy on the instance, reused by every task. That
        # also removes download time from every subsequent start, which is minutes per restart.
        task_def.add_volume(name="modelcache",
                            host=ecs.Host(source_path=HOST_MODEL_CACHE))

        # Optional Hugging Face token for gated models, read from a Secrets Manager secret the user
        # created (README, "Gated models"). Passed as an ECS secret, so it reaches the engine as HF_TOKEN
        # and never appears in the template, the outputs or the logs. Without it, public models work and
        # gated ones fail with a 401 while pulling.
        hf_secret_name = str(_given(cfg.get("hfTokenSecretName"), "")).strip()
        if hf_secret_name.startswith("arn:"):
            # from_secret_name_v2 would build a second ARN around this one; the task then fails at
            # start with ResourceNotFound, after a full deploy.
            raise ConfigError(
                "hfTokenSecretName takes the secret's NAME, not its ARN: the part after 'secret:',\n"
                "  without the six-character suffix, e.g. gpu-llm-serving/hf-token.")
        container_secrets = {}
        if hf_secret_name:
            hf_secret = secretsmanager.Secret.from_secret_name_v2(self, "HfToken", hf_secret_name)
            container_secrets["HF_TOKEN"] = ecs.Secret.from_secrets_manager(hf_secret)

        env = {
            "MODEL_ID": cfg["modelId"],
            "PORT": str(CONTAINER_PORT),
            "TENSOR_PARALLEL": str(tuning["tensorParallel"]),
            "DATA_PARALLEL": str(tuning["dataParallel"]),
            # 0 means "omit the flag and let the engine choose" for both of these.
            "MAX_MODEL_LEN": str(tuning["maxModelLen"]),
            "MAX_NUM_BATCHED_TOKENS": str(tuning["maxNumBatchedTokens"]),
            "MAX_NUM_SEQS": str(tuning["maxNumSeqs"]),
            "GPU_MEMORY_UTILIZATION": str(tuning["gpuMemoryUtilization"]),
            "ENABLE_PREFIX_CACHING": "true" if tuning["enablePrefixCaching"] else "false",
            "KV_CACHE_DTYPE": tuning["kvCacheDtype"],
            "ENABLE_EXPERT_PARALLEL": "true" if tuning["enableExpertParallel"] else "false",
        }
        if cfg.get("quantization"):
            env["QUANTIZATION"] = cfg["quantization"]
        # Tool calling and reasoning parsers are names from the engine's own lists, so a value that is
        # not a bare name is a typo, not an option. The container adds the enabling flag for each one.
        for key, var in (("toolCallParser", "TOOL_CALL_PARSER"), ("reasoningParser", "REASONING_PARSER")):
            value = _given(cfg.get(key), "").strip()
            if value and not re.fullmatch(r"[a-z0-9_]+", value):
                raise ConfigError(f"{key} must be a parser name as `vllm serve --help` lists it, e.g. hermes (got {value!r}).")
            if value:
                env[var] = value
        # Escape hatch for engine flags this project does not model - container/serve appends these
        # verbatim to `vllm serve`. Plumbed because the docs point at it as *the* way to try an
        # unmodelled flag (data parallelism, for one) and to diagnose a reluctant engine, and without
        # this there was nowhere to actually put it. Unvalidated by design: anything here bypasses the
        # checks in validate_tuning, which is the point, and the risk.
        if cfg.get("extraArgs"):
            env["EXTRA_ARGS"] = cfg["extraArgs"]
        # Environment variables for the engine process. Many vLLM knobs are only reachable this way
        # (kernel backends, attention backend, download behaviour), and a model card's recipe often
        # sets them. Validated as names so a typo cannot shadow one of the variables this stack sets.
        for name, value in _mapping(cfg.get("extraEnv"), "extraEnv").items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)) or str(name) in env:
                raise ConfigError(
                    f"extraEnv: {name!r} is not a valid variable name, or is one this stack already sets "
                    f"({', '.join(sorted(env))})."
                )
            if not isinstance(value, (str, int, float)):   # bool is an int; rendered as true/false
                raise ConfigError(f"extraEnv: {name} must be a string or number (got {value!r}).")
            env[str(name)] = str(value).lower() if isinstance(value, bool) else str(value)

        container_mib = inst.container_memory_mib // tuning["replicas"]
        shm_mib = max(8192, container_mib // 2)
        container = task_def.add_container(
            "vllm",
            image=self._container_image(cfg["image"]),
            # Derived from HOST RAM, not GPU count - see hardware.py for why that distinction
            # matters. ECS reserves the whole amount, so this also caps replicas per instance.
            memory_limit_mib=container_mib,
            gpu_count=tuning["tensorParallel"] * tuning["dataParallel"],
            environment=env,
            secrets=container_secrets or None,
            # Non-blocking: the default mode blocks the engine's stdout when CloudWatch Logs is
            # unreachable, and vLLM logs per request, so a logging outage would stall inference.
            logging=ecs.LogDrivers.aws_logs(stream_prefix="vllm", log_group=log_group,
                                            mode=ecs.AwsLogDriverMode.NON_BLOCKING,
                                            max_buffer_size=Size.mebibytes(25)),
            port_mappings=[ecs.PortMapping(container_port=CONTAINER_PORT)],
            # /dev/shm, set UNCONDITIONALLY - do not make this depend on tensorParallel. The per-GPU
            # workers pass tensors through POSIX shared memory, and Docker's 64 MiB default kills any
            # degree above 1 seconds after start. TP=1 never touches that path, so a conditional value
            # looks fine until someone raises the degree. See docs/troubleshooting.md.
            # Half the container's memory, floor 8 GiB: vLLM also puts the CPU KV offload buffer here
            # (`--kv-offloading-size N` in extraArgs), and a fixed 8 GiB refused to start the engine with
            # `Insufficient space in /dev/shm: 32768 MiB required, 8192 MiB free`. tmpfs pages are
            # allocated on write, so a larger cap costs nothing until the buffer is configured and used.
            linux_parameters=ecs.LinuxParameters(self, "Linux", shared_memory_size=shm_mib),
            # What protects an in-flight request on scale-in or redeploy is the 180 s deregistration
            # delay above: the target is drained before ECS sends SIGTERM, and this vLLM aborts
            # in-flight requests on SIGTERM. This is the wait before SIGKILL after that.
            stop_timeout=Duration.seconds(120),
        )
        # One copy of the weights per instance, shared by every task and surviving restarts.
        container.add_mount_points(ecs.MountPoint(container_path=CONTAINER_MODEL_PATH,
                                                 source_volume="modelcache",
                                                 read_only=False))

        # ------------------------------------------------------------------ engine metrics
        # The load balancer can say THAT the fleet is slow; only the engine can say WHY. It publishes
        # Prometheus metrics on /metrics, and this sidecar scrapes them over localhost (awsvpc puts
        # both containers in one network namespace) and writes them to CloudWatch as embedded-metric-
        # format log records, which CloudWatch turns into metrics under `<stack>/Engine`.
        #
        # No dimensions. CloudWatch then aggregates every engine's samples per minute, so
        # `Maximum` is the worst engine and `Average` the typical one - which is what the dashboard
        # needs - and the bill is ten metrics plus one series per band edge plus one regardless of fleet size. A per-task dimension would
        # let you name the sick engine, at a cost that grows with the fleet; the task's own logs in
        # the log group already serve that purpose.
        engine_metrics_namespace = f"{self.stack_name}/Engine"
        prompt_bands = validate_token_bands(cfg.get("promptTokenBands"), "promptTokenBands",
                                            DEFAULT_PROMPT_TOKEN_BANDS)
        output_bands = validate_token_bands(cfg.get("outputTokenBands"), "outputTokenBands",
                                            DEFAULT_OUTPUT_TOKEN_BANDS)
        # Second scrape of the same endpoint for the request-size bands. The receiver folds a histogram's
        # `_bucket` series into one histogram, and the exporter then keeps only its sum and count, so the
        # buckets are renamed out from under it (`_bucket` -> `_le`), which makes them plain series;
        # `transform` turns those into counters so `cumulativetodelta` can take per-scrape deltas.
        le_values = lambda bands: "|".join([f"{b}.0" for b in bands] + [r"\\+Inf"])  # noqa: E731
        collector_config = f"""
receivers:
  prometheus:
    config:
      scrape_configs:
        - job_name: engine
          scrape_interval: 30s
          static_configs:
            - targets: ["localhost:{CONTAINER_PORT}"]
        - job_name: bands
          scrape_interval: 30s
          static_configs:
            - targets: ["localhost:{CONTAINER_PORT}"]
          metric_relabel_configs:
            - source_labels: [__name__, le]
              regex: "vllm:request_prompt_tokens_bucket;({le_values(prompt_bands)})|vllm:request_generation_tokens_bucket;({le_values(output_bands)})"
              action: keep
            - source_labels: [__name__]
              regex: "vllm:request_prompt_tokens_bucket"
              target_label: __name__
              replacement: "vllm:request_prompt_tokens_le"
            - source_labels: [__name__]
              regex: "vllm:request_generation_tokens_bucket"
              target_label: __name__
              replacement: "vllm:request_generation_tokens_le"
processors:
  # By metric name, after the receiver has assembled histograms. A scrape-time keep rule sees the raw
  # _bucket/_sum/_count series instead and dropped every histogram (found in the pre-deploy check).
  filter/shortlist:
    metrics:
      include:
        match_type: strict
        metric_names: {list(ENGINE_METRICS) + ["vllm:request_prompt_tokens_le", "vllm:request_generation_tokens_le"]}
  transform/bands:
    metric_statements:
      - context: metric
        statements:
          - convert_gauge_to_sum("cumulative", true) where IsMatch(name, "_le$")
  cumulativetodelta:
    include:
      match_type: regexp
      metrics: {[f"^{m}$" for m in ENGINE_CUMULATIVE] + ["_le$"]}
exporters:
  awsemf:
    region: {self.region}
    namespace: {engine_metrics_namespace}
    log_group_name: {log_group.log_group_name}
    log_stream_name: engine-metrics
    dimension_rollup_option: NoDimensionRollup
    metric_declarations:
      - dimensions: [["le"]]
        metric_name_selectors: ["_le$"]
      - dimensions: [[]]
        metric_name_selectors: {[f"^{m}$" for m in ENGINE_METRICS]}
service:
  pipelines:
    metrics:
      receivers: [prometheus]
      processors: [filter/shortlist, transform/bands, cumulativetodelta]
      exporters: [awsemf]
"""
        task_def.add_container(
            "metrics",
            image=ecs.ContainerImage.from_registry(METRICS_SIDECAR_IMAGE),
            # Taken from the 40% of host memory the engine does not reserve, so the engine's own
            # limit - and the replicas-per-instance arithmetic built on it - is unchanged.
            memory_limit_mib=METRICS_SIDECAR_MEMORY_MIB,
            # Not essential: if the collector dies the engine keeps serving and the dashboard's engine
            # row goes blank. The alternative - a metrics bug taking down inference - is the wrong trade.
            essential=False,
            # ECS never restarts a non-essential container on its own, so without this an OOM at the
            # memory limit would silently blank this engine's share of the metrics until the task was
            # replaced. 60 s is the smallest period ECS allows; a container that dies within its first
            # 60 s is not restarted, which is the one gap this leaves.
            enable_restart_policy=True, restart_attempt_period=Duration.seconds(60),
            command=["--config=env:OTEL_CONFIG"],
            environment={"OTEL_CONFIG": collector_config},
            logging=ecs.LogDrivers.aws_logs(stream_prefix="metrics", log_group=log_group),
        )
        # What the collector needs: to write embedded-metric-format records into this stack's log
        # group. CloudWatch turns those records into metrics itself; no PutMetricData is involved, so
        # none is granted. Eight tasks writing one log stream is fine: PutLogEvents has not required
        # sequence tokens since 2023.
        task_def.add_to_task_role_policy(iam.PolicyStatement(
            actions=["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
            resources=[log_group.log_group_arn, log_group.log_group_arn + ":*"]))

        tasks = instance_count * tuning["replicas"]
        service = ecs.Ec2Service(
            self, "Service",
            cluster=cluster,
            task_definition=task_def,
            # With awsvpc each task gets its own ENI and IP address, so replicas can all bind the
            # same container port on one instance and the ALB round-robins across them.
            #
            # ALWAYS declared, including with autoscaling on, and that is a trade.
            #
            # Omitting it means CloudFormation omits DesiredCount, and ECS then defaults a new service
            # to ONE task - so a first deploy would serve from a single task on a full fleet of
            # instances, waiting for something to raise it. Registering a scalable target with a higher
            # MinCapacity is not documented to do that on creation (only on update of an existing one),
            # so the initial size has to be stated here.
            #
            # The cost: a later `cdk deploy` resets a scaled-out service back to
            # this number, and the scaling policy then takes ~11 minutes to climb again. The deploy
            # terminates nothing directly - the ASG above keeps its instances because DesiredCapacity is
            # absent from the template - but managed termination protection is DISABLED, so the capacity
            # provider notices the surplus and scales in about 15 minutes later, and the warm
            # /opt/modelcache on those instances goes with them. Still the milder half of the problem it
            # replaces, where the deploy terminated instances immediately and mid-request.
            desired_count=tasks,
            capacity_provider_strategies=[
                ecs.CapacityProviderStrategy(capacity_provider=capacity_provider.capacity_provider_name,
                                             weight=1)],
            # 0, because a redeploy on a fixed fleet has no free GPU to start the new engine on: at the
            # default 100 the deployment could never place a task and hung until CloudFormation gave up
            # hours later. At 0 ECS stops one engine, starts its replacement, then swaps the rest once
            # it is healthy (measured: 5, then 1, healthy of 6 for ~11 minutes). With a spare instance
            # (maxInstanceCount above instanceCount) the same setting rolls one engine at a time with
            # none down. README.md, "Changing the model, tuning or image later".
            min_healthy_percent=0,
            # Circuit breaker OFF. Enabled, it reverts to the previous task definition
            # after repeated start failures - and on a single-service GPU deployment that revision is
            # frequently also broken, so the two contend for the same GPU and the service never
            # converges. What you give up: a bad deploy stays broken until you notice. Right when an
            # operator is watching and GPUs are scarce; turn it on for an unattended fleet with spare
            # capacity. See docs/troubleshooting.md.
            circuit_breaker=None,
            health_check_grace_period=Duration.minutes(30),
        )
        service.attach_to_application_target_group(target_group)

        # ------------------------------------------------------------------ autoscaling
        #
        # Only when maxInstanceCount exceeds instanceCount (the shipped 16 -> 24 does). Burst absorption,
        # not right-sizing: a new task is minutes from decision to serving, so the minimum must cover
        # steady state. Scales on RequestCountPerTarget, which leads latency and needs no custom metric;
        # the threshold is workload-specific and config.yaml shows the derivation.
        # requests_per_target stays None without a policy, so the dashboard draws no threshold line.
        requests_per_target = None
        if autoscaling_enabled:
            requests_per_target = _num(_given(cfg.get("scalingRequestsPerTarget"), 925),
                                       # minimum=1: a zero or negative target is rejected at synth
                                       # rather than by the Application Auto Scaling API after the
                                       # VPC, ALB and ASG already exist.
                                       "scalingRequestsPerTarget", minimum=1)
            scaling = service.auto_scale_task_count(
                # Bounds are TASK counts, so both are multiplied by replicas-per-instance. Using
                # instance counts directly would cap a multi-engine fleet at a fraction of its tasks.
                min_capacity=tasks,
                max_capacity=configured_max * tuning["replicas"],
            )
            scaling.scale_on_request_count(
                "RequestsPerTarget",
                requests_per_target=requests_per_target,
                target_group=target_group,
                # Asymmetric: scale out readily, scale in reluctantly, because discarding a warm
                # engine costs minutes of weight loading to undo. Measured, these cooldowns are NOT the
                # dominant term - scale-out took ~11 min to usable capacity, mostly metric lag and
                # weight loading, so lowering the 3 minutes would change almost nothing. See "What
                # autoscaling actually does" in docs/tuning.md.
                scale_out_cooldown=Duration.minutes(3),
                scale_in_cooldown=Duration.minutes(15),
            )
        # The ALB must be allowed to reach the tasks. `connections.allow_to` writes both halves of
        # the rule; adding ingress alone leaves the ALB's egress blocked and health checks silently
        # never arrive.
        alb.connections.allow_to(service, ec2.Port.tcp(CONTAINER_PORT),
                                 "ALB to vLLM tasks")

        # ------------------------------------------------------------------ observability
        #
        # One dashboard, on every deployment, with nothing to switch on. Two kinds of metric feed it:
        #
        # * What the load balancer and the Auto Scaling group already publish: latency, request rate,
        #   errors, healthy tasks, instances. Free, and there is no collection path that can break.
        # * What the engine itself reports, via the metrics sidecar defined with the task above: queue
        #   depth, batch occupancy, KV cache usage, preemptions. These are the numbers that say WHY
        #   latency is rising rather than just that it is; ten metrics plus one series per band edge plus one.
        #
        # Everything user-facing here is written in plain language on purpose. The reader is someone
        # woken by an alarm who has never seen this stack, so a widget titled "TargetResponseTime p95"
        # is worth less than one saying what a bad reading means and what to do about it.

        # The p95 you are willing to serve, if you have one. There is no defensible default: a chat UI
        # and a nightly batch job disagree by two orders of magnitude, and shipping someone else's number
        # here would put an arbitrary red line on the graph and page whoever crossed it. So 0 means off -
        # the p50/p95/p99 graph is drawn either way, and the budget line and its alarm appear only once
        # you say what the budget is.
        latency_alarm = _num(_given(cfg.get("latencyAlarmSeconds"), 0),
                             "latencyAlarmSeconds", float, minimum=0)

        # `.metrics.` rather than the `metric_*` methods on the constructs: those are deprecated, and
        # CDK prints a warning per call, which buries the memory-pressure warning above - the one this
        # stack actually needs you to read.
        tg_metrics, alb_metrics = target_group.metrics, alb.metrics
        minute = Duration.minutes(1)

        def response_time(percentile: str, **kw) -> cloudwatch.Metric:
            return tg_metrics.target_response_time(statistic=percentile, period=minute, **kw)

        def asg_metric(name: str, label: str) -> cloudwatch.Metric:
            # Raw, because CDK exposes no helper for ASG group metrics. Enabled on the ASG above.
            return cloudwatch.Metric(
                namespace="AWS/AutoScaling", metric_name=name, label=label,
                dimensions_map={"AutoScalingGroupName": asg.auto_scaling_group_name},
                statistic="Maximum", period=minute)

        # Named after the stack so it is findable without hunting through a hashed logical id, and so
        # the console URL can be printed as a plain string rather than a Ref nobody can click.
        # Region in the name: CloudWatch dashboard names are account-wide, not regional. Two stacks in
        # two regions with the same fixed name cannot both exist, and the second deploy fails at change
        # set validation with "already exists".
        dashboard_name = f"{self.stack_name}-{self.region}"
        dashboard = cloudwatch.Dashboard(self, "Dashboard", dashboard_name=dashboard_name)
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                # The load balancer times a request until the engine starts answering: the whole
                # generation for a non-streamed call, the first token for a streamed one.
                title=("Is it slow? - time to the answer (whole answer if not streaming, first token if streaming)"
                       + (f" vs your {latency_alarm:g}s budget" if latency_alarm else "")), width=12,
                left=[response_time("p50", label="typical request (p50)"),
                      response_time("p95", label="slow request (p95)"),
                      response_time("p99", label="slowest requests (p99)")],
                left_y_axis=cloudwatch.YAxisProps(label="seconds", show_units=False),
                left_annotations=([cloudwatch.HorizontalAnnotation(
                    value=latency_alarm, label=f"your budget: {latency_alarm:g}s",
                    color="#d13212")] if latency_alarm else None),
            ),
            cloudwatch.GraphWidget(
                title="How hard is each engine working? - requests a minute per task", width=12,
                left=[tg_metrics.request_count_per_target(
                    period=minute, label="requests/min handled by one task")],
                left_y_axis=cloudwatch.YAxisProps(label="requests/min", show_units=False),
                left_annotations=([cloudwatch.HorizontalAnnotation(
                    value=requests_per_target,
                    label=f"adds capacity above {requests_per_target}", color="#ff7f0e")]
                    if requests_per_target else None),
            ),
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Is the load balancer failing? - usually no healthy task", width=6,
                left=[alb_metrics.http_code_elb(elbv2.HttpCodeElb.ELB_5XX_COUNT, period=minute,
                                                label="requests the ALB could not serve")],
            ),
            cloudwatch.GraphWidget(
                title="Are the engines erroring? - the model returned a 5XX itself", width=6,
                left=[tg_metrics.http_code_target(elbv2.HttpCodeTarget.TARGET_5XX_COUNT,
                                                  period=minute, label="errors from the model")],
            ),
            cloudwatch.GraphWidget(
                title="Did an engine die mid-request? - dropped connections", width=6,
                left=[alb_metrics.target_connection_error_count(
                    period=minute, label="connections that failed")],
            ),
            cloudwatch.GraphWidget(
                title=(f"Is CloudFront timing out? - non-streamed answers over "
                       f"{CLOUDFRONT_READ_TIMEOUT_S}s"), width=6,
                # CloudFront publishes its metrics in us-east-1 whatever region the stack is in.
                left=[cloudwatch.Metric(
                    namespace="AWS/CloudFront", metric_name="5xxErrorRate", region="us-east-1",
                    dimensions_map={"DistributionId": distribution.distribution_id,
                                    "Region": "Global"},
                    statistic="Average", period=minute, label="% of requests failing at the edge")],
                left_y_axis=cloudwatch.YAxisProps(label="percent", show_units=False, min=0),
            ),
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title=(f"Are the engines up? - {tasks} healthy is normal" if tasks
                       else "Are the engines up? - fleet parked, none expected"), width=8,
                left=[tg_metrics.healthy_host_count(label="healthy and serving", period=minute),
                      tg_metrics.unhealthy_host_count(label="failing health checks", period=minute)],
                left_y_axis=cloudwatch.YAxisProps(label="tasks", show_units=False),
            ),
            cloudwatch.GraphWidget(
                title="Did AWS give us the instances? - a gap means no capacity", width=8,
                left=[asg_metric("GroupInServiceInstances", "instances running"),
                      asg_metric("GroupDesiredCapacity", "instances wanted")],
                left_y_axis=cloudwatch.YAxisProps(label="instances", show_units=False),
            ),
            cloudwatch.GraphWidget(
                title="How much traffic is arriving? - whole fleet", width=8,
                left=[alb_metrics.request_count(period=minute, label="requests/min")],
            ),
        )

        def engine_metric(name: str, statistic: str, label: str) -> cloudwatch.Metric:
            return cloudwatch.Metric(namespace=engine_metrics_namespace, metric_name=name,
                                     statistic=statistic, label=label, period=minute)

        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Is work queueing inside the engines? - waiting means saturated", width=8,
                left=[engine_metric("vllm:num_requests_waiting", "Maximum", "waiting, busiest engine"),
                      engine_metric("vllm:num_requests_waiting", "Average", "waiting, typical engine"),
                      engine_metric("vllm:num_requests_running", "Average",
                                    f"running, typical engine (max {tuning['maxNumSeqs']})")],
                left_y_axis=cloudwatch.YAxisProps(label="requests", show_units=False, min=0),
            ),
            cloudwatch.GraphWidget(
                title="Is the KV cache filling up? - near 100% means preemption is next", width=8,
                left=[engine_metric("vllm:kv_cache_usage_perc", "Maximum", "fullest engine"),
                      engine_metric("vllm:kv_cache_usage_perc", "Average", "typical engine")],
                left_y_axis=cloudwatch.YAxisProps(label="fraction of cache", show_units=False,
                                                  min=0, max=1),
            ),
            cloudwatch.GraphWidget(
                title="Are engines redoing work? - preemptions, any is bad", width=8,
                left=[engine_metric("vllm:num_preemptions_total", "Sum", "preemptions per minute")],
                left_y_axis=cloudwatch.YAxisProps(label="preemptions", show_units=False, min=0),
            ),
        )
        # Averages over the requests completed that minute, whole fleet: Sum/Count of the engine
        # histograms. Not percentiles; see ENGINE_METRICS.
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="How long does a request take inside the engine? - averages that minute", width=12,
                left=[engine_metric("vllm:time_to_first_token_seconds", "Average",
                                    "time to first token (queue wait + prefill)"),
                      engine_metric("vllm:e2e_request_latency_seconds", "Average", "whole request")],
                left_y_axis=cloudwatch.YAxisProps(label="seconds", show_units=False, min=0),
            ),
            cloudwatch.GraphWidget(
                title="What shape are the requests being served? - average tokens per request", width=12,
                left=[engine_metric("vllm:request_prompt_tokens", "Average", "input tokens"),
                      engine_metric("vllm:request_generation_tokens", "Average", "output tokens")],
                left_y_axis=cloudwatch.YAxisProps(label="tokens", show_units=False, min=0),
            ),
        )

        def bands_widget(title: str, metric: str, bands: tuple[int, ...]) -> cloudwatch.GraphWidget:
            """Requests per minute in each size band, stacked. The engine's buckets are cumulative
            (le=1000 includes le=500), so a band is the difference of two adjacent buckets."""
            def bucket(le: str, ident: str) -> cloudwatch.Metric:
                return cloudwatch.Metric(namespace=engine_metrics_namespace, metric_name=f"{metric}_le",
                                         dimensions_map={"le": le}, statistic="Sum", period=minute,
                                         label=ident)
            edges = [f"{b}.0" for b in bands] + ["+Inf"]
            using = {f"b{i}": bucket(le, f"b{i}") for i, le in enumerate(edges)}
            series = [cloudwatch.MathExpression(expression="b0", using_metrics={"b0": using["b0"]},
                                                label=f"up to {bands[0]:,}", period=minute)]
            for i in range(1, len(edges)):
                upper = "and more" if edges[i] == "+Inf" else f"to {bands[i]:,}"
                series.append(cloudwatch.MathExpression(
                    expression=f"b{i} - b{i - 1}", label=f"{bands[i - 1]:,} {upper}", period=minute,
                    using_metrics={f"b{i}": using[f"b{i}"], f"b{i - 1}": using[f"b{i - 1}"]}))
            return cloudwatch.GraphWidget(
                title=title, width=8, left=series, stacked=True,
                left_y_axis=cloudwatch.YAxisProps(label="requests per minute", show_units=False, min=0))

        dashboard.add_widgets(
            bands_widget("What size are the prompts? - requests a minute by input tokens",
                         "vllm:request_prompt_tokens", prompt_bands),
            bands_widget("How long are the answers? - requests a minute by output tokens",
                         "vllm:request_generation_tokens", output_bands),
            cloudwatch.GraphWidget(
                title="Is the prefix cache paying off? - % of prompt tokens already cached", width=8,
                left=[cloudwatch.MathExpression(
                    expression="100 * hits / queries", label="hit rate", period=minute,
                    using_metrics={"hits": engine_metric("vllm:prefix_cache_hits_total", "Sum", "hits"),
                                   "queries": engine_metric("vllm:prefix_cache_queries_total", "Sum",
                                                            "queries")})],
                left_y_axis=cloudwatch.YAxisProps(label="percent", show_units=False, min=0, max=100),
            ),
        )

        # Two alarms, plus an optional latency one. More would mostly restate these, and an alarm
        # nobody trusts is worse than no alarm: an earlier round of work on this project lost real time
        # to a check that cried wolf on every normal model swap.
        #
        # No SNS topic is created here, because a topic with no subscription notifies nobody while
        # looking like it does. Supply `alarmTopicArn` and the alarms notify it; leave it out and they
        # still change state in the console and in `describe-alarms`.
        topic_arn = str(_given(cfg.get("alarmTopicArn"), "")).strip()
        if topic_arn and not re.match(r"^arn:[a-z0-9-]+:sns:[a-z0-9-]+:\d{12}:.+$", topic_arn):
            raise ConfigError(
                f"alarmTopicArn must be an SNS topic ARN (got {topic_arn!r}).\n"
                "  Expected arn:aws:sns:REGION:ACCOUNT:topic-name. Leave it out for alarms that change "
                "state without notifying anyone.")
        actions = ([cw_actions.SnsAction(sns.Topic.from_topic_arn(self, "AlarmTopic", topic_arn))]
                   if topic_arn else [])

        # NOTE the metrics below carry no `label`. A labelled metric cannot be expressed in
        # CloudFormation's simple alarm form, so CDK silently renders the alarm as a metric-math query
        # instead - which works, but leaves MetricName absent from the template and makes the alarm
        # harder to read in the console and in `describe-alarms`. Label for widgets, not for alarms.
        #
        # The descriptions are written to be read on a phone by someone who did not deploy this.
        alarms = [
            tg_metrics.unhealthy_host_count(period=minute).create_alarm(
                self, "UnhealthyTargetsAlarm",
                alarm_name=f"{self.stack_name}-engines-unhealthy",
                alarm_description=(
                    f"At least one engine has been failing its health check for 15 minutes "
                    f"({tasks} should be healthy). Normal during a deploy while weights load; "
                    f"otherwise the container is crashing - check the task's logs in "
                    f"{log_group.log_group_name}."),
                threshold=0, evaluation_periods=15,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                # Fifteen minutes, because a starting task is legitimately unhealthy while it pulls the
                # image and loads tens of GiB of weights. Measured on a fresh instance with a cold cache:
                # about 7 minutes unhealthy in the target group, 14 from launch to healthy. A 5-minute
                # window fired on every first deploy, which is how an alarm gets ignored.
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            ),
            alb_metrics.http_code_elb(elbv2.HttpCodeElb.ELB_5XX_COUNT,
                                      period=minute).create_alarm(
                self, "AlbErrorAlarm",
                alarm_name=f"{self.stack_name}-load-balancer-erroring",
                alarm_description=(
                    "Clients are getting 5XX from the load balancer rather than from the model. "
                    "503 means there was no healthy task to send it to; 504 means a request ran past "
                    "the load balancer's 300-second timeout. Check whether the engines are up."),
                # Not zero. A deploy briefly has no healthy target and a scale-in drains a task, so a
                # zero threshold fires on both. Sustained is what matters.
                threshold=10, evaluation_periods=2,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            ),
        ]
        if latency_alarm:
            alarms.append(response_time("p95").create_alarm(
                self, "LatencyAlarm",
                alarm_name=f"{self.stack_name}-too-slow",
                alarm_description=(
                    f"Requests are taking longer than {latency_alarm:g}s at p95. The fleet is "
                    f"overloaded or an engine is unhealthy. Check requests/min per task on the "
                    f"{dashboard_name} dashboard: if it is high, the fleet needs more "
                    f"instances (raise maxInstanceCount, or instanceCount if it is already capped)."),
                threshold=latency_alarm,
                # Three minutes, not one: a single minute over budget is what a task starting or
                # draining looks like, and a fleet that cannot grow faster than ~11 minutes gains
                # nothing from being told sooner.
                evaluation_periods=3,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                # An idle fleet publishes no response times at all. Without this the alarm sits in
                # INSUFFICIENT_DATA whenever traffic stops, which trains everyone to ignore it.
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            ))
        for alarm in alarms:
            for action in actions:
                alarm.add_alarm_action(action)


        # ------------------------------------------------------------------ outputs
        CfnOutput(self, "Endpoint", value=endpoint,
                  description="HTTPS, via CloudFront. Send the API key as Authorization: Bearer <key>")
        CfnOutput(self, "ResponsesApi", value=f"{endpoint}/v1/responses")
        CfnOutput(self, "ChatCompletionsApi", value=f"{endpoint}/v1/chat/completions")
        CfnOutput(self, "DistributionId", value=distribution.distribution_id,
                  description="The CloudFront distribution in front of the load balancer")
        CfnOutput(self, "ApiKeyValue", value=key_value,
                  description="Send as Authorization: Bearer <key>. Also visible in the template, so not "
                              "confidential from anyone who can read the stack")
        CfnOutput(self, "ModelName", value=cfg["modelId"],
                  description="The model id to put in requests; also what /v1/models returns")
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "AsgName", value=asg.auto_scaling_group_name)
        CfnOutput(self, "LogGroup", value=log_group.log_group_name)
        # "DashboardName", not "Dashboard": the Dashboard construct above already owns that id in this
        # scope, and two children with the same id is a synth error.
        CfnOutput(self, "DashboardName", value=dashboard_name)
        # The URL, not just the name. A dashboard nobody can find is not observability, and the console
        # path for a named dashboard is not something anyone guesses. `self.region` resolves to the
        # literal region here and to the AWS::Region pseudo-parameter in an env-agnostic stack, so this
        # is correct either way. Printed by scripts/endpoint_info.py too.
        CfnOutput(self, "DashboardUrl",
                  value=(f"https://{self.region}.console.aws.amazon.com/cloudwatch/home"
                         f"?region={self.region}#dashboards:name={dashboard_name}"),
                  description="Open this to see whether the fleet is healthy")
        CfnOutput(self, "ResolvedTensorParallel", value=str(tuning["tensorParallel"]))
        CfnOutput(self, "ContainerMemoryMib", value=str(inst.container_memory_mib
                                                       // tuning["replicas"]))
        CfnOutput(self, "PurchaseModel", value="spot" if use_spot else "on-demand")

    def _container_image(self, image_uri: str) -> ecs.ContainerImage:
        """Resolve the image so ECS is actually allowed to pull it.

        `from_registry` passes the URI through as an opaque string, so CDK cannot tell it names an ECR
        repository and grants the task execution role nothing beyond `ecr:GetAuthorizationToken`. That
        authenticates but cannot fetch a manifest or layers, so the task fails at start with:

            CannotPullContainerError: ... denied

        CDK warns about it ("Proper policies need to be attached before pulling from ECR repository,
        or use 'fromEcrRepository'"), and that warning is easy to read as boilerplate.

        The repository is built from the ARN implied by the URI's OWN account and region rather than by
        comparing them to this stack's: the CDK CLI overwrites `CDK_DEFAULT_ACCOUNT` from the ambient
        credentials, so `self.account` is often not the account in the URI, and a comparison silently
        fell through to `from_registry` with no pull permissions at all.

        Anything that is not an ECR URI - Docker Hub, a public gallery - falls through unchanged.
        """
        # `[^:@]+` for the repository name, and the tag separator matched explicitly, so a DIGEST URI
        # (repo@sha256:...) is handled rather than mangled: `[^:]+` swallowed the `@sha256` into the
        # repository name, producing an invalid ARN and leaving the task unable to pull - the exact
        # failure this method exists to prevent. `\.cn` because ECR in China is amazonaws.com.cn.
        # Stripped ONCE, and the stripped value used down both branches. Cleaning the string for the
        # match and then passing the raw one to from_registry put the padding of
        # `image: "  vllm/vllm-openai:v0.11  "` straight into the task definition, where it failed at
        # task start.
        uri = str(image_uri).strip()
        match = re.match(
            r"^(\d{12})\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?/([^:@]+)"
            r"(?::(.+)|@(sha256:[0-9a-f]+))?$",
            uri)
        if not match:
            return ecs.ContainerImage.from_registry(uri)
        account, region, name, tag, digest = match.groups()
        # A trailing slash is newly reachable now that the tag is optional, and it would produce an
        # ARN ending in `repository/name/` - accepted by IAM, matching nothing.
        name = name.rstrip("/")
        repository = ecr.Repository.from_repository_attributes(
            self, "ImageRepo",
            repository_arn=f"arn:{self.partition}:ecr:{region}:{account}:repository/{name}",
            repository_name=name)
        if digest:
            return ecs.ContainerImage.from_ecr_repository(repository, digest)
        # An untagged reference is legal and means :latest. Requiring a tag meant it fell through to
        # from_registry with no pull grant - the failure this method exists to prevent.
        return ecs.ContainerImage.from_ecr_repository(repository, tag or "latest")
