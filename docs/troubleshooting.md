# Troubleshooting

Organised by symptom. Most GPU deployment failures present as "nothing is happening", so start with
the triage commands.

---

## Start here: three commands that locate almost any problem

Set `REGION` first. Against the wrong region every command fails as if the stack does not exist:

```bash
REGION=eu-west-2        # the region in your config

CLUSTER=$(aws cloudformation describe-stacks --stack-name GpuLlmServing --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`ClusterName`].OutputValue' --output text)
SERVICE=$(aws ecs list-services --cluster "$CLUSTER" --region "$REGION" \
  --query 'serviceArns[0]' --output text | awk -F/ '{print $NF}')
LOG_GROUP=$(aws cloudformation describe-stacks --stack-name GpuLlmServing --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`LogGroup`].OutputValue' --output text)
ASG=$(aws cloudformation describe-stacks --stack-name GpuLlmServing --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`AsgName`].OutputValue' --output text)
TASK_DEF=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" \
  --query 'services[0].taskDefinition' --output text)

# 1. Does the service have a running task, and is a deployment stuck?
aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" \
  --query 'services[0].[runningCount,desiredCount,pendingCount,events[0].message]'

# 2. Do the instances actually expose GPUs to ECS?   <-- the highest-value check
CI=$(aws ecs list-container-instances --cluster "$CLUSTER" --region "$REGION" \
  --query 'containerInstanceArns' --output text)
aws ecs describe-container-instances --cluster "$CLUSTER" --container-instances $CI --region "$REGION" \
  --query 'containerInstances[].{type:attributes[?name==`ecs.instance-type`].value|[0],
           registeredGPU:registeredResources[?name==`GPU`].stringSetValue|[0],
           freeGPU:remainingResources[?name==`GPU`].stringSetValue|[0]}'

# 3. What did the engine say?
aws logs tail "$LOG_GROUP" --since 30m --region "$REGION"
```

---

## Symptom: you cannot reach the endpoint at all

Later sections assume the request reaches the load balancer.

**Cause 1: the distribution is still deploying.** CloudFront takes a few minutes to propagate after
the stack completes. Until then the hostname may not resolve, or CloudFront returns its own error page.

```bash
DIST=$(aws cloudformation describe-stacks --stack-name GpuLlmServing --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`DistributionId`].OutputValue' --output text)
aws cloudfront get-distribution --id "$DIST" --query 'Distribution.Status' --output text
# InProgress -> wait; Deployed -> read on
```

**Cause 2: you called `http://`, not `https://`.** CloudFront answers plain http with a 403 and does
not redirect (a redirect would turn POST into GET). Use the `Endpoint` output as printed.

**Cause 3: a 504 after about two minutes, on a non-streamed request.** CloudFront waits at most 120 s
for a non-streaming response, then returns 504 while the engine finishes anyway. Ask for less output or
set `"stream": true`. README.md, *Access*, has the arithmetic.

**Cause 4: the environment was never bootstrapped.** `cdk deploy` fails with "this stack has not been
bootstrapped". Bootstrapping is per account and per region:

```bash
cd infra && cdk bootstrap && cd ..
```

---

## Symptom: `cdk destroy` ends in `DELETE_FAILED` on a subnet, security group or the VPC

`DependencyViolation ... has dependencies and cannot be deleted`. Something outside the stack holds a
network interface in the VPC. In an account with GuardDuty Runtime Monitoring and automated agent
management, GuardDuty creates a `guardduty-data` VPC endpoint and a `GuardDutyManagedSecurityGroup-*`
security group in every VPC that runs instances, and does not always remove them when the instances go.
Neither belongs to the stack, so CloudFormation cannot delete them and the subnets they sit in.

Remove them, then destroy again; the retry picks up where it stopped:

```bash
REGION=<region>; STACK=GpuLlmServing
VPC=$(aws cloudformation describe-stack-resources --region "$REGION" --stack-name "$STACK" \
  --query "StackResources[?ResourceType=='AWS::EC2::VPC'].PhysicalResourceId" --output text)
aws ec2 delete-vpc-endpoints --region "$REGION" --vpc-endpoint-ids $(aws ec2 describe-vpc-endpoints \
  --region "$REGION" --filters Name=vpc-id,Values="$VPC" Name=service-name,Values="com.amazonaws.$REGION.guardduty-data" \
  --query 'VpcEndpoints[].VpcEndpointId' --output text)
# wait until this prints 0, about a minute
aws ec2 describe-network-interfaces --region "$REGION" --filters Name=vpc-id,Values="$VPC" \
  --query 'length(NetworkInterfaces)' --output text
aws ec2 delete-security-group --region "$REGION" --group-id $(aws ec2 describe-security-groups \
  --region "$REGION" --filters Name=vpc-id,Values="$VPC" Name=group-name,Values='GuardDutyManagedSecurityGroup-*' \
  --query 'SecurityGroups[0].GroupId' --output text)
cdk destroy
```

Any other interface the first command lists (a Lambda, a Client VPN, another team's endpoint) is the same
story: find its owner, remove it, retry.

A different `DELETE_FAILED`, on the ECS cluster with `The specified capacity provider is in use`, is an
ordering race inside CloudFormation: it detached the capacity provider before the service was gone.
Nothing to clean up; destroy again.

---

## Symptom: `cdk destroy` fails with `delete is not allowed for this vpc origin`

You destroyed a stack whose CloudFront VPC origin was still being created. CloudFront refuses to delete a
VPC origin until it reaches `Deployed`, which takes a few minutes after the ALB exists, and the stack
lands in `DELETE_FAILED`. Wait for it, then destroy again; the retry picks up where it stopped.

```bash
aws cloudfront list-vpc-origins --query 'VpcOriginList.Items[].[Name,Status]' --output text
```

---

## Symptom: `cdk deploy` fails with `409 ... VPC origin is currently associated with one or more distributions`

You are upgrading a deployment created before 2026-09-06. The VPC origin's name gained the region so two
regions can coexist in one account, and CloudFront refuses to change a VPC origin while a distribution
uses it; the stack rolls back before touching the engines. There is no in-place path: `cdk destroy`,
then deploy again. The endpoint URL changes. Fresh deployments never see this.

---

## Symptom: a first deploy sits in `CREATE_IN_PROGRESS` with tasks stopping

The image cannot be pulled (the build failed after `--write-config` wrote the URI, or a Docker Hub pull
was rate-limited during the build) or the engine crashes on start. CloudFormation waits on the ECS
service for up to three hours before rolling back, and `cancel-update-stack` does not apply to a create.

```bash
aws ecs describe-tasks --cluster "$CLUSTER" --region "$REGION" \
  --tasks $(aws ecs list-tasks --cluster "$CLUSTER" --region "$REGION" --desired-status STOPPED \
            --query 'taskArns[0]' --output text) \
  --query 'tasks[0].[stopCode,stoppedReason,containers[0].reason]'
```

Do not wait. `aws cloudformation delete-stack --stack-name GpuLlmServing --region "$REGION"`, fix the
cause (`python3 scripts/build_image.py --status <id>` for the build), deploy again.

---

## Symptom: `stack is in UPDATE_IN_PROGRESS and can not be updated`

Every `cdk deploy` is rejected and the stack has been `UPDATE_IN_PROGRESS` far longer than a deploy
takes, often over an hour.

```bash
aws cloudformation describe-stacks --stack-name GpuLlmServing --region "$REGION" \
  --query 'Stacks[0].[StackStatus,LastUpdatedTime]' --output text
```

A stack update that touches the service waits silently for it to stabilise. CloudFormation waits for
its own timeout, up to three hours; the stack is locked until then. Two things stop it stabilising.

**Cause A: capacity was changed while a deploy was still waiting.** Scaling the ASG to zero or changing
the instance type during that wait leaves nothing to place tasks on.

**Cause B: the new engine configuration cannot start, so the task crash-loops.** The service has no
deployment circuit breaker (a restart loop is also how a spot reclaim recovers), so ECS keeps starting
the task and CloudFormation keeps waiting. Check for a run of stopped tasks with exit code 1:

```bash
aws ecs list-tasks --cluster "$CLUSTER" --region "$REGION" --desired-status STOPPED --query 'taskArns' --output text
aws logs tail "$LOG_GROUP" --since 30m --region "$REGION" | grep -E "Error|ValueError|RuntimeError" | head
```

The engine log names the cause. Seen so far: `not divisible by weight quantization block_n` (a
block-quantised FP8 checkpoint at a tensor-parallel degree that does not divide its expert size into
whole tiles; drop the degree or enable expert parallelism, see [tuning.md](tuning.md)), and the
`maxModelLen` startup failure on a nearly full card.

**Fix: cancel the update, let it roll back, then retry.**

```bash
aws cloudformation cancel-update-stack --stack-name GpuLlmServing --region "$REGION"

# Wait for the rollback to finish before deploying again.
aws cloudformation wait stack-update-rollback-complete \
  --stack-name GpuLlmServing --region "$REGION"
```

**Avoiding it: do not change ASG capacity or `instanceType` while a deploy is in flight, and after any
engine configuration change watch the first task start before walking away.**

---

## Symptom: `MaxSpotInstanceCountExceeded`, and the fleet will not grow

Scaling activities report it and no instance launches, though the quota looks sufficient.

```bash
aws autoscaling describe-scaling-activities --auto-scaling-group-name "$ASG" --region "$REGION" \
  --max-items 5 --query 'Activities[].[StatusCode,StatusMessage]' --output text
```

**Cause: something else is holding the quota**, most likely an instance of the type you just stopped
using. When `instanceType` changes, the ASG may replace an old-type instance before the launch template
updates; the stranded instance holds quota the new type needs. Observed: a leftover `g7e.12xlarge` held
48 of a 64 vCPU spot quota while the ASG tried to launch six `g7e.2xlarge` (also 48).

Count what is running. vCPU is the unit, not instances:

```bash
aws ec2 describe-instances --region "$REGION" \
  --filters Name=instance-state-name,Values=running,pending \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,InstanceLifecycle]' --output text
```

**Fix: scale to zero, confirm zero instances, then change the type.** Terminating the stranded instance
also works. Retrying does not.

Spot and on-demand are separate quotas (`L-3819A6DF` and `L-DB2E81BA`); check the one matching
`useSpot`.

---

## Symptom: tasks stay in PROVISIONING forever, no error anywhere

`containerInstanceArn` is `null`, the instance is healthy, and nothing explains why.

**Cause: the GPU driver never attached, so ECS registered the instance with an empty GPU list.**

The instance passes every EC2 and ASG health check, joins the cluster, reports `agentConnected: true`,
and advertises no GPUs, so a task requesting one can never be placed. Check:

```bash
CI=$(aws ecs list-container-instances --cluster "$CLUSTER" --region "$REGION" \
  --query 'containerInstanceArns' --output text)
aws ecs describe-container-instances --cluster "$CLUSTER" --container-instances $CI --region "$REGION" \
  --query 'containerInstances[].{
      instance:ec2InstanceId,
      agent:agentConnected,
      registeredGPU:registeredResources[?name==`GPU`].stringSetValue|[0],
      runningTasks:runningTasksCount}'
```

Empty or absent `registeredGPU` with `agent` `true` is conclusive. If `registeredGPU` lists device
ids, read Cause 2.

The Amazon Linux 2 ECS GPU AMI ships a driver that predates this GPU generation. Instance console
output:

```
NVRM: The NVIDIA GPU ... is not supported by the NVIDIA <version> driver release.
nvidia: probe of 0000:2b:00.0 failed with error -1
NVRM: None of the NVIDIA devices were initialized.
```

**Fix:** the Amazon Linux 2023 GPU AMI has a supporting driver and this project already uses it. If
you changed `machine_image` in `infra/serving_stack.py`, restore it:

```python
machine_image=ecs.EcsOptimizedImage.amazon_linux2023(ecs.AmiHardwareType.GPU)
```

Confirm on the host:

```bash
aws ssm send-command --instance-ids <id> --document-name AWS-RunShellScript --region "$REGION" \
  --parameters 'commands=["nvidia-smi --query-gpu=name,driver_version --format=csv"]'
```

**Cause 2: the instance is not in the capacity provider's ASG.** Same symptoms, but `registeredGPU` is
populated and `agentConnected` is `true`. An instance launched by hand with `run-instances` joins the
cluster, but the service places through `capacityProviderStrategy`, which only uses instances its own
ASG owns.

The workaround doubles as the best placement diagnostic:

```bash
aws ecs run-task --cluster "$CLUSTER" --launch-type EC2 --region "$REGION" \
  --task-definition "$TASK_DEF" --query 'failures'
```

It bypasses the capacity provider and returns a `failures` list naming why each instance was rejected:
`RESOURCE:GPU`, `RESOURCE:MEMORY`, `MemberOf placement constraint unsatisfied`. The service path never
shows this.

**Fix:** let the ASG launch instances (change `instanceCount` or the ASG's desired capacity).

---

## Symptom: `TaskFailedToStart`, no exit code, no container reason

**Cause A: the container memory limit is too small for the model.**

The kernel OOM-kills the container mid-load; ECS reports only a generic start failure. Memory must
derive from host RAM, not GPU memory: a `g7e.2xlarge` has 96 GiB of VRAM but 64 GiB of host RAM.
`infra/hardware.py` uses 60% of host RAM. Check:

```bash
aws cloudformation describe-stacks --stack-name GpuLlmServing --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`ContainerMemoryMib`].OutputValue' --output text
```

**Cause B: the ECS pause container image is missing.**

```
Error response from daemon: No such image: amazon/amazon-ecs-pause:0.1.0
```

Every `awsvpc` task needs this image for its network namespace; without it the pause container fails
and every task dies within seconds. Almost always caused by `docker image prune -a` in a cleanup
script, since nothing references the pause image. Restart the ECS agent to reload it from local cache:

```bash
aws ssm send-command --instance-ids <id> --document-name AWS-RunShellScript --region "$REGION" \
  --parameters 'commands=["systemctl restart ecs","sleep 20","docker images | grep pause"]'
```

**Never use `docker image prune -a` on an ECS host.** Use `docker container prune -f`; stopped
containers hold the writable layers that take the space.

---

## Symptom: the ASG never launches an instance

Read the reason:

```bash
aws autoscaling describe-scaling-activities --auto-scaling-group-name "$ASG" --region "$REGION" \
  --max-items 5 --query 'Activities[].[StatusCode,StatusMessage]' --output text
```

| Message contains | Meaning | Fix |
|---|---|---|
| `MaxSpotInstanceCountExceeded` | spot quota exceeded, a separate and much smaller quota than on-demand | request an increase on `L-3819A6DF` |
| `VcpuLimitExceeded` | on-demand vCPU quota exceeded | request an increase on `L-DB2E81BA` |
| `InsufficientInstanceCapacity` | AWS has none of this shape in that zone right now | try `useSpot: true`, another region, or wait |
| `Unsupported ... not supported in your requested Availability Zone` | that zone never offers this type | deterministic for that zone; the group retries in others. Set `availabilityZones` to the zones that offer it (see below) |
| `not authorized to use launch template` | IAM | the principal running `cdk deploy` needs `ec2:RunInstances` and `iam:PassRole` on the instance role; the instance role itself is not involved |

- Quotas are per-region and per-purchase-model. On-demand quota says nothing about spot.
  `g7e.24xlarge` (96 vCPU) and `g7e.48xlarge` (192 vCPU) both exceed the typical 64 vCPU spot default.
- Ignore the "you can currently get capacity by choosing zone X" hint. It comes from the list of zones
  that offer the type, not live inventory, and pinning to it reduces the zones the ASG can try.

**On `InsufficientInstanceCapacity` during a first deploy:** CloudFormation keeps waiting on the ECS
service for up to an hour before it rolls back, and a stack in `CREATE_IN_PROGRESS` cannot be updated.
Do not wait. `cdk destroy`, set `useSpot: true`, deploy again: about ten minutes to a working endpoint,
measured here after on-demand had no `g7e.2xlarge` in either zone of `us-east-2`.

**On `Unsupported`:** the ASG retries forever. Find the zones that offer the type and pin the fleet:

```bash
aws ec2 describe-instance-type-offerings --region <region> \
  --location-type availability-zone \
  --filters Name=instance-type,Values=g7e.2xlarge \
  --query 'InstanceTypeOfferings[].Location'
```

```yaml
# config.yaml
availabilityZones: ["us-east-2a", "us-east-2b"]
```

---

## Symptom: 503 from the endpoint

The load balancer has no healthy target.

```bash
TG=$(aws elbv2 describe-target-groups --region "$REGION" --query 'TargetGroups[?contains(TargetGroupName,`Targets`)].TargetGroupArn|[0]' --output text)
aws elbv2 describe-target-health --target-group-arn "$TG" --region "$REGION" \
  --query 'TargetHealthDescriptions[].[Target.Id,TargetHealth.State,TargetHealth.Reason]'
```

**If the target is `unhealthy` or `initial`:** most likely still loading. A large model takes several
minutes, during which `/health` does not answer, and the health check then needs consecutive
successes. Watch the engine log for `Application startup complete`.

**If there is no target at all:** the task is not running. See the `TaskFailedToStart` section.

**If the target is healthy but you still get 503:** mid-deployment. The old task drained before the
new one passed its health check.

---

## Symptom: 504 from the endpoint on a long request

Something in front of the engine gave up. Two timers sit between caller and model, and a non-streaming
completion sends nothing until generation finishes, so both cover the whole request:

| Hop | Timer | Returns |
|---|---|---|
| CloudFront | read timeout, **120 s** (`CLOUDFRONT_READ_TIMEOUT_S`; the quota ceiling without a support request) | 504 from CloudFront, with a CloudFront error page |
| Load balancer | idle timeout, **300 s** | 504 from the ALB, JSON body |

CloudFront's is the one you will hit. The engine log will show the request completed:

```bash
aws logs tail "$LOG_GROUP" --region "$REGION" --since 10m | grep -i "finished\|generated"
```

Fixes, in order:

1. **Stream.** Both timers reset on every byte, so generation can run as long as it needs.
2. **Reduce the offered load.** A 504 on a short output means requests are queueing and the fleet is
   past its knee. See "Choosing an operating concurrency" in [tuning.md](tuning.md).
3. **Raise the CloudFront read timeout** only for non-streamed requests over two minutes. Above 120 s
   this needs a CloudFront quota increase (*Response timeout per origin*) first.

Decode speed depends on hardware and model, not fleet size. More instances raise throughput, not
per-request speed.

## Symptom: 403 from the endpoint

Working as designed: the load balancer's default action rejects anything without a valid key.

```bash
aws cloudformation describe-stacks --stack-name GpuLlmServing --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiKeyValue`].OutputValue' --output text
```

Send it as `Authorization: Bearer <key>`. Any other header name gets the same 403.

---

## Symptom: 404 on `/v1/responses`

The Responses API is newer than Chat Completions and absent from older engine builds.

```bash
curl -s "$ENDPOINT/v1/models" -H "Authorization: Bearer $KEY"
```

If Chat Completions works and `/v1/responses` returns 404, the pinned engine does not expose it. Use
`/v1/chat/completions`, or bump the base image in `container/Dockerfile` and rebuild.

`scripts/test_endpoint.py` reports which of the two is available.

---

## Symptom: the engine exits during start with `Insufficient space in /dev/shm: 32768 MiB required`

Configuring a host-memory KV tier (`extraArgs: --kv-offloading-size 32`) and the engine start ends in
`RuntimeError: Insufficient space in /dev/shm: 32768 MiB required, 8192 MiB free`, raised from
`vllm/v1/kv_offload/cpu/shared_offload_region.py`. ECS restarts the task, so this presents as a
crash loop and a `cdk deploy` that never finishes. **Cause:** the tier is an mmap file in `/dev/shm`
(the log line above it reads `Created mmap file /dev/shm/vllm_offload_<engine-id>.mmap`), and the
container's shared memory is a fixed size. **Fix:** this stack sets `/dev/shm` to half the container's
memory, so a tier up to that size starts; a larger tier needs a smaller `--kv-offloading-size` or a
larger instance. Cancel the stuck update with `aws cloudformation cancel-update-stack` first.
docs/tuning.md, *Offloading the cache to host memory moves the capacity wall*.

## Symptom: the engine refuses `--kv-cache-memory-bytes` with "larger than the available KV cache memory"

`ValueError: To serve at least one request with the model's max seq len (262144), 12.0 GiB KV cache is
needed, which is larger than the available KV cache memory`. **Cause:** the engine checks that one request
at `maxModelLen` fits in the cache, and a pinned cache size is compared against the model's full context,
not against what you intend to send. **Fix:** set `maxModelLen` in the same change to something one
request's worth of KV fits inside.

## Symptom: `Insufficient space in /dev/shm`

```
RuntimeError: Insufficient space in /dev/shm: 160 MiB required, 64 MiB free.
```

The task exits within seconds of starting. Only with `tensorParallel` above 1: one worker per GPU
passes tensors through a POSIX shared-memory ring buffer in `/dev/shm`, and Docker's default 64 MiB is
not enough. TP=1 does not use this path, so a working single-GPU deployment proves nothing.

**Already fixed in this project:** the task definition sets 8 GiB unconditionally
(`shared_memory_size` in `infra/serving_stack.py`). Do not make it conditional on the tensor-parallel
degree: the degree can also be raised through `EXTRA_ARGS`, where the stack cannot see it.

Check:

```bash
aws ecs describe-task-definition --task-definition "$TASK_DEF" --region "$REGION" \
  --query 'taskDefinition.containerDefinitions[0].linuxParameters.sharedMemorySize'
```

---

## Symptom: an engine dies under load with nothing in its log

The task's log ends with the API server reporting `EngineDeadError: EngineCore encountered an issue`
and exiting 0; the engine process itself wrote nothing. ECS replaces the task, the load balancer counts
502s and 504s meanwhile. When the engine process is killed from outside, the reason is on the **host**:

```bash
aws ssm send-command --region "$REGION" --instance-ids <instance> --document-name AWS-RunShellScript \
  --parameters 'commands=["dmesg -T | grep -i -E \"out of memory|killed process|xid\" | tail"]'
```

- `Out of memory: Killed process ... python3`: host RAM. The container gets 60% of the instance's RAM;
  a model that needs more host memory than that at runtime needs a size with more RAM per GPU.
- `NVRM: Xid 13 / 31 / 43 ... name=python3`: a GPU kernel faulted (illegal memory access, MMU fault).
  Not a capacity problem; a kernel bug for this model, precision and GPU combination. Seen on this
  hardware with a 120B NVFP4 mixture-of-experts under 4,000-token prompts at 32 or more requests per
  engine. Try the alternative kernel backend the model card names, through `extraEnv` (for FP4 MoE:
  `VLLM_USE_FLASHINFER_MOE_FP4: "0"`), or a different checkpoint precision.

---

## Symptom: the engine starts, then dies once traffic arrives

Usually `502`s under load after a clean startup: the target closed the connection mid-request. Look
for:

```
MemoryError: CUDA out of memory. Tried to allocate 2.02 GiB.
GPU 0 has a total capacity of 94.97 GiB of which 1.82 GiB is free.
  FusedMoeRunner::getWorkspaceInfo(...)
```

The cause is too little VRAM headroom: startup profiling passes, then real concurrency needs more. Two
configurations produce it.

**Cause A: `gpuMemoryUtilization` is too high.** Keep it at or below 0.97 (synth rejects higher) and
prefer the tested 0.95. Dense models are less tolerant: a dense 27B crashed at 0.97 where a comparable
mixture-of-experts model was stable.

**Cause B: weights that fill most of the card, plus `kvCacheDtype: fp8`.** The trigger is the VRAM
footprint, not the precision; a large quantised model hits it too, and `cdk synth` warns on the same
grounds. The chain:

1. fp8 halves bytes per cached token, so the cache holds roughly twice as many tokens.
2. More cached tokens means the scheduler runs a larger batch.
3. A larger batch means a larger per-step workspace for the fused mixture-of-experts kernels.
4. bf16 weights already take most of the card (around 57 of 95 GiB for a ~30B model), so nothing is
   left for step 3.

A 30B in fp8 (about 29 GiB of weights) has headroom and does not hit this. A 70B in fp8 (about 65 GiB
on one card) does.

**Fix for cause A: lower `gpuMemoryUtilization`.**

**Fix for cause B: set `kvCacheDtype: auto`**, or use an instance with more GPUs so the weights occupy
less of each. Lowering utilisation also stops the crash but is a net loss: at 0.90 the same deployment
measured 37,877 tok/s against 46,808 for an unquantised cache at 0.95.

Confirm what the engine used:

```bash
aws logs tail "$LOG_GROUP" --since 1h --region "$REGION" | grep -iE "kv.cache|gpu_memory_utilization|out of memory"
```

`--kv-cache-memory`, which the engine suggests at startup, measured as a no-op and will not help. See
[tuning.md](tuning.md).

---

## Symptom: the engine dies when an evaluation asks for prompt logprobs

`scripts/quality.py`, or any client sending `/v1/completions` with `echo` and `logprobs`, kills a healthy
engine within seconds; the load balancer answers `502 Bad Gateway` until ECS has restarted the task.
The log:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.97 GiB. GPU 0 has a total capacity
of 94.97 GiB of which 2.42 GiB is free.
```

**Cause:** prompt logprobs make the engine keep the full vocabulary of logits for every token of the
prefill chunk (151,936 × up to 8,192 tokens in fp32, plus the copies sorting makes), outside the KV
cache budget. The serving default of `gpuMemoryUtilization: 0.95` leaves no room; 0.85 did not either.

**Fix:** for the evaluation, deploy with `gpuMemoryUtilization: 0.80` and `maxNumBatchedTokens: 2048`
(the buffer scales with the prefill chunk), and keep the log-likelihood requests at a few in flight.
Measured to hold at 4 in flight with 10-shot prompts. Neither setting changes what is scored; put the
serving values back afterwards.

---

## Symptom: the request carries `tools`, and the answer is text that looks like a tool call

The model wrote its tool call in its own text format and nothing turned it into the `tool_calls` field,
so the client sees `content` such as `<tool_call>{"name": ...}</tool_call>` and the agent stalls on its
first step. **Cause:** no tool parser configured; the engine only parses tool calls when told which
format to expect. **Fix:** set `toolCallParser` in `config.yaml` to the model family's parser (`hermes`
for Qwen3, `qwen3_coder` for Qwen3-Coder, `openai` for gpt-oss; `vllm serve --help` lists them) and
redeploy. A wrong parser for the family fails the same way, silently. docs/tuning.md, *Tool calling*.

## Symptom: an agent's steps fail with `504 Gateway Timeout` from CloudFront, and retries never succeed

The step's request took longer than 120 s, the endpoint's limit for a non-streamed answer, so CloudFront
gave up on the origin while the engine was still generating. A coding agent's step carries its whole
trajectory, tens of thousands of tokens, and on a busy engine that is over two minutes. **Fix:** stream
the request (no limit applies), lower the load on the engine, or shorten the trajectory. Retrying the
same request does not help: it times out the same way while the engine also finishes the abandoned one.
docs/tuning.md, *Tool calling*.

## Symptom: `400` from `/v1/completions` with `stop` in the request

The engine accepts a stop string or a list of at most four; a fifth returns `400` with a `too_long`
validation error. lm-evaluation-harness's HumanEval task sends five, which is one reason
`scripts/quality.py` leaves code tasks out. Trim the list at the client.

---

## Symptom: an engine dies during the weight download with `429 Too Many Requests`

The engine log ends in a traceback from the Hugging Face client: `HTTP status client error (429 Too
Many Requests)`. Several instances pulling the same model anonymously at the same time hit the
unauthenticated rate limit; ECS replaces the task and the second attempt usually succeeds, so the only
cost is a slower rollout (21 minutes instead of 14 on an eight-instance fleet). Set `hfTokenSecretName`
even for public models; authenticated requests have a far higher limit.

---

## Symptom: the engine fails to start after a `modelId` change, and the error names no cause

The new task logs `WorkerProc initialization failed due to an exception in a background process` or a
weight-loading traceback with nothing useful above it, and restarts every few minutes. The previous model
worked on the same instance.

**Cause: the host disk is full.** The weights cache is a host directory sized by the root volume (500 GiB
here), and it keeps every model the instance has ever served. Two large checkpoints do not fit: a 236 GB
FP8 build plus the 470 GB bf16 build of the same model overran the volume by a wide margin, and the
download failed mid-file without saying so.

```bash
aws ssm start-session --target "$INSTANCE_ID" --region "$REGION"
df -h /
du -sh /opt/modelcache/hf/hub/models--*
```

**Fix: delete the cache directory of the model you no longer serve, then let the task restart.** The
download resumes from the completed files. Delete `*.incomplete` blobs left by the failed attempts as
well; they are not reused. If you switch models often, replace the instance instead (scale the ASG to
zero and back), which gives you an empty volume.

---

## Symptom: model loading takes far longer than expected

**Cause A: the first task on a new instance is downloading the weights from Hugging Face.** Tens of
GiB through the NAT gateway; later tasks and restarts on that instance hit the shared host cache.

**Cause B: the root volume is throttling.** At the gp3 default of 125 MB/s, a 57 GiB model takes ~8
minutes to read. This project provisions 500 MB/s. Check:

```bash
aws ec2 describe-volumes --filters Name=attachment.instance-id,Values=<id> --region "$REGION" \
  --query 'Volumes[].[Size,VolumeType,Throughput,Iops]'
```

Raise it on a running volume with no downtime:

```bash
aws ec2 modify-volume --volume-id <vol-id> --throughput 1000 --iops 6000 --region "$REGION"
```

**Cause C: the disk is full.** This project mounts a host directory for the weights cache so restarts
share one copy. Without it, each container start writes its own copy into its writable layer, which a
stopped container keeps; a crash-looping task can fill a 500 GB volume in under an hour. Replacing a
model can also leave old weights behind. Loading never finishes, with no explicit error:

```bash
aws ssm send-command --instance-ids <id> --document-name AWS-RunShellScript --region "$REGION" \
  --parameters 'commands=["df -h /","docker container prune -f","df -h /"]'
```

---

## Symptom: a deployment reverts to an older configuration by itself

The ECS deployment circuit breaker rolled back after repeated start failures. This project disables
it: while iterating, it reverts to a revision that may also be broken, and the two contend for the same
GPUs indefinitely. If you enabled it, the rollback shows in:

```bash
aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" \
  --query 'services[0].deployments[].[taskDefinition,rolloutState,rolloutStateReason]'
```

---

## Symptom: no GPU free, but no task running

A leaked GPU reservation: the ECS agent holds the GPU against a task that no longer exists, usually
after repeated failed deployments. New tasks sit in PROVISIONING with no error anywhere. Look for an
instance with zero free GPUs and no running tasks:

```bash
aws ecs describe-container-instances --cluster "$CLUSTER" --region "$REGION" \
  --container-instances $(aws ecs list-container-instances --cluster "$CLUSTER" \
    --query 'containerInstanceArns[]' --output text) \
  --query 'containerInstances[].{
      instance:ec2InstanceId,
      runningTasks:runningTasksCount,
      registeredGPU:registeredResources[?name==`GPU`].integerValue|[0],
      freeGPU:remainingResources[?name==`GPU`].stringSetValue}'
```

`registeredGPU` non-zero, `freeGPU` empty, `runningTasks` 0 → leaked reservation. (`freeGPU` empty
and `registeredGPU` 0 is the driver problem; see the PROVISIONING section above.)

Drain the service to zero, wait for the reservation to clear, then scale back up:

```bash
aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" --desired-count 0 --region "$REGION"
# wait until list-tasks returns nothing and freeGPU is populated again
aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" --desired-count 1 --region "$REGION"
```

---

## Symptom: you scaled the service to zero and it came back

You set `--desired-count 0` by hand. When `maxInstanceCount > instanceCount`, the stack registers a
scalable target whose `MinCapacity` is `instanceCount`, and Application Auto Scaling pushes the service
straight back to it. Any capacity set with the CLI is drift: the next `cdk deploy` restores the template.

**Fix:** set `instanceCount: 0` and `maxInstanceCount: 0` in your config and deploy. README,
*Scale to zero without tearing down*.

If you also registered a scalable target by hand, CloudFormation does not own it and it survives every
deploy. Remove it:

```bash
aws application-autoscaling deregister-scalable-target --service-namespace ecs --region "$REGION" \
  --resource-id "service/$CLUSTER/$SERVICE" --scalable-dimension ecs:service:DesiredCount
```

---

## Symptom: an instance is `InService` in the ASG but does not exist

```
aws ec2 describe-instances --instance-ids i-... --region "$REGION"
An error occurred (InvalidInstanceID.NotFound)
```

while `describe-auto-scaling-groups` still lists it as `InService`/`Healthy`.

Cause: the instance was reclaimed (spot interruption or any termination) while the ASG's replacement
processes were suspended. The ASG cannot remove or replace it, so the phantom counts toward desired
capacity and nothing new launches. The service stays at zero tasks indefinitely.

```bash
aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$ASG" --region "$REGION" \
  --query 'AutoScalingGroups[0].SuspendedProcesses'
aws autoscaling resume-processes --auto-scaling-group-name "$ASG" --region "$REGION"   # then it self-heals
```

See *Spot and instance protection* in [README.md](../README.md) for when suspending is worth it.

---

## Which kernels did the engine pick?

Two deployments of the same weights on two GPUs, or two formats on one GPU, are not comparable until
you know which kernels ran. The engine names them at startup:

```bash
aws logs tail "$LOG_GROUP" --since 30m --region "$REGION" \
  | grep -E "attention backend|MoE backend|Marlin|native support|cudagraph|Available KV cache memory"
```

What to expect, and what it means:

| Line | Meaning |
|---|---|
| `Using FLASHINFER attention backend` or `Using FLASH_ATTN attention backend` | which attention kernels; they differ in fp8-cache handling |
| `Using DEEPGEMM Fp8 MoE backend` (this GPU) or `Using TRITON Fp8 MoE backend` (H100) | native fp8 expert kernels |
| `Using 'MARLIN' Mxfp4 MoE backend` | 4-bit weights dequantised to bf16 for the matmul: weight-only, not native 4-bit compute. The only MXFP4 path vLLM 0.28.0 has for this GPU generation |
| `Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM` | native 4-bit compute |
| `Your GPU does not have native support for FP4 computation` | a fallback is in use; expect weight-only performance |
| `Graph capturing finished in N secs, took X GiB` | CUDA graphs on; X is subtracted from the KV cache |

A fallback is not a bug, but a number measured on one is a number for that kernel, not for the format.
docs/tuning.md records the kernel next to every 4-bit result for that reason.

---

## Symptom: throughput is far below what the hardware should give

Compute the decode ceiling and compare. See *Interpreting your own measurements* in
[tuning.md](tuning.md). Briefly:

- **Reaching >60% of the ceiling:** bandwidth-bound. Working as intended; reduce bytes per token
  (quantization) or get a faster GPU.
- **20–40%:** overhead-bound. Usually too much tensor parallelism. Try a lower degree, or fp8 to fit
  fewer GPUs.
- **<20%:** check `tensorParallel` first, then whether the GPUs share a slow interconnect.

Measure the right thing: aggregate throughput rises with concurrency while per-request speed falls.
Tuning on one alone misleads.

---

## Symptom: a configuration change has no effect, and the deployment looks healthy

The service goes green, requests succeed, and the engine runs its previous configuration. Two causes;
both record measurements under the wrong label.

**Cause A: an ECS `command` does not override the image's `ENTRYPOINT`.** `container/Dockerfile` sets
`ENTRYPOINT ["/usr/local/bin/serve"]`, so `command` is passed to `serve` as arguments, and `serve`
ignores arguments it does not recognise. The deployment is healthy and your flag was never applied.

To replace the entrypoint, override `entryPoint` and set `command: []`. **For engine flags, use
`extraArgs` in `config.yaml`.**

The container logs the exact command it runs:

```bash
aws logs tail "$LOG_GROUP" --since 10m --region "$REGION" | grep '\[serve\]'
```

For multi-GPU configurations, check the GPUs are in use:

```bash
aws ssm send-command --instance-ids <id> --document-name AWS-RunShellScript --region "$REGION" \
  --parameters 'commands=["nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"]'
```

An arm labelled as using two GPUs that reports `0, 1 MiB` and `1, 93991 MiB` is using one.

**Cause B: you edited `config.yaml` but did not redeploy.** Without `cdk deploy` the service keeps the
old task definition. Confirm the running revision:

```bash
aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" \
  --query 'services[0].taskDefinition'
```

---

## Getting a shell on the host

The instance role includes SSM, so no SSH key or bastion is needed:

```bash
aws ssm start-session --target <instance-id> --region "$REGION"
```

Useful once there:

```bash
nvidia-smi                                    # driver and GPU health
tail -100 /var/log/ecs/ecs-agent.log          # why a task did not start
docker ps -a                                  # container state
df -h /                                       # disk
```
