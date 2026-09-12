# AGENTS.md

Instructions for coding agents working in this repository. Read it fully before changing anything.
Humans: start with `README.md`.

## Project

One CDK stack that serves an open-weight LLM with vLLM on ECS GPU instances (g7e, plus p5 for comparison
runs), behind an internal
ALB and CloudFront, with an OpenAI-compatible API and a Bearer key. One config file. It is built around
vLLM on purpose (entrypoint flags, tuning keys, engine metrics on the dashboard); do not generalise it
to other engines. The docs carry as much value as the code: every number in them was measured on this
hardware, and the point of the sample is that a reader can deploy it, understand it, and tune it without
reading the source.

Python 3.11 or 3.12, CDK v2 (Python), pytest. No other build system.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && npm install -g aws-cdk
```

## Commands

| command | when |
|---|---|
| `python3 -m pytest tests/ -q` | before every commit; must pass; needs no AWS credentials |
| `cd infra && npx cdk synth` | after any stack change; needs credentials (one prefix-list lookup) |
| `cd infra && npx cdk deploy` | ~15 min for infrastructure, then minutes while the model loads |
| `python3 scripts/build_image.py` | after ANY change under `container/`; a deploy alone runs the old entrypoint |
| `python3 scripts/endpoint_info.py` | prints endpoint, key, model name, dashboard, and the two commands below |
| `python3 scripts/test_endpoint.py <url> --key <key>` | smoke test: health, both APIs, streaming, 64 concurrent |
| `python3 scripts/benchmark.py <url> --key <key>` | concurrency sweep; the numbers to size a fleet from |
| `python3 scripts/size_fleet.py --engine-rps ... --demand-rps ... --price-per-hour ...` | one engine's measured rate into instances and price per million tokens |
| `python3 scripts/quality.py <url> --key <key>` | standard benchmark suite through the endpoint (lm-eval); compare two deployments |
| `python3 scripts/extraction.py <url> --key <key>` | structured extraction into a JSON object, three ways to ask; compare two deployments |

## Layout

```
config.yaml              the only file a user edits; each key has a comment and a docs/tuning.md section
config.local.yaml        gitignored overrides: API key, image URI, anything account-specific
infra/app.py             loads and validates config, generates the API key once, synthesises
infra/hardware.py        instance catalog and derived values; pure functions, no AWS calls
infra/serving_stack.py   the stack: VPC, ECS, ALB, CloudFront, service, metrics sidecar, dashboard, alarms
container/serve          entrypoint: environment variables to vLLM flags; baked into the image
container/Dockerfile     vLLM base image plus the entrypoint
scripts/                 build_image, endpoint_info, test_endpoint, benchmark, size_fleet, quality, extraction
tests/                   test_hardware (catalog, validation), test_template (synthesised template), test_app (config loading, key file), test_scripts (the scripts import, --help runs, their pure functions)
docs/tuning.md           measurements per config key, model and precision choice, sizing, autoscaling, reading the dashboard
docs/troubleshooting.md  symptom, cause, fix
measurements/            every row behind the docs: benchmarks.csv, quality.csv, agentic.csv
```

## Rules

1. **Do not add machinery for problems that have not happened.** Every construct, script and config key
   here earned its place by a failure or a measurement. A new one needs the same. Removing code is
   usually the better change.
2. **A new config key needs all of:** a comment in `config.yaml` saying what it does and the one trap,
   a default, validation in `hardware.py` or `app.py` raising a one-line `ConfigError`, a test, and a
   section in `docs/tuning.md` that the comment names. `0` means "let the engine decide".
3. **Nothing account-specific in tracked files.** Account ids (`111122223333` in tests only), hostnames,
   distribution ids, tokens, keys, company or customer names. If it came from a real deployment, it
   belongs in `config.local.yaml` or nowhere.
4. **Tests stay credential-free.** Context lookups are seeded in `AZ_CONTEXT` in `tests/test_template.py`.
   Test names are sentences saying what would break; docstrings say what broke before.
5. **Assert values, not presence.** Synth cannot catch a wrong IAM action, header name, alarm threshold
   or an apostrophe in a security-group description. All of those reached a live deploy once.
6. **Keep headings and anchors.** `config.yaml` comments, the README and this file link to sections in
   `docs/` by title.
7. **Never delete a measurement or a documented failure to shorten a file.** Shorten the prose around it.
8. **Numbers come from measurements, quality scores from published benchmarks.** A throughput claim
   needs a `benchmark.py` run; a quality claim needs a standard task through `quality.py` or a published
   agentic harness. `extraction.py` is a same-prompt comparison between deployments, never a leaderboard
   score. Say what was not measured instead of estimating.

## Writing

Engineer to engineer. Short sentences. State the fact and the action; one sentence of justification per
default at most. Numbers in tables. No em-dashes. No "deliberately", "worth knowing", "genuinely",
"verified", "world-class", "battle-tested", "seamless", "robust", "leverage". A statement is a statement;
when something is estimated or not measured, say so instead of labelling the rest as verified.

Code comments explain why, not what, and record the failure that motivated a non-obvious choice.

## Common changes

- **Engine flag or startup behaviour**: edit `container/serve`, rebuild the image, deploy, confirm the
  `[serve] vllm serve ...` line in the task log shows the change.
- **Stack resource**: edit `serving_stack.py`, add or update a template test, synth with the shipped
  `config.yaml` in isolation (not only your local overrides), deploy once.
- **Instance type**: add to the catalog in `hardware.py` with vCPU, GPUs and host RAM from
  `ec2 describe-instance-types`; the tests check derived values. Check the GPU count: `g7e.8xlarge` has
  one GPU, not four.
- **Dashboard or alarm**: widget titles are questions in plain language; alarm descriptions are written
  for someone on a phone who did not deploy this. Test the threshold value.

## Helping someone choose a deployment

Most people arriving with an agent want a running endpoint that fits a workload, not a code change.
The user decides; your job is to ask the questions whose answers change the configuration, offer the
measured default for each, and say what the choice costs. Ask them together, early, in plain words. Do
not ask about things that do not change the deployment.

| Ask | Why it matters | Default if they do not know | Config and reading |
|---|---|---|---|
| **What will call it: a chat UI, a batch job, a coding agent or agent framework, your own tool-calling harness?** | Agents send `tools` and read `tool_calls`; without a parser the call comes back as text and the agent stalls silently. Agent steps are long requests. | Enable tool calling whenever anything but plain chat is possible: `toolCallParser` for the model family (`hermes` Qwen3, `qwen3_coder` Qwen3-Coder, `openai` gpt-oss, `llama3_json`, `mistral`). Tell agent users to stream (120 s non-streamed limit). | `toolCallParser`; tuning.md *Tool calling*, *What quantisation costs an agent* |
| **Which model?** The user's choice, or the job it is for. | The model decides more of the quality than any hosting or precision setting does: on the same agentic benchmark the families measured differ by 20 to 40 points, the precisions within a family by one. The docs measure precision effects across families, not models against each other. | The shipped `Qwen/Qwen3-30B-A3B-Instruct-2507` for a first deployment; otherwise the model the user names, checked against `Will my model fit?`. Offer to score their candidates on their own task through the endpoint rather than ranking them from the docs. | `modelId`, `estimatedParamsBillions`; tuning.md *Choosing a model to host*, *Will my model fit?* |
| **Does it think?** Reasoning models, and whether they want the chain of thought. | Reasoning effort is the capacity setting (tokens per answer 1.7× to 3.5×); a chain of thought eats every answer cap and every benchmark; clients need it separated from the answer. | Thinking off for a Qwen3 thinking model unless they ask for it (`extraArgs: --default-chat-template-kwargs '{"enable_thinking": false}'`); with thinking on set `reasoningParser` and size answer caps for it. | `reasoningParser`, `extraArgs`; tuning.md *Reasoning models* |
| **How exact must answers be, and against what?** | fp8 weights and fp8 KV cost nothing measurable on any task or agent benchmark; 4-bit costs 0.5 to 2 points and one answer in twenty, and one community 4-bit build was broken for tool use while passing chat benchmarks. | Publisher fp8 (or `quantization: fp8` on a bf16 checkpoint: same result), `kvCacheDtype: fp8`. 4-bit only from a calibrated publisher, and only after `quality.py` and, for agents, a tool-use score against the fp8 deployment. | `quantization`, `kvCacheDtype`; tuning.md *Quantisation is two independent decisions*, *What quantisation costs in answers*, *What quantisation costs an agent* |
| **Prompt and answer shape.** Typical and maximum input tokens, output tokens, how much of a prompt repeats (system prompt, documents, multi-turn). | Prefill slows with prompt length (36k tok/s at 4k, 8k at 64k on one GPU); a cached prefix is the whole game for long prompts; multi-turn traffic needs stickiness to hit the cache; when the conversations in flight outgrow the GPU cache the hit rate falls to zero with no preemptions to show for it; structured output costs 15% of decode. | `maxModelLen: 0` (the model's own maximum); `stickySessions: true` when clients keep a cookie per conversation and traffic is multi-turn; `enablePrefixCaching` stays on. No host-memory KV tier (`--kv-offloading-size`) unless the working set that comes back exceeds the cache and the tier can be sized above it: a 32 GiB tier behind a 58 GiB cache at 1.4x served zero hits, and the tier is per engine. Structured output as the client asks for it, with the cost budgeted. | `maxModelLen`, `stickySessions`, `extraArgs`; tuning.md *Long prompts*, *Prefix caching is a routing decision*, *Offloading the cache to host memory*, *Structured output* |
| **Latency budget and demand.** p95 seconds per request, requests per second at peak, whether a slow answer is a failure or a delay. | One GPU holds a knee at a given concurrency; the budget picks the concurrency and the concurrency picks the fleet. Autoscaling is 11 minutes to capacity: burst insurance, not sizing. | Size a fixed fleet from one measured engine (`benchmark.py` then `size_fleet.py`), 15% headroom, `maxInstanceCount` equal or slightly above. | `instanceCount`, `maxInstanceCount`, `scalingRequestsPerTarget`, `latencyAlarmSeconds`; tuning.md *Choosing an operating concurrency*, *Sizing a fleet* |
| **Where, and how they buy.** Region, AZs, spot or on-demand, quotas. | g7e exists in few regions and fewer zones; capacity is thin in the afternoon in several regions, spot and on-demand alike; quotas are in vCPU. | The region closest to callers that offers `g7e.2xlarge`; `useSpot: true` for a first deployment; `availabilityZones` limited to zones that offer the type. | `region`, `availabilityZones`, `useSpot`; README *Before you start*; troubleshooting *the ASG never launches an instance*, *MaxSpotInstanceCountExceeded* |
| **Gated or private weights?** | Llama, Gemma and private repos need a token; eight instances pulling anonymously hit 429. | A `gpu-llm-serving/hf-token` secret in the region, `hfTokenSecretName` set, for public models too. | `hfTokenSecretName`; troubleshooting *429*, *403* |
| **Who watches it, and who gets paged?** | Alarms change state silently without a topic. | `alarmTopicArn` set if anyone should know; `latencyAlarmSeconds` at their budget. | README *Operating*; tuning.md *Watching a running fleet* |

Defaults that need no question: `gpuMemoryUtilization: 0.95`, `enablePrefixCaching: true`,
`enableExpertParallel: false`, `tensorParallel: 0` (derived), `replicas: 0` (one engine per GPU),
`maxNumBatchedTokens: 0`, no host-memory KV tier, no speculative decoding unless a publisher draft or MTP head exists for the model
(then it is one line in `extraArgs`, +9% to +41%). Each is measured in *Engine tuning* and *Settings that
sound useful and measurably are not*; do not change one without a measurement.

Two things to say before they commit to anything: the weights are a choice with a measured price and a
measured quality cost, both in the README's Configure table and in *Every weight option, measured*; and
the fleet size follows from a measurement on one instance, not from a guess. Offer to run
`benchmark.py` with their prompt shape and, when they are weighing a precision, `quality.py` against
both deployments. For agent workloads add a tool-use score; `docs/tuning.md` names the harnesses used.

If they only want to learn, point them at the reading paths at the top of `docs/tuning.md`, then answer
from the docs and the measurements, citing the section. Do not deploy anything to answer a question the
docs already answer.

## Helping someone deploy

Walk them through this order and check each step before the next.

1. **Preflight, before anything is created.** Region offers `g7e.2xlarge`
   (`aws ec2 describe-instance-type-offerings ... --location-type availability-zone`); if only some
   zones do, put them in `availabilityZones`. Quota for the purchase model they will use: on-demand
   `L-DB2E81BA`, spot `L-3819A6DF`, in vCPU, 8 per `g7e.2xlarge`. `cdk bootstrap` once per account and
   region. CloudFront VPC origins must be supported in the region (they are in all commercial regions
   that offer g7e as of this writing).
2. **Configure**, from the answers above: `region`, `instanceType`, `modelId`, `toolCallParser`,
   `quantization`, `instanceCount` and `maxInstanceCount` in `config.yaml`; the API key and image URI
   land in `config.local.yaml`. The default fleet is 16 and needs 128 vCPU of quota; with less, lower
   both counts. For a first deployment suggest `useSpot: true`: on-demand g7e.2xlarge had no capacity in
   several regions on the same afternoon, and spot in the same regions launched within a minute more
   often than not.
3. **Gated model?** Create the `gpu-llm-serving/hf-token` secret in the deployment region and set
   `hfTokenSecretName`. A 401 while pulling means no token; a 403 means the token's account has not
   accepted that model's licence.
4. **Build, then deploy.** `build_image.py --write-config`, then `cdk deploy`. Three to five minutes
   into the deploy, read the ASG scaling activities. `InsufficientInstanceCapacity` or
   `UnfulfillableCapacity` will not resolve soon: cancel the update, switch purchase model or region,
   deploy again. Do not let CloudFormation wait; it will, for up to an hour, and a crash-looping engine
   locks the stack for longer.
5. **Confirm.** `endpoint_info.py`, then the smoke-test command it prints. If tool calling was asked
   for, send one request with a `tools` array and check that `tool_calls` comes back, not text. Then
   `benchmark.py` with their prompt shape if they need capacity numbers; set `maxInstanceCount` equal to
   `instanceCount` first so autoscaling does not move under the measurement.
6. **Read the dashboard with them.** `DashboardUrl` output. Top rows are the load balancer's view,
   bottom three rows are the engine's: waiting requests mean saturation, KV cache near 100% means preemption
   next, any preemptions mean lost work. Alarms fire only on sustained conditions.
7. **Leaving.** Park with both counts at 0 and deploy; instances are gone in about five minutes, and in
   a quota-bound region the next deploy must wait until they are, because spot requests hold quota until
   the instance is terminated. Destroy only after they are gone. A destroy started while the CloudFront
   VPC origin is still `Deploying` fails; wait for `Deployed` and retry.

Names that are account-wide, not regional, and already carry the region so two stacks can coexist:
the CloudWatch dashboard and the CloudFront VPC origin. IAM roles are CDK-generated and unique.

## Operating a test deployment

- Use `instanceCount: 6`, `maxInstanceCount: 8`, `useSpot: true` in `config.local.yaml`. The shipped
  default is 16 and needs a quota increase.
- `UPDATE_IN_PROGRESS` blocks further deploys. `aws cloudformation cancel-update-stack` rolls back to
  the previous task definition.
- A changed `modelId` on a 500 GiB root volume can fill the disk with two sets of weights; the
  troubleshooting doc has the symptom.
- GPU instances are the entire cost. Never leave them running after a test unless asked to.

## Commits

Imperative subject under 70 characters. The body says why, names the failure or measurement behind the
change, and states what was deployed or tested. One logical change per commit.

## Done means

- Tests pass.
- Stack changes were synthesised from the shipped config and deployed once.
- `container/` changes were rebuilt into the image and seen in a task log.
- Docs changes: no em-dash character in the repo, every number that was there is still there.
- The fleet is parked.
