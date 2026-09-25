# Choosing an instance type and tuning the engine

The defaults in `config.yaml` come from the measurements in this document. They are here so you can
tell when your workload justifies a different value, and so you know what was not measured.

Three questions settle most of the configuration:

1. **Does the model fit one GPU in fp8?** Yes: one engine per GPU, `tensorParallel: 1`, the smallest
   instance size (*Choosing an instance type*). No: the smallest tensor-parallel degree that fits, then
   as many engines as the host allows (*Topology*).
2. **Do your prompts share a prefix, and with whom?** With every request (one system prompt): the cached
   figures apply and the fleet is about half the unique-prompt size. Within a conversation or a tenant:
   the cache only helps if the same engine sees the follow-up, so set `stickySessions` or route on the
   prefix (*Prefix caching is a routing decision*). Not at all: size on the unique-prompt column.
3. **Which wall are you at?** Long prompts and short answers: prefill, so compute and the prefix cache
   decide. Short prompts and long answers: decode, so bandwidth, fp8 weights and an fp8 cache decide.
   One command tells you (*Interpreting your own measurements*).

Every number here was measured with vLLM 0.28.0; the kernels it picked are named where they matter, and
the rows themselves are in [measurements/](../measurements/README.md).

Reading paths, if you have one job today:

- **Sizing a fleet:** *Choosing an instance type*, then *Choosing an operating concurrency*, then *Sizing
  a fleet*. `scripts/benchmark.py` on one instance and `scripts/size_fleet.py` turn the measurement into
  instances and a price per million tokens.
- **Choosing a model or a precision:** *Choosing a model to host*, then *Quantisation is two independent
  decisions*, then *What quantisation costs in answers*; `scripts/quality.py` runs the check.
- **A fleet that is slower than it should be:** *Which wall are you at?*, *Interpreting your own
  measurements*, then *Watching a running fleet* and [troubleshooting.md](troubleshooting.md).
- **A model that needs several GPUs:** *Tensor parallelism*, then *Topology*.
- **Serving agents or tool calls:** *Tool calling*, *Structured output*, then *What quantisation costs an
  agent*; `scripts/extraction.py` checks structured extraction on your own deployment.

---

## The one mental model worth having

A request has two phases with opposite performance characteristics.

**Prefill** reads the prompt. All input tokens go through the model in one pass. Compute-bound; sets
time-to-first-token.

**Decode** generates output one token at a time, each depending on the previous ones.
Memory-bandwidth-bound: every forward pass streams the model's activated weights out of VRAM to
produce one token per sequence.

Same operation, different batch sizes, either side of a crossover:

| Tokens in one forward pass | Bound by | GPU arithmetic units busy |
|---|---|---|
| 1 (decode, one request) | **memory** | almost idle |
| ~8–64 (decode, batched) | memory | rising |
| ~400+ (prefill, or heavy batching) | **compute** | saturated |

Two consequences:

- **Decode speed per request is capped by bandwidth ÷ bytes-read-per-token.** Only reading fewer bytes
  makes a single request faster.
- **Batching is what makes a GPU efficient.** One weight read serving 64 tokens instead of 1 is 64× the
  useful work; KV cache size caps how many requests are in flight.

A third follows from the first two and decides which lever works:

- **Bytes read per token means *activated* weights, for one request.** A mixture-of-experts model with
  30B parameters but 3B active per token reads a tenth of what a dense 27B reads, and a single request
  decodes accordingly. Under load the picture changes: each token in the batch picks its own experts,
  and a batch of 64 touches nearly all of them, so a loaded step reads close to the whole model. The
  mixture of experts still wins because that read is shared by the batch and the arithmetic per token
  stays a tenth. Measured on one fleet, the dense model delivered a third of the MoE's request rate at
  the same precision (*Choosing a model to host*); the batch-1 and loaded ceilings are worked out in
  *Interpreting your own measurements*. Total parameter count says how much VRAM you need; active
  parameter count says how fast it runs.

A multi-GPU engine adds a third wall: the per-step cost of keeping N GPUs in lockstep (an all-reduce per
layer, expert dispatch, and every kernel launched N times), which does not shrink as N grows. A 235B
mixture-of-experts at TP=8 spent about 5% of each decode step reading weights and the rest in that
overhead. Two TP=4 engines beat one TP=8 engine on every loaded shape for that reason (*Topology*).

### Which wall are you at?

Every workload on every model is limited by one of two things at a time, and the fixes do not overlap.
The symptoms below point at it; *Interpreting your own measurements* has the one-command measurement
(memory controller busy: ~50% prefill-bound, 75 to 80% decode-bound).

| Signal | Prefill-bound (compute) | Decode-bound (bandwidth) |
|---|---|---|
| Where the time goes | time to first token grows with load | first token is quick, the rest is slow |
| Prefix cache hits | large gain (measured 2.7× at 4,000-token prompts) | little or none (measured 0 to +46%) |
| Longer prompts | hurt request rate but not tokens/sec | barely matter |
| Longer answers | barely matter | hurt in proportion |
| What helps | fewer input tokens (cache, shorter context), a faster GPU, fp8/NVFP4 | fewer bytes per token (quantisation, MoE), speculative decoding, fewer requests per GPU |
| What does not | batch-size knobs | prefix caching, prompt trimming |

The prefill wall has a number. On this GPU, measured with 4,000-token prompts at 16 to 64 in flight on one
engine, prefill saturated at 27,000 tok/s for a 3.8B-active mixture-of-experts in bf16, 45,000 for the same
weights in fp8, 11,700 for a dense 12B in bf16, 4,300 for a dense 31B in bf16 and 8,100 for it in fp8, and
about 36,000 for a 3.3B-active mixture-of-experts in fp8. Multiply each by twice its activated parameters
and the same figure comes out: 200 to 280 TFLOPS in bf16, 340 to 500 in fp8. So before measuring, expect
prefill tok/s of about 250e12 / (2 × active parameters) at bf16 and roughly twice that at fp8, on this card,
whatever the architecture; a model that lands far below it is on a slow kernel (*Choosing a model to host*).

Read it off the dashboard: *How long does a request take inside the engine?* shows time to first token
against the whole request, and *Is the prefix cache paying off?* says whether hits are even possible.
Short prompts with long answers on a dense model are decode-bound; long documents through a
mixture-of-experts are prefill-bound; most chat traffic is decode-bound on one side of the knee and
prefill-bound past it.

---

## Choosing an instance type

The catalog also knows `p5.4xlarge` and `p5.48xlarge` (H100 SXM, 80 GiB HBM3 at 3,350 GB/s, 1 and 8
GPUs) so the same stack can be measured on Hopper; every figure in this document is g7e unless a table
says otherwise, and NVFP4 checkpoints do not run on an H100. Measured on eight of each with the same
matrix (*Choosing a model to host*, point 10): the H100 is +26% on the fp8 mixture-of-experts and +124%
on a dense bf16 model, at roughly 1.8× the spot price per GPU-hour. On Gemma 4 the gap followed the
kernel, not the spec sheet: its 26B mixture-of-experts ran within −13 to +12% per engine of the g7e (both
GPUs on the Triton attention kernel), its dense 31B in fp8 +20 to +58% (a dense step streams all the
weights, so bandwidth shows), at 2× the spot price per GPU-hour that day ($2.80 against $1.20 to $1.50).

What makes one GPU faster than another for this work is not its count or its memory size. Three
things are:

- **Memory bandwidth** sets decode speed: every generated token streams the active weights and the KV
  cache out of memory once. GDDR7 at ~1,600 GB/s against HBM3 at 3,350 GB/s is 2.1×, so the
  H100's lead is largest on bf16 (twice the bytes per token) and smallest on fp8.
- **Tensor compute** sets prefill speed, and so time to first token and long-prompt throughput.
- **Kernel maturity.** A GPU generation that is months old runs some kernels through fallback paths or
  not at all (*Choosing a model to host*, point 8, and the kernel note under *The evidence*); part of any
  gap between generations is software
  that will close.

The two GPUs measured here differ by about the same factor on both walls, so a cross-GPU delta alone
cannot say which wall a workload is at:

| | RTX PRO 6000 Blackwell Server Edition (g7e) | H100 SXM (p5) | Ratio |
|---|---|---|---|
| Memory bandwidth | 1,597 GB/s | 3,350 GB/s | 2.1× |
| Dense bf16 tensor throughput (vendor peak, no sparsity) | ~0.5 PFLOPS | ~1.0 PFLOPS | ~2× |
| Dense fp8 tensor throughput | ~1.0 PFLOPS | ~2.0 PFLOPS | ~2× |
| GPU to GPU | PCIe 5.0, ~64 GB/s per direction | NVLink, ~450 GB/s per direction | ~7× |

A workload that is bound by either wall should run about 2× faster on the H100. The dense 27B in bf16
did (+124%). The fp8 mixture-of-experts did not (+26%): on both cards it is limited by something that
did not double, which points at per-step overhead and kernel efficiency rather than at either wall.
*Which wall are you at?* has the direct measurement.

Memory size only decides what fits: a card with more of it holds more KV cache (96 GiB here against 80
on the H100) and starts models the smaller card cannot. Read a GPU's bandwidth and its tensor
throughput from the spec sheet, and *Interpreting your own measurements* turns them into a decode ceiling
before you buy.

Every g7e size carries the same GPU: 96 GiB of VRAM at about 1,600 GB/s (the Server Edition runs its GDDR7
at 25 Gbps; the 1,792 GB/s often quoted is the workstation card). Larger sizes add GPUs, vCPU
and host RAM.

Per-GPU columns:

| Instance | GPUs | VRAM | Host RAM | vCPU/GPU | Host RAM/GPU | Use it when |
|---|---|---|---|---|---|---|
| `g7e.2xlarge` | 1 | 96 GiB | 64 GiB | **8** | 64 GiB | The model fits one GPU and you want the lowest cost. **Start here.** |
| `g7e.4xlarge` | 1 | 96 GiB | 128 GiB | 16 | 128 GiB | Same GPU, twice the host: the hedge if 8 vCPU is thin |
| `g7e.8xlarge` | 1 | 96 GiB | 256 GiB | **32** | 256 GiB | Same GPU, the most host per GPU in the family |
| `g7e.12xlarge` | 2 | 192 GiB | 512 GiB | 24 | 256 GiB | The model needs more than 96 GiB |
| `g7e.24xlarge` | 4 | 384 GiB | 1 TiB | 24 | 256 GiB | The model needs more than 192 GiB |
| `g7e.48xlarge` | 8 | 768 GiB | 2 TiB | 24 | 256 GiB | The model needs more than 384 GiB |

vCPU per GPU is not monotonic: the `g7e.8xlarge` gives 32 per GPU, the multi-GPU sizes 24, the
`g7e.2xlarge` 8. This workload is overhead-bound rather than bandwidth-bound (see *Topology* below), so
host CPU per GPU can be part of the ceiling.

### For a model that fits one GPU, buy the smallest instance

Two effects compound.

**Per-GPU price is not flat.** The `g7e.2xlarge` is roughly **19% cheaper per GPU** than every larger
size, which are all priced alike.

**Granularity waste.** The per-GPU rate from *How to size N* below (16,075 tok/s, measured on a
`g7e.2xlarge`) against a demand of 70,028 tok/s is 4.36 GPUs:

| Shape | GPUs | $/GPU/hr | Instances needed | Fleet $/hr | Idle GPUs paid for |
|---|---|---|---|---|---|
| **`g7e.2xlarge`** | 1 | **3.36** | 5 | **16.82** | 0.64 |
| `g7e.4xlarge` | 1 | 4.00 | 5 | 19.99 | 0.64 |
| `g7e.12xlarge` | 2 | 4.14 | 3 | 24.86 | 1.64 |
| `g7e.24xlarge` | 4 | 4.14 | 2 | 33.14 | 3.64 |
| `g7e.48xlarge` | 8 | 4.14 | 1 | 33.14 | 3.64 |

**Five `g7e.2xlarge` cost 32% less than three `g7e.12xlarge`, and 49% less than one `g7e.48xlarge`,
for the same served throughput.**

The `g7e.48xlarge` is the worst value for a model that fits on one GPU. Buy it when a model needs 8
GPUs' worth of VRAM, not for 8 GPUs of throughput.

Unique prompts, same engine configuration, on a `g7e.2xlarge` with a third of the host per GPU:

| | vCPU given to the engine | Uncached prefill @128 | Decode | p95 |
|---|---|---|---|---|
| One engine on a `g7e.12xlarge`, the whole host (48 vCPU, second GPU idle) | 48 | 16,176 | 35.5 | 7.37 s |
| **`g7e.2xlarge`** | **8** | **16,195** | 36.5 | 7.35 s |

Identical, +0.1%.

**But not for prefix-sharing workloads.** On cached prompts the `2xlarge` is **15–25% slower** (23,095
against 29,888 at concurrency 128; 40,227 against 46,645 at 256; single runs, hence the range). If
your prompts share a long prefix, prefer a size with more host per GPU.

Prices are illustrative on-demand rates for one region; re-check them for yours.

### How much host CPU an engine needs, and when it matters

One engine, pinned to a subset of one instance's CPUs with `taskset`; only CPU count varied:

| CPUs given to the engine | Cached @128 | Cached @256 | Uncached @128 | Uncached @256 |
|---|---|---|---|---|
| 48 | 30,024 | 46,747 | 16,176 | 21,665 |
| 8 | 29,888 | 46,645 | 15,886 | 22,549 |
| 4 | 29,691 | **32,455** | 15,969 | 21,691 |

The 21,665 in the first row is also the TP=2 figure in *Measured results, and what they imply*; the two runs were
consecutive and the coincidence has not been re-measured. Read the CPU rows against each other, not
against other tables.

**Roughly 8 vCPU per engine is enough, and more is not better.** 8 to 48 changes nothing measurable in
any column.

**Prefix cache hits move the bottleneck from the GPU to the CPU.** A cache hit removes the prefill work
but none of the per-request CPU work (tokenising, detokenising, HTTP, scheduling), so on cached traffic
CPU binds first: 4 CPUs costs **30% of cached throughput at concurrency 256** (46,747 → 32,455) while
barely touching uncached throughput.

| Your traffic | Does vCPU per GPU matter? |
|---|---|
| Unique prompts | **No.** 8 vCPU per engine matches 48. Buy the cheapest GPU. |
| Heavy prefix reuse at high concurrency | **Yes.** Prefer a size with more host per GPU. |

### Will my model fit?

Weights are roughly `parameters × bytes-per-parameter`:

| Precision | Bytes/param | A 30B model | A 70B model |
|---|---|---|---|
| bf16/fp16 | 2 | ~56 GiB | ~130 GiB |
| fp8 | 1 | ~28 GiB | ~65 GiB |
| 4-bit | 0.5 | ~14 GiB | ~33 GiB |

Some VRAM must stay free for the KV cache, activations and CUDA graphs. This project reserves a **fixed 16 GiB per GPU**,
subtracted from what the engine claims, so the weight budget is
`96 GiB × gpuMemoryUtilization − 16 GiB` = **75.2 GiB per GPU** at the default 0.95. Fixed rather than
proportional because activation peaks and CUDA graphs are roughly constant in absolute terms. The
"does not fit" error quotes the same numbers.

One 96 GiB GPU holds a 30B model in bf16, or a 70B in fp8.

**The weights are not the only fixed cost.** The engine refuses to start unless the cache left after the
weights holds one request at `maxModelLen`, and that cost is the model's KV bytes per token times its
context length, which the model card does not print and which varies 6x between architectures. Measured
on this card with vLLM 0.28.0: a dense 31B with hybrid attention (60 layers, 16 local KV heads of 256, 4
global KV heads of 512, 262,144 context) needed **33.3 GiB for one 262,144-token request**; its 62.5 GB
of bf16 weights left 29.2 GiB, so it did not start until `maxModelLen: 131072`, and then held 164,056
tokens of cache (186 KiB per token counting every layer; 1.25 requests of full length). The same
family's 26B mixture-of-experts (30 layers, 8 and 2 KV heads) costs 33 KiB per token and started with
39 GiB to spare, 1.23M tokens. The engine prints the two numbers it compared; the fix is the one in
*`maxModelLen`* below, or quantised weights, which for the 31B leave room for the full context.

---

## Tensor parallelism: a way to make a model fit, not a way to go faster

`tensorParallel` splits one model across N GPUs. Set it to `0` and it is derived: the smallest degree
whose combined VRAM holds the model. Smallest, because splitting costs an all-reduce on every layer of
every forward pass; the GPUs advance in lockstep. With one request that pays. At saturation the synchronisation cancels the parallelism: on two GPUs
with unique prompts, the second GPU added nothing to prefill under TP=2 (22,685 → 21,665), while the
same GPU running an independent engine added 43% (→ 32,467).

Over a slow interconnect it is worse: at TP=4 on PCIe-connected GPUs the model reached a lower fraction
of its memory-bandwidth ceiling than at TP=1 and delivered less in absolute terms.

**Use tensor parallelism when the weights do not fit on one GPU, not to make a fitting model faster.**
If the model does not fit, quantizing to fp8 to get back to TP=1 is usually a better trade than adding
GPUs. Spare GPUs get an independent engine each; see *Topology: replicas or tensor parallelism?*.

**That rule was measured over PCIe. Over NVLink it holds only above the knee.** The same dense 31B in
fp8 on eight H100s (NVLink), host-level concurrency, request rate relative to eight TP=1 engines:

| Topology | 1 in flight (decode tok/s per request) | 64 in flight | 128 | 256 | 512 |
|---|---|---|---|---|---|
| 8 × TP=1 | 64 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4 × TP=2 | 97 | **+26%** | **+16%** | +6% | −9% |
| 2 × TP=4 | 134 | **+34%** | **+26%** | −2% | −12% |

A dense model streams its whole weight set every decode step; splitting it halves the bytes per GPU per
step, and NVLink's all-reduce is cheap enough to keep most of that gain. Above about 32 requests per
GPU the eight independent batches win it back, and on 4,000-token unique prompts at 512 in flight
TP=4 costs 37%. So on an NVLink instance serving a dense model below the knee, TP=2 or TP=4 is faster,
not only a way to fit; on a PCIe instance (the g7e, where TP=2 measured 0.1% on prefill) one engine per
GPU stays right. The mixture-of-experts case was not measured on NVLink.

The degree must be a power of two (1, 2, 4, 8) and must divide the model's attention head count.

**Block-quantised FP8 checkpoints add a third constraint.** Their weights are quantised in 128×128
tiles, and tensor parallelism slices each expert's gate and up matrices along the intermediate
dimension. Every slice must be whole tiles: `intermediate_size / tensorParallel` must be a multiple of
128. A 235B mixture-of-experts checkpoint with an expert intermediate size of 1,536 started at TP=4
(384 per GPU, three tiles) and refused TP=8 (192, a tile and a half) with:

```
ValueError: The output_size of gate's and up's weight = 192 is not divisible by weight quantization block_n = 128.
```

The engine exits within seconds, before any model loading, and ECS restarts it forever; see
[troubleshooting.md](troubleshooting.md). Two ways out: stay at the degree that divides, or set
`enableExpertParallel: true`, which places whole experts on GPUs instead of slicing them and so removes
the constraint. bf16 weights have no tiles and no such limit.

**If you raise the degree outside this project's config, raise `/dev/shm` too.** Workers exchange
tensors through POSIX shared memory; with Docker's 64 MiB default any degree above 1 exits within
seconds with `Insufficient space in /dev/shm`. The task definition here sets 8 GiB. TP=1 never touches
that path, so a working single-GPU deployment proves nothing about a multi-GPU one.

---

## Measured results, and what they imply

All figures below: same model class (~30B mixture-of-experts), same GPU generation, same harness,
1,000-token prompts. Decode is tokens/sec per request under load; prefill is peak tokens/sec.

**How to read these numbers.** Repeat runs of an identical configuration came out **0.7% and 2.1%**
apart. Treat differences of **8% or more as real** and a few percent as noise, including where a table
shows a "winner" by 2–3%. Single runs unless stated.

**Precision, on one GPU.** Relative fleet cost is the cost to serve a fixed request rate; lower is
better.

| Configuration | Decode | Prefill | Relative fleet cost |
|---|---|---|---|
| **FP8, TP=1, 1 GPU** | **118.5** | **9,670** | **1.00** |
| bf16, TP=1, 1 GPU | 109.9 | 6,313 | 1.53 |

Treat 1.53 as a floor on the cost of bf16: it is a single-GPU figure at moderate concurrency with
prefix cache hits. At each precision's own latency-passing knee the gap is closer to **2×** (7,903
input tok/s per GPU against 15,941). See *Every weight option, measured*.

**How to spend two GPUs.** The answer depends on the concurrency and on whether prompts share a prefix.
At **saturation** (concurrency 256, FP8 weights, fp8 KV cache), same model, same instance:

| Topology | GPUs | Cached prefill | Cached decode | Uncached prefill | Uncached decode |
|---|---|---|---|---|---|
| TP=1, one GPU, other idle | 1 | 46,747 | 51.7 | 22,685 | 26.3 ✗ |
| **TP=1 × 2 replicas** | 2 | 47,913 | 52.6 | **32,467** | **36.5 ✓** |
| TP=2 | 2 | **49,201** | **56.5** | 21,665 | 29.1 ✗ |

✓/✗ is a 31.7 tok/s per-request decode gate. **With unique prompts, two replicas are the only two-GPU
topology that passes it**, and they beat TP=2 on prefill by 50%. With cache hits the three are within
5% and TP=2 edges ahead.

The same comparison measured earlier at **concurrency 8–32 with cache hits**, the wrong place to
measure:

| Precision | Topology | Decode | Prefill |
|---|---|---|---|
| bf16 | TP=2 | 119.9 | 9,850 |
| bf16 | 2 replicas, TP=1 each | 94.4 | 9,107 |
| bf16 | TP=1, second GPU idle | 109.0 | 8,584 |
| FP8 | TP=2 | 140.4 | 10,943 |
| FP8 | 2 replicas, TP=1 each | 90.0 | 7,218 |

There TP=2 looks 52% ahead of replicas on prefill. At saturation the same comparison is **+2.7% with
cache hits and −33% without**.

### Quantisation is two independent decisions

Precision is two choices, set by different keys:

| Decision | Key | Default here | What it affects |
|---|---|---|---|
| **Weight** precision | `modelId` / `quantization` | whatever the model ships | the model's own parameters |
| **KV cache** precision | `tuning.kvCacheDtype` | `fp8` | cached attention state, not the weights |

Either can be taken without the other: quantised weights change the model you run, a quantised KV
cache only how attention state is stored.

**Take both if you can.** FP8 weights measured **~35% lower fleet cost** than bf16, with **+53%
prefill** and **+8% decode**, a larger effect than any engine flag. An fp8 KV cache adds **+12%
decode**, growing with concurrency. The defaults assume both.

The weight gain exceeds "half the bytes" because prefill is compute-bound and this GPU generation has
native FP8 tensor cores.

They interact in one direction:

| Weights | Use for the KV cache | Why |
|---|---|---|
| quantised (FP8/4-bit) | **`fp8`** | +12% decode, and small weights leave ample headroom |
| unquantised (bf16) | **`auto`** | fp8 raises memory pressure here rather than lowering it; see `gpuMemoryUtilization` |

#### What quantisation costs in answers, on standard benchmarks

Throughput says nothing about answers, so every precision was scored against its own bf16 with the
same harness through the endpoint (`scripts/quality.py`: lm-evaluation-harness 0.4.13, the set quantised
checkpoints are published with). Log-likelihood tasks over `/v1/completions`: MMLU 5-shot (50 per subject,
2,850), ARC-Challenge 25-shot, HellaSwag 10-shot, Winogrande 5-shot, TruthfulQA mc2 (500 each), and
WikiText word perplexity (60 documents). Generative tasks over chat completions, greedy, 1,024-token cap:
GSM8K 5-shot (500) and IFEval (541). Thinking models were served with thinking off, and the harness sends the
model's beginning-of-sequence token on the completions path when it has one (Gemma 4's tokenizer adds
none by itself). Gemma 4's instruction-tuned checkpoints are out of distribution on raw text even then
(a plain paragraph read perplexity 654 to 8,923 raw and 12 to 19 inside their chat turn), so their
log-likelihood tasks were run with the chat template and without wikitext (`quality.py --chat-loglik`;
troubleshooting, *log-likelihood scores at chance*). One standard error is
about 0.7 points on MMLU, 2 on the other tasks, 1.1 to 1.9 on GSM8K, 1.6 on IFEval. Each row below is one
served configuration on one 96 GB GPU; deltas are against the family's bf16 row.

**Qwen3-30B-A3B (mixture of experts, 3B active), bf16 = MMLU 81.0, ARC 68.8, HellaSwag 67.2, Winogrande 70.8, TruthfulQA 52.7, GSM8K 92.6, IFEval 83.5, perplexity 10.89**

| Served as | MMLU | ARC | HellaSwag | Winogrande | TruthfulQA | GSM8K | IFEval | Perplexity | Answers flipped |
|---|---|---|---|---|---|---|---|---|---|
| publisher fp8, fp8 KV | −0.5 | +0.2 | +0.8 | +0.4 | +1.8 | −0.8 | −0.4 | +0.2% | 2 to 4% |
| publisher fp8, bf16 KV | 0.0 | 0.0 | +0.4 | −0.6 | +1.9 | −0.2 | −0.6 | +0.3% | 2 to 4% |
| NVIDIA NVFP4 (calibrated) | −0.6 | −1.2 | +0.6 | +0.4 | −2.1 | −1.2 | −1.3 | +3.4% | 5 to 6% |
| Qwen GPTQ-Int4 | −0.8 | −2.0 | +0.6 | +0.2 | +0.5 | −1.8 | −1.3 | +3.6% | 5 to 6% |

**Qwen3-32B (dense), bf16 = MMLU 84.0, ARC 72.4, HellaSwag 74.2, Winogrande 77.4, TruthfulQA 57.4, GSM8K 94.2, IFEval 84.5, perplexity 9.37**

| Served as | MMLU | ARC | HellaSwag | Winogrande | TruthfulQA | GSM8K | IFEval | Perplexity | Answers flipped |
|---|---|---|---|---|---|---|---|---|---|
| publisher fp8, fp8 KV | −0.3 | 0.0 | −0.4 | +0.2 | +0.2 | −0.6 | −2.1 | +0.7% | 2 to 4% |
| Red Hat NVFP4 (calibrated) | −1.0 | −1.4 | +0.2 | +0.6 | −1.0 | 0.0 | −2.1 | +5.3% | 4 to 6% |

**Qwen3-8B (dense), bf16 = MMLU 76.8, ARC 65.8, HellaSwag 67.6, Winogrande 70.4, TruthfulQA 53.8, GSM8K 78.4, IFEval 80.8, perplexity 12.25**

| Served as | MMLU | ARC | HellaSwag | Winogrande | TruthfulQA | GSM8K | IFEval | Perplexity | Answers flipped |
|---|---|---|---|---|---|---|---|---|---|
| publisher fp8, fp8 KV | 0.0 | +0.4 | −0.8 | +1.2 | −0.6 | −3.8 | +1.5 | +0.7% | 4% MMLU, 16% GSM8K |
| publisher fp8, bf16 KV | −0.7 | −0.2 | −0.6 | −0.2 | 0.0 | −0.8 | +0.7 | +1.1% | 4% MMLU, 13% GSM8K |
| bf16 weights, fp8 KV | +0.2 | +0.2 | −0.8 | +0.8 | 0.0 | +3.4 | 0.0 | +0.2% | 2% MMLU, 10% GSM8K |
| Qwen AWQ Int4 | −0.6 | +0.4 | −0.8 | +0.6 | −0.4 | +2.0 | +0.2 | +3.7% | 7% MMLU, 16% GSM8K |
| Red Hat NVFP4 (calibrated) | −1.7 | −0.8 | −0.4 | −0.6 | −0.2 | +1.2 | +0.5 | +3.9% | 8% MMLU, 17% GSM8K |

**Qwen3-30B-A3B-Instruct-2507 (the shipped model), bf16 = MMLU 83.4, ARC 72.8, HellaSwag 70.2, Winogrande 75.0, TruthfulQA 61.8, GSM8K 93.6, IFEval 81.7, perplexity 8.86**

| Served as | MMLU | ARC | HellaSwag | Winogrande | TruthfulQA | GSM8K | IFEval | Perplexity | Answers flipped |
|---|---|---|---|---|---|---|---|---|---|
| publisher fp8, fp8 KV | −0.5 | −0.6 | 0.0 | 0.0 | −0.9 | −0.2 | 0.0 | +0.7% | 2 to 4% |
| load-time fp8 (`quantization: fp8`), fp8 KV | −0.8 | −1.0 | −0.8 | +0.6 | −0.6 | −0.6 | +0.9 | +0.6% | 2 to 4% |

**Qwen3.8-27B (dense, thinking off), bf16 = MMLU 83.4, ARC 72.8, HellaSwag 66.4, Winogrande 79.2, TruthfulQA 52.3, GSM8K 97.4, IFEval 80.4, perplexity 8.48**

| Served as | MMLU | ARC | HellaSwag | Winogrande | TruthfulQA | GSM8K | IFEval | Perplexity | Answers flipped |
|---|---|---|---|---|---|---|---|---|---|
| publisher fp8, fp8 KV | +0.1 | 0.0 | +0.2 | −0.6 | +0.3 | −0.6 | −0.9 | +0.4% | 2 to 4% |
| community NVFP4 | −1.1 | +0.6 | +1.6 | +1.4 | −0.6 | 0.0 | +0.4 | +1.9% | 4 to 6% |

**Gemma 4 26B-A4B (mixture of experts, 3.8B active; a sixth family, scored with chat-templated multiple choice, no perplexity), bf16 = MMLU 84.2, ARC 70.8, HellaSwag 62.4, Winogrande 70.2, TruthfulQA 62.9, GSM8K 93.8, IFEval 87.1**

| Served as | MMLU | ARC | HellaSwag | Winogrande | TruthfulQA | GSM8K | IFEval | Answers flipped |
|---|---|---|---|---|---|---|---|---|
| Red Hat fp8, fp8 KV | +0.1 | −1.2 | −0.6 | +0.6 | +1.1 | 0.0 | +1.8 | 2 to 5% |
| NVIDIA NVFP4 (calibrated) | −0.4 | −2.0 | −0.6 | +3.4 | +0.6 | −0.2 | +1.5 | 3 to 6% |

**Gemma 4 31B (dense; chat-templated multiple choice, no perplexity), bf16 = MMLU 87.6, ARC 68.8, HellaSwag 62.0, Winogrande 75.8, TruthfulQA 64.0, GSM8K 96.0, IFEval 90.0**

| Served as | MMLU | ARC | HellaSwag | Winogrande | TruthfulQA | GSM8K | IFEval | Answers flipped |
|---|---|---|---|---|---|---|---|---|
| Red Hat fp8, fp8 KV | +0.1 | 0.0 | 0.0 | −0.8 | −0.4 | −0.4 | +0.2 | 1 to 6% |
| Google QAT W4A16 (the publisher's own quantisation-aware int4) | −0.6 | −1.2 | +2.6 | −0.2 | −0.8 | −0.2 | +0.6 | 1 to 8% |
| NVIDIA NVFP4 (calibrated) | −0.1 | −1.6 | −0.2 | −1.0 | −1.5 | +0.8 | +0.2 | 2 to 7% |

**Gemma 4 12B (dense, encoder-free), bf16 = MMLU 80.1, ARC 64.8, HellaSwag 62.6, Winogrande 72.2, TruthfulQA 61.6, GSM8K 92.4, IFEval 86.9**

| Served as | MMLU | ARC | HellaSwag | Winogrande | TruthfulQA | GSM8K | IFEval | Answers flipped |
|---|---|---|---|---|---|---|---|---|
| Red Hat fp8, fp8 KV | −0.4 | +2.0 | +0.4 | −1.2 | −1.0 | −1.4 | +1.1 | 3 to 5% |
| Google QAT W4A16 (the publisher's own quantisation-aware int4) | −1.0 | −1.6 | +1.0 | 0.0 | −2.7 | −1.8 | −0.2 | 7 to 9% |

The Gemma rows are not comparable with the Qwen rows in absolute terms (chat-wrapped prompts score lower on
HellaSwag and higher on MMLU than raw ones); the deltas against their own bf16 are like for like, and they
say what every family before them said: fp8 within noise with balanced flips, 4-bit 0.5 to 2 points down on
the knowledge and math tasks with one answer in twelve to twenty flipped. Google's quantisation-aware int4
bought the same 4-bit as everyone else's calibration, not a free one.

**Mistral Small 3.2 24B (dense, a second model family), a partial check.** Its tokenizer is not the Hub
tokenizer, so the harness had to send text instead of token ids, and the official bf16 checkpoint in the
engine's mistral tokenizer mode returned chance-level log-likelihoods, so only its generative scores
count. Red Hat fp8 against the official bf16 on those: GSM8K (flexible extraction, the model does not
write the `####` format) 93.4 against 91.6, IFEval 75.1 against 73.9: fp8 free again. Red Hat NVFP4
against fp8 on the log-likelihood tasks: ARC 71.4 against 72.2, HellaSwag 75.8 against 76.6, Winogrande
77.0 against 79.0, TruthfulQA 59.3 against 61.5, a 1 to 2 point cost, the same as every other 4-bit
build. Its generative passes were lost to an engine disconnect and were not repeated.

What generalises across the five families:

- **fp8 weights cost nothing measurable.** Every task within one standard error of bf16 on every model,
  perplexity 0.2 to 0.7% worse, 2 to 4% of answers flipped with gains about equal to losses. Load-time
  fp8 of the bf16 checkpoint measured the same as the publisher's fp8 build. This is the 2× throughput
  option and it is free.
- **The fp8 KV cache costs nothing measurable either.** Perplexity +0.2 to +0.4%, tasks within noise,
  flips balanced; on the 30B, bf16 and fp8 caches under fp8 weights are indistinguishable on every task.
  `fp8_e5m2` is refused by the engine on fp8 checkpoints; the e4m3 default is the only choice there.
- **4-bit weights are the first precision with a consistent cost, and it is small and the same for every
  format.** NVFP4, GPTQ-Int4 and AWQ all land 0.5 to 2 points below bf16 on MMLU, ARC, GSM8K and IFEval,
  3.4 to 5.3% worse in perplexity, and flip 5 to 8% of answers with losses outnumbering gains by about
  four to three. NVIDIA's and Red Hat's calibrated NVFP4, Qwen's GPTQ and AWQ, and the community NVFP4
  of the 27B are indistinguishable from each other on quality; the difference between them is throughput
  (native FP4 compute on this GPU against weight-only dequantisation). The 3 to 5 GSM8K points that same
  27B build appeared to lose in an earlier check were the thinking chain hitting the generation cap, not
  the weights: with thinking off it is within noise on GSM8K and one point down on MMLU like every other
  4-bit build. Score the checkpoint you would deploy, under the settings you would serve it with.
- **Perplexity is the metric that separates the classes.** bf16 12.25, fp8 12.33 to 12.38, 4-bit 12.70 to
  12.73 on the 8B, in that order on every family, while every task score overlaps its interval. Report it.
- **The flip rate is the honest number for a fleet.** fp8 changes one answer in thirty against bf16; 4-bit
  one in twenty; on a small model, greedy math changes one answer in six between any two configurations,
  including two that differ only in cache precision. A ±3-point GSM8K swing on an 8B is noise.
- **Small models are noisier, not much worse.** The 8B's 4-bit cost on MMLU (−0.6 to −1.7) is the same as
  the 30B's and the 32B's; what grows is the variance of the generative metrics.
- **A model release moves more than any quantisation.** The July release of the 30B beats the April
  release in bf16 by 2 to 8 points on every task, with 8 to 11% of answers flipped. Never compare a
  quantised build with a different release of the model.
- **Thinking models need thinking off for these settings**, or the chain of thought eats the cap: the same
  NVFP4 30B went from 68.6 to 91.2% on GSM8K and from 33.6 to 83.6% on IFEval with
  `extraArgs: --default-chat-template-kwargs '{"enable_thinking": false}'`. Within-model comparisons under
  one setting hold; absolute numbers across models do not unless the setting matches how you serve.

What the check is not: 500 questions per task and 2,850 for MMLU is enough to see a precision that
answers worse, not to certify one that does not; two tasks per aggregate are generative, and code was
left out because no code task runs correctly for an instruct model over an API (`scripts/quality.py`
explains). Models whose tokenizer is not the Hub's (Mistral) need `--text-prompts` and lose the
perplexity row. Run it on your own prompts before switching a fleet.

#### What quantisation costs an agent, on standard agentic benchmarks

The suite above scores knowledge and short answers. An agent fails differently: it emits a tool call
with the wrong argument, loses the thread on turn six, or writes a patch that does not apply. So every
precision was also scored on four agentic benchmarks through the endpoint, with their published
scaffolds unchanged and the engine's tool parser in the loop. As in the section above, every row is a
precision measured against its own family's bf16; the families are there to show whether the precision
effect depends on architecture, not to compare models with each other:

- **BFCL v4** (Berkeley Function Calling Leaderboard), the single-turn and multi-turn categories, 4,441
  tests, graded by AST match and by final state. Tools go in the request, `tool_calls` come back, so
  this scores the model and `toolCallParser` together, as every agent framework sees them.
- **τ-bench** (tau2-bench), retail 114 tasks and airline 50, two trials, the agent at temperature 0 and
  the simulated user and the judge fixed to one external model for every row. Metric pass^1.
- **SWE-bench Verified**, the first 100 instances in dataset order, driven by mini-swe-agent (a bash
  tool, 75 steps, 8 workers) and graded by the official harness. Metric: resolved out of 100.
- **Extraction**, `scripts/extraction.py`: CoNLL-2003 named entities as a fixed JSON object, 1,000
  sentences, entity-level F1. Standard data and metric, our prompt, so it compares deployments with each
  other and nothing else.

Four engines per row (two for the 30B rows), the serving defaults, no evaluation-only settings.
Before the precision rows, the one row that decides what a difference means:

**Run-to-run noise: the same bf16 30B-A3B-2507 deployment, scored twice.** BFCL 35.27 against 35.25 with
3.2% of items flipped (+74 −67); τ-bench retail 59.6 against 60.5 with 27% of tasks flipped; SWE-bench 13
against 13 with 12% of instances flipped (+6 −6); extraction F1 56.3 against 55.4 with 2.0% of sentences
flipped. Agent trajectories diverge at temperature 0: one different token early and the run ends
elsewhere. A τ-bench difference under 3 points, a SWE-bench difference under 5 tasks, or a BFCL flip rate
near 3% is not a result.

**Qwen3-Coder-30B-A3B-Instruct (mixture of experts, `toolCallParser: qwen3_coder`), bf16 = BFCL 32.8 overall (non-live 85.0, live 79.0, multi-turn 29.0, irrelevance 76.7), τ-bench retail 22.8 / airline 34.0, SWE-bench 22/100, extraction F1 69.9**

| Served as | BFCL overall | multi-turn | irrelevance | τ retail | τ airline | SWE-bench | extraction F1 | Items flipped (BFCL, SWE, extraction) |
|---|---|---|---|---|---|---|---|---|
| publisher fp8 | −0.3 | −0.6 | −1.4 | +1.3 | +8.0 | +5 | −0.1 | 7.4% (+156 −171), 19%, 4.2% (+21 −21) |
| community AWQ Int4 | −1.3 | −3.5 | −0.8 | +0.4 | +5.0 | +2 | −1.4 | 9.7% (+188 −242), 14%, 10.0% (+36 −64) |
| community NVFP4, uncalibrated | −6.0 | −9.6 | −27.5 | +3.5 | 0.0 | −15 | −0.8 | 17.6% (+188 −592), 19% (+2 −17), 11.2% |

**Qwen3-32B (dense, thinking off, `toolCallParser: hermes`), bf16 = BFCL 32.4 (87.4, 80.8, 25.4, 79.5), τ-bench retail 52.2 / airline 23.0, SWE-bench 5/100, extraction F1 65.2**

| Served as | BFCL overall | multi-turn | irrelevance | τ retail | τ airline | SWE-bench | extraction F1 | Items flipped (BFCL, SWE, extraction) |
|---|---|---|---|---|---|---|---|---|
| publisher fp8 | 0.0 | +0.1 | −1.2 | −1.3 | +2.0 | +6 | 0.0 | 3.8% (+84 −84), 6%, 3.6% (+15 −21) |
| Qwen AWQ Int4 | −0.6 | −1.3 | −2.0 | +0.9 | −1.0 | +4 | +0.3 | 6.4% (+123 −163), 6%, 5.1% (+22 −29) |
| Red Hat NVFP4, calibrated | −0.9 | −2.1 | −0.8 | +4.8 | −2.0 | +2 | −0.5 | 6.3% (+114 −167), 4%, 6.8% (+26 −42) |

**Qwen3-30B-A3B-Instruct-2507 (the shipped model, `toolCallParser: hermes`, two engines), bf16 = BFCL 35.3 (87.1, 78.1, 35.8, 80.2), τ-bench retail 59.6 / airline 34.0, SWE-bench 13/100, extraction F1 56.3**

| Served as | BFCL overall | multi-turn | irrelevance | τ retail | τ airline | SWE-bench | extraction F1 | Items flipped (BFCL, SWE, extraction) |
|---|---|---|---|---|---|---|---|---|
| the same bf16 again (noise) | 0.0 | −0.3 | −0.1 | +0.9 | 0.0 | 0 | −0.9 | 3.2% (+74 −67), 12% (+6 −6), 2.0% |
| publisher fp8 | +0.2 | +1.0 | −0.5 | 0.0 | +2.0 | +1 | −0.5 | 4.7% (+103 −107), 9% (+5 −4), 5.5% (+28 −27) |

**gpt-oss-120b (MXFP4, the only precision published, `toolCallParser: openai`, `reasoningParser: openai_gptoss`, two engines)**:
BFCL 36.2 overall with the best multi-turn (55.9) and irrelevance (87.1) of every model, and 0.0 on every
parallel category: through the engine's parser it returns one tool call per message, so a test that
expects several calls in one answer fails outright. τ-bench retail 75.4 / airline 60.0, 15 to 25 points
above the Qwen models. SWE-bench 13/100 with 74 empty patches: it rarely submits a diff within the step
budget. Extraction F1 50.2 at a 300-token answer cap with a quarter of the answers unparsed, 71.9 (the
best) at 2,000: a reasoning model spends the cap on its thinking first, so every generative check needs
the cap sized for it (*Reasoning models*). One row, no comparison: a reference point for what a larger
open-weight model does on the same harnesses and hardware.

What generalises:

- **fp8 is free for agents too.** On all three families every aggregate is inside the repeat spread,
  flips are balanced (Coder +156 −171, 32B +84 −84, 30B +103 −107), and the SWE-bench and τ-bench
  differences are the same size as the bf16 repeat's own. The throughput case for fp8 stands
  unchanged for tool-calling and agent workloads.
- **4-bit costs about a point of BFCL and one to three points of multi-turn, with losses ahead of
  gains.** AWQ and calibrated NVFP4 flip 6 to 10% of BFCL items against 3 to 4% for fp8, and the flips
  run about four losses to three gains; extraction flips go the same way (Coder AWQ +36 −64). It is the
  same shape as the Q&A result, and no bigger: the agent's multi-step trajectory does not amplify a
  small per-token perturbation into a large end-to-end loss.
- **One checkpoint was broken, and only the agentic suite showed it.** The uncalibrated community NVFP4
  of the Coder scored within a point of bf16 on extraction and on non-live BFCL, then lost 27.5 points of
  irrelevance detection (it calls a tool when it should answer), 9.6 of multi-turn, and 15 of 22
  SWE-bench tasks (43 empty patches, 40 that did not apply). The same format from a calibrated
  publisher (Red Hat's 32B) costs a point. Format is not the risk; the build is. Score a 4-bit
  checkpoint on tool use before serving it to agents, whatever its perplexity says.
- **Agent metrics are noisy in a way Q&A metrics are not.** Two identical deployments disagree on 12% of
  SWE-bench instances and 27% of τ-bench tasks; these are the flip rates of *nothing*. Compare only
  aggregates, on the same instance list, and repeat the baseline once before believing a delta.
- **The model moves the score more than the precision does.** Between the three families measured
  here, the same benchmark differs by 20 to 40 points (τ-bench retail, SWE-bench, BFCL multi-turn),
  while the precision moves it by one. These rows are not a ranking of models: each family was picked
  as a different architecture to test the precision effect on, and the families differ on which task
  each is better at. Choose the model for your task on your task, then choose the precision by these
  rules.
- **Constrained decoding does not change extraction quality.** The strict schema, `json_object` and a
  plain "answer with JSON" prompt agree within a point of F1 on every configuration (Coder 69.9, 69.7,
  69.7; 32B 65.2, 65.2, 65.2), and the free answers all parsed. The schema buys a guaranteed shape at
  the 15% decode cost measured in *Structured output*, not better answers.

What the check is not: 100 SWE-bench instances and 164 τ-bench tasks give a standard error of about
5 points, enough to catch a broken build, not a one-point one; τ-bench's user simulator is an external
model, so its absolute scores depend on that choice; the extraction score is a same-prompt comparison,
not a CoNLL result. Two SWE-bench instances were dropped from two rows (marked 98) when their steps
exceeded the endpoint's 120 s non-streamed limit on every attempt, which is itself a finding for agent
traffic (*Tool calling*).

#### Every weight option, measured

Every row **meets** the latency budget, at the highest concurrency where it does. Plan against the
unique-prompt column unless you know your prompts share a prefix:

| Weights | Unique prompts | GPUs | Shared prefix | GPUs |
|---|---|---|---|---|
| AWQ 4-bit ⚠ | **33,496** @256 | **2.1** | **82,112** @512 | **0.9** |
| official FP8 | 32,467 @256 | 2.2 | 74,687 @512 | 0.9 |
| load-time FP8 | 28,363 @256 | 2.5 | 75,368 @512 | 0.9 |
| bf16 | 15,805 @128 | 4.4 | 46,808 @256 | 1.5 |

Prefill tokens/sec on two GPUs as 2 × TP=1 with an fp8 KV cache (`auto` for the bf16 row); GPUs to
serve a fixed 70,000 input tok/s. One `g7e.2xlarge` is one GPU, so this is also the instance count on
that size; *How to size N* derives the same number from a single-GPU measurement.

Only the unique-prompt column shows:

- **Unquantised weights cost most without prefix reuse.** 15,805 against 46,808 with cache hits: 4.4
  GPUs rather than 1.5.
- **The operating concurrency drops to 256 across the board**: per-request decode falls through the
  latency floor before throughput peaks.

⚠ **AWQ 4-bit is the fastest measured option and is not a recommendation.** See below.

#### Load-time FP8 measures the same as an official FP8 build

`quantization: fp8` applied to bf16 weights measured **75,368** tok/s against **74,687** for the
publisher's own FP8 build: parity within 1%. Both routes store the same fp8 numbers in VRAM and run the
same kernels. The differences are not performance:

| | Official FP8 build | `quantization: fp8` |
|---|---|---|
| Download | ~29 GiB | ~57 GiB, converted after loading |
| Startup | loads directly | one conversion pass first |
| Scale factors | the publisher's calibration | per-tensor absolute-maximum |

**Prefer the official build when one exists**: smaller download on every task start, and scale factors
chosen with access to the model and data.

**Planning consequence:** bf16 is *not* the fallback for a model with no official FP8 build. Choose it
only on quality grounds.

#### NVFP4: Blackwell's native 4-bit, measured on eight instances

`nvidia/Qwen3-30B-A3B-NVFP4` with an fp8 KV cache, against load-time FP8 on the same eight
`g7e.2xlarge`, unique 1,000-token prompts. **Read this as a kernel and format measurement, not a
precision comparison with the fp8 rows:** the checkpoint's model card names `Qwen/Qwen3-30B-A3B`, the
earlier hybrid thinking release, as its base, not the Instruct-2507 weights the fp8 rows use. Against
its own base it is lossless: scored with thinking disabled, 91.2% gsm8k and 83.6% ifeval against 90.8%
and 83.2% for the bf16 base, at twice the throughput (*What quantisation costs in answers*). On the
full suite every 4-bit build measured, calibrated or not, costs 0.5 to 2 points and about one point of
tool-calling accuracy against its own bf16; one uncalibrated community build was broken for tool use
while passing the chat tasks (*What quantisation costs an agent*). Who quantised matters as much as
the format.

| Concurrency | FP8 input tok/s | NVFP4 | Gain | p95 FP8 | p95 NVFP4 |
|---|---|---|---|---|---|
| 256 | 59,213 | 73,923 | **+25%** | 4.03 s | 3.34 s |
| 512 | 95,539 | 118,125 | **+24%** | 5.23 s | 4.38 s |
| 768 | 119,223 | 152,672 | **+28%** | 6.33 s | 5.06 s |

p99 spiked in two of the five levels (8.2 s at 256, 9.6 s at 768) where FP8 did not; a 60-second level
is too short to say whether that is noise. Weights are half the size of FP8 again, so the KV cache gets
another ~15 GiB. FP8 stays the default because it costs nothing measurable in answers and 4-bit costs a
little on every task (*What quantisation costs in answers*). To use a 4-bit build, set `modelId` to the
checkpoint and clear `quantization`, check its base model first, and score its tool use if agents will
call it.

**Online NVFP4 does not run on this GPU.** vLLM 0.28 also has `quantization: nvfp4_per_token`, which
quantises bf16 weights at load time the way `fp8` does. On g7e the engine refuses to start:
`nvfp4_per_token online quantization requires a Blackwell (SM100) GPU`. The RTX PRO 6000 is Blackwell
SM120, not the SM100 of B200. So on this hardware NVFP4 means a calibrated checkpoint, and the load-time
convenience of fp8 has no NVFP4 equivalent.

**NVFP4 and EAGLE-3 stack.** The two together, on six instances at the same 96 requests per engine:
133,463 input tok/s, 22,244 per instance against 14,903 for the shipped FP8, **+49% per GPU**, p95
4.54 s against 6.33 s. Acceptance length fell from 2.2 to 1.95, since the speculator was trained against
the bf16 model, and it still added 17% on top of NVFP4 alone.

#### 4-bit is two different products on this GPU: native FP4 compute, or weight-only dequantisation

Measured on Gemma 4 with the same matrix on one engine, each 4-bit build against its own family's fp8:

| Model and build | Kernel the log named | 1 in flight decode tok/s | 8 in flight (unique, cached) | 64 in flight unique | 64 cached | Prefill wall (4k prompts) |
|---|---|---|---|---|---|---|
| 26B MoE, NVIDIA NVFP4 | FlashInfer CUTLASS NvFp4 MoE | 159 vs 186 (−15%) | 0%, −13% | +8% | 0% | 46.9k vs 45.9k tok/s (+2%) |
| 12B dense, Google QAT W4A16 | Marlin (compressed-tensors WNA16) | 132 vs 89 (+48%) | +21%, +26% | −26% | −47% | 12.0k vs 21.4k tok/s (−44%) |
| 31B dense, Google QAT W4A16 | Marlin | 62 vs 39 (+58%) | +23%, +55% | −30% | +20% | 3.9k vs 8.0k tok/s (−52%); 4k unique prompts at 64 in flight 0.20 vs 1.06 req/s |
| 31B dense, NVIDIA NVFP4 | FlashInfer CUTLASS NvFp4 GEMM | 41 vs 39 (+4%) | +8%, 0% | +13% | +8% | 8.6k vs 8.0k tok/s (+8%) |

A weight-only kernel unpacks int4 to bf16 and runs the matmul at the bf16 rate, so it wins exactly where
the wall is memory bandwidth (few requests in flight, long answers, cached prefixes) and gives the fp8
prefill gain back where the wall is compute (many unique prompts). Native FP4 keeps the fp8 compute rate
and cuts bytes; on the 26B's 704-wide experts that bought almost nothing over fp8, and Google does not
publish a 4-bit build of that model for quality reasons. So pick a 4-bit checkpoint by the wall you are at
(*Which wall are you at?*) and the kernel the startup log names, not by the bit count: an interactive
fleet at low concurrency is the W4A16 case; a batch or long-document fleet is not. What the two paths
cost in answers was the same: 0.5 to 2 points and one answer in twelve to twenty (*What quantisation
costs in answers*).

#### AWQ 4-bit: measured fastest, quality unvalidated

A 4-bit AWQ build measured the best throughput on both prompt conditions: **82,112** cached and
**33,496** unique, against 74,687 and 32,467 for FP8. About 15 GiB of weights instead of 29 leaves more
of the card for KV cache, and cache space caps concurrency.

**This is not a recommendation.** Everything measured here is throughput; **output quality was never
evaluated.** That matters most at 4 bits: it is a far larger perturbation than FP8, a sparse
mixture-of-experts has less redundancy per expert to absorb the error than a dense model of the same
size, and community 4-bit uploads vary in calibration quality.

Validate it against your own evaluations, as you would any model change. Do not adopt it on a
throughput number alone.

#### If you cannot quantise the weights

Choose bf16 only if quantised weights are unacceptable on quality grounds; "no official FP8 build" is
**not** a reason, since load-time `quantization: fp8` matches one. Then:

```yaml
modelId: <publisher's bf16 model>
quantization: ""          # leave empty - do not quantize at load time
tuning:
  kvCacheDtype: auto      # NOT fp8 with unquantised weights - see below
```

**With unquantised weights, do not quantise the cache.** The combination either runs out of VRAM under
load or, at the lower utilisation that survives, is slower than not doing it (37,877 against 46,808).
The stack warns about it at synth time.

The cost **depends heavily on whether your prompts share a prefix**:

| Weights | Prompts | Operating concurrency | Prefill tok/s | Relative fleet |
|---|---|---|---|---|
| FP8 | shared prefix | 512 | 74,687 | **1.00** |
| bf16 | shared prefix | 256 | 46,808 | 1.60 |
| FP8 | unique | 256 | 32,467 | 2.30 |
| **bf16** | **unique** | **128** | **15,805** | **4.70** |

Per GPU, at each precision's own latency-passing knee, bf16 sustained **7,903 input tok/s against
15,941 for FP8**: almost exactly **2×**.

- **With prefix reuse, bf16 costs about 1.6× the fleet** at half the operating concurrency.
- **bf16 loses on two counts at once, so the gap is 2× rather than the ~1.5× weight size predicts.** It
  is 1.5× behind FP8 at the same concurrency, *and* its usable load is lower: 64 concurrent per GPU
  against 128, because per-request latency reaches the budget sooner.

Comparing FP8 against bf16 on your own evaluations is cheaper than doubling a fleet.

Two further consequences:

- **The weights need roughly twice the VRAM** (*Will my model fit?*). If bf16 pushes a model past one
  card, use `tensorParallel: 2`. **Do not raise it while the model still fits**: a ~30B model is
  ~57 GiB in bf16, inside one 96 GiB card, and TP=2 measured **6,991 tok/s per GPU against 6,989 for
  TP=1**. Two containers, one per GPU, measured **13% better** than TP=2 on the same two GPUs (15,805
  against 13,981). The one-engine-per-GPU rule holds at both precisions.
- **Expert parallelism may help in bf16 where it hurts FP8.** Measured **+15%** in bf16 and **−13%** in
  FP8, so test `enableExpertParallel: true` on a bf16 deployment.

#### If you cannot quantise the KV cache either

Set `kvCacheDtype: auto`. **An fp8 KV cache is a lossy store for attention state**: the computation is
unchanged, but the cached intermediate values lose precision. If you ruled out weight quantisation on
quality grounds, decide this one explicitly too.

`auto` costs the +12% decode above and nothing else. Use it as the quality baseline; if fp8 compares
acceptably, take it. With unquantised weights, `auto` is required anyway.

### Topology: replicas or tensor parallelism?

**If the model fits on one GPU, run one engine per GPU.** Use tensor parallelism when the model does
not fit, or when your prompts share a long common prefix.

The mechanism:

**Tensor parallelism splits every request across GPUs**, with an all-reduce per layer, so the GPUs
advance in lockstep. At high concurrency **the second GPU contributed nothing to uncached prefill**
(22,685 on one GPU, 21,665 across two).

**Independent engines split the requests instead of the request**, with no coordination at any layer:
22,685 → 32,467 uncached prefill for the same second GPU.

TP buys lower latency on one request; replicas buy more requests at once. A saturated server needs the
second.

**The distinction is one model split versus independent replicas, not one process versus several.**
vLLM can run N independent replicas inside one process with `--data-parallel-size N`, and that
configuration clears the split-model plateau.

#### Data parallelism: an option that exists, and loses on realistic traffic

`--data-parallel-size N` runs N replicas inside one process with vLLM routing between them, which could
route by cache affinity where a load balancer round-robins blindly. **Tested directly, it did not win.**

Three prompt shapes at concurrency 256, on two GPUs:

| Prompt shape | Two containers (ALB) | DP=2 (vLLM routing) | Winner |
|---|---|---|---|
| Every prompt unique | **32,467** | 27,793 | containers, **+17%** |
| **8 shared prefixes + unique tails** | **36,106** | 32,399 | containers, **+11%** |
| Every prompt identical | 47,913 | **54,019** | DP, +13% |

The middle row (a common system prompt with per-request content) is the realistic one and was
constructed to favour DP's routing. DP lost it by 11%.

**DP only wins when every request is literally the same prompt**, a property of benchmarks, not
workloads. Use one container per GPU.

DP has **better decode** (45.3 against 41.3 tok/s) and **worse prefill and time-to-first-token**
(3.05 s against 1.84 s). The engine runs one API server per data-parallel rank, so the frontend is not
the cause. With expert parallelism the ranks step in lockstep, every forward pass waits for the slower
attention group, and one busy rank stalls the other. On unique prompts at concurrency
256 it also fails the latency budget, p95 8.57 s against 8 s.

**Why DP wins the identical-prompt case is unexplained.** The middle row rules out cache affinity.

Re-tested at eight GPUs with a 235B mixture-of-experts (`dataParallel: 2` × TP=4 with expert parallelism,
against two separate TP=4 engines): 5.7 against 10.4 rps at 64 in flight, 13.1 against 22.5 at 512, and
4,000-token unique prompts at 512 collapsed to 0.4 rps as one rank's prefill backlog held the other
rank's step. Same GPUs, same weights, same flags apart from how the two halves are joined.

Two further reasons for separate containers: a DP process is one load balancer target, so a sick
replica inside it is invisible to health checks; and one process is one failure domain.

To try it anyway, set `dataParallel:` in `config.yaml` (it requires `enableExpertParallel: true`;
synth refuses the combination otherwise) and measure against the same GPUs as separate engines.

#### When the model needs several GPUs: the smallest degree that fits, then replicas

The rule above was measured with a model that fits one GPU. It holds, harder, when the model does not.
A 235B mixture-of-experts (22B active, FP8, 236 GB of weights) needs at least three 80 GB GPUs, so on an
eight-GPU host the choice is two engines at TP=4 or one engine at TP=8. Same host, same model, same total
offered load, unique 1,000-token prompts with 190-token answers:

| In flight (whole host) | 2 engines × TP=4 | 1 engine × TP=8 (+EP) | 1 engine, DP=2 × TP=4 (+EP) |
|---|---|---|---|
| 64 | **10.4 rps**, p50 5.6 s | 7.9 rps, p50 8.2 s | 5.7 rps, p50 10.8 s |
| 256 | **17.4 rps**, p50 12.5 s | 13.1 rps, p50 17.1 s | 10.3 rps, p50 21.4 s |
| 512 | **22.5 rps**, p50 18.1 s | 13.4 rps, p50 27.6 s | 13.1 rps, p50 28.7 s |

Long answers (800 tokens) at 512: 18.4 rps against 10.2. Long prompts (4,000 tokens, unique): 7 rps
against 3.5, so **prefill over eight GPUs was half the prefill of two independent groups of four.**

Doubling the tensor-parallel degree halves each GPU's share of the work but leaves the per-layer
all-reduce and the per-step kernel launch count where they were, so one TP=8 step is not twice as fast
as one TP=4 step, while two TP=4 engines really do run two steps at once. TP=8 won one case: cached
prompts at low load (25.7 against 22.3 rps at 64 in flight), where a lightly loaded request gets eight
GPUs instead of four. Under load the two engines won every shape.

A second, mechanical reason the single engine lost at 512: **one engine caps concurrent sequences at
`maxNumSeqs`** (256 by default), so half the requests queued behind the other half (cached p50 8.5 s
against 5.1 s). Two engines hold 2 × 256 without changing anything.

The same host also ran the bf16 build of the model (470 GB, TP=8 is the only fit). bf16 at TP=8 measured
8.3 / 11.5 / 15.1 / 15.4 rps on the reference shape at 64 / 128 / 256 / 512 in flight, against 7.9 / 10.0 /
13.1 / 13.4 for FP8 at TP=8 (which needs expert parallelism to start), so **at TP=8 the weight precision
made no useful difference**. The arithmetic says why: the active parameters read per decode step are 22B
× 2 bytes ÷ 8 GPUs ≈ 5.5 GB per GPU, about 1.6 ms of HBM time, in a step that measured around 30 ms. When
the weight stream is 5% of the step, halving it cannot show. The step is spent in 94 layers of
all-reduce, expert dispatch and small kernels in lockstep across eight GPUs: a fixed per-step cost that
more GPUs do not shrink. FP8 was still worth 1.5× on this host (22.5 against 15.4 rps), but through the
topology it unlocked (two engines at TP=4), not through faster arithmetic.

**So: the smallest tensor-parallel degree that fits the weights with useful KV headroom, and as many
engines as the host then allows.** In-engine data parallelism (`dataParallel: 2` × TP=4, one process,
one load balancer target) was the worst of the three on every shape, as it was on two GPUs.

**Where each wins:**

| | One engine per GPU | Tensor parallelism |
|---|---|---|
| Model does not fit on one GPU | impossible | **the only option** |
| Unique prompts, saturated | **+50% prefill, and the only topology that met the latency gate** | fails the gate |
| Shared prefixes, saturated | within 3% | **slightly ahead** (49,201 vs 47,913) |
| Low concurrency | worse | **better**: nothing else can use the idle GPU |
| Fault isolation | **one engine dying costs 1/N of capacity** | one failure domain |
| Rolling model updates | **yes** | no |
| Reasoning about it | **one model, one GPU, no interaction** | collectives, shared KV, lockstep |

**"Replicas are not a throughput strategy" holds only for cached workloads at moderate concurrency**,
where it was measured; without cache hits the picture inverts.

For throughput, in order:

1. **Prefer FP8** so the model fits on fewer GPUs.
2. **One engine per GPU** if it fits: `tensorParallel: 1`, `replicas:` = the instance's GPU count.
3. **Tensor parallelism only when the weights need it**, or when your prompts reliably share a prefix.
4. **Scale by adding instances.**

`replicas × tensorParallel` must not exceed the instance's GPU count.

---

## Larger instances: what is measured and what is not

**Measured:** one GPU at TP=1, and two GPUs three ways (TP=2, TP=1 with one GPU idle, 2 × TP=1
replicas) at concurrency 256, cached and uncached, on this GPU. On eight H100s (`p5.48xlarge`): eight
engines at TP=1 with models that fit one GPU, and a 235B mixture-of-experts that does not, as 2 × TP=4,
1 × TP=8 and DP=2 × TP=4, with and without expert parallelism (*Topology*).

**Not measured: TP=4 and TP=8 on this GPU.** Four- and eight-GPU instances of this generation were not
obtainable during testing. The H100 results are the best guide: the collective cost grew with the
degree there and the interconnect is faster, so expect no better here.

**First, check you want an 8-GPU instance at all**; if the model fits on one GPU, single-GPU instances
are cheaper for the same throughput (*For a model that fits one GPU, buy the smallest instance*).

### Recommended starting point for an 8-GPU instance: 8 replicas × TP=1

```yaml
instanceType: g7e.48xlarge
tuning:
  tensorParallel: 1
  replicas: 8
```

One engine per GPU:

1. **It is the measured shape, wider.** 2 × TP=1 won at two GPUs; 8 × TP=1 is the same topology.
2. **The model does not need more than one GPU.** A ~30B model in FP8 is about 29 GiB and fits a 96 GiB
   card with room for KV cache.
3. **Tensor parallelism does not scale under load.** The second GPU added nothing to uncached prefill
   at TP=2 (22,685 → 21,665). Eight GPUs will not do better: the collective gets wider and the
   synchronisation more expensive.
4. **It fails gracefully.** One engine dying costs an eighth of capacity. With 1 × TP=8 it costs all of
   it.
5. **It is simpler to reason about.** One model, one GPU, no collectives.

### The alternative: 4 replicas × TP=2, for high-cache-hit workloads

```yaml
tuning:
  tensorParallel: 2
  replicas: 4
```

Only if your prompts reliably share a long prefix: TP=2 measured slightly ahead of 2 × TP=1 (49,201 vs
47,913 prefill, about 3%). Without cache hits, at two GPUs TP=2 *failed* the latency gate that
2 × TP=1 passed.

### What to measure first

In order:

1. **Host contention at 8 engines.** Contention for vCPU, host memory bandwidth and PCIe was measured
   with **two** engines, not eight; an 8-GPU instance has proportionally more host per GPU, but four
   times the engines is well outside what was tested.
2. **8 × TP=1 against 4 × TP=2 on your own prompts.** The crossover is prefix-sharing.
3. **8 × TP=1 against 1 × TP=8.** Measured on H100s with a model that needs the GPUs: two TP=4
   engines beat one TP=8 engine by 40% at load. With a model that fits one GPU the gap can only be
   wider.

If host contention bites at 8 engines, the fix is fewer, wider engines: 4 × TP=2.

### Two practical points for a multi-engine instance

- **Host memory is divided by the number of engines, and getting it wrong is silent.** ECS reserves the
  full memory limit per task, so eight tasks each asking for the whole instance's share means seven
  never place: one engine at an eighth of the throughput you pay for. This project divides
  `container_memory_mib` by `replicas` automatically; if you set memory yourself, divide it.
- **`/dev/shm` is already sufficient** for any topology up to TP=8.

---

## Long prompts: prefill slows with length, and the cache is the whole game

Everything above was measured at 1,000 to 4,000 prompt tokens. Measured at 16k, 32k and 64k with the 30B
fp8 mixture-of-experts, streamed, unique prompts and then the same prompts with their prefix already
cached:

| Prompt tokens | One g7e engine, unique: input tok/s, TTFT p50, per-request decode | Eight H100s, unique: input tok/s, TTFT p50 | Cached prefix: TTFT p50 |
|---|---|---|---|
| 4,000 | ~36,000 inside one request | 212,000 at 256 in flight, 0.18 s | |
| 16,000 | 11,000 to 18,000 at 2 to 16 in flight, 0.6 to 1.5 s, 106 to 18 tok/s | 215,000 at 128, 0.58 s | 60 to 90 ms (g7e), 0.18 s (H100 at 64k) |
| 32,000 | 11,500 to 12,800 at 2 to 4, 2.4 to 3.0 s | 169,000 at 64, 1.6 s | 80 to 110 ms |
| 64,000 | 8,300 at 1 to 2, 5.2 s; 5,800 at 8, 21 s | 110,000 at 32, 4.5 s (p95 11.8 s) | 140 to 170 ms |

Four things follow:

- **Prefill tokens per second is not a constant of the GPU; it falls with prompt length.** Attention's
  share of prefill grows with the square of the prompt while the expert work grows linearly, so the same
  engine that prefills 36,000 tok/s at 4k does 8,300 at 64k, and a host that peaks at 215,000 between 4k
  and 16k is at 110,000 at 64k. Size a long-prompt fleet from a long-prompt measurement.
- **A cached long prefix costs nothing.** Time to first token at 64k fell from 5 s to 0.14 s when the
  prefix was resident, and the cached input rate reached 138,000 tok/s on one g7e engine and 879,000 on
  eight H100s: 8× the unique rate at 64k, against 1.7× at 1k. For long-context traffic the prefix cache
  and the routing that keeps a conversation on its engine (*Prefix caching is a routing decision*) decide
  the fleet size; the GPU's prefill speed is the fallback path.
- **Concurrency at long context costs latency before it costs memory.** At 64k on one engine, decode per
  request fell from 131 tok/s alone to 4 tok/s at 8 in flight, and TTFT p95 reached 42 s, while the KV
  cache never passed 37% and nothing was preempted: chunked prefill of the neighbours' prompts occupies
  the engine. On the H100 host, 64k requests hold about 3 GiB of KV each in fp8, so 43 GiB per GPU holds
  about 14 of them; that, not compute, capped the batch there.
- **The model's 262k context limit was never the limit; time to first token was.** `maxModelLen` only
  matters where the weights nearly fill the card (*`maxModelLen`*).

---

## Reasoning models: the effort setting is the capacity setting

A reasoning model spends tokens thinking before it answers, and the request decides how many. Measured
with the 120B mixture-of-experts on one engine, 1,000-token prompts, answers capped at 2,000 tokens,
`scripts/benchmark.py --reasoning-effort`:

| Effort | Output tokens per answer | Requests/s at 8 in flight | at 32 | p50 at 32 |
|---|---|---|---|---|
| low | 465 | 1.3 | 2.6 | 9.1 s |
| medium (the default) | 803 | 0.7 | 1.6 | 18.8 s |
| high | 1,607 (many hit the cap) | 0.3 | 0.8 | 36.0 s |

The engine produced the same 580 tokens/s at 8 and 1,170 at 32 whatever the effort, and decode per
request was the same 79 and 42 tokens/s. Effort changed only how many tokens an answer costs: 1.7×
from low to medium, 2× from medium to high, 3.5× end to end, and latency with it. Time to first token
did not move, because the first token is the first thought.

Two consequences for a fleet: size it on output tokens per second, and set the effort per route, because a
fleet sized for low-effort traffic serves a third of the requests when clients ask for high. A quality
harness has to see the answer, too: with the cap below the model's chain of thought, answers never
leave the reasoning channel and score as empty (*What quantisation costs in answers*).

The same law on a second family with a thinking switch rather than an effort setting. Gemma 4 thinks when
the request (or `--default-chat-template-kwargs '{"enable_thinking": true}'`) asks; measured on one engine
with 1,000-token prompts, answers left to run to their end under a 2,000-token cap:

| Model | Thinking off: tokens per answer, req/s at 8 and 32 | Thinking on | Decode tok/s per request, off / on |
|---|---|---|---|
| 26B MoE, fp8 | 265, 2.83 and 6.66 | 1,017, 0.72 and 1.72 | 100 / 102 at 8, 60 / 65 at 32 |
| 12B dense, fp8 | 241, 2.36 and 6.49 | 910, 0.63 and 1.68 | 75 / 80, 50 / 60 |
| 31B dense, fp8 | 350, 0.68 and 1.86 | 939, 0.25 and 0.55 | 34 / 35, 24 / 27 |

3.8× the tokens per answer on the two smaller models, 2.7× on the 31B, which already writes longer answers;
the request rate falls by the same factor and decode per request does not move. Time to first token did not
move either. With a 190-token cap and thinking on, the request rate read the same as with thinking off,
because every answer was cut off inside the thought: `content` empty, the reasoning truncated. A fleet with a
small `max_tokens` and thinking on looks healthy on every throughput graph and answers nothing; size the cap
for the chain of thought (about 1,000 tokens here) or turn thinking off per request
(`chat_template_kwargs: {"enable_thinking": false}`) or by default. The tool-call turn is affected too: the
same `get_weather` call cost 23 output tokens with thinking off and 104 to 130 with it on.

---

## Tool calling: the parser is part of the deployment

An agent framework sends `tools` with the request and reads `tool_calls` from the answer. The model
does not produce that field; it produces text in its own tool-call format (`<tool_call>{...}</tool_call>`
for Qwen3, an XML dialect for Qwen3-Coder, the Harmony channels of gpt-oss), and the engine needs a
parser for that format to turn the text into the field. Without one, a request that carries `tools` gets
the model's tool-call text back as plain `content`, no client library recognises it, and the agent stalls
on its first step with no error anywhere. `toolCallParser` in `config.yaml` is that one flag, per model
family: `hermes` for Qwen3 and Qwen3-30B-A3B, `qwen3_coder` for Qwen3-Coder, `openai` for gpt-oss,
`gemma4` for Gemma 4, `llama3_json` for Llama 3.x, `mistral` for Mistral; `vllm serve --help` lists them all. For a thinking
model served with thinking on, `reasoningParser` (`qwen3`, `openai_gptoss`, `gemma4`, `deepseek_r1`) moves the chain
of thought into `reasoning_content` so the client gets the answer; with thinking off it is not needed, with one
measured exception: Gemma 4 emits an empty `<|channel>thought\n<channel|>` block after a tool-result turn
even with thinking off, and without `reasoningParser: gemma4` those tokens arrived in `content` (a plain
question came back clean). With thinking requested and no parser, the answer was 1,336 tokens of nothing:
empty `content`, no reasoning field. For that family the two parsers are one setting.

Three things measured while scoring agents through this stack (*What quantisation costs an agent*):

- **The parser adds nothing measurable to serving cost**, and its output was scored on 4,441 BFCL tests
  per configuration: the engine's own parser plus the model reached 85 to 87% on single-turn function
  calls for every 30B-class model, with the model, not the parser, deciding the rest.
- **Agent steps are long requests.** A coding agent's step carries the whole trajectory, tens of
  thousands of tokens, and on a busy engine a non-streamed step took over 120 s, which is the endpoint's
  non-streamed limit (README, *Access*): the client saw `504 Gateway Timeout` from CloudFront and
  retried forever. Agents behind this stack should stream, or the origin timeout must be raised. The
  same load never timed out on chat-sized requests.
- **A checkpoint can be fine for chat and broken for tools.** The uncalibrated NVFP4 Coder build above
  scored within a point of bf16 on extraction and short function calls and lost 27 points of
  irrelevance detection. Score tool use before serving a 4-bit build to agents.

---

## Structured output costs about 15% of decode speed

Tool calls and agents ask for JSON that matches a schema; the engine compiles a grammar per request and
masks every sampled token against it. Measured with `scripts/benchmark.py --schema` (a fixed object
schema, strict) against plain generation on one g7e engine with the 30B fp8, 1,000-token prompts,
streamed:

| In flight | Plain: decode tok/s per request, TTFT p50 | Structured: decode tok/s per request, TTFT p50 |
|---|---|---|
| 8 | 90.4, 0.16 s | 80.8, 0.07 s |
| 32 | 55.5, 0.34 s | 46.2, 0.10 s |
| 64 | 42.6, 0.39 s | 35.2, 0.12 s |

Decode per token is 11 to 17% slower under the grammar at every level, and e2e p95 was 14% higher at 64.
Requests per second went the other way, because a closed schema ends the answer sooner; time to first
token fell because the first token is the mandatory brace. Size a tool-calling fleet on tokens, not on
requests, and budget about 15% of decode for the grammar.

What the grammar buys is shape, not accuracy. `scripts/extraction.py` asked three models for the same
JSON object three ways, a strict schema, `json_object`, and a plain request for JSON, on 1,000
CoNLL-2003 sentences: entity F1 agreed within a point in every case (69.9 / 69.7 / 69.7 for the Coder,
65.2 / 65.2 / 65.2 for the 32B), and every free-form answer parsed. Use the schema when a malformed
answer would break the caller; do not expect it to extract better.

---

## Engine tuning

### `gpuMemoryUtilization: 0.95`: the biggest single effect for a model that fills the card

The +27% below is for a model occupying most of the card, where the last 5% is a large share of what
remains for the KV cache. For a model that leaves tens of GiB free, the same 5% is a few percent of the
cache and this setting is not a lever.

Not the engine's 0.90 default. The KV cache is whatever remains after the weights; on a model that
fills most of the card, the 5% of VRAM that 0.90 leaves unclaimed can be larger than the entire KV
cache.

Measured, 0.90 → 0.95 on a model occupying most of a card: **+27% throughput, −22% cost per token**.
The single largest win of any parameter tested.

**Do not exceed 0.97, and 0.95 is the better default anyway.** Above it the engine passes startup
memory profiling and fails under real load. 0.95 → 0.97
measured **+1.6%**; at 0.97 the mixture-of-experts model above was stable, while a dense 27B on the
same hardware crashed under load.

**0.95 itself is not universally safe.** Two independent cases where it is too high:

1. **Dense models**, as the 27B above.
2. **Unquantised weights with an fp8 KV cache.** Starts cleanly, passes its health check, then fails
   with CUDA out-of-memory under traffic, surfacing as `502`. An fp8 cache holds twice as many tokens
   per byte, so batches grow, so the per-step fused-expert workspace grows, and bf16 weights have
   already taken most of the card.

   **The fix is a plain KV cache, not lower utilisation.** Lowering it stops the crash and is still a
   net loss: at 0.90 the same deployment measured **37,877** tokens/sec against **46,808** for a plain
   cache at 0.95.

   The stack warns at synth time when it detects this; [troubleshooting.md](troubleshooting.md) has
   the diagnostic.

   | weights | KV cache | why |
   |---|---|---|
   | FP8 | **fp8** | +12% decode, and 29 GiB of weights leaves ample headroom |
   | bf16 | **default** | fp8 either crashes or, once made safe, is slower than not doing it |

**`--kv-cache-memory` changes nothing measurable; use it for a fixed pool, not for speed.** The engine
suggests it at startup:

```
Actual usage is 52.7 GiB for consumed memory (weights + non-torch), 5.65 GiB for peak
activation, and 0.47 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with
`--kv-cache-memory=38082119680` (35.47 GiB) to fully utilize gpu memory.
```

On throughput it is a no-op: set explicitly it measured 74,491 against 74,687 prefill for the fraction
alone. What it does change is that the pool is an absolute size, independent of what else is resident on
the GPU, so every host gets the same cache and a result reproduces. That is worth having on a card the
weights nearly fill, where 0.95 has crashed under load (*troubleshooting.md*).

### `enablePrefixCaching: true`: always on, but read the caveat

Reuses computed attention state across requests sharing a prompt prefix. For a common system prompt or
tool schema (most agent and RAG traffic) it is worth up to an order of magnitude on time-to-first-token
under load. It removes prefill work only; decode still re-reads the whole cache for every token, so it
does nothing at concurrency 1.

**Leave it on whatever your workload.** Turning it off on prompts that share nothing measured
**−1.2%**.

**The caveat is about concurrency, not the flag.** Every throughput figure here was measured with
prompts that share a prefix; *Choosing an operating concurrency* gives both numbers. Sizing a fleet from
the cached figure when your traffic has no shared prefix **under-provisions by about 2×**.

#### Prefix caching is a routing decision

The cache is per engine. A prefix computed on engine 3 helps only a request that reaches engine 3, and
the load balancer sends each request to the next engine in turn. That is harmless for the one kind of
reuse the cached figures above were measured with, a prefix every request shares, because every engine
warms it up independently. It fails for every other kind: the second turn of a conversation, an agent
loop resending its growing context, a tenant's document. Those hit their prefix about 1 in N times on N
engines, and the fraction falls as the fleet grows.

Measured on eight engines with six-turn conversations (a 600-token unique opener, 150-token answers,
each turn resending the conversation so far), streamed, 120 s per level, fleet hit rate from the engine
counters:

| Routing | 512 in flight | p50 | Hit rate |
|---|---|---|---|
| Round robin (default) | 118.5 req/s | 4.27 s | 21% |
| Load balancer cookie per session (`stickySessions: true`) | **135.3 req/s** | **3.69 s** | **75%** |
| One engine, all turns on it, scaled ×8 | 149 req/s | 3.34 s | 74% |

Every turn after the first can hit, which is 76% of the prompt tokens in this shape; the cookie reaches
it, round robin gets a third of the way there (connection reuse between the CDN and the load balancer
gives some accidental affinity). The engine-side gain is a modest +14% at this prompt length, because a
1,000-token prefill is cheap on this GPU; the same mechanism on 8,000-token contexts is a multiple.
Single-turn unique prompts measured 0% hits in both configurations, so nothing here changes the
unique-prompt sizing figures.

`stickySessions: true` turns on a one-hour load balancer cookie; the CDN forwards cookies both ways.
The cost is balance: one upstream service calling with a single cookie jar pins all of its traffic to
one engine, so leave it off unless clients keep a cookie per end-user conversation. The next step up is
a router that knows which engine holds which prefix and the queue depth of each; this project does not
include one (*What this project does not do*).

#### Offloading the cache to host memory moves the capacity wall, it does not remove it

The cache is per engine and lives in GPU memory, so there are two independent problems: getting a request
to the engine that holds its blocks (routing, above) and keeping the blocks at all. vLLM can keep them
somewhere other than HBM: `extraArgs: --kv-offloading-size 32` allocates a 32 GiB host-memory tier and
copies blocks into it as they leave the GPU, so a later turn reloads them over PCIe instead of prefilling
them again. Measured on one `g7e.4xlarge` with the 30B fp8 model, six-turn conversations of 8,000 tokens,
streamed. Its GPU cache is 58.0 GiB, which is 1,266,816 tokens at 48 KiB of KV per token, so the working
set is stated as a multiple of that:

| Working set | Tier | GPU cache hits | Host tier hits | req/s | TTFT p50 | Host memory written |
|---|---|---|---|---|---|---|
| 0.5M tokens, 0.4x the GPU cache | none | 75.7% | | 7.459 | 0.139 s | |
| 0.5M tokens, 0.4x the GPU cache | 32 GiB | 75.7% | 0% | 7.459 | 0.138 s | 98 GB in 120 s |
| 1.4M tokens, 1.1x | 32 GiB | 0% | 0% | 3.614 | 4.76 s | 456 GB in 240 s |
| 1.8M tokens, 1.4x | none | 0% | | 3.626 | 15.45 s | |
| 1.8M tokens, 1.4x | 32 GiB | 0% | 0% | 3.676 | 15.25 s | 590 GB in 240 s |
| 0.26M tokens, 3.2x a deliberately small 81,376-token cache | 32 GiB, 9x the cache | 0% | **79.0%** | 3.124 | 6.55 s | 257 GB in 240 s |

Read the table as one ratio, not as a verdict on the feature. The host tier serves the same fraction the
GPU cache would have served, 79% against 75.7%, when it is large against the working set (last row). It
serves nothing when it is small against the working set: at 32 GiB against a 1.8M-token working set it was
written 590 GB in four minutes, overwriting itself about seventeen times, so a conversation's blocks were
gone from the host tier too by the time its next turn arrived. Between those two rows the end-to-end
numbers move by 1.4% on requests per second and 1.3% on time to first token, inside the run-to-run band.

Three things follow. **The wall is a ratio.** Size a tier against the working set you expect to reuse and
against how long reuse takes to come back, not against HBM. **It is not free when it does nothing.** In the
shape that already fits HBM the numbers are identical to three decimals and 98 GB still crossed the bus,
because the store path runs on eviction whether or not anything will read it back. **It is node-local.**
Each engine creates its buffer as a file in its own container (`/dev/shm/vllm_offload_<engine-id>.mmap`),
so a request routed to a different replica cannot see it: this does nothing for the routing problem above,
and the fix for that is either the cookie or a store the fleet shares. vLLM 0.28.0 can reach shared stores
without patching (its connector registry includes LMCache, Mooncake, FlexKV, HF3FS and NIXL), which needs
the client library in the image and a backend inside the VPC; not measured here.

The tier reports itself: `vllm:external_prefix_cache_queries_total` and `vllm:external_prefix_cache_hits_total`
are the tokens looked up in and served from it, `vllm:kv_offload_total_bytes_total` is what it cost the bus.
The dashboard does not scrape them (*Engine metrics*, and custom metrics are billed per name); read them from
`/metrics` through the endpoint during a representative hour before deciding the tier earns its place.

Two traps, both of which refused to start the engine. The tier is allocated in `/dev/shm`, so the
container's shared memory must exceed it; this stack sizes `/dev/shm` at half the container's memory for
that reason. And pinning the cache small to test a tier (`--kv-cache-memory-bytes`) is refused unless one
request at `maxModelLen` fits in what is left, so cap `maxModelLen` in the same change.

### `maxModelLen: 0` (the model's own maximum)

Maximum tokens per request, input plus output.

**"Size the context window to your workload" is widely repeated and measurably false here.** Cutting it
8× (32,768 → 4,096 on a ~1,200-token workload) changed throughput by −1.6%, noise. vLLM allocates KV
cache in fixed-size blocks on demand; the ceiling is not a reservation, so lowering it frees nothing.

Set it only to **reject** requests longer than some limit. Not to go faster.

One exception, and it is about starting rather than speed: the engine refuses to start unless the KV
cache left after the weights can hold at least one request of `maxModelLen` tokens. On a card the
weights nearly fill (a 57 GB bf16 model on an 80 GB card, 16 GiB left, 24 GiB needed for a 262k
request) the model's default context makes the engine fail at startup; `maxModelLen: 32768` starts it.
On the 96 GB cards this document is measured on, none of the models tested came close.

### `maxNumSeqs: 256`

Ceiling on concurrent sequences. A ceiling, not a reservation: it reserves no memory; too low leaves
the GPU idle.

128 → 256 measured **+1.4%**; 256 → 512 was noise. Both on 1,000-token prompts.

**Keep it well above the concurrency you load-test at**, or you are measuring this flag, not the
hardware: requests past it queue instead of batching.

The cap is per engine, so a host with N engines holds N × 256 before it binds. It bound exactly once in
these measurements: one TP=8 engine given 512 in flight queued half of them (cached p50 8.5 s against
5.1 s for two TP=4 engines holding 256 each). Raising it to 512 on the two-engine host changed nothing
(22.6 against 22.5 rps), because it was never the limit there.

**With long prompts the cap has to come from the KV cache, not from this flag.** The cache holds a fixed
number of tokens: on this GPU with FP8 weights and an fp8 cache, about 63 GiB ÷ 48 KiB per token ≈
1.3 million tokens. 256 sequences of 1,200 tokens fit ten times over; 256 sequences of 12,000 tokens do
not, and the engine preempts rather than refuse. Derive the cap for your longest common request:
`cache tokens ÷ (input + output tokens)`, and set `maxNumSeqs` at or below it. The dashboard's KV cache
and preemption widgets show when this is the limit.

### `maxNumBatchedTokens: 0` (engine default)

Token budget per scheduler step.

**No measured effect from 8k to 64k; leave it alone.** 64k landed within noise (decode 136.0 → 136.2,
prefill 19,069 → 19,274), as did cutting it 4× to 8k (prefill 49,603 vs 49,774). The prefill batch is
never the constraint at this scale, so the default omits the flag.

Measured again at 4,000-token prompts against the engine default: 16,384 changed no shape and no level by
more than 1% (*Choosing a model to host*). Measured a third time on a prefill-bound configuration built
to give it a chance (a 235B mixture-of-experts at TP=4 on H100s, where 4,000-token unique prompts held a
flat 7 rps from 64 to 512 in flight): 16,384 moved every shape by under 5%, inside run-to-run noise. The
chunked-prefill budget is not what limits prefill; the tensor cores and the all-reduce are.

### `kvCacheDtype: fp8`

Stores the KV cache at 8 bits instead of 16. Measured **+12% decode** at full load (163.7 vs 146.7,
like-for-like), up from +4.9% at low concurrency. On a sliding-window mixture-of-experts (the 120B
MXFP4) it was +5% on the reference shape and +7% on 4,000-token prompts, and it started without complaint.

At high concurrency the KV cache is roughly half of all decode memory traffic, and at concurrency 1
almost none, so halving it is worth close to 10% with a full batch and nothing alone. Smaller entries
also fit more requests in the same VRAM.

**Measured in three places, fp8 was neutral to positive in all but one.** On this GPU with the 30B, fp8
won every shape including long cached prompts (17.1 against 12.3 req/s at 64 on 4,000-token shared
prompts), and the gap widened with context: at 16k and 32k prompts `auto` was 10 to 24% slower on every
level, unique and cached, with half the token capacity in the same 58 GiB. On eight H100s with the same model, from 1k to 64k prompts: nothing separates them below 16k
(±5%), fp8 wins 6 to 18% on unique prompts from 16k up because twice the tokens fit and the batch grows,
and cached long prompts are a wash. The one exception was a 235B mixture-of-experts at TP=4 on H100s,
where 4,000-token cached prompts ran 20 to 30% faster with `auto`, reproduced twice on that model and
not on the 30B at TP=1: a property of that model and topology, not of the GPU. Decode speed per request
did not depend on the cache precision anywhere. Quality: no measurable effect on gsm8k or ifeval
(*What quantisation costs in answers*).

**A second exception, with a mechanism: the cache precision can decide which attention kernel you get.**
Gemma 4's global layers use 512-wide heads, and in vLLM 0.28.0 on H100s only the Triton attention kernel
accepts an fp8 KV cache for them; with a bf16 cache FlashAttention is offered and chosen. Measured on
eight H100s with the 31B in fp8 weights, `auto` against `fp8`: +8 to +15% on the reference shape, +9 to
+18% on long answers, +28 to +69% on 4,000-token unique prompts (14.4 against 8.5 req/s at 512 in
flight), and identical decode per request at 1 to 32 in flight, with half the tokens in the cache. The
26B mixture-of-experts, same swap: −1 to +7% on the reference shape, +5 to +13% on long answers and
mixed traffic, +17 to +46% on 4,000-token unique prompts (74.9 against 51.3 req/s at 512 in flight),
−10 to 0% at 1 to 32 in flight: the same mechanism, with less of the step spent in attention. The
log says which kernel ran (`Using ... attention backend out of potential backends: [...]`); when `fp8`
narrows the list to Triton, `auto` is the faster setting. On the g7e the list is Triton either way for
this family (FlashAttention 4 does not support the head size there), so its rows stand as measured.

**On by default here.** It is a lossy store for cached attention state; to rule that out, set
`kvCacheDtype: auto`, and do so when it buys a better attention kernel.

### CUDA graph mode: leave the engine default

vLLM 0.28.0 captures full CUDA graphs for decode-only batches and piecewise graphs for batches that mix
prefill and decode (`FULL_AND_PIECEWISE`). `FULL_DECODE_ONLY`, set with
`extraArgs: -cc.cudagraph_mode=FULL_DECODE_ONLY`, skips the piecewise graphs to give their memory back
to the KV cache. Measured on one engine with the 30B fp8 mixture-of-experts: graph capture fell from 9 s
and 0.62 GiB to 4 s and 0.21 GiB, and the KV cache grew from 57.99 to 58.29 GiB, half a percent. Every
unique-prompt shape at 8 to 64 in flight was within 3%; the cached reference shape ran 10 to 33% faster
in a single run, which is unexplained and unrepeated. The memory argument does not exist on a 96 GiB
card; there is nothing here to change until the cached result is reproduced.

### `enableExpertParallel: false`: measure it, do not assume

For mixture-of-experts models, pins whole experts to specific GPUs instead of sharding every expert
across all of them. Communication changes shape: one all-reduce per layer becomes an all-to-all
dispatch and combine.

**The measurements disagree, and that is the finding:**

| Hardware and configuration | Effect on decode |
|---|---|
| bf16, TP=2, fast interconnect | **+15%** (138.4 vs 119.9) |
| FP8, TP=2, fast interconnect, at full load | **−13%** (142.5 vs 163.7) |
| bf16, TP=4, PCIe-connected GPUs | **−18%** |
| FP8 235B MoE, TP=4 × 2 engines, H100 NVLink, 512 in flight | **+5%** unique (23.6 vs 22.5 rps), **−10%** cached (85.8 vs 95.5) |
| bf16 235B MoE, TP=8, H100 NVLink | **−7 to −15%** unique (13.1 vs 15.4 rps at 512), **−20%** cached (39.7 vs 46.8 at 256) |

Two variables move the answer. **Interconnect:** each token routes to a handful of experts scattered
across GPUs, and all-to-all punishes a slow link far harder than all-reduce. **Precision:** in FP8 the
GPU finishes its share sooner, so communication is a larger fraction of the step.

No correct default exists, so this project ships it off. The bf16 gain seen at TP=2 did not survive at
TP=8 on the same class of interconnect: with 128 experts spread over eight GPUs the all-to-all touches
every GPU on every layer, and the loss grew with the degree. Expect a loss in FP8, a significant loss over
PCIe, and in bf16 a result that depends on the degree. Measure it: one flag, one restart.

One case where it is not optional: a block-quantised FP8 checkpoint at a tensor-parallel degree that
does not divide its expert size into whole tiles refuses to start without it (see *Tensor parallelism*).

---

## Settings that sound useful and measurably are not

Each was tested and made things worse.

### `--enforce-eager`: measured **−85% decode**, and it is the advice you will be given

Disables CUDA graphs: every forward pass is dispatched op by op from Python.

Of 23 configurations tested, this was **the only one that failed a latency budget**: decode fell from
117.5 to **17.8 tokens/sec per request**. A decode step does little GPU work per token, so launch
overhead dominates without CUDA graphs.

It is the standard first suggestion when an engine will not start, and it does fix startup problems
(lower memory use, no graph capture). Use it to *diagnose*, through `EXTRA_ARGS`, and remove it before
measuring or serving.

`validate_tuning` rejects `enforceEager: true` outright with this number attached.

### A host-memory KV tier smaller than the working set: measured **0 hits**, 590 GB of bus traffic

`--kv-offloading-size 32` behind a 58 GiB GPU cache, with 1.4x the cache in conversations in flight: every
block was overwritten in the tier before its conversation returned, throughput and time to first token
within 1.5% of no tier, and 590 GB crossed PCIe in four minutes. The same tier served 79% of prompt tokens
when it was nine times the cache. Size it against the working set that comes back or leave it off
(*Offloading the cache to host memory moves the capacity wall* above).

### How speculative decoding works, and when it pays

The two speculative sections below make sense once the mechanism is clear.

Decode is slow for one reason: each generated token needs its own forward pass, and a pass cannot start until the
previous one has produced its token. Each pass streams the model's active weights out of memory to produce one token
per request, while the arithmetic units sit mostly idle (*The one mental model worth having*).

A forward pass does more than predict the last token. Given a sequence, it predicts the next token at every position
at once; prefill uses exactly this to read a whole prompt in one pass. Generation ignores all but the last position,
because only the last one is new. Speculative decoding puts the other positions to work:

1. A small draft model guesses the next few tokens as one sequence, not as alternatives. It writes them one at a
   time like any model, but it is small (Gemma 4's drafters have four layers), so the guesses cost little.
2. The target model runs one forward pass over the text so far plus the guesses. At every position it produces the
   token it would have generated there. Attention only looks backwards, so its prediction after a guessed token is
   the same one it would have made had it written that token itself.
3. Walking the positions in order, each guess that matches the target's prediction is kept. At the first mismatch
   the target's own prediction is kept instead, and everything after it is discarded.

```
text so far: "The cat"                       draft guesses: "sat on the mat"
one target pass over "The cat sat on the mat":
  after "cat" the target predicts "sat"      the guess matches: keep
  after "sat" the target predicts "on"       the guess matches: keep
  after "on"  the target predicts "a"        the guess said "the": keep "a", discard "the mat"
result: "sat on a", three tokens from one pass of the target model
```

The output is what the target model would have written on its own: every kept token is its own prediction on a
prefix it would have produced. Only the speed changes. One pass yields at least one token, no worse than without a
draft, and at most one more than the number drafted. The attention state of the kept tokens stays in the KV cache,
so nothing is recomputed. The engine logs the average as `Mean acceptance length` (2.2 of 3 for the 30B's EAGLE-3
draft, 3.0 to 3.1 of 4 for Gemma 4's drafters).

Why it is cheap: the pass over five positions reads the weights once, like a pass over one, and the extra arithmetic
lands on units that were idle. Why it is not free: that arithmetic is real work, rejected guesses are wasted, and the
draft takes some KV cache (2.5 to 2.8 GiB in the drafts measured here, a few percent of the pool). So the gain depends
on spare compute. With few requests in flight there is plenty: +107% single-stream on a dense 12B. With many unique
prompts in flight, prefill needs that compute, and a fixed draft length lost 27% at 64 per engine on gpt-oss-120b. The
batch-size schedule in the EAGLE-3 section below turns drafting down as the batch grows and keeps the low-load gain
without the loss.

### N-gram speculative decoding: measured **−58%**

Drafts several tokens ahead from the prompt, then verifies them in one pass. Output is identical, so it
is pure speedup when it works. It failed for three compounding reasons:

1. **It spends compute to buy latency**: drafting 5 tokens means ~6× the work per forward pass,
   refunded only when the draft is accepted, and at high concurrency there is no spare compute.
2. **Acceptance was near zero**: matching against the prompt works where output quotes input
   (summarisation, code editing, extraction), not for open-ended generation.
3. **It silently disables asynchronous scheduling.**

Revisit only for a large model, at low concurrency, on a workload whose output quotes its input.

### Speculative decoding with EAGLE-3: the largest gain measured, and it is a flag

An EAGLE-3 draft model is a small separate checkpoint trained against the target model. vLLM 0.28 loads
it with `--speculative-config`; nothing has to ship inside the target checkpoint. Publishers release them
alongside the model. For the shipped model:

```yaml
extraArgs: --speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3","num_speculative_tokens":3}'
```

Measured on eight `g7e.2xlarge`, FP8 weights, fp8 KV cache, unique 1,000-token prompts, 190-token
answers, 60 s per level after warm-up:

| Concurrency | Without, input tok/s | With EAGLE-3 | Gain | p95 without | p95 with | Decode tok/s per request |
|---|---|---|---|---|---|---|
| 8 | | 7,381 | | | 1.22 s | 189 |
| 64 | | 32,842 | | | 2.02 s | 105 |
| 256 | 59,213 | 83,707 | **+41%** | 4.03 s | 3.17 s | 48.9 → 67.9 |
| 512 | 95,539 | 123,924 | **+30%** | 5.23 s | 4.32 s | 39.2 → 50.6 |
| 768 | 119,223 | 147,914 | **+24%** | 6.33 s | 5.62 s | 33.0 → 40.7 |

Mean acceptance length 2.2 of 3 drafted tokens. Output is the same distribution as without speculation;
the engine warns that `min_p` and `logit_bias` are unsupported with it. Not shipped as the default because
the draft model is specific to the target: change `modelId` and this line must change with it, and a
mismatch fails at startup.

The same flag on a different family, the 120B MXFP4 mixture-of-experts with its publisher's EAGLE-3 draft
(`nvidia/gpt-oss-120b-Eagle3-v3`): mean acceptance 2.5, long answers (1,000 in / 800 out) **31.8
against 19.1 req/s at 512 in flight (+66%)**, 97 against 65 tokens/s per request at 64, the mixed shape
+43%. On the reference shape with unique prompts the picture inverts with load: +44% at 8 in flight per
engine, +47% at 16, +8% at 32 with a heavy tail (p95 23 s), −27% at 64; with the prompts cached, +57%
at 64 and no tail. Verification needs compute, and prefill of unique prompts at high concurrency takes it
(*Choosing a model to host*, point 9).

**The loss at load is avoidable: make the draft length follow the batch size.** vLLM 0.28.0 accepts a
schedule in the same flag, `num_speculative_tokens_per_batch_size`, a list of `[from, to, K]` ranges over
the number of running requests. Measured on one engine with the 120B model and its draft, unique
1,000-token prompts, against the same engine without speculation:

```yaml
extraArgs: --speculative-config '{"method":"eagle3","model":"nvidia/gpt-oss-120b-Eagle3-v3","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,16,3],[17,32,1],[33,4096,0]]}'
```

| Per engine in flight | K by schedule | Reference shape | Long answers | Mixed shape | All prompts cached |
|---|---|---|---|---|---|
| 8 | 3 | **+23%** | **+41%** | **+25%** | +9% |
| 16 | 3 | **+27%** | **+54%** | **+37%** | **+27%** |
| 32 | 1 | **+17%** | **+38%** | **+22%** | **+22%** |
| 64 | 0 | −6% | 0% | −3% | +7% |

The static K=3 measured −27% at 64 on the same shape. With the schedule the low-load gain stays and the
high-load loss becomes noise, so one configuration holds across the day. At one request in flight the
draft gave 199 against 168 tokens per second.

**Warm every batch regime before you trust the first minute.** The first time the engine ran a new
draft length at a new batch size it stalled for about 30 s (p95 31 to 35 s at those levels, a clean 2 to
8 s at every level once seen). The identical sweep run a second time in reverse order showed no stall
anywhere. After a deploy with speculation on, sweep the concurrencies you intend to serve before taking
traffic, or the warm-up in `scripts/benchmark.py` at one level is not enough.

**A built-in draft head is the cheap kind of speculation, and topology decides whether it pays.** The
80B hybrid mixture-of-experts ships its own multi-token prediction head
(`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`): one extra token per step,
verified in the same forward pass, no second model to load. The engine reported the draft right four
times in five on both hosts, and the results differed by topology:

| Deployment | Effect of the MTP head |
|---|---|
| One engine on this GPU (TP=1), one draft token | decode per request **+9 to +18%** at every level up to 64 in flight; unique-prompt request rate +3 to +24%; the head took 2.8 of 11.5 GiB of KV cache, so cached shapes that were pool-bound lost |
| Same, two draft tokens | **+9 to +41%** on every unique shape, better than one token at every level including 64 in flight; one request in flight +30% |
| Four engines at TP=2 on eight H100s | decode per request −6% to +5%; long prompts −15 to −31%: the verify step across two GPUs cost about what the saved token was worth |

So: turn on a shipped MTP head on a single-GPU engine, try two draft tokens, and measure it before
trusting it on a tensor-parallel one. The "speculation loses at load" rule of the EAGLE-3 draft did
not apply here: a 3B-active model leaves compute to spare for verification. Judge any speculation by request rate and per-request decode on your traffic, never
by the acceptance rate the log prints, which was the same in both rows.

**A publisher's MTP drafter on a dense model was the largest gain measured, and it needs vLLM 0.29.0.**
Google ships a four-layer draft model for each Gemma 4 size (`google/gemma-4-<size>-it-assistant`); it
shares the target's KV cache and costs 2.7 GiB of it. On vLLM 0.28.0, the release this project pins, it
crash-loops at start (`Cannot copy between CPU and CUDA tensors during CUDA graph capture` in
`gemma4_mtp.py`; troubleshooting has the entry); 0.29.0 fixes it, and served the same model without the
drafter within ±3% of 0.28.0 at every level, so the comparison below is engine-neutral. The 12B in fp8
with `--speculative-config '{"model": "google/gemma-4-12B-it-assistant", "num_speculative_tokens": 4}'`,
one engine, mean acceptance 3.1 of 4 drafted tokens:

| Per engine in flight | Reference shape, req/s | Decode tok/s per request | All prompts cached | Long answers | 4,000-token unique prompts |
|---|---|---|---|---|---|
| 1 | 1.00 vs 0.48 (**+107%**) | 176 vs 86 (207 vs 90 on the decode shape) | | +90% at 8 to EOS | |
| 8 | 6.08 vs 3.23 (**+88%**) | 133 vs 71 | +256% | +103% | +44% |
| 16 | 9.54 vs 5.34 (**+79%**) | 104 vs 60 | +235% | +93% | +28% |
| 32 | 12.35 vs 8.03 (**+54%**) | 70 vs 45 | +56% | +63% | +17% |
| 64 | 14.23 vs 10.77 (**+32%**) | 40 vs 31 | +14% | +37% | +6% |

No level lost, p95 at 64 in flight 5.4 s against 6.0 s without. The 26B mixture-of-experts with its own
drafter, same settings, acceptance 3.0 to 3.1 of 4: +58% at 8 per engine, +60% at 16, +64% at 32, **+51% at
64** on the reference shape (21.0 against 13.9 req/s), +48% with every prompt cached, +61 to +76% on long
answers, and +34% to +53% single-stream (235 against 175 tok/s on 1,000-token prompts, 285 against 186 on the
decode shape), where its decode was already fast. The EAGLE-3
rule above (verification competes with prefill at high load) did not bite on either model: a dense 12B at fp8 has a 21,000 tok/s prefill wall
and 64 unique 1,000-token prompts every six seconds use half of it, and a four-layer drafter that reads the
target's KV is cheap to run. Where decode is the wall, this is the setting to turn on first.

An earlier version of this document said EAGLE and multi-token prediction were not configuration options
and had to ship inside the checkpoint. That was true of older engine versions and is wrong for 0.28.

---

## Choosing a model to host: what decides throughput

Hosting an open-weight model is a choice of four things: architecture (dense or mixture-of-experts),
size, precision, and the shape of your traffic. Their effects multiply, and they are cheap to measure
before you commit a fleet. Everything below was measured with one method on one fleet (8 × `g7e.2xlarge`,
one engine per GPU) so the effects can be compared; the model names are the evidence, not the point.

### The evidence: five sets of weights, one matrix

Three families: a 30B mixture-of-experts with ~3B active parameters, a dense 27B with hybrid attention,
each in every precision it is published in, and a 120B mixture-of-experts with ~5B active parameters
published only in MXFP4. A fourth, a 120B Mamba-hybrid mixture-of-experts in NVFP4, did not survive the
matrix (below). Six shapes (1,000 in / 190 out, the same fully cached, 4,000 in
/ 190 out unique and cached, 1,000 in / 800 out, and a 300 to 1,700 in / 100 to 300 out mix), four
concurrency levels (64 to 512 in flight, 8 to 64 per engine), 60 s each, unique prompts unless stated,
driven from an in-region client. At 512 in flight, whole fleet:

| weights | 1,000/190 req/s (p50) | same, all cached | 4,000/190 input tok/s | 1,000/800 req/s |
|---|---|---|---|---|
| 30B MoE, fp8 | **93.5** (5.2 s) | 136 | **168,000** | **30.2** |
| 30B MoE, bf16 | 59.1 (8.2 s) | 85 | 92,000 | 17.5 |
| 27B dense, NVFP4 (community build) | 42.5 (10.5 s) | 44 | 52,000 | 9.8 |
| 27B dense, fp8 | 33.8 (13.2 s) | 34 | 33,000 | 8.5 |
| 27B dense, bf16 | 19.0 (20.6 s) | 38 | 13,000, saturated | 4.7 |
| 120B MoE (5B active), MXFP4, Marlin kernel (see note) | 64.1 (7.4 s) | 102 | 102,000 | 19.1 |
| 120B Mamba-hybrid MoE (12B active), NVFP4 | 30.5 (14.8 s) | 34 | engines crashed | not reached |
| 8B dense, fp8 (one engine, scaled ×8 from 17.0 req/s at 64 per engine) | ~136 | ~270 (cached, 128 per engine) | 155,000 | ~57 |
| 32B dense, fp8 (one engine, scaled ×8 from 4.3 req/s at 64 per engine) | ~34 | ~86 | ~32,000 (saturated at 4k prompts) | ~10 |
| 80B hybrid MoE (3B active, linear attention on 3 of 4 layers), fp8, 4 × TP=2 on eight H100s | 34.0 at 256 | 36.0 | 76,000 | 22.1 |
| 80B hybrid MoE, fp8, one engine on this GPU (scaled ×8 from 7.5 req/s at 64 per engine) | ~60 | ~89 | ~140,000 | ~34 |
| 26B MoE (3.8B active, hybrid attention), fp8 (one engine, scaled ×8 from 13.85 req/s at 64 per engine) | ~111 (4.4 s) | ~196 | ~211,000 | ~78 |
| 26B MoE (3.8B active), bf16 (one engine, scaled ×8 from 8.32 req/s at 64 per engine) | ~67 (7.5 s) | ~119 | ~116,000 | ~45 |
| 31B dense (hybrid attention), fp8 (one engine, scaled ×8 from 3.78 req/s at 64 per engine) | ~30 (14.7 s) | ~71 | ~31,000 | ~20 |
| 31B dense, bf16, `maxModelLen: 131072` (one engine, scaled ×8 from 1.67 req/s at 64 per engine) | ~13 (36.0 s) | ~34 | ~13,000 | ~7 |
| 12B dense (encoder-free), bf16 (one engine, scaled ×8 from 6.08 req/s at 64 per engine) | ~49 (10.5 s) | ~205 | ~53,000 | ~39 |

Which kernel ran matters as much as which weights. The startup log names it. On this GPU the 120B
MXFP4 experts ran through the Marlin backend, which dequantises to bf16 for the matmul; in vLLM 0.28.0
that is the only MXFP4 path it offers this GPU generation, and on the H100 the same model used the
Triton MXFP4 kernel. The 27B NVFP4 build ran on the native FlashInfer CUTLASS FP4 kernel here. Read the
lines `Using '...' Mxfp4 MoE backend`, `Using ... NvFp4 MoE backend`, `Using ... attention backend` and
`Using ... Fp8 MoE backend` before comparing two GPUs or two formats; the numbers in this document carry
the kernels of 0.28.0 and no other release. The three Gemma 4 rows (26B, 31B, 12B) ran on vLLM's Triton
attention kernel, not FlashInfer or FlashAttention: their global layers use 512-wide heads next to
256-wide local ones, and the log says so (`heterogeneous head dimensions ... forcing TRITON_ATTN backend`).
A later engine release may move those rows; the ones above them will not move with it.

### What generalises

1. **Architecture first.** At the same precision the dense model needed 2.8× the GPUs of the
   mixture-of-experts for the same request rate, 5× in bf16. Before comparing precisions or tuning
   anything, compare activated parameters. The decode ceiling in *Interpreting your own measurements*
   predicts the ratio from the model card alone. A fourth family repeated it: a 26B mixture-of-experts
   with 3.8B active served 5.0× the requests of its dense 31B sibling in bf16 and 3.7× in fp8, and a
   dense 12B fell below the 26B mixture-of-experts at every level (6.1 against 8.3 req/s at 64 per
   engine): activated parameters decide, total parameters only cost VRAM.
2. **Precision second, and it is worth more than any knob.** fp8 over bf16 was +58% to +83% on both
   models, most on long prompts; on the fourth family +55% to +94% for the mixture-of-experts and +80%
   to +230% for the dense 31B, whose bf16 weights left it starved of KV cache as well as of bandwidth. A 4-bit format added +26% on short prompts and +57% on long ones over
   fp8. Weight precision is the only setting in this repository with a 2× effect; every engine knob is
   under 30%. Throughput says nothing about answers, so the precisions were scored on the standard
   benchmarks across four model families (*What quantisation costs in answers*): fp8 weights and an fp8
   cache cost nothing measurable; every 4-bit format costs 0.5 to 2 points and flips one answer in
   twenty. That is a check, and your own prompts are the evaluation you owe your users before switching.
3. **Know which wall you are at before you buy anything** (*Which wall are you at?*). The prefix cache
   was worth 2.7× on long prompts and nothing on the dense model's short prompts. A GPU with more
   bandwidth helps a decode-bound workload and does nothing for a prefill-bound one. The same fleet
   can be both, at different times of day.
4. **Size the fleet on tokens, scale it on requests.** Long prompts moved twice the tokens per second
   in half the requests (168k tok/s at 46 req/s against 88k at 93.5 req/s). Capacity is a token rate;
   the request-count threshold that autoscaling watches has to be re-derived for your prompt length
   (*`scalingRequestsPerTarget`: derive it, do not inherit it*).
5. **Find the knee, and do not run past it.** Throughput rises with concurrency until the engine's
   queue forms, then latency rises alone. The dense model in bf16 with 4,000-token prompts fell from 6.1
   req/s at 256 in flight to 3.6 at 512, p50 52 s, with 63 running and 63 waiting per engine. Past the
   knee, more clients only add queue; the fix is fewer requests per GPU or more GPUs, never a setting.
6. **Measure the way your users call.** Unique prompts, the real output length, the real mix. A
   benchmark with one shape at one concurrency and cache hits overstates a fleet by 2× or more; every
   number in this document says which shape it came from for that reason.
7. **Batch-size knobs do not move the needle.** `maxNumBatchedTokens` at 16,384 against the engine
   default was within 1% on every shape and level, now measured at 1,000 and 4,000-token prompts.
8. **A new architecture in a new format on a new GPU can be kernel-unstable in the current engine
   release. Test for survival before speed.** The Mamba-hybrid 120B in NVFP4 loaded cleanly and served
   1,000-token prompts, then killed three of eight engines under 4,000-token prompts with nothing in
   its own log; the hosts' `dmesg` showed NVIDIA Xid 13/31/43 in the engine process, a GPU kernel
   fault. The model card's recipe sets two environment variables to avoid the FlashInfer FP4 kernels;
   the first removed the long-prompt fault and was faster, but engines still died on cached prompts;
   both together ran clean at a 30 to 40% throughput cost. `extraEnv` exists for this. Judge such a
   model by whether it survives the 4,000-token shape at 64 per engine, not by its first rows.
9. **Speculative decoding is the biggest decode lever, and it is conditional on batch size.** An
   EAGLE-3 draft on the 120B MXFP4 model, mean acceptance 2.5 of 3 drafted tokens: on the reference
   shape +44% at 8 per engine, +47% at 16, +8% at 32 with a p95 of 23 s against 6 s, and **−27% at 64
   per engine**. With every prompt cached, +57% at 64 per engine and no tail; on long answers +66% at
   64. So the loss is not batch size as such but prefill load: verifying drafts needs compute, and a
   GPU busy prefilling unique prompts at 32 or more per engine has none to spare (a dense 12B with a
   four-layer MTP drafter, whose prefill wall is far from 64 unique prompts, gained at every level, +32%
   at 64: *Speculative decoding*). Speculation is worth
   most where decode dominates (long answers, cache hits, moderate concurrency) and a loss on a
   prefill-heavy fleet at its knee. The draft must match the target checkpoint. Measured after a
   four-minute warm-up; the tail is not a warm-up effect.
10. **A faster GPU pays in proportion to how bandwidth-bound the model is.** The same matrix on eight
    H100 80 GB (one `p5.48xlarge`, spot ~$20/h) against the eight g7e.2xlarge: +26% on the fp8
    mixture-of-experts, +82% on it in bf16, +70% on the dense 27B in fp8, +124% on the dense 27B in
    bf16. Precision matters less there: fp8 over bf16 is +9% on the H100 against +58% here. Cost is
    requests per second divided by dollars per hour, and spot prices move: at the prices seen on the
    measurement day ($2.02 per g7e.2xlarge, $19.9 per p5.48xlarge, both spot), the fp8
    mixture-of-experts came out even (5.8 against 5.9 req/s per dollar-hour), bf16 favoured the H100
    by 50%, and the dense model by 40 to 75%. On-demand ($5.85 against $55.04) the same ratios hold.
    Two things the ratio does not show: a g7e fleet grows one GPU at a time and 4-bit checkpoints
    (+26 to 57% on this GPU) do not run on the H100, while p5 spot capacity was unavailable in every
    EU region tried. Buy the GPU for the wall you are at, and recompute with the day's prices.
    Gemma 4 on the same host class ten days later ($22.5 per `p5.48xlarge`, $1.20 to $1.50 per
    g7e.2xlarge, spot) says the kernel decides as much as the bandwidth: the 26B mixture-of-experts came
    out within −13 to +12% per engine of the g7e, because both GPUs ran the Triton attention kernel for
    its 512-wide heads (with a bf16 KV cache and FlashAttention the H100 led by 3 to 20% on unique
    prompts and matched it cached); the dense 31B in fp8 was +20 to +58% (single-stream 62 against 38
    tokens/s). Per dollar-hour the g7e won by about 2× on the mixture-of-experts and 1.3 to 1.6× on the
    dense model at load. Qwen3-30B fp8, which gets FlashAttention on the H100, was +26% on the same day.
11. **On a card the weights nearly fill, `maxModelLen` decides whether the engine starts.** The 30B in
    bf16 (57 GB) on an 80 GB H100 refused to start at the model's 262k default context: one maximum-
    length request needs 24 GiB of KV and 16 GiB were left. `maxModelLen: 32768` started it. The same
    card then hit 100% KV and 150 preemptions under 4,000-token prompts at 64 per engine: bf16 on 80 GB
    is KV-starved for long prompts. On 96 GB nothing of this shows.
12. **Below the model that fits, smaller is cheaper per request but not per prompt token.** An 8B dense
    model in fp8 on one engine held 17.0 req/s at 64 in flight against the 30B mixture-of-experts' 12.8
    (+33%, the cheapest request measured on this GPU), but prefilled 4,000-token prompts *slower*
    (19,400 against 22,400 input tok/s): eight billion parameters of arithmetic per prompt token against
    three billion active. Short prompts favour the small dense model; long prompts favour the mixture
    of experts. Its batch-1 decode ran at 72% of its bandwidth ceiling, dense behaviour, against the
    MoE's 33%. Above the model that fits, a dense 32B in fp8 on the same GPU did a third of the MoE's
    request rate at every load (4.3 against 12.8 req/s at 64) and an eleventh of its prefill (4,000 against
    48,000 input tok/s), at 82% of its own batch-1 ceiling from the first request: every dense model measured
    (8B, 27B, 32B) sits at the bandwidth wall alone, and the mixture of experts is the one that batches.
13. **A model that needs several GPUs pays a third tax, and the topology decides how much.** Beyond
    compute and bandwidth there is the cost of keeping N GPUs in lockstep: an all-reduce per layer,
    expert dispatch, every kernel launched N times. A 235B mixture-of-experts (22B active) on eight
    H100s spent about 5% of each decode step reading weights and the rest in that overhead, so bf16
    and FP8 measured the same at TP=8, and two TP=4 engines beat one TP=8 engine by 40% (*Topology*).
    Per request it cost about six times what a 30B mixture-of-experts (3B active) cost on the same
    dollar basis: partly the seven times more active parameters, partly the lockstep. The rule that
    follows: the smallest tensor-parallel degree that fits, then replicas, and FP8 because it lowers
    that degree, not because the arithmetic is faster.
14. **Prefix-cache economics depend on the architecture.** An 80B mixture-of-experts with 3B active
    parameters and linear (Gated-DeltaNet) attention on three layers in four gained only 2 to 6% from a
    cached 1,000-token prefix, against the attention-based 30B's 46%: linear-attention layers carry a
    recurrent state, not a KV cache, so only the attention layers reuse the prefix. At 4,000 tokens the
    cached gain rose to 58 to 136% as attention's share of prefill grew. The same model decoded a single
    stream at 184 to 220 tokens/s on two H100s, the fastest measured on any model, and delivered 43% of
    the 30B's request rate on the same eight GPUs (half the engines at TP=2, the lockstep tax, and
    younger kernels: 483 s to initialise). The same 80B fits one 96 GB card in fp8 (75.5 GiB of weights,
    11.5 GiB of KV cache at a 32k context) and there it showed the other half of the mixture-of-experts
    rule: with the same 3B active parameters as the 30B it delivered 21 to 41% fewer unique-prompt
    requests and 41 to 55% fewer cached ones, because a loaded step reads the union of the experts the
    batch touches, and 80B of experts is 2.7× the bytes of 30B, while the small KV pool caps the batch.
    It won one shape, 800-token answers, by 30 to 48% at low and mid load: the linear-attention layers
    make each extra output token cheap. **Total parameters decide the bytes per loaded step; active
    parameters decide the arithmetic; the attention type decides how cost grows with length.** A new
    architecture has to be measured on its own shape of traffic before the rules above are applied to it.
15. **Warm up compile-heavy engines before measuring.** The 120B MXFP4 model showed a p95 of 24 to
    37 s in the first shape after every deploy on both GPU families (lazy kernel compilation and
    autotuning), then ran with p95 5 to 9 s. The benchmark script's `--warmup-seconds` covers the
    first requests; for such models run a throwaway minute first.

### Two things about measuring itself

- **A client that abandons a request behind CloudFront leaves the engine working on it.** The engine
  never sees the disconnect and finishes the generation. A load tool that ends a run with requests
  open, or a real client with a short timeout and a retry, stacks zombie work on the engines. An earlier
  version of `scripts/benchmark.py` did exactly that: the next level started behind up to one
  concurrency of abandoned requests, 4,000-token prompts read a fifth of their real rate, and a fully
  cached run came out slower than an uncached one. It now drains in-flight requests before a level
  reports. Set client timeouts above the real p99 and never retry a generation blindly.
- **Rolling out to many instances at once pulls the same weights many times.** Eight instances pulling
  anonymously hit Hugging Face's 429 limit; one engine died mid-download and was replaced. Set
  `hfTokenSecretName` for public models too. Instances with several GPUs download once for all their
  engines through the shared host cache.

## Choosing an operating concurrency

Measured on the recommended topology (two GPUs as 2 × TP=1, FP8 weights, fp8 KV cache, ~1,200-token
prompts) against an 8-second p95 budget and a 31.7 tok/s per-request decode floor:

| Concurrency | Prompts | Prefill tok/s | Decode per request | p95 | Verdict |
|---|---|---|---|---|---|
| 256 | shared prefixes | 46,235 | 51.9 | 4.8 s | pass, wide margins |
| **512** | **shared prefixes** | **74,687** | **42.3** | **5.9 s** | **best measured** |
| 512 | unique | 44,705 | 25.1 | 10.6 s | **fails both** |
| 768 | either | 35,669 ↓ | 10.9 | 19.3 s | **fails badly** |

**Two answers:**

| Your traffic | Operating concurrency | Usable prefill | Instances for 70,000 input tok/s |
|---|---|---|---|
| Requests share a prefix (common system prompt, tool schema, shared document) | **512** | 74,687 | **0.9** |
| Every prompt is unique | **256** | 32,467 | **2.2** |

Quote 74,687 *with its condition attached*.

**Per GPU, the unique-prompt limit is ~128 concurrent requests** (256 across two GPUs). An 8-instance
fleet of single-GPU instances hit its knee at the same 128 per instance (*What a deployed fleet
actually held up to*).

**If you do not know which row you are on, plan for the unique-prompt row.** It is a factor of ~2.4 in
fleet size. Prefix reuse is easy to overestimate: a shared system prompt only helps as identical bytes
from the start of the request.

**Past 512 the cliff is sharp.** At 768, prefill halves (74,687 → 35,669), per-request decode falls to
10.9, and p95 more than triples to 19.3 s.

**Topology at the load that matters, both precisions:** FP8 at concurrency 512 with cache hits, 2 × TP=1
delivered 74,687 against TP=2's 65,152 (**+15%**). bf16 at 256, the highest concurrency either passes
the budget: 46,808 against 42,947 (**+9%**). At 256 in FP8 TP=2 is marginally ahead, so compare at the
load you will run. Comparing bf16 at 512 would reverse this and mean nothing: both bf16 configurations
fail the budget there (decode 27.7 and 29.6 against 31.7 required).

### What a deployed fleet actually held up to

Measured on the shipped default (`g7e.2xlarge` on spot, one engine per GPU): **1,000-token unique
prompts, 190 output tokens**, over `/v1/responses`.

| Instances | Concurrency | Input tok/s | Requests/sec | p50 | p95 | Errors |
|---|---|---|---|---|---|---|
| 1 | 128 | 16,075 | 17.2 | not recorded | 7.42 s | not recorded |
| **6** | 768 | **95,645** | 102.1 | 7.45 s | 8.38 s | 0.01% |
| **8** | 1024 | **132,772** | 141.8 | 6.92 s | 7.60 s | 0.00% |

**Fleet capacity is linear in instance count, and a single-instance benchmark predicts it within ~3%.**
Six instances delivered 99.2% of six times the one-instance figure; eight delivered 103% of eight
times. Each engine has its own GPU, KV cache and process; nothing shared appears as the fleet widens.

#### The whole envelope, on eight instances

Same workload, same fleet, concurrency swept:

| Concurrency | Per instance | Requests/sec | Input tok/s | p50 | p95 | p99 | Within 8 s |
|---|---|---|---|---|---|---|---|
| 256 | 32 | 59.7 | 55,917 | 4.12 s | 4.51 s | 4.67 s | yes |
| 512 | 64 | 96.7 | 90,579 | 5.15 s | 5.74 s | 6.28 s | yes |
| **1024** | **128** | **141.8** | **132,772** | 6.92 s | **7.60 s** | 13.10 s | **yes** |
| 1536 | 192 | 167.6 | 156,998 | 8.49 s | 11.04 s | 12.47 s | **no** |

**The knee is at ~128 concurrent per instance, where a single instance also saturated.** The limit
belongs to one engine on one GPU and does not move behind a load balancer.

**Past the knee you buy throughput with latency.** 1024 to 1536 concurrent (+50% offered load) bought
**+18% throughput and +45% p95**, which left the budget.

**The tail degrades first.** At 1024 concurrent p95 read 7.60 s while **p99 was 13.10 s**. Size on p99.

An earlier datum: the same 6-instance fleet served **~115 requests/second at ~0.1% errors** with
**1-token** outputs, a request-rate figure only; it shows the load balancer, the header-auth listener
rule and 6-way balancing hold up.

These are **unique-prompt** figures. With every prompt cached the same hardware measured +46% on
1,000-token prompts and 2.7× on 4,000-token prompts on the MoE, and nothing on a dense 27B (*Choosing a
model to host*);
see *Choosing an operating concurrency*.

### If you are benchmarking this yourself

- **Do not stop at 32.** Extending one sweep from 32 to 64 nearly doubled measured peak prefill
  (10,943 → 20,293). A number from concurrency 32 roughly **doubles the instance count** you conclude
  you need.
- **Sweep until throughput stops rising**, then step back one point. Here 512 was only identifiable by
  seeing 768 fall; the interesting region was 256–768.
- **Raise `maxNumSeqs` above your highest test point** first, or you are measuring the flag.
- **Report aggregate and per-request numbers together.** Size the fleet on the first, check the latency
  budget on the second. At 512 uncached only the second fails, and a throughput-only sweep would have
  called it the winner.
- **Take both numbers from the same run.** Peak throughput from one concurrency and a passing latency
  check from a lower one overstated per-instance capacity by up to 21% in one harness. **A throughput
  number is only usable if the same run met the latency budget.**
- **Ask for token counts explicitly when streaming** (`"stream_options": {"include_usage": true}` on
  the OpenAI-compatible API). Otherwise `prompt_tokens` reads as zero and aggregate prefill silently
  computes as zero or falls back to counting chunks.
- **Wait for healthy, then warm up, then measure.** Benchmarking from the instant a target reports
  healthy understated throughput by ~21% here: the first requests pay for CUDA graph capture and kernel
  autotuning.
- **If throughput FALLS as you add concurrency, suspect your load generator.** A real server plateaus;
  it does not halve. One Python process driving 768 connections measured **33,688 tok/s against a true
  95,645** (53,492 at 384, then 33,688 at 768); 12 processes gave the correct answer. Even at 384 the
  single-process result was 16% low.
- **Watch p99, not just p95.** At 1024 concurrent, p95 was 7.60 s while p99 was 13.10 s.
- **Use your own prompt shapes and prefix-sharing.** It moves the answer by 2×.
- **Stream, and read three numbers, not one.** End-to-end latency folds queueing, prefill and decode into
  one figure that moves for three unrelated reasons. Time to first token is queue plus prefill; the
  decode rate after it is generation speed; goodput is the requests per second that met the budget.
  `--stream` adds the first, measures the second properly, and `--budget-seconds` adds the third.
- **Measure conversations as conversations.** A shared prefix on every request is the one reuse pattern
  round-robin routing handles. `--turns N` resends a growing conversation on one session, which is what
  multi-turn traffic does to the prefix cache (*Prefix caching is a routing decision*).
- **The p95 of time to first token is noisy.** Two identical runs at 64 in flight on one engine
  differed by 25% on it; treat a change under that as no change.

`scripts/test_endpoint.py` is a smoke test: it shows the endpoint holds up, not where its ceiling is.
`scripts/benchmark.py` does the sweep above: multi-process, warm-up first, aggregate and per-request
numbers from the same run, unique prompts unless you pass `--shared-prefix`, time to first token and
goodput with `--stream` and `--budget-seconds`, conversations with `--turns`.

---

## Sizing a fleet, and when to autoscale

Everything above configures one instance. This is how many to buy.

### Default shape: one task per GPU, on the smallest instance that fits

Express it as *N instances of size X, one task per GPU*:

```yaml
instanceType: g7e.2xlarge     # 1 GPU
instanceCount: 5              # 5 instances -> 5 tasks
tuning:
  tensorParallel: 1
  replicas: 1                 # tasks per instance = GPUs per instance
```

`instanceCount × replicas` is the task count; `replicas × tensorParallel` must not exceed the
instance's GPUs. On a 2-GPU instance set `replicas: 2`.

Four reasons:

1. **~19% cheaper per GPU** than any larger size (*Choosing an instance type*).
2. **Scales in 1-GPU steps.** Most demand figures do not land on a multiple of 8.
3. **Failure isolation.** Losing one instance costs 1/N of capacity.
4. **Spot availability, often the deciding factor.** A 1-GPU `g7e.2xlarge` was obtained on spot in
   **22 seconds**. `g7e.24xlarge` and `g7e.48xlarge` returned `InsufficientInstanceCapacity` on **300+
   consecutive attempts across all four availability zones**, with 768 vCPU of quota free: capacity,
   not quota. **Large GPU instances are hard to obtain; small ones are not.** Recovery from spot
   reclaims follows the same asymmetry.

### How to size N

1. Deploy **one** instance.
2. Find the highest concurrency at which it meets your latency budget (sweep as in *Choosing an
   operating concurrency*, watching p95 and per-request decode).
3. Convert to the unit your demand is in (requests/second, or input tokens/second).
4. Divide demand by it and round **up**.

One `g7e.2xlarge` sustained **16,075 input tok/s** inside an 8-second p95 budget, about **16
requests/second** at this prompt size. Against demand of 70,028 input tok/s:

```
70,028 / 16,075 = 4.36  ->  5 instances minimum, 6 with headroom
```

Round up, then add headroom (next section).

The one-instance figure predicted the 6- and 8-instance fleets within ~3%
(*What a deployed fleet actually held up to*).

### What to scale on

**Autoscaling is ON in the shipped config**: `instanceCount: 16` with `maxInstanceCount: 24`, which
creates target tracking on the load balancer's request count per target. Set the two equal (or leave
`maxInstanceCount` unset) for a fixed-size fleet, and no scaling policy is created.

Signals:

| Signal | Verdict |
|---|---|
| GPU utilisation | **Bad.** Sits near 100% while latency is still fine. It has no relationship to the SLA, so it either scales constantly or never. |
| ALB `TargetResponseTime` p95 | Direct, but **lagging**: it only rises once requests are already slow, so you scale *after* breaching the budget. |
| **ALB `RequestCountPerTarget`** | **Recommended.** Leads both, native to ECS target tracking, no custom metrics to publish. |
| Engine `num_requests_waiting` | Best signal in principle (a queue forming *is* saturation), but it must be published as a custom metric first. |

### `scalingRequestsPerTarget`: derive it, do not inherit it

**The shipped 445 is specific to one workload**; expect to change it.

It is the **requests per minute, per task**, at which capacity is added. Derivation:

```
one g7e.2xlarge sustained 17.2 requests/sec inside an 8 s p95 budget  (1,000-token prompts)
17.2 × 60              = 1,032 requests/minute per task at saturation
1,032 × 0.85           ≈ 870                                          <- for 1,000-token prompts
```

The **shipped default is 445, not 870**: the example workload it is sized for mixes two request shapes with a
request-weighted average input of ~1,780 tokens rather than 1,000. `16,000 / 1,780 = 9.0` req/s per
task, x60 x0.85 = 460; the shipped 445 comes from the 15,500 tok/s measured the day it was set, and
either is fine. Longer prompts mean fewer requests carrying the same tokens, so the threshold comes
down (next section). Split into one fleet per shape and the thresholds become ~870 and ~315; a blended
threshold serves neither shape well.

The 0.85 is the margin, and 85% is *late* given the ~11 minutes scale-out takes; to grow sooner, lower
the multiplier, not the measured rate.

#### Estimate it from traffic you already have, before deploying anything

**If you know your average prompt length**: per-GPU input tokens/sec is roughly constant across prompt
sizes (16,075 against 15,277 for a 4x difference in prompt length, table below). So:

```
requests/sec per task  =  ~16,000  ÷  your average INPUT tokens per request
threshold              =  that  × 60  × 0.85
```

**If you only know aggregate volume**, derive the average request size:

```
average tokens per request  =  (tokens per minute ÷ 60)  ÷  requests per second
average INPUT tokens        =  that  −  your average output length
```

Do this even if you think you know your prompt size.

Treat ~16,000 as an order-of-magnitude figure for this model class on this GPU. It moves with weight
precision (unquantised weights are roughly half) and with the model. Being 20% out costs a little early
scaling; being 4x out means never scaling at all.

#### Then confirm it by measuring

1. Deploy one instance and find the highest concurrency it sustains inside your latency budget (see
   *Choosing an operating concurrency*).
2. Take the requests/second it achieved there and multiply by 60.
3. Multiply by 0.85.

If the measurement disagrees with the estimate by more than about 30%, trust the measurement and check
whether your real prompts are longer than you assumed, the usual cause.

#### Why this number does not transfer between workloads

**Request rate scales inversely with prompt length, while tokens/sec per GPU stays roughly constant.**
Measured on identical hardware:

| Prompt size | Input tok/s per GPU | Requests/sec per GPU | Correct threshold |
|---|---|---|---|
| 1,000 tokens | 16,075 | 17.2 | **~870/min** |
| 4,000 tokens | 15,277 | 4.2 | **~215/min** |

Same GPU work in both rows, packaged into a quarter as many requests, so the request threshold comes
down by the same factor.

**Quadruple your prompt size and keep 870:** the task saturates at 4.2 requests/sec, 252/minute, so
870 is **3.5× higher than the task can ever reach**. The alarm never fires, the fleet never grows, and
there is no error.

Rule of thumb without re-measuring: scale the threshold by `1,000 ÷ your average prompt tokens`.

#### The blind spot

**Request count is indifferent to request cost.** A hundred 200-token requests and a hundred
4,000-token requests look identical to this metric, and they are twenty times apart in GPU work. If
your traffic mixes prompt sizes unpredictably, either split it into separate fleets per shape, or
publish `num_requests_waiting` as a custom metric and scale on that.

#### The whole scaling surface

Four values, all in `config.yaml`:

| Setting | Controls |
|---|---|
| `instanceCount` | the minimum, and the fixed size when autoscaling is off |
| `maxInstanceCount` | the ceiling, **and whether autoscaling exists at all**: equal to `instanceCount` (or 0) creates no scaling policy |
| `scalingRequestsPerTarget` | the threshold above |
| `useSpot` | unrelated to scaling, but it decides how obtainable each added instance is |

The cooldowns (3 minutes out, 15 in) are not exposed. They are not the dominant term in either
direction (see the timings below).

As a safety net against a slow-but-not-yet-queueing regression, add a step-scaling alarm on p95
`TargetResponseTime` above about **75% of your budget** (6 s for an 8 s budget).

### What autoscaling actually does, measured

Timings from a real deployment: 6 -> 8 single-GPU instances, ~29 GiB of weights read from S3. They
scale with weight size and fleet shape.

| Transition | Elapsed | Waiting on |
|---|---|---|
| load starts -> tasks 6 -> 8 | **~6 min** | load balancer metric publication, then a 3-datapoint alarm |
| -> new tasks healthy | **+~5 min** | instance boot, image pull (2 min), weights download, engine start |
| load stops -> tasks 8 -> 7 | **~17 min** | the 15-datapoint low alarm |
| -> instance terminated | **+~15 min** | the capacity provider's own scale-in evaluation |
| full 8 -> 6 convergence | **~45–60 min** | one step per cooldown |

Where the engine's own start goes, measured on one restart of the 30B fp8 model with the weights already
on the host: model loading 81 s, torch compilation 28 s, profiling and graph capture 40 s, about 3 min
from container start to serving. With the engine's cache directory on the host volume (what the
entrypoint does) the second start loaded in 14 s, compiled in 0.1 s and served after 50 s. That removes
two thirds of the engine start from every restart and redeploy on an existing instance. A new instance
still pays its boot, the image pull and the first weights download, so the scale-out figure above moves
by about a minute, not five.

Consequences:

- **Scale-out is ~11 minutes to usable capacity, not 3.** Metric lag, three alarm datapoints, then
  weight loading; the cooldown is almost irrelevant. The minimum has to cover steady state on its own.
- **Target tracking scales in ONE STEP AT A TIME.** 8 -> 6 is two steps of roughly 15 minutes. Expect
  a slow shrink after a spike.
- **The instance outlives the task by about 15 minutes.** The ECS capacity provider runs its own
  scale-in evaluation after the service removes a task: tasks drop to 7 while the Auto Scaling Group
  stays at 8 for another cooldown, and **you keep paying for the GPU instance**.
- **Do not treat autoscaling as a way to save money on a GPU fleet.** The shrink is slow. Size the
  minimum for steady state and treat scale-out as insurance.

**What it is for.** A 15-minute sustained run at 768 concurrent scaled 6 → 8 mid-run and averaged
**111.7 requests/second at p95 7.11 s, with 2 failures in 100,520 requests**. The identical load
against a *fixed* 6 instances sat at p95 **8.38 s**, outside an 8-second budget. Recovering the latency
budget, not cost, is the case for enabling it.

Two things about scale-in that are not obvious from the settings: the Auto Scaling group picks the
instance to terminate by its own policy, not by which one is busiest, so a scale-in after a burst can
drain a loaded engine while an idle one survives, and its replacement cold-starts elsewhere. And on a
fixed fleet a redeploy takes engines down for the reload time, because with fully reserved GPUs no new
task can be placed until an old one stops; with headroom ECS rolls instead. README.md, "Changing the
model, tuning or image later" has the measurements and the choice.

---

## Watching a running fleet

Every deployment gets one CloudWatch dashboard and two alarms (a third, on latency, once you set what
slow means). Nothing to switch on: the URL is a stack output (`DashboardUrl`) and
`python3 scripts/endpoint_info.py` prints it next to the endpoint and the key.

The top rows use metrics the load balancer and the Auto Scaling group already publish. The bottom three
rows come from inside the engines through a sidecar (*Engine metrics* below).

Widgets are titled as questions:

| Widget | Read it for |
|---|---|
| *Is it slow?* (p50/p95/p99, with `latencyAlarmSeconds` drawn on it if set) | The only number with an SLA. The load balancer times a request until the engine starts answering: the whole answer for a non-streamed call, the first token for a streamed one. **A rising p99 against a flat p50 means queueing, not a slow model.** |
| *How hard is each engine working?* (requests/min per task, with `scalingRequestsPerTarget` drawn on it) | The leading indicator, and the metric the scaling policy compares against. |
| *Is the load balancer failing?* (ALB 5XX) | The *load balancer* failing: 503 = no healthy target, 504 = a request outran its 300 s idle timeout. |
| *Is CloudFront timing out?* (CloudFront 5xx rate) | A non-streamed answer that took longer than CloudFront's 120 s read timeout. Invisible to the load balancer, so it has its own widget. |
| *Are the engines erroring?* (target 5XX) | The *engine* answering with an error. A different problem from the row above, so a separate widget. |
| *Did an engine die mid-request?* (target connection errors) | A container that went away with a request in flight. |
| *Are the engines up?* (healthy / unhealthy targets) | Healthy targets falling while instances stay in service means engines are dying, not capacity going away. |
| *Did AWS give us the instances?* (in service vs desired) | Whether the ASG is still trying to grow. Capacity it cannot get looks like a persistent gap here. |
| *How much traffic is arriving?* (requests/min, fleet total) | Offered load, for correlating everything else against. |

Every alarm treats missing data as *not breaching*: an idle load balancer publishes nothing,
and an alarm in `INSUFFICIENT_DATA` every quiet hour is an alarm nobody reads.

| Alarm | Fires on | Why that delay |
|---|---|---|
| `<stack>-too-slow` (only if `latencyAlarmSeconds` is set) | p95 above it for 3 minutes | One minute over budget is what a task starting or draining looks like. A fleet that cannot grow faster than ~11 minutes gains nothing from being told sooner. |
| `<stack>-engines-unhealthy` | any unhealthy target for 15 minutes | A fresh instance is unhealthy for about 7 minutes while it pulls the image and loads weights (14 from launch to healthy). A 5-minute window fired on every first deploy. |
| `<stack>-load-balancer-erroring` | more than 10 ALB 5XX/min for 2 minutes | Not zero: a long prompt hitting the idle timeout produces a 504, and a deploy briefly has no healthy target. Sustained is what matters. |

Each description says what the alarm means and what to check.

Set `alarmTopicArn` to notify an SNS topic. None is created for you: a topic with no subscription
looks like a working notification and is not one.

### Engine metrics: what the load balancer cannot see

The numbers that say *why* the fleet is slow are on vLLM's Prometheus endpoint at `/metrics`. A
sidecar in every task scrapes ten of them over localhost and writes them to CloudWatch under the
namespace `<stack>/Engine`, the bottom three rows of the dashboard. Nothing to enable.

| Metric | Widget | What a bad reading means |
|---|---|---|
| `vllm:num_requests_waiting` | *Is work queueing inside the engines?* | **The one to watch.** Requests admitted but not yet running. A queue forming *is* saturation, and it forms before latency moves - the leading indicator `RequestCountPerTarget` only approximates. |
| `vllm:num_requests_running` | same widget | Sequences in the current batch. Pinned at `maxNumSeqs` means the batch ceiling is the limit; well below it while requests wait means KV cache is. |
| `vllm:kv_cache_usage_perc` | *Is the KV cache filling up?* | 0 to 1. Near 1 is the real ceiling on concurrency, and what `gpuMemoryUtilization: 0.95` buys more of. |
| `vllm:num_preemptions_total` | *Are engines redoing work?* | **Anything above zero is trouble.** The engine evicted a running sequence to free cache and will recompute it from scratch. Rising preemptions are the mechanism behind a collapsing p99. |

| `vllm:time_to_first_token_seconds` | *How long does a request take inside the engine?* | Average time to first token: queue wait plus prefill. Rising while *waiting* is zero means prefill itself is the cost (long prompts). |
| `vllm:e2e_request_latency_seconds` | same widget | Average whole-request time as the engine saw it. Compare with the load balancer's p50: a gap is the network path, not the engine. |
| `vllm:request_prompt_tokens` | *What shape are the requests being served?* | Average input tokens per request. The number `scalingRequestsPerTarget` is derived from; if it drifts, so should the threshold. |
| `vllm:request_generation_tokens` | same widget | Average output tokens per request. Pinned at a round number means callers hit their `max_output_tokens`. |

| `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total` | *Is the prefix cache paying off?* | Hit rate = hits / queries, in prompt tokens, per minute. A shared system prompt shows up here; unique prompts read 0%, which is what the fleet is sized for. |

The four latency and size metrics are histograms in the engine, but the collector passes CloudWatch
their sum and count, not their buckets, so the dashboard shows **averages** over the requests completed
that minute and no percentiles. Latency percentiles come from the load balancer widget. (Tested: the
exported record is `{Sum, Count}` even with the collector's detailed-metrics option.)

#### Request-size bands

*What size are the prompts?* and *How long are the answers?* stack requests per minute into token
bands. The edges come from `promptTokenBands` and `outputTokenBands` in `config.yaml`; shipped: prompts
500, 1,000, 2,000, 5,000, 10,000, 20,000 and answers 100, 200, 500, 1,000, 2,000, so the bands read
"up to 500", "500 to 1,000", ..., "20,000 and more". Each edge must be one of the engine's histogram
edges (1, 2, 5, 10, 20, 50, 100, 200, 500, 1,000, 2,000, 5,000, 10,000, 20,000, 50,000, 100,000,
200,000): the collector only sees those buckets, so an edge like 8,000 cannot be drawn and is rejected
at synth. Fewer edges, fewer lines: `[1000, 10000]` gives three bands. One metric series per edge plus
one, about $0.30 a month each.

How it works: the collector scrapes the engine's histogram buckets a second time, renames them so they
arrive as plain counters with an `le` dimension, and the widget subtracts adjacent buckets. The bands are
coarse by nature, but a shift in traffic shape is visible at a glance, and this is the number to check
`scalingRequestsPerTarget` against.

The metrics carry no per-task dimension; CloudWatch aggregates every engine's samples each minute.
**Maximum** is the busiest engine, **Average** the typical one. A large gap between them is an uneven
load balancer or one sick task, not a capacity problem.

On a real fleet, eight `g7e.2xlarge` engines, FP8, 1,000-token prompts, 190-token answers:

| Offered concurrency | req/s | input tok/s | p95 | running (busiest) | waiting (busiest) | KV cache (fullest) | preemptions |
|---|---|---|---|---|---|---|---|
| 768 | 121 | 113,000 | 6.6 s | 92 | 0 | 9% | 0 |
| 1,920 | 191 | 179,000 | 12.0 s | **244** | 13 | 20% | 0 |

Second row: throughput rose 58% and p95 doubled. *Running* is pressed against the `maxNumSeqs: 256`
batch ceiling, the KV cache is a fifth full, nothing was preempted. The fleet was out of **batch
slots**, not memory, so the lever is `maxNumSeqs`, not more instances or a smaller `maxModelLen`.

The first four or five scrapes in the sidecar's log fail with `Failed to scrape Prometheus endpoint`:
the collector starts in seconds, the engine takes minutes to load weights. It retries every 30 seconds.
Warnings that continue past startup mean the engine is not listening on its port.

How it is built:

* The unmodified upstream OpenTelemetry collector image, contrib build (`METRICS_SIDECAR_IMAGE` in
  `infra/serving_stack.py`), 256 MiB, non-essential so a metrics problem cannot stop inference. Its
  configuration is a ~60-line string in the same file, passed inline through the `OTEL_CONFIG`
  environment variable. Upstream rather than the AWS distribution because the request-size bands need
  the `transform` processor, which the AWS build does not include.
* It scrapes `localhost:8080/metrics` every 30 seconds; `awsvpc` networking puts both containers in
  one network namespace.
* Metrics are written as embedded-metric-format records into the stack's own log group (stream
  `engine-metrics`), so retention and teardown are the stack's.
* To add a metric, append its name to `ENGINE_METRICS` and give it a widget. The endpoint exposes ~86
  families; the shortlist is short because **custom metrics are billed per name** (about $0.30 each
  per month) and the rest are derivable from these, duplicated by the load balancer, or histograms,
  which CloudWatch receives as Min/Max/Sum/Count and cannot turn into a p95. Time-to-first-token
  therefore stays on the load balancer's response time graph.

Not collected: **GPU utilisation** needs a host-level agent and NVIDIA's DCGM exporter, and on a
decode-heavy workload it reads near 100% while the memory bus is the limit. **Per-task engine
metrics** would name the sick engine but cost per task per metric; the task's own log stream already
does that.

For real percentiles of queue time, or to scale on `vllm:num_requests_waiting` directly, the same
sidecar can remote-write to Amazon Managed Service for Prometheus by swapping the `awsemf` exporter for
`prometheusremotewrite`. That adds a workspace and a Grafana.

### What happens when a container is overloaded

**vLLM 0.28.0 does not shed load, and it has no request timeout.** Checked in the engine source: no
queue-depth limit, no queued-token limit, no request priority header. Re-check on an upgrade; a
later engine that rejects with 503 above a queue depth changes this section.

An arriving request is tokenised and put on a **waiting queue**. Each scheduler step admits waiting
requests as three limits allow: `maxNumSeqs`, `maxNumBatchedTokens`, and free KV cache blocks. No
queue-depth limit, no admission control, no deadline: a request waits as long as the client holds the
connection.

When the KV cache fills, the scheduler **preempts** a running sequence and later recomputes it from the
beginning. Under sustained overload that recomputation competes with new work, so throughput *falls*
as load rises: doubling offered concurrency from 768 to 1536 lowered aggregate throughput and blew p99
out.

The engine never returns "busy". A queued request ends only by:

- **CloudFront's 120 s read timeout, then the ALB's 300 s idle timeout.** Either returns a **504 while
  the engine is perfectly healthy**; neither applies to a streamed response, whose bytes reset both
  timers;
- **the client's own timeout**, the only backstop you control directly, so keep it shorter than your
  latency budget;
- **finishing**, eventually.

An overloaded fleet degrades silently until something external gives up. Size the *minimum* for steady
state; autoscaling takes ~11 minutes. Real load shedding has to go in front of the engine: a
concurrency limit at the client, or `maxNumSeqs` plus a short client timeout so over-limit work fails
fast.

---

## Where the numbers come from

Every figure in this document was measured on vLLM 0.28.0 in its shipped container, through the
CloudFront endpoint, with `scripts/benchmark.py` on an in-region EC2 client. Conditions that change
between sections are stated where they matter; the campaigns behind them:

| Measurements | Hardware | Models | Shape of the run |
|---|---|---|---|
| Single- and two-GPU topology, CPU, quantisation, EAGLE-3 on the 30B, autoscaling, fleet linearity | 1 to 8 × `g7e.2xlarge`, one `g7e.12xlarge` | 30B MoE fp8, bf16, AWQ, load-time fp8 | 60 to 120 s per level after warm-up, 64 to 1,920 in flight per fleet |
| Every weight option, model families, EAGLE-3 on the 120B | 8 × `g7e.2xlarge` | 30B MoE, 27B dense (bf16, fp8, NVFP4), 120B MXFP4 (Marlin kernel), 120B NVFP4 hybrid | same matrix, 64 to 512 per fleet |
| H100 comparison, multi-GPU topologies (TP, EP, DP) | one `p5.48xlarge` (8 × H100) | the same, plus a 235B MoE in fp8 and bf16 | same matrix, 64 to 512 per host |
| Routing and the prefix cache, decode ceilings and the memory-controller measurement, KV precision by kernel, CUDA graph mode, warm restart, dynamic speculation, dense contrast, quality | 1 and 8 × `g7e.2xlarge` | 30B MoE fp8, 27B dense, 120B MXFP4 | streamed, 1 to 128 per engine, 60 to 120 s per level |
| Repeatability across regions and days, long prompts to 64k, KV precision by context length, structured output, the 8B and 32B dense points, the 80B hybrid MoE on one GPU and at TP=2 with its MTP head, quality on two tasks for eleven configurations | one `g7e.2xlarge` in three regions, one `p5.48xlarge` | 30B MoE, 8B, 27B, 32B dense, 80B hybrid MoE, 120B MXFP4 | streamed, 90 s levels, 1 to 256 in flight; lm-eval gsm8k 500 and ifeval 541 |
| Output quality of every precision against its own bf16 (*What quantisation costs in answers*) | one `g7e.2xlarge` or `g7e.8xlarge` in four regions | 30B MoE (two releases), 8B, 27B, 32B dense, Mistral Small 3.2 24B; bf16, fp8, NVFP4, GPTQ, AWQ, fp8 KV | lm-eval 0.4.13 standard suite: MMLU 2,850, five log-likelihood tasks at 500, WikiText 60 docs, GSM8K 500, IFEval 541; ~40 min per configuration |
| Agentic quality of every precision (*What quantisation costs an agent*, *Tool calling*) | 4 × `g7e.2xlarge` in two regions, 2 × `g7e.8xlarge` in a third | Qwen3-Coder-30B-A3B, Qwen3-32B, Qwen3-30B-A3B-2507 in bf16, fp8, AWQ, NVFP4; gpt-oss-120b | BFCL v4 single and multi-turn (4,441), τ-bench retail and airline (164 tasks, 2 trials), SWE-bench Verified first 100 with mini-swe-agent, CoNLL-2003 extraction 1,000 sentences in three JSON modes; one bf16 configuration repeated for the noise floor; 1.5 to 4 h per configuration |
| Gemma 4 (*Choosing a model to host*, *Reasoning models*, *Speculative decoding*, *`kvCacheDtype: fp8`*, *Tensor parallelism*): fit, kernels, bf16, fp8, NVFP4, the publisher's int4, thinking, the MTP drafter on 0.29.0, quality; on H100s the KV precision by kernel, TP=1, 2 and 4 over NVLink, and the Qwen reference repeated | one `g7e.2xlarge` in three regions, one `p5.48xlarge` | Gemma 4 26B-A4B MoE, 31B dense, 12B encoder-free; Qwen3-30B-A3B-2507 fp8 | same matrix, 1 to 64 per engine and 64 to 512 per host; lm-eval log-likelihood tasks chat-wrapped, GSM8K 500, IFEval 541 |
| KV cache offload to host memory (*Offloading the cache to host memory moves the capacity wall*) | one `g7e.4xlarge` (128 GiB host RAM) in eu-west-2 | 30B MoE fp8 | six-turn conversations of 8,000 tokens, streamed, 120 to 240 s per level, 32 to 224 conversations in flight; host tier 0 or 32 GiB; one run with the GPU cache pinned to 81,376 tokens |

Run-to-run noise, measured by repeating configurations: an eight-engine H100 host reproduced every row
within ±2% back to back and across two days and two regions, and the same configuration on another
spot instance ten days later within +0 to +3% on every shape and level; a single g7e engine within ±3 to 4%
(cached shapes ±4%), and two regions' single-engine baselines matched within 4%. The p95 of time to
first token moves ±25% between identical runs. A difference inside those bands is not a result.
A later engine release moves the kernel choices named in *The evidence* and *troubleshooting.md*, and
with them every 4-bit figure and the KV precision result.

## What this project does not do, and when to revisit

Considered and left out, each with the condition that would bring it back:

| Not done | Why not here | Revisit when |
|---|---|---|
| A prefix-aware router (a scheduler that knows which engine holds which prefix and each queue depth) | The load balancer plus `stickySessions` recovers most of the multi-turn gain (21% to 75% hit rate) with no new component | conversations span clients that cannot keep a cookie, or one upstream client fans out on behalf of many users |
| Prefill and decode on separate engines (disaggregated serving) | Homogeneous single-GPU engines with a moderate prompt:answer ratio; the KV transfer between engines over PCIe would cost more than it saves | prompts grow past several thousand tokens with strict time-to-first-token targets, or the model needs TP>1 anyway |
| A shared KV store across engines (LMCache, Mooncake, NIXL) | Host-memory offload is measured and is node-local (*Offloading the cache to host memory moves the capacity wall*); a shared store needs the client library in the image and a backend in the VPC | conversations or documents are reused across replicas and the cookie cannot pin them |
| Multi-instance GPU (four 24 GB slices per card) | A 30B fp8 model needs the whole card | a model under 20 GB with strict per-tenant isolation |
| Pipeline parallelism, multi-node engines | Every model measured fits one instance | a model over 640 GB in fp8 |
| Another engine (TensorRT-LLM, SGLang) | One engine, measured deeply, beats two measured shallowly for a sample | a kernel gap on a GPU generation this engine does not serve well |
| Speculative decoding on by default | the draft is specific to the model, so it cannot ship with a `modelId` the user chooses; with a batch-size schedule it measured +17 to +54% below 32 per engine and neutral above, and a shipped MTP head +9 to +18% on one GPU (*Speculative decoding with EAGLE-3*) | a publisher draft or an MTP head exists for your model: add the one line |
| Load shedding in the engine | vLLM 0.28.0 has no queue limit or admission control; it lives at the client | an engine release that rejects above a queue depth |
| Suffix decoding (a better n-gram) | needs a package the engine image does not ship; n-gram itself measured −58% | agentic or code-editing traffic with heavy repetition, and an image rebuild |
| A prefix-aware or agent-aware gateway (per-conversation routing, request queueing, a longer origin timeout for agent steps) | The stack is one load balancer and a CDN; agent steps over 120 s hit the CDN's non-streamed limit (*Tool calling*) and the fix is to stream | your agents cannot stream and their steps run past two minutes |

## Interpreting your own measurements

Two ceilings, from the spec sheet, and one direct measurement that says which wall you are at.

**Ceiling 1, one request in flight:** the fastest a single request can decode.

```
per-request ceiling (tokens/sec) = GPU bandwidth ÷ (activated weight bytes ÷ tensor-parallel degree)
```

`infra/hardware.py` has this as `decode_ceiling_tokens_per_sec`. A 30B mixture-of-experts with 3B active
parameters in fp8 on a 1,597 GB/s GPU: about 530 tokens/s. Measured at one in flight: 174 to 178
tokens/s, a third of it. That is normal, not a fault: at batch 1 the step is 48 layers of small kernels
and the memory controller is busy 39% of the time (measured below). Nothing in the configuration raises
it; a faster GPU raises it only in proportion to its per-kernel speed.

**Ceiling 2, under load:** the most the engine can decode in aggregate. Every decode step reads the
weights once for the whole batch, and for a mixture of experts the batch decides *which* weights: one
token touches 8 of 128 experts, a batch of 64 touches nearly all of them. So under load the bytes per
step approach the whole model, and the ceiling is:

```
aggregate ceiling (tokens/sec) = batch × GPU bandwidth ÷ (total weight bytes ÷ tensor-parallel degree)
```

Measured on the same engine, 100-token prompts, ~130-token answers, streamed:

| In flight | Tokens/s per request | Aggregate tokens/s | Memory controller busy |
|---|---|---|---|
| 1 | 174 | 166 | 39% |
| 4 | 109 | 416 | 55% |
| 16 | 67 | 1,024 | 72% |
| 64 | 47.5 | 2,901 | 77% |
| 128 | 41.2 | 4,971 | 77% |
| 256 | 32.5 | 7,760 | 68% (scheduler preempting) |

At 128 in flight the engine takes 38.8 steps/s. Reading all 29 GiB of weights per step at that rate is
1.2 TB/s, 75% of the GPU's bandwidth, which is what the memory controller reports. **Under load a
mixture of experts decodes like a dense model of its total size, amortised over the batch.** That is why
per-request speed falls from 174 to 41 while the aggregate rises 30×, and why the per-request column of
a loaded benchmark can never be compared with ceiling 1.

A dense model has no such transition. The dense 27B in fp8 on the same GPU and shape: 45 tokens/s per
request at one in flight, 76% of its ceiling 1 (59 tokens/s for 27 GB of weights), with the memory
controller 84% busy across the whole sweep. It is at the bandwidth wall from the first request and its
aggregate tops out at 2,129 tokens/s at 128 in flight against the MoE's 4,971. Its prefill saturated at
9,100 input tokens/s against 48,000: nine times the arithmetic per token, five times slower.

**The direct measurement.** GPU utilisation as normally reported (SM active) reads 98 to 100% for every
shape above and says nothing. The memory controller's busy fraction does:

```bash
# on the instance (aws ssm start-session), while the load runs; mem = memory controller busy %
nvidia-smi dmon -s um -d 2
```

| Memory controller busy | Meaning | What helps |
|---|---|---|
| **75 to 80%**, and it stays there as load grows | decode-bound, at the practical bandwidth limit | a GPU with more bandwidth, fewer bytes (fp8 weights, fp8 cache), more engines |
| **about 50%**, flat as load grows | prefill-bound: the tensor cores and kernels are the limit | a GPU with more compute, fewer prompt tokens (prefix cache), more engines |
| **about 40%** at one request in flight (mixture of experts) | latency-bound: per-kernel and per-layer overhead | nothing in the config; more requests use the idle bandwidth. A dense model shows 75 to 85% here instead |
| falling while SM active also falls | the scheduler is the limit: preemption, a full KV cache, or the client | fewer requests per engine, or the KV cache fixes in *troubleshooting.md* |

The same engine, 4,000-token prompts and 8-token answers: 48,000 input tokens/s from 16 in flight
upward with the memory controller 52% busy. That is prefill-bound, at about 15% of the vendor's fp8
tensor peak: the compute wall in practice is kernel efficiency, well below the roofline number.

Do this before buying hardware. A decode-bound fleet at 77% gains from bandwidth; a prefill-bound fleet
at 52% does not, and the cross-GPU comparison in *Choosing an instance type* could not tell the two
apart on its own.
