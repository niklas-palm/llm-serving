"""The instance catalog (g7e, and p5 for an H100 comparison), everything derivable from it, and config validation.

This module makes no AWS calls, so it is fast and testable. It exists to stop the user having to get
details right that follow mechanically from their instance choice, and to reject a bad configuration
at synth time rather than after a twenty-minute deployment.

Two facts drive most of what follows:

  * Every g7e size carries the SAME GPU - an NVIDIA RTX PRO 6000 Blackwell with 96 GiB of VRAM.
    Larger sizes add GPUs, vCPU and host RAM. So "which g7e" is a question about how many GPUs and
    how much host RAM you need, never about GPU speed. The p5 entries carry a different GPU (H100,
    80 GiB, 3,350 GB/s), so VRAM and bandwidth live on the instance, not in a constant.

  * VRAM and host RAM scale independently and the gap is wide. A g7e.2xlarge has 96 GiB of VRAM but
    only 64 GiB of host RAM. Any sizing that assumes host RAM tracks GPU memory will be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

# GPU memory bandwidth, GB/s. Decode streams the model's activated weights out of VRAM once per
# generated token, so this sets the ceiling on tokens/sec for a single request.
# 1,597, not 1,792: g7e carries the RTX PRO 6000 Blackwell SERVER Edition, whose GDDR7 runs at 25 Gbps
# (nvidia-smi reports a 12,481 MHz memory clock, two bits per clock per pin) on a 512-bit bus. The 1,792
# figure is the workstation card at 28 Gbps. The difference is 11% on every ceiling derived here.
GPU_MEMORY_BANDWIDTH_GBS = 1597   # g7e; the H100 in p5 is 3,350
GPU_VRAM_GIB = 96                 # g7e; p5 is 80

# Per-GPU VRAM that must stay free for something other than weights, when deciding how many GPUs a
# model needs. Three things share it: peak activations (measured ~5.7 GiB for a 30B on this card),
# CUDA graphs (~0.5 GiB), and enough KV cache to serve a useful number of concurrent requests - the
# largest of the three and the reason this is 16 rather than 8. A model that leaves no cache behind
# technically loads and then cannot batch, which is the whole source of GPU efficiency.
#
# Subtracted from the claimed budget rather than applied as a second fraction; see
# derive_tensor_parallel for why compounding the two was a bug.
WORKING_RESERVE_BYTES = 16 * 1024**3

# Fraction of host RAM to give the container. ECS reserves the full amount whether it is used or
# not, so this also decides how many tasks fit on one instance. vLLM streams weights from disk
# straight to GPU rather than holding the model in host RAM, so a large reservation buys nothing
# and only blocks placement.
CONTAINER_MEMORY_FRACTION = 0.60

# Root volume. It holds the ~9 GB serving image and the containers' writable layers; the weights cache is on
# the instance's local NVMe (serving_stack.py). Kept at the size that held two 30B checkpoints before the
# cache moved, because it is still where the weights land if the NVMe mount fails.
ROOT_VOLUME_GIB = 500
# The gp3 default of 125 MB/s would make reading a 57 GiB model from it take ~8 minutes.
ROOT_VOLUME_THROUGHPUT_MBPS = 500
ROOT_VOLUME_IOPS = 6000


@dataclass(frozen=True)
class Instance:
    name: str
    gpus: int
    vcpu: int
    host_mem_gib: int
    gpu: str = "RTX PRO 6000 Blackwell"
    gpu_vram_gib: int = GPU_VRAM_GIB
    gpu_bandwidth_gbs: int = GPU_MEMORY_BANDWIDTH_GBS

    @property
    def total_vram_gib(self) -> int:
        return self.gpus * self.gpu_vram_gib

    @property
    def vcpu_per_gpu(self) -> float:
        """Host CPU available to each engine, when running one engine per GPU.

        Decision-relevant rather than trivia, and NOT monotonic across the family: the 8xlarge gives
        32 vCPU per GPU while every multi-GPU size gives 24 and the 2xlarge gives 8.

        Measured: about 8 vCPU per engine is enough and more changes nothing - 8 and 48 CPUs given to
        the same engine on the same GPU performed identically. The exception is prefix-cache hits,
        which remove the GPU-heavy prefill work but not the per-request CPU work (tokenisation,
        detokenisation, HTTP, scheduling), so cached serving is the one regime where host CPU binds
        first: 4 CPUs cost 30% of cached throughput at high concurrency while barely touching
        uncached. So this ratio matters for prefix-sharing workloads and is safe to ignore for
        unique-prompt ones. See "How much host CPU an engine needs" in docs/tuning.md.
        """
        return self.vcpu / self.gpus

    @property
    def host_mem_gib_per_gpu(self) -> float:
        return self.host_mem_gib / self.gpus

    @property
    def container_memory_mib(self) -> int:
        """Container memory limit, derived from HOST RAM rather than GPU count.

        Sizing this off GPU count is a trap: it produces a limit far below what loading a large
        model needs, the kernel OOM-kills the container mid-load, and ECS surfaces it as
        `stopCode: TaskFailedToStart` with no exit code and no container reason - a symptom that
        points at scheduling rather than at memory.
        """
        return int(self.host_mem_gib * 1024 * CONTAINER_MEMORY_FRACTION)


# vCPU and host RAM confirmed against ec2:DescribeInstanceTypes.
INSTANCES: dict[str, Instance] = {
    "g7e.2xlarge":  Instance("g7e.2xlarge",  gpus=1, vcpu=8,   host_mem_gib=64),
    "g7e.4xlarge":  Instance("g7e.4xlarge",  gpus=1, vcpu=16,  host_mem_gib=128),
    "g7e.8xlarge":  Instance("g7e.8xlarge",  gpus=1, vcpu=32,  host_mem_gib=256),
    "g7e.12xlarge": Instance("g7e.12xlarge", gpus=2, vcpu=48,  host_mem_gib=512),
    "g7e.24xlarge": Instance("g7e.24xlarge", gpus=4, vcpu=96,  host_mem_gib=1024),
    "g7e.48xlarge": Instance("g7e.48xlarge", gpus=8, vcpu=192, host_mem_gib=2048),
    # H100 SXM (80 GiB HBM3, 3,350 GB/s). For comparison runs; every number in the docs is g7e.
    # NVFP4 checkpoints need Blackwell and do not run here; fp8 and bf16 do.
    "p5.4xlarge":   Instance("p5.4xlarge",   gpus=1, vcpu=16,  host_mem_gib=256,  gpu="H100", gpu_vram_gib=80, gpu_bandwidth_gbs=3350),
    "p5.48xlarge":  Instance("p5.48xlarge",  gpus=8, vcpu=192, host_mem_gib=2048, gpu="H100", gpu_vram_gib=80, gpu_bandwidth_gbs=3350),
    # B200 (179 GiB HBM3e, 8,000 GB/s, Blackwell SM100: NVFP4 runs natively). Usually bought as a Capacity Block.
    "p6-b200.48xlarge": Instance("p6-b200.48xlarge", gpus=8, vcpu=192, host_mem_gib=2048, gpu="B200", gpu_vram_gib=179, gpu_bandwidth_gbs=8000),
}


class ConfigError(ValueError):
    """A configuration problem the user can fix, reported with what to do about it."""


def get_instance(name: str) -> Instance:
    """Look up an instance type, rejecting anything outside the supported family.

    Restricted to the catalog on purpose. Memory sizing and the tuning defaults are calibrated per
    GPU, and silently accepting an unknown type would produce a deployment that looks fine and
    performs nothing like the documentation says.
    """
    if name in INSTANCES:
        return INSTANCES[name]
    raise ConfigError(
        f"Unsupported instanceType {name!r}. This project knows these types:\n"
        + "\n".join(f"    {i.name:<14} {i.gpus} x {i.gpu:<24} {i.total_vram_gib:>4} GiB VRAM  "
                    f"{i.vcpu:>3} vCPU  {i.host_mem_gib:>4} GiB RAM"
                    for i in INSTANCES.values())
        + "\n  See docs/tuning.md for how to choose."
    )


def model_bytes(params_billions: float, bytes_per_param: float = 2.0) -> int:
    """Approximate weight size. bf16/fp16 is 2 bytes per parameter, fp8 is 1, 4-bit is 0.5."""
    return int(params_billions * 1e9 * bytes_per_param)


# Markers that a checkpoint is ALREADY quantised, so `quantization` is correctly left empty.
# Case-insensitive substring match on the model id.
# fp4 covers nvfp4 and mxfp4. gpt-oss is a family whose weights are only published in MXFP4, and the
# repo ids carry no format marker, so the family name is the marker.
FOUR_BIT_MARKERS = ("awq", "gptq", "int4", "w4a16", "fp4", "nf4", "4bit", "4-bit", "gpt-oss")
QUANTISED_MODEL_MARKERS = FOUR_BIT_MARKERS + ("fp8", "int8", "w8a8", "8bit", "bnb")


def weights_are_quantised(model_id: str, quantization: str | None) -> bool:
    """Whether the weights are quantised, by either route: at load time, or already in the checkpoint.

    Two ways to end up with fp8 weights, and only one of them sets `quantization`:

        quantization: "fp8"                       quantised at load time from a bf16 checkpoint
        modelId: .../Qwen3-30B-A3B-Instruct-FP8   already fp8 on disk; quantization stays EMPTY

    Inspecting only `quantization` therefore misreads an official quantised build as full precision.
    That is not a cosmetic error - it doubles the estimated weight size, which can inflate the derived
    tensor-parallel degree, and it makes memory_pressure_warning advise clearing `kvCacheDtype` and
    give up a measured 12% of decode for no reason.

    Heuristic, biased toward assuming quantised: under-warning is better than telling
    someone to undo a correct configuration. A publisher who ships an fp8 build without saying so in
    the repo name is the case this misses, and the cost is a warning that does not appear.
    """
    if str(quantization or "").strip():
        return True
    name = str(model_id or "").lower()
    return any(marker in name for marker in QUANTISED_MODEL_MARKERS)


def bytes_per_param_for(model_id: str, quantization: str | None) -> float:
    """0.5 bytes for 4-bit formats, 1 for 8-bit, 2 for full precision. See weights_are_quantised.

    Treating every quantised format as 1 byte over-estimated 4-bit checkpoints by 2x, enough to derive
    tensorParallel: 2 for a model that fits one GPU.
    """
    name = f"{model_id or ''} {quantization or ''}".lower()
    if any(m in name for m in FOUR_BIT_MARKERS):
        return 0.5
    return 1.0 if weights_are_quantised(model_id, quantization) else 2.0


def derive_tensor_parallel(inst: Instance, weight_bytes: int,
                           gpu_memory_utilization: float = 0.95) -> int:
    """Smallest tensor-parallel degree whose combined VRAM holds the weights plus working space.

    Smallest, not largest, and that is the point. Tensor parallelism is a way to make a model FIT
    across GPUs, not a way to make a fitting model faster. Every layer's partial results must be
    recombined with an all-reduce, so the GPUs advance in lockstep; at saturation that
    synchronisation cancels the parallelism it bought. Measured on two GPUs with unique prompts, the
    second GPU added nothing to prefill under TP=2, while the same GPU running an independent engine
    added 43%.

    So: fewest GPUs per engine, and give any GPUs left over their own engine (see `replicas`).
    Quantizing to fp8 to reach a lower degree is usually a better trade than adding GPUs.

    Tensor parallel degree must also be a power of two and divide the model's attention heads; 1, 2,
    4 and 8 are the only values these instances can offer, and all are valid head counts for
    mainstream architectures.
    """
    # Budget = what the engine claims, MINUS a fixed reserve. Both halves matter:
    #
    # `gpuMemoryUtilization` is the fraction of the card vLLM claims at all, so deriving against the
    # full 96 GiB would contradict the setting - with `gpuMemoryUtilization: 0.50` and a 65 GiB model,
    # a fixed fraction derived TP=1 on the basis of ~72 GiB while vLLM would claim only 48 GiB, a
    # guaranteed out-of-memory at load. Lowering utilisation must therefore RAISE the derived degree.
    #
    # The reserve is SUBTRACTED, not a second multiplier. Multiplying the two compounded them: at the
    # default 0.95 the budget became 96 x 0.95 x 0.75 = 68.4 GiB, TIGHTER than the flat 72 GiB it
    # replaced, so a 70.8 GiB model was rejected from a 96 GiB card with no utilisation able to rescue
    # it - and it could also pick TP=2 for a model that fits one card, which measured -43% on prefill.
    #
    # A fixed reserve is also the physically honest shape: activation peaks and CUDA graphs are roughly
    # constant in absolute terms rather than proportional to the card, so a bigger card should leave
    # proportionally MORE for weights, not the same fraction.
    # Guarded here as well as in validate_tuning, because this is public, takes the utilisation as a
    # defaulted argument, and below about 0.167 the budget goes negative - at which point no degree
    # fits and the error blames the model rather than the setting.
    util = _num(gpu_memory_utilization, "gpuMemoryUtilization", float, minimum=0.20)
    usable_per_gpu = inst.gpu_vram_gib * 1024**3 * util - WORKING_RESERVE_BYTES
    for tp in (1, 2, 4, 8):
        if tp > inst.gpus:
            break
        if weight_bytes <= usable_per_gpu * tp:
            return tp
    budget = max(usable_per_gpu, 0) / 1024**3
    raise ConfigError(
        f"A {weight_bytes / 1024**3:.1f} GiB model does not fit on {inst.name}.\n"
        f"  {inst.gpus} x {inst.gpu_vram_gib} GiB card(s), of which the engine claims "
        f"{util:.0%} (`gpuMemoryUtilization`), less "
        f"{WORKING_RESERVE_BYTES / 1024**3:.0f} GiB per GPU for activations, CUDA graphs and a usable\n"
        f"  KV cache: {budget:.1f} GiB per GPU, {budget * inst.gpus:.1f} GiB across "
        f"{inst.gpus}.\n"
        f"  Options: set `quantization: fp8` to halve the weights, choose an instance with more GPUs,\n"
        f"  or check `estimatedParamsBillions` matches your model - the size above is derived from it."
    )


# Applied for any key the caller omits, so a partial config is valid and these values live in one
# place. docs/tuning.md has the measurement behind each; 0 means the engine decides.
DEFAULT_TUNING = {
    "tensorParallel": 0,            # 0 = derive from the model size
    "maxModelLen": 0,               # 0 = the model's own maximum; measured no effect from lowering it
    "maxNumSeqs": 256,              # 128 -> 256 measured +1.4%; 256 -> 512 is noise
    "maxNumBatchedTokens": 0,       # 0 = engine default; measured no effect anywhere from 8k to 64k
    "gpuMemoryUtilization": 0.95,   # 0.97 is +1.6% at best and crashes dense models under load
    "enablePrefixCaching": True,    # load-bearing at high concurrency, not an optimisation
    "kvCacheDtype": "fp8",          # +12% decode, and the gain grows with concurrency
    "enableExpertParallel": False,  # -13% in FP8; +15% in bf16, so it stays configurable
    "dataParallel": 1,              # in-engine data parallelism; >1 only with expert parallelism on a MoE
    "replicas": 0,                  # 0 = one engine per GPU, which is the measured best topology
}
# `enforceEager` is absent above. It is not a knob to tune, it is a trap: see the
# rejection in validate_tuning for the measured cost of turning it on.

# Values of `kvCacheDtype` the engine accepts. "auto" means the model's own KV precision.
KV_CACHE_DTYPES = ("auto", "fp8", "fp8_e4m3", "fp8_e5m2")


def _given(value: object, default: object) -> object:
    """`default` when a key is absent or written blank, but NEVER when it is written as 0.

    `cfg.get(key) or default` is the obvious form and it is wrong for anything countable, because 0 is
    falsy: an explicit zero was replaced by the default and never reached the range check that would
    have accepted or rejected it. `scalingRequestsPerTarget: 0` silently became 925, and
    `maxInstanceCount: 0` silently became `instanceCount`, which made the "below
    instanceCount" error unreachable for the one value someone parking a fleet would write. The quoted
    forms were rejected all along, so the two disagreed.

    A key written with nothing after it parses to None in YAML, so blank must still mean "default".
    """
    return default if value is None or value == "" else value


def _num(value: object, key: str, cast: type = int,
         minimum: float | None = None, maximum: float | None = None) -> int | float:
    """Coerce a config value, reporting a bad one as a ConfigError rather than a raw ValueError.

    Needed because `ConfigError` subclasses `ValueError`, so the relationship is the wrong way round
    for `except ConfigError` to catch a coercion failure: `int("")` from a config key left blank in
    YAML raised a bare ValueError that escaped the handler and printed a traceback instead of the
    one-line message every other config mistake gets.

    OverflowError is in the tuple because YAML parses `.inf` to a float, and `int(float("inf"))`
    raises OverflowError rather than ValueError - so `maxNumSeqs: .inf` escaped this otherwise.

    The finiteness check is separate from that, and needed because a `float` cast SUCCEEDS on infinity
    and NaN: `estimatedParamsBillions: .inf` coerced cleanly here and then raised OverflowError from
    `model_bytes`, outside both this guard and app.py's handler. `minimum` closes the matching hole for
    values that are finite but nonsensical - a negative parameter count produced negative weight bytes
    and derived a happy TP=1.

    A counting key TRUNCATES rather than rounds, so it must refuse a fraction outright. `int(0.5)` is
    0, and zero is a legal fleet size, so `instanceCount: 0.5` deployed a stack that came up healthy
    with no instances and served 503s. `maxInstanceCount: 8.7` and `scalingRequestsPerTarget: 2.7`
    lost their fractions just as quietly.
    """
    # An int key is parsed as a float FIRST, so the fraction is still visible when it is rejected
    # below. Casting straight to int loses it, and `int("0.5")` raises a ValueError that says nothing
    # about the actual problem.
    try:
        exact = float(value) if cast is int else cast(value)      # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as e:
        raise ConfigError(
            f"{key} must be a number (got {value!r}). Leave it out to use the default."
        ) from e
    if isinstance(exact, float) and not math.isfinite(exact):
        raise ConfigError(
            f"{key} must be a finite number (got {value!r}). YAML reads `.inf` and `.nan` as numbers, "
            "so they pass a numeric check and then fail arithmetic later."
        )
    number = int(exact) if cast is int else exact
    if number != exact:
        raise ConfigError(
            f"{key} counts things, so it must be a whole number (got {value!r}).\n"
            f"  It would be truncated to {number}, not rounded."
        )
    if minimum is not None and number < minimum:
        raise ConfigError(f"{key} must be at least {minimum:g} (got {value!r}).")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{key} must be at most {maximum:g} (got {value!r}).")
    return number


_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0", ""}


# The engine's histogram bucket edges for tokens per request (vLLM, fixed in its code). A size band on the
# dashboard is the difference of two adjacent buckets, so a configured band edge has to be one of these.
VLLM_TOKEN_BUCKET_EDGES = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
                           100000, 200000)
DEFAULT_PROMPT_TOKEN_BANDS = (500, 1000, 2000, 5000, 10000, 20000)
DEFAULT_OUTPUT_TOKEN_BANDS = (100, 200, 500, 1000, 2000)


def validate_token_bands(value: object, key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """Band edges for a request-shape widget: ascending, each one of the engine's bucket edges.

    Absent or blank means the default. The edges make the bands: up to the first, first to second, ...,
    last and more. An edge the engine has no bucket for (8000, say) cannot be drawn, because the
    collector only sees the engine's buckets, so it is rejected here rather than showing an empty band.
    """
    given = _given(value, None)
    if given is None:
        return default
    if not isinstance(given, (list, tuple)):   # not _list: that coerces items to strings
        raise ConfigError(f"{key} must be a LIST of edges, e.g. [1000, 10000] (got {given!r}).")
    edges = list(given)
    allowed = ", ".join(str(e) for e in VLLM_TOKEN_BUCKET_EDGES)
    if not edges:
        raise ConfigError(f"{key} needs at least one edge (or leave it out for the default).")
    clean: list[int] = []
    for e in edges:
        if isinstance(e, bool) or not isinstance(e, int) or e not in VLLM_TOKEN_BUCKET_EDGES:
            raise ConfigError(
                f"{key}: {e!r} is not one of the engine's histogram edges.\n"
                f"  Allowed, ascending: {allowed}"
            )
        clean.append(e)
    if clean != sorted(set(clean)):
        raise ConfigError(f"{key} must be ascending with no repeats (got {clean}).")
    return tuple(clean)


def _list(value: object, key: str) -> list[str]:
    """Coerce a list-shaped config value, rejecting a scalar written where a list belongs.

    A bare string is iterable, so `availabilityZones: us-west-2a` became ten single-character "zones"
    and surfaced as an internal jsii error from deep inside the VPC construct, rather than as the
    one-line config mistake it is.
    """
    if value is None:
        return []
    if isinstance(value, str):
        raise ConfigError(
            f"{key} must be a LIST, not a single value (got {value!r}).\n"
            f"  Write it as:  {key}: [{value!r}]"
        )
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{key} must be a list (got {type(value).__name__}).")
    return [str(v) for v in value]


def _mapping(value: object, key: str) -> dict:
    """The same guard as `_list`, for the keys that take a block of settings rather than a value.

    `tuning: fast` is a natural thing to write and reads as if it should work. It
    reached `.get()` on a string and raised a bare AttributeError - a traceback about `str` having no
    attribute `get`, which points at the code rather than at the line of YAML that caused it.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(
            f"{key} must be a BLOCK of settings, not a single value (got {value!r}).\n"
            f"  Write it as:  {key}:\n"
            f"                  <setting>: <value>"
        )
    return value


def _flag(value: object, key: str) -> bool:
    """Coerce a config boolean, treating a QUOTED one as what it says rather than as truthy.

    `bool("false")` is True, and quoting a boolean in YAML is an easy mistake to make - so every one
    of these failed in the direction opposite to the user's intent: `enablePrefixCaching: "false"`
    stayed on, `useSpot: "false"` turned spot ON, and `enforceEager: "false"` raised the error telling
    the user to disable the very thing they had just written as disabled.

    Anything not recognisable is rejected rather than guessed, because a silently-wrong boolean is
    exactly the failure this exists to stop.
    """
    if value is None or isinstance(value, (int, float)):   # bool is an int
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(
        f"{key} must be true or false (got {value!r}).\n"
        "  Note YAML treats a quoted \"false\" as a string, which is not the same thing."
    )


def validate_tuning(inst: Instance, tuning: dict) -> dict:
    """Check tuning values and fill in anything omitted. Returns a fully resolved copy.

    Fails loudly on values that produce a deployment which starts and then misbehaves, because those
    are far more expensive to diagnose than a synth-time error.
    """
    # None and a blank string both mean "the default": `kvCacheDtype:` with nothing after it and
    # `kvCacheDtype: ""` are the same intent, and "" used to turn a boolean key off and fail a numeric one.
    t = {**DEFAULT_TUNING,
         **{k: v for k, v in _mapping(tuning, "tuning").items()
            if v is not None and not (isinstance(v, str) and not v.strip())}}

    util = _num(t["gpuMemoryUtilization"], "gpuMemoryUtilization", float)
    if not 0.50 <= util <= 0.97:
        raise ConfigError(
            f"gpuMemoryUtilization must be between 0.50 and 0.97 (got {util}).\n"
            "  Above 0.97 the engine passes its startup memory profiling and then fails under "
            "load, when activation peaks exceed the remaining headroom - the worst failure mode, "
            "because it looks like a successful deployment."
        )
    t["gpuMemoryUtilization"] = util

    tp = _num(t["tensorParallel"], "tensorParallel")
    if tp and tp not in (1, 2, 4, 8):
        raise ConfigError(f"tensorParallel must be 1, 2, 4 or 8 (got {tp}).")
    # Written back, like every other coerced value. Left unwritten, `tensorParallel: "2"` from YAML
    # stayed a string and reached resolve_topology's `inst.gpus // t["tensorParallel"]` as an uncaught
    # TypeError, and `tensorParallel: true` shipped TENSOR_PARALLEL=True to the container.
    t["tensorParallel"] = tp
    if tp > inst.gpus:
        raise ConfigError(
            f"tensorParallel={tp} needs {tp} GPUs but {inst.name} has {inst.gpus}."
        )

    # 0 means "derive": one engine per GPU. Resolved by resolve_topology once the tensor-parallel
    # degree is known, since the two are linked - replicas x tensorParallel must equal the GPU count.
    # In-engine data parallelism: one engine owns tp x dp GPUs and runs dp attention groups over its own
    # request batches with the experts sharded across all of them (the MoE-serving topology). Without
    # expert parallelism it is only a more expensive way to run replicas, so it is refused.
    dp = _num(t["dataParallel"], "dataParallel", minimum=1)
    if dp > 1 and not _flag(t.get("enableExpertParallel"), "enableExpertParallel"):
        raise ConfigError(
            f"dataParallel={dp} without enableExpertParallel: true is just replicas with more overhead.\n"
            "  For independent engines use `replicas`; data parallelism inside one engine pays only when\n"
            "  the experts are sharded across its GPUs."
        )
    if dp > inst.gpus:
        raise ConfigError(f"dataParallel={dp} needs {dp} GPUs but {inst.name} has {inst.gpus}.")
    t["dataParallel"] = dp

    replicas = _num(t["replicas"], "replicas")
    if replicas < 0:
        raise ConfigError("replicas must be 0 (one engine per GPU) or a positive integer.")
    if tp and replicas and replicas * tp * dp > inst.gpus:
        raise ConfigError(
            f"replicas={replicas} x tensorParallel={tp} x dataParallel={dp} needs {replicas * tp * dp} GPUs, "
            f"but {inst.name} has {inst.gpus}.\n"
            f"  Each replica gets its own GPUs; they are not shared."
        )
    t["replicas"] = replicas

    # maxModelLen and maxNumBatchedTokens accept 0, meaning "omit the flag and let the engine
    # choose". Both measured no effect across the range worth testing, so a wrong guess is more
    # likely to hurt than the default is.
    for key in ("maxModelLen", "maxNumBatchedTokens"):
        t[key] = _num(t[key], key, minimum=0)
    t["maxNumSeqs"] = _num(t["maxNumSeqs"], "maxNumSeqs", minimum=1)

    kv = str(t["kvCacheDtype"])
    if kv not in KV_CACHE_DTYPES:
        raise ConfigError(
            f"kvCacheDtype must be one of {', '.join(KV_CACHE_DTYPES)} (got {kv})."
        )
    t["kvCacheDtype"] = kv

    for flag in ("enablePrefixCaching", "enableExpertParallel", "enforceEager"):
        t[flag] = _flag(t.get(flag), flag)   # enforceEager has no default entry

    if t["enforceEager"]:
        raise ConfigError(
            "enforceEager: true disables CUDA graphs, which are load-bearing for decode.\n"
            "  Measured effect: decode fell from 117.5 to 17.8 tokens/sec per request (-85%). It was\n"
            "  the only configuration of 23 tested that failed a latency budget.\n"
            "  It is common advice for getting a reluctant engine to start. If you need it to\n"
            "  diagnose a startup problem, set it through EXTRA_ARGS so it cannot be left on by\n"
            "  accident, and remove it before measuring anything."
        )

    return t


def resolve_topology(inst: Instance, tuning: dict, weight_bytes: int) -> dict:
    """Fill in tensorParallel and replicas together, because neither is meaningful alone.

    They are one decision: how the instance's GPUs are divided into engines. `tensorParallel` is GPUs
    per engine, `replicas` is how many engines, and their product cannot exceed the GPU count. Asking
    the user to compute the second from the first invites the exact mistake that makes a multi-GPU
    instance behave like a single-GPU one - eight GPUs running one engine, with seven idle.

    Defaults, both measured:
      tensorParallel = the smallest degree whose combined VRAM holds the weights (usually 1)
      replicas       = GPUs / tensorParallel, i.e. one engine per GPU, using the whole instance

    One engine per GPU is the recommended shape. At saturation with unique prompts, two independent
    engines beat one engine spread over the same two GPUs by 50% on prefill, and were the only
    topology of the four tested to meet the latency budget. Splitting a model that already fits buys
    nothing: the halves advance in lockstep with an all-reduce per layer.
    """
    # Validate FIRST, so the values are coerced before any arithmetic touches them. Reading
    # `tuning` raw meant `tensorParallel: "2"` from YAML reached the `//` below as a string and raised
    # an uncaught TypeError. Idempotent, and the call at the end still re-checks the derived pair.
    t = validate_tuning(inst, tuning)
    if not t.get("tensorParallel"):
        t["tensorParallel"] = derive_tensor_parallel(
            inst, weight_bytes,
            # The engine will only claim this much of each card, so the derivation has to agree with
            # it - otherwise lowering the utilisation leaves a model "fitting" a budget vLLM never
            # asks for.
            gpu_memory_utilization=t["gpuMemoryUtilization"])
    if not t.get("replicas"):
        t["replicas"] = max(1, inst.gpus // (t["tensorParallel"] * t["dataParallel"]))
    # Re-validate: a derived pair still has to satisfy replicas x tp <= gpus, and an explicit
    # tensorParallel combined with a derived replica count is the case most likely to overshoot.
    return validate_tuning(inst, t)


def memory_pressure_warning(weight_bytes_per_gpu: int, tuning: dict,
                            quantised: bool = False, gpu_vram_gib: int = GPU_VRAM_GIB) -> str | None:
    """Flag the one configuration combination known to start cleanly and then fail under load.

    Unquantised weights plus an fp8 KV cache at high utilisation runs out of VRAM once traffic
    arrives, and the mechanism is the opposite of the intuition - see the entry in
    docs/troubleshooting.md.

    The trigger is VRAM FOOTPRINT, not parameter count. `quantised` only words the message; it does
    not change whether the check fires. An earlier version returned early for any quantised model, which
    made the size gate below unreachable and suppressed a true positive: at 65.2 GiB per GPU with an
    fp8 cache at 0.95 the warning fired for unquantised weights and stayed silent for quantised ones -
    the same VRAM footprint, opposite outcomes. The mechanism this warns about depends on what is left
    over after the weights, which is exactly the number the function is given.

    A warning rather than an error because only the operator knows their model's real size, and the
    estimate here is derived from parameter count. The recommended fix is to drop the fp8 cache rather
    than to lower utilisation: lowering it does stop the crash, and measured 37,877 tok/s against
    46,808 for an unquantised cache at 0.95 - so rescuing the combination costs more than the
    combination returns.
    """
    if not str(tuning.get("kvCacheDtype") or "").startswith("fp8"):
        return None
    util = float(tuning.get("gpuMemoryUtilization", 0.95))
    if util <= 0.90:
        return None
    # ONE threshold, applied whatever the weights' precision. Below about half the card there is
    # enough headroom to absorb the larger batches an fp8 cache allows; above it there is not, and the
    # mechanism does not care WHY the VRAM is full. A 30B in fp8 sits at 29% and never warns; a 70B in
    # fp8 sits at 68% and does, correctly - it has the same footprint as an unquantised 35B, and
    # exempting it was how the earlier version turned this gate into dead code.
    if weight_bytes_per_gpu < 0.50 * gpu_vram_gib * 1024**3:
        return None
    weights = "quantised" if quantised else "unquantised"
    return (
        f"{weights} weights occupying {weight_bytes_per_gpu / 1024**3:.1f} GiB per GPU, with "
        f"kvCacheDtype: fp8 at "
        f"gpuMemoryUtilization: {util}.\n"
        "  An fp8 cache holds about twice as many tokens in the same budget, which raises the batch\n"
        "  size the scheduler runs, which raises the per-step workspace the model allocates. Large\n"
        "  weights leave nothing to absorb that, so the engine starts, passes its health check, and\n"
        "  then dies with CUDA out-of-memory once real traffic arrives - reaching callers as 502.\n"
        "  Fix: set kvCacheDtype: auto, or use an instance with more GPUs so the weights occupy less\n"
        "  of each one. Lowering gpuMemoryUtilization to 0.90 also stops the crash but costs more\n"
        "  than it saves - measured 37,877 tok/s against 46,808. See docs/troubleshooting.md."
    )


def decode_ceiling_tokens_per_sec(weight_bytes: int, tp: int, active_fraction: float = 1.0,
                                  bandwidth_gbs: int = GPU_MEMORY_BANDWIDTH_GBS) -> float:
    """Upper bound on output tokens/sec for ONE request with nothing else in flight.

    Generating a token requires reading the activated weights out of VRAM, so this is a hard
    physical limit: bandwidth divided by bytes-read-per-token. Measured engines reach about a third
    of it at one request in flight (per-kernel overhead), which is normal. It is NOT the ceiling
    under load: a batch shares one weight read, and for a mixture of experts a large batch touches
    nearly every expert, so the aggregate ceiling is batch * bandwidth / total weight bytes. See
    docs/tuning.md, "Interpreting your own measurements".

    `active_fraction` is activated parameters over total. Dense models read everything, so 1.0. A
    mixture-of-experts model reads only the experts a token routes to, plus the always-active
    attention and embedding layers - typically 10-20% of total, so such models decode far
    faster than their parameter count suggests.
    """
    bytes_per_token_per_gpu = weight_bytes * active_fraction / tp
    if bytes_per_token_per_gpu <= 0:
        return 0.0
    return bandwidth_gbs * 1e9 / bytes_per_token_per_gpu
