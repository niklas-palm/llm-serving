"""Tests for the logic that runs before any AWS call.

Everything here guards a mistake that would otherwise surface as a broken deployment rather than an
error message - which is the expensive kind, because a GPU deployment takes 20+ minutes to fail.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "infra"))

from hardware import (ConfigError, bytes_per_param_for, weights_are_quantised, DEFAULT_TUNING, INSTANCES, _given, _mapping, _num,
                      decode_ceiling_tokens_per_sec, derive_tensor_parallel, get_instance,
                      memory_pressure_warning, model_bytes, resolve_topology,
                      validate_tuning)


# ---------------------------------------------------------------- instance validation
def test_every_supported_instance_resolves():
    for name in INSTANCES:
        assert get_instance(name).name == name


def test_unsupported_family_is_rejected_with_the_valid_options():
    """The error must list what IS allowed. A bare rejection sends the user to the source."""
    with pytest.raises(ConfigError) as e:
        get_instance("g5.12xlarge")
    msg = str(e.value)
    assert "g5.12xlarge" in msg          # what they asked for
    assert "g7e.2xlarge" in msg          # and what they can have
    assert "g7e.48xlarge" in msg



def test_larger_sizes_never_have_fewer_resources():
    """Ordering matters for the 'choose an instance' guidance to make sense."""
    ordered = [INSTANCES[n] for n in
               ("g7e.2xlarge", "g7e.4xlarge", "g7e.8xlarge",
                "g7e.12xlarge", "g7e.24xlarge", "g7e.48xlarge")]
    for a, b in zip(ordered, ordered[1:]):
        assert b.vcpu > a.vcpu
        assert b.host_mem_gib > a.host_mem_gib
        assert b.gpus >= a.gpus


# ---------------------------------------------------------------- memory derivation
def test_container_memory_follows_host_ram_not_gpu_count():
    """The distinction this whole module exists for.

    g7e.2xlarge and g7e.8xlarge have the SAME single GPU but 64 vs 256 GiB of host RAM. Anything
    deriving memory from GPU count would give them the same limit, which is wrong for both.
    """
    small = INSTANCES["g7e.2xlarge"]
    large = INSTANCES["g7e.8xlarge"]
    assert small.gpus == large.gpus == 1
    assert large.container_memory_mib > small.container_memory_mib * 3


def test_container_memory_leaves_headroom_for_the_os():
    """ECS reserves the full amount, so claiming all of host RAM makes the task unplaceable."""
    for inst in INSTANCES.values():
        assert inst.container_memory_mib < inst.host_mem_gib * 1024


def test_container_memory_is_large_enough_for_a_big_model():
    """A limit in the low tens of GB gets the container OOM-killed loading a 30B model."""
    assert INSTANCES["g7e.2xlarge"].container_memory_mib >= 32_000


# ---------------------------------------------------------------- tensor parallelism
def test_a_model_that_fits_one_gpu_gets_tp_1():
    """Fewest GPUs wins: tensor parallelism costs an all-reduce per layer."""
    inst = INSTANCES["g7e.48xlarge"]                       # 8 GPUs available
    assert derive_tensor_parallel(inst, model_bytes(30)) == 1   # 30B bf16 = ~56 GiB, fits 96 GiB


def test_a_model_too_large_for_one_gpu_scales_up():
    inst = INSTANCES["g7e.48xlarge"]
    assert derive_tensor_parallel(inst, model_bytes(100)) >= 2


def test_quantization_can_avoid_tensor_parallelism():
    """Halving the weights is usually a better trade than adding a GPU."""
    inst = INSTANCES["g7e.12xlarge"]
    # A 70B model: 140 GB in bf16 needs two GPUs, 70 GB in fp8 fits one.
    assert derive_tensor_parallel(inst, model_bytes(70, 2.0)) == 2
    assert derive_tensor_parallel(inst, model_bytes(70, 1.0)) == 1


def test_a_model_too_large_for_the_whole_instance_fails_with_advice():
    with pytest.raises(ConfigError) as e:
        derive_tensor_parallel(INSTANCES["g7e.2xlarge"], model_bytes(200))
    assert "fp8" in str(e.value) and "more GPUs" in str(e.value)


# ---------------------------------------------------------------- tuning validation
def test_utilization_above_the_safe_ceiling_is_rejected():
    """0.98 starts successfully and dies under load - the most expensive failure mode, so it must be
    caught at synth time rather than discovered in production."""
    with pytest.raises(ConfigError) as e:
        validate_tuning(INSTANCES["g7e.2xlarge"], {"gpuMemoryUtilization": 0.98})
    assert "0.97" in str(e.value)



def test_tensor_parallel_beyond_the_instance_gpu_count_is_rejected():
    with pytest.raises(ConfigError) as e:
        validate_tuning(INSTANCES["g7e.2xlarge"], {"tensorParallel": 4})
    assert "1 GPU" in str(e.value) or "has 1" in str(e.value)


def test_non_power_of_two_tensor_parallel_is_rejected():
    with pytest.raises(ConfigError):
        validate_tuning(INSTANCES["g7e.24xlarge"], {"tensorParallel": 3})


def test_replicas_times_tensor_parallel_cannot_exceed_the_gpus():
    """Each replica gets its own GPUs; they are not shared. 4 replicas x TP=2 needs 8 GPUs."""
    inst = INSTANCES["g7e.24xlarge"]                       # 4 GPUs
    validate_tuning(inst, {"tensorParallel": 2, "replicas": 2})     # 4 GPUs exactly - fine
    with pytest.raises(ConfigError) as e:
        validate_tuning(inst, {"tensorParallel": 2, "replicas": 3})  # needs 6
    assert "replicas" in str(e.value)



# ---------------------------------------------------------------- decode ceiling
def test_tensor_parallelism_raises_the_bandwidth_ceiling():
    """Splitting a model halves the bytes each GPU reads per token, so the ceiling doubles. Whether
    real throughput follows is a different question - communication cost is not modelled here."""
    w = model_bytes(30)
    assert decode_ceiling_tokens_per_sec(w, 2) == pytest.approx(
        decode_ceiling_tokens_per_sec(w, 1) * 2, rel=0.01)


def test_sparse_models_have_a_far_higher_ceiling_than_their_size_suggests():
    """A mixture-of-experts model reads only the experts a token routes to, so its decode ceiling is
    set by activated parameters, not total."""
    w = model_bytes(30)
    dense = decode_ceiling_tokens_per_sec(w, 1, active_fraction=1.0)
    sparse = decode_ceiling_tokens_per_sec(w, 1, active_fraction=0.11)
    assert sparse > dense * 8


def test_ceiling_is_zero_rather_than_raising_on_degenerate_input():
    assert decode_ceiling_tokens_per_sec(0, 1) == 0.0


# --------------------------------------------------------------------------- measured defaults

def test_defaults_match_the_shipped_config():
    """config.yaml is the file users read and edit; DEFAULT_TUNING is what applies when they omit a
    key. If the two drift, the documented default and the real one differ silently."""
    import yaml
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    with open(os.path.join(root, "config.yaml")) as fh:
        shipped = yaml.safe_load(fh)["tuning"]

    assert set(shipped) == set(DEFAULT_TUNING), \
        f"config.yaml and DEFAULT_TUNING disagree on which keys exist: " \
        f"{set(shipped) ^ set(DEFAULT_TUNING)}"
    for key, value in shipped.items():
        assert value == DEFAULT_TUNING[key], \
            f"{key}: config.yaml says {value!r}, DEFAULT_TUNING says {DEFAULT_TUNING[key]!r}"


def test_enforce_eager_is_rejected_with_its_cost():
    """The only configuration of 23 tested that failed a latency budget: -85% decode. It is also the
    standard advice for a reluctant engine, so it must be hard to leave on."""
    with pytest.raises(ConfigError) as e:
        validate_tuning(get_instance("g7e.2xlarge"), {"enforceEager": True})
    assert "17.8" in str(e.value), "the error must carry the measured cost, not just say no"


def test_zero_means_let_the_engine_choose():
    """maxModelLen and maxNumBatchedTokens measured no effect across every value worth testing, so
    the default omits the flag rather than guessing."""
    t = validate_tuning(get_instance("g7e.2xlarge"), {})
    assert t["maxModelLen"] == 0
    assert t["maxNumBatchedTokens"] == 0
    # But a positive value must still be accepted, for anyone who wants to cap request length.
    t = validate_tuning(get_instance("g7e.2xlarge"), {"maxModelLen": 4096})
    assert t["maxModelLen"] == 4096
    with pytest.raises(ConfigError):
        validate_tuning(get_instance("g7e.2xlarge"), {"maxModelLen": -1})


def test_max_num_seqs_must_be_positive():
    """Unlike the two above, 0 here is meaningless rather than 'engine default'."""
    with pytest.raises(ConfigError):
        validate_tuning(get_instance("g7e.2xlarge"), {"maxNumSeqs": 0})


def test_kv_cache_dtype_is_checked_against_what_the_engine_accepts():
    assert validate_tuning(get_instance("g7e.2xlarge"), {})["kvCacheDtype"] == "fp8"
    with pytest.raises(ConfigError):
        validate_tuning(get_instance("g7e.2xlarge"), {"kvCacheDtype": "int8"})


@pytest.mark.parametrize("given", ["auto", "", None])
def test_unquantised_kv_cache_is_a_supported_path(given):
    """A reader who cannot accept a lossy store for attention state must have a path that works
    ("auto"), and a blank or omitted value must mean the default rather than an error."""
    t = validate_tuning(get_instance("g7e.2xlarge"), {"kvCacheDtype": given})
    # Blank (absent, `kvCacheDtype:` or `kvCacheDtype: ""`) means the default, like every other key.
    # "" used to resolve to auto while None resolved to fp8.
    assert t["kvCacheDtype"] == ("auto" if given == "auto" else "fp8")


def test_expert_parallelism_ships_off():
    """-13% in FP8, +15% in bf16. No correct default, so it must not be silently on."""
    assert validate_tuning(get_instance("g7e.2xlarge"), {})["enableExpertParallel"] is False


# ------------------------------------------------------- the one combination that fails under load

def _tuning(**over):
    return validate_tuning(get_instance("g7e.2xlarge"), over)


def test_large_bf16_weights_with_fp8_cache_are_flagged():
    """Starts cleanly, then dies with CUDA out-of-memory once concurrency rises, reaching callers as
    502. Counter-intuitive: the smaller cache dtype is what raises memory pressure, because it lets
    the scheduler run bigger batches than the leftover VRAM can support."""
    bf16_30b = model_bytes(30, 2.0)                  # ~56 GiB, most of a 96 GiB card
    warning = memory_pressure_warning(bf16_30b, _tuning(kvCacheDtype="fp8"))
    assert warning is not None
    assert "gpuMemoryUtilization" in warning, "must say which knob provoked it"
    # The recommended fix is to drop the fp8 cache, NOT to lower utilisation. Lowering it works and
    # measured slower than not quantising the cache at all, so a warning that recommended it would
    # send the reader to the worse of the two fixes.
    assert "kvCacheDtype: auto" in warning, "must name the recommended fix"


def test_no_warning_when_there_is_headroom_to_absorb_it():
    fp8_30b = model_bytes(30, 1.0)                   # ~28 GiB, ample headroom
    assert memory_pressure_warning(fp8_30b, _tuning(kvCacheDtype="fp8")) is None
    # An unquantised cache does not enlarge batches, so large weights alone are fine.
    assert memory_pressure_warning(model_bytes(30, 2.0), _tuning(kvCacheDtype="auto")) is None
    # Lowering utilisation is not the recommended fix, but it does remove the pressure the warning is
    # about, so continuing to warn would be noise on a configuration that works.
    assert memory_pressure_warning(
        model_bytes(30, 2.0), _tuning(kvCacheDtype="fp8", gpuMemoryUtilization=0.85)) is None



def test_per_gpu_host_resources_match_the_documented_table():
    """docs/tuning.md and README.md both publish vCPU-per-GPU and RAM-per-GPU, and a reader now picks
    an instance size partly on those numbers. If the catalog and the tables drift, the advice points
    at the wrong size."""
    expected = {                       # (vcpu_per_gpu, host_mem_gib_per_gpu)
        "g7e.2xlarge":  (8, 64),
        "g7e.4xlarge":  (16, 128),
        "g7e.8xlarge":  (32, 256),
        "g7e.12xlarge": (24, 256),
        "g7e.24xlarge": (24, 256),
        "g7e.48xlarge": (24, 256),
    }
    for name, (vcpu, mem) in expected.items():
        inst = INSTANCES[name]
        assert inst.vcpu_per_gpu == vcpu, f"{name}: vCPU/GPU"
        assert inst.host_mem_gib_per_gpu == mem, f"{name}: host RAM/GPU"


def test_vcpu_per_gpu_is_not_monotonic_in_instance_size():
    """The reason that column is worth publishing. A reader assuming 'bigger instance = more host per
    engine' would pick a 48xlarge over an 8xlarge and get less CPU per engine, not more."""
    assert INSTANCES["g7e.8xlarge"].vcpu_per_gpu > INSTANCES["g7e.48xlarge"].vcpu_per_gpu
    # And the cheapest size is the thinnest, which is exactly the untested assumption the docs flag.
    assert INSTANCES["g7e.2xlarge"].vcpu_per_gpu < INSTANCES["g7e.12xlarge"].vcpu_per_gpu


# --------------------------------------------------------------------------------------------
# Fixes for bugs that synthesised cleanly and failed at load time
# --------------------------------------------------------------------------------------------

def test_a_lower_utilization_raises_the_derived_degree():
    """The engine only claims `gpuMemoryUtilization` of each card, so the derivation has to agree.

    Assuming a fixed fraction let the two contradict: a model derived TP=1 against a ~72 GiB budget
    while vLLM would claim only 48 GiB - a guaranteed out-of-memory at load."""
    inst = get_instance("g7e.12xlarge")
    weights = 56 * 1024 ** 3            # a 30B in bf16
    assert derive_tensor_parallel(inst, weights, gpu_memory_utilization=0.95) == 1
    assert derive_tensor_parallel(inst, weights, gpu_memory_utilization=0.50) == 2


def test_a_model_that_fits_one_card_is_not_split_across_two():
    """The headroom must be SUBTRACTED from the claimed budget, not multiplied by it.

    Compounding the two factors put the per-GPU budget at 96 x 0.95 x 0.75 = 68.4 GiB - tighter than
    the flat 72 GiB it replaced - so a 70.8 GiB model was rejected from a 96 GiB card with a message
    quoting the card's full size, and no allowed utilisation could rescue it. On a multi-GPU instance
    the same arithmetic silently chose TP=2 for a model that fits one card, which measured -43% on
    prefill.
    """
    weights = int(70.8 * 1024 ** 3)
    assert derive_tensor_parallel(get_instance("g7e.8xlarge"), weights, 0.95) == 1
    assert derive_tensor_parallel(get_instance("g7e.12xlarge"), weights, 0.95) == 1


def test_the_does_not_fit_message_names_the_setting_that_caused_it():
    """It used to quote raw card totals, which cannot explain a rejection driven by utilisation and
    headroom - and suggested remedies that did not include the actual cause."""
    with pytest.raises(ConfigError) as e:
        derive_tensor_parallel(get_instance("g7e.8xlarge"), 200 * 1024 ** 3, 0.95)
    msg = str(e.value)
    assert "gpuMemoryUtilization" in msg
    assert "estimatedParamsBillions" in msg, "the size is derived from it, so say so"
    assert "75.2 GiB per GPU" in msg, f"state the real budget, not the card total:\n{msg}"


def test_the_memory_pressure_warning_keys_off_footprint_not_precision():
    """The trigger is VRAM left over, which is what the function is given - not parameter count.

    Two versions of this were wrong in opposite directions. First it ignored quantisation entirely, so
    a correctly configured 70B FP8 spread over two GPUs was told to set `kvCacheDtype: auto` and give
    up a measured 12% of decode for nothing. The fix then returned early for ANY quantised model, which
    made the size gate unreachable: at 65.2 GiB per GPU the warning fired for unquantised weights and
    stayed silent for quantised ones - the same footprint, opposite outcomes.
    """
    tuning = {**DEFAULT_TUNING, "kvCacheDtype": "fp8", "gpuMemoryUtilization": 0.95}
    g = 1024 ** 3

    # Identical footprint, identical verdict, whatever the precision.
    assert memory_pressure_warning(int(65.2 * g), tuning, quantised=True) is not None
    assert memory_pressure_warning(int(65.2 * g), tuning, quantised=False) is not None

    # A small footprint is silent either way - a 70B fp8 SPLIT over two GPUs is 32.6 GiB each.
    assert memory_pressure_warning(model_bytes(70, 1.0) // 2, tuning, quantised=True) is None
    assert memory_pressure_warning(model_bytes(30, 1.0), tuning, quantised=True) is None

    # And the escape hatches still work.
    assert memory_pressure_warning(int(65.2 * g), {**tuning, "kvCacheDtype": "auto"},
                                   quantised=False) is None
    assert memory_pressure_warning(int(65.2 * g), {**tuning, "gpuMemoryUtilization": 0.90},
                                   quantised=False) is None


@pytest.mark.parametrize("written, means", [
    ("false", False), ("FALSE", False), ("no", False), ("off", False), ("0", False),
    ("true", True), ("TRUE", True), ("yes", True), ("on", True), ("1", True),
])
def test_a_quoted_boolean_means_what_it_says(written, means):
    """`bool("false")` is True, and quoting a boolean in YAML is an easy mistake.

    Every one of these failed in the direction opposite to intent: `enablePrefixCaching: "false"`
    stayed on, and `enforceEager: "false"` raised the error telling the user to disable the very thing
    they had written as disabled.
    """
    inst = get_instance("g7e.2xlarge")
    assert validate_tuning(inst, {"enablePrefixCaching": written})["enablePrefixCaching"] is means
    # The same string on enforceEager must not trip the enforce-eager guard when it means false.
    if means:
        with pytest.raises(ConfigError):
            validate_tuning(inst, {"enforceEager": written})
    else:
        assert validate_tuning(inst, {"enforceEager": written})["enforceEager"] is False


def test_an_unrecognisable_boolean_is_rejected_rather_than_guessed():
    with pytest.raises(ConfigError) as e:
        validate_tuning(get_instance("g7e.2xlarge"), {"enablePrefixCaching": "maybe"})
    assert "true or false" in str(e.value)


@pytest.mark.parametrize("written", ["2", 2, " 2 "])
def test_a_stringy_tensor_parallel_is_coerced_before_it_is_used_arithmetically(written):
    """The coerced value was computed and discarded, so the raw string reached
    `inst.gpus // tuning["tensorParallel"]` as an uncaught TypeError - and `true` shipped
    TENSOR_PARALLEL=True to the container."""
    inst = get_instance("g7e.12xlarge")
    assert validate_tuning(inst, {"tensorParallel": written})["tensorParallel"] == 2
    resolved = resolve_topology(inst, {"tensorParallel": written}, model_bytes(30, 1.0))
    assert resolved["tensorParallel"] == 2 and resolved["replicas"] == 1


def test_an_infinite_numeric_value_is_a_config_error():
    """YAML parses `.inf` to a float, and int(inf) raises OverflowError - not ValueError - so it
    escaped the coercion guard and printed a traceback."""
    with pytest.raises(ConfigError):
        validate_tuning(get_instance("g7e.2xlarge"), {"maxNumSeqs": float("inf")})


def test_a_blank_numeric_config_value_is_a_config_error_not_a_traceback():
    """`ConfigError` subclasses `ValueError`, so the relationship is the wrong way round for
    `except ConfigError` to catch a coercion failure. A key left blank in YAML printed a raw
    ValueError traceback instead of the one-line message every other config mistake gets."""
    inst = get_instance("g7e.2xlarge")
    for bad in ({"maxNumSeqs": "abc"}, {"gpuMemoryUtilization": "abc"}, {"tensorParallel": "x"}):
        with pytest.raises(ConfigError):
            validate_tuning(inst, bad)


@pytest.mark.parametrize("value", [".inf", "-.inf", ".nan", float("inf"), float("nan")])
def test_a_number_that_is_not_finite_is_rejected(value):
    """YAML parses `.inf` and `.nan` into real floats, so they arrive as numbers and pass `float()`.

    They then poison every derived quantity silently until something far away raises: an infinite
    parameter count made the weight size infinite, and `int(float("inf"))` raises OverflowError from
    inside the tensor-parallel arithmetic - a traceback about integer conversion rather than a message
    about the setting that caused it."""
    with pytest.raises(ConfigError) as e:
        _num(value, "estimatedParamsBillions", float, minimum=0.001)
    assert "estimatedParamsBillions" in str(e.value)


def test_a_number_below_its_floor_is_rejected_by_name():
    """A negative parameter count produced negative weight bytes, which fit on any card, so it derived
    a happy TP=1 and synthesised cleanly."""
    with pytest.raises(ConfigError) as e:
        _num(-30, "estimatedParamsBillions", float, minimum=0.001)
    assert "estimatedParamsBillions" in str(e.value)


@pytest.mark.parametrize("value", [0.5, 8.7, "0.5", -0.5])
def test_a_counting_key_refuses_a_fraction_rather_than_truncating(value):
    """int() truncates, and truncation is silent. `instanceCount: 0.5` became 0 - a legal fleet size
    since parking became supported - so the typo deployed a healthy stack with no instances that
    returned 503 forever. `maxInstanceCount: 8.7` and `scalingRequestsPerTarget: 2.7` lost their
    fractions the same way, just less visibly."""
    with pytest.raises(ConfigError) as e:
        _num(value, "instanceCount", int, minimum=0)
    assert "whole number" in str(e.value)


def test_a_whole_number_written_as_a_float_is_fine():
    """YAML reads `6.0` as a float, and it is unambiguous - reject the fraction, not the notation."""
    assert _num(6.0, "instanceCount", int, minimum=0) == 6
    assert _num("6", "instanceCount", int, minimum=0) == 6


@pytest.mark.parametrize("value", [0, 0.0])
def test_an_explicit_zero_is_not_mistaken_for_an_absent_key(value):
    """`cfg.get(key) or default` treats 0 as absent, so an explicit zero never reached the range check
    that would have accepted or rejected it: `scalingRequestsPerTarget: 0` silently became 925 and
    `maxInstanceCount: 0` silently became instanceCount. The QUOTED forms were rejected all along, so
    the two disagreed about the same value."""
    assert _given(value, 925) == value


@pytest.mark.parametrize("blank", [None, ""])
def test_an_absent_or_blank_key_still_gets_the_default(blank):
    """A key written with nothing after it parses to None, so blank has to keep meaning "default"."""
    assert _given(blank, 925) == 925


@pytest.mark.parametrize("value", ["fast", 3, ["a"]])
def test_a_scalar_where_a_block_of_settings_belongs_says_so(value):
    """`tuning: fast` and `access: public` read as if they should work. They reached .get() on a str and
    raised a bare AttributeError - a traceback naming `str`, which points at the code rather than at
    the line of YAML responsible."""
    with pytest.raises(ConfigError) as e:
        _mapping(value, "tuning")
    assert "tuning" in str(e.value)


def test_an_absent_block_is_not_an_error():
    assert _mapping(None, "tuning") == {}


@pytest.mark.parametrize("model_id, quantization, expected", [
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", "", 2.0),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", "fp8", 1.0),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "", 1.0),
    ("nvidia/Qwen3-30B-A3B-NVFP4", "", 0.5),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", "awq", 0.5),
    ("some-org/model-GPTQ-Int4", "", 0.5),
])
def test_bytes_per_parameter_follows_the_quantisation_width(model_id, quantization, expected):
    """4-bit formats were counted as 1 byte, doubling the estimate and deriving TP=2 for a model that
    fits one GPU."""
    assert bytes_per_param_for(model_id, quantization) == expected


def test_nf4_is_sized_as_four_bit():
    """nf4 was in the quantised markers but not the 4-bit ones, so an NF4 checkpoint was sized at
    1 byte per parameter: the 2x over-estimate that derives TP=2 for a model that fits one GPU."""
    assert bytes_per_param_for("org/Model-NF4", "") == 0.5


def test_the_recommended_nvfp4_checkpoint_counts_as_quantised():
    """The 4-bit markers knew fp4 but the quantised markers did not, so the NVFP4 build the README
    recommends was sized at half a byte per parameter and still reported as unquantised."""
    assert weights_are_quantised("nvidia/Qwen3-30B-A3B-NVFP4", "")
    assert bytes_per_param_for("nvidia/Qwen3-30B-A3B-NVFP4", "") == 0.5


def test_token_band_edges_must_be_engine_bucket_edges_in_order():
    """A band is the difference of two adjacent engine buckets, so an edge the engine has no bucket for
    (8000) would draw an empty band; unsorted or repeated edges would draw negative ones."""
    from hardware import (DEFAULT_PROMPT_TOKEN_BANDS, VLLM_TOKEN_BUCKET_EDGES, validate_token_bands)
    assert validate_token_bands(None, "promptTokenBands", DEFAULT_PROMPT_TOKEN_BANDS) == DEFAULT_PROMPT_TOKEN_BANDS
    assert validate_token_bands("", "promptTokenBands", DEFAULT_PROMPT_TOKEN_BANDS) == DEFAULT_PROMPT_TOKEN_BANDS
    assert validate_token_bands([1000, 10000], "promptTokenBands", DEFAULT_PROMPT_TOKEN_BANDS) == (1000, 10000)
    assert validate_token_bands(list(VLLM_TOKEN_BUCKET_EDGES), "x", ()) == VLLM_TOKEN_BUCKET_EDGES, "every edge at once is allowed"
    for bad, match in [([8000], "not one of the engine's histogram edges"),
                       ([2000, 1000], "ascending"), ([1000, 1000], "no repeats"),
                       ([], "at least one edge"), ("1000", "must be a LIST"), (1000, "must be a LIST"),
                       ([True], "not one of"), (["1000"], "not one of"), ([1000.0], "not one of")]:
        with pytest.raises(ConfigError, match=match):
            validate_token_bands(bad, "promptTokenBands", DEFAULT_PROMPT_TOKEN_BANDS)


def test_the_p5_entries_carry_their_own_gpu_and_size_against_it():
    """VRAM and bandwidth were module constants for the one g7e GPU; an H100 has 80 GiB and 3,350 GB/s,
    so a p5 must derive its topology, memory warning and decode ceiling from its own numbers."""
    p5 = get_instance("p5.4xlarge")
    assert (p5.gpu, p5.gpu_vram_gib, p5.total_vram_gib) == ("H100", 80, 80)
    assert get_instance("p5.48xlarge").total_vram_gib == 640
    assert derive_tensor_parallel(p5, model_bytes(30, 1.0)) == 1, "30B fp8 fits one H100"
    assert derive_tensor_parallel(p5, model_bytes(30, 2.0)) == 1, "30B bf16 (56 GiB) fits inside 80 x 0.95 - 16"
    with pytest.raises(ConfigError, match="does not fit on p5.4xlarge"):
        derive_tensor_parallel(p5, model_bytes(70, 2.0))
    assert derive_tensor_parallel(get_instance("p5.48xlarge"), model_bytes(70, 2.0)) == 4, "130 GiB over 60 GiB per GPU"
    assert memory_pressure_warning(model_bytes(30, 2.0), _tuning(kvCacheDtype="fp8"), gpu_vram_gib=80), \
        "56 GiB is more than half of an 80 GiB card, so the fp8-KV warning must fire there"
    w = model_bytes(30, 1.0)
    assert decode_ceiling_tokens_per_sec(w, 1, bandwidth_gbs=3350) == pytest.approx(
        decode_ceiling_tokens_per_sec(w, 1) * 3350 / 1597, rel=0.01)
    assert get_instance("g7e.2xlarge").gpu_vram_gib == 96, "g7e entries unchanged"
    with pytest.raises(ConfigError, match="p5.48xlarge"):
        get_instance("p4d.24xlarge")


def test_the_b200_entry_sizes_the_largest_checkpoints_against_its_own_vram():
    """A B200 holds 179 GiB, more than twice an H100. Sized against 80 GiB, a 756 GB fp8 model would be refused
    on the one instance in the catalog it fits."""
    b200 = get_instance("p6-b200.48xlarge")
    assert (b200.gpu, b200.gpus, b200.gpu_vram_gib, b200.total_vram_gib) == ("B200", 8, 179, 1432)
    assert derive_tensor_parallel(b200, model_bytes(756, 1.0)) == 8, "704 GiB over 154 GiB per GPU needs 5, so 8"
    assert derive_tensor_parallel(b200, model_bytes(230, 1.0)) == 2, "214 GiB over 154 GiB per GPU"
    with pytest.raises(ConfigError, match="does not fit on p6-b200.48xlarge"):
        derive_tensor_parallel(b200, model_bytes(1600, 1.0))


def test_natively_4bit_families_without_a_marker_in_the_id_are_still_sized_as_4bit():
    """openai/gpt-oss-120b ships only in MXFP4 and its id says nothing about precision; sized at 2 bytes
    it was 240 GB and 'did not fit' a card it fits with room to spare."""
    assert bytes_per_param_for("openai/gpt-oss-120b", "") == 0.5
    assert derive_tensor_parallel(get_instance("g7e.2xlarge"), model_bytes(120, 0.5)) == 1


def test_data_parallel_reserves_tp_times_dp_gpus_and_needs_expert_parallelism():
    """In-engine data parallelism shards the experts across dp attention groups; the container must own
    tensorParallel x dataParallel GPUs, replicas divide what is left, and without expert parallelism the
    setting is refused because it would only be replicas with extra overhead."""
    p5 = get_instance("p5.48xlarge")
    t = resolve_topology(p5, {"tensorParallel": 4, "dataParallel": 2, "enableExpertParallel": True}, model_bytes(235, 1.0))
    assert (t["tensorParallel"], t["dataParallel"], t["replicas"]) == (4, 2, 1)
    t = resolve_topology(p5, {"tensorParallel": 2, "dataParallel": 2, "enableExpertParallel": True}, model_bytes(30, 1.0))
    assert t["replicas"] == 2, "8 GPUs / (2 x 2)"
    with pytest.raises(ConfigError, match="without enableExpertParallel"):
        validate_tuning(p5, {"tensorParallel": 4, "dataParallel": 2})
    with pytest.raises(ConfigError, match="needs 16 GPUs"):
        validate_tuning(p5, {"tensorParallel": 4, "dataParallel": 2, "replicas": 2, "enableExpertParallel": True})
    assert validate_tuning(get_instance("g7e.2xlarge"), {})["dataParallel"] == 1, "off by default"


def test_a_blank_value_means_the_default_for_every_tuning_key():
    """`enablePrefixCaching: ""` used to turn prefix caching OFF (the blank matched the false spellings)
    while `kvCacheDtype: ""` meant the default and `maxNumSeqs: ""` was an error: three rules for one
    way of writing "nothing"."""
    t = validate_tuning(get_instance("g7e.2xlarge"), {"enablePrefixCaching": "", "maxNumSeqs": "", "tensorParallel": ""})
    assert t["enablePrefixCaching"] is True and t["maxNumSeqs"] == 256 and t["tensorParallel"] == 0
