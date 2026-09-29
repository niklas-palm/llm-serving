"""app.py's file side effect: the generated API key must land in config.local.yaml exactly once."""
import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "infra"))
_spec = importlib.util.spec_from_file_location("app", os.path.join(HERE, "..", "infra", "app.py"))
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)


def _persist(tmp_path, monkeypatch, existing: str | None) -> tuple[str, str]:
    path = tmp_path / "config.local.yaml"
    if existing is not None:
        path.write_text(existing)
    monkeypatch.setattr(app, "LOCAL_CONFIG_PATH", str(path))
    key = app._persist_generated_api_key()
    return path.read_text(), key


def test_a_missing_file_is_created_with_the_key(tmp_path, monkeypatch):
    body, key = _persist(tmp_path, monkeypatch, None)
    assert body.count("apiKey:") == 1 and key in body


def test_a_file_without_a_trailing_newline_gets_one_before_the_key(tmp_path, monkeypatch):
    body, key = _persist(tmp_path, monkeypatch, "region: us-east-2")
    assert body == f"region: us-east-2\napiKey: {key}\n"


def _load_from(tmp_path, monkeypatch, body: str) -> dict:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(body)
    monkeypatch.setattr(app, "CONFIG_PATH", str(cfg_path))
    monkeypatch.setattr(app, "LOCAL_CONFIG_PATH", str(tmp_path / "config.local.yaml"))
    return app.load_config()


def test_an_existing_key_is_left_alone_by_load_config(tmp_path, monkeypatch):
    """load_config only generates when no key is configured."""
    cfg = _load_from(tmp_path, monkeypatch,
                     "region: us-east-2\ninstanceType: g7e.2xlarge\nmodelId: m\napiKey: keep-this-key-1234\nimage: x/y:z\n")
    assert cfg["apiKey"] == "keep-this-key-1234"


@pytest.mark.parametrize("blank", ["apiKey:", 'apiKey: ""', "apiKey: ''", "apiKey: null", "apiKey: ~",
                                   'apiKey: "   "', "apiKey:   # set on first deploy"])
def test_every_spelling_of_a_blank_key_is_replaced_not_duplicated(tmp_path, monkeypatch, blank):
    """Appending gave two apiKey lines; the file worked only because PyYAML keeps the last one."""
    body, key = _persist(tmp_path, monkeypatch, f"region: us-east-2\n{blank}\ninstanceCount: 2\n")
    assert body == f"region: us-east-2\napiKey: {key}\ninstanceCount: 2\n"


def test_the_local_config_is_not_world_readable(tmp_path, monkeypatch):
    _persist(tmp_path, monkeypatch, None)
    assert oct(os.stat(tmp_path / "config.local.yaml").st_mode & 0o777) == "0o600"


def test_missing_serving_image_is_rejected_at_synth(tmp_path):
    """The upstream engine image would deploy and then serve nothing: this project's Dockerfile
    replaces the entrypoint with `serve`, which is what turns these env vars into
    engine flags. A default would trade a one-second failure for a 20-minute one that looks like a
    broken model.

    Driven through the real entry point as a subprocess, because app.py synthesises on import.
    """
    import subprocess
    import textwrap
    root = os.path.join(HERE, "..")

    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent("""
        region: us-west-2
        instanceType: g7e.2xlarge
        modelId: some-org/some-model
    """))

    env = {**os.environ, "CONFIG": str(cfg), "CDK_DEFAULT_ACCOUNT": "111122223333"}
    env.pop("SERVING_IMAGE", None)
    r = subprocess.run([sys.executable, os.path.join(root, "infra", "app.py")],
                       capture_output=True, text=True, env=env)

    assert r.returncode != 0, "a config with no image must not synthesise"
    out = r.stdout + r.stderr
    assert "image" in out.lower(), f"the error must say what is missing:\n{out}"
    assert "build_image.py" in out, f"and how to produce one:\n{out}"




def test_a_whitespace_only_required_value_is_missing(tmp_path, monkeypatch):
    """`modelId: "  "` passed the required-values check, synthesised MODEL_ID="" and the task died
    after a full deploy with 'MODEL_ID is required'."""
    with pytest.raises(app.ConfigError, match="missing required values: modelId"):
        _load_from(tmp_path, monkeypatch,
                   'region: us-east-2\ninstanceType: g7e.2xlarge\nmodelId: "   "\napiKey: some-stable-key-1234\n')


def test_a_non_string_apikey_is_not_treated_as_blank(tmp_path, monkeypatch):
    """load_config called `apiKey: 0` blank and generated a key, but the line-replacer did not, so the
    file got a second apiKey line. One predicate now: 0 is a value, and synth rejects it."""
    cfg = _load_from(tmp_path, monkeypatch,
                     "region: us-east-2\ninstanceType: g7e.2xlarge\nmodelId: org/m\napiKey: 0\nimage: x\n")
    assert cfg["apiKey"] == 0
    assert not (tmp_path / "config.local.yaml").exists()


@pytest.mark.parametrize("body, match", [
    ("region: us-east-2\ninstanceType: [g7e.2xlarge]\nmodelId: org/m\n", "instanceType"),
    ("apiKey:\tx\n", "not valid YAML"),
    ("- a\n- b\n", "must be a mapping"),
])
def test_a_broken_config_file_is_a_config_error_not_a_traceback(tmp_path, monkeypatch, body, match):
    """A list where a string belongs died as an unhashable type in the instance catalog; a tab after a
    colon was a PyYAML ScannerError traceback; a list document was an AttributeError."""
    with pytest.raises(app.ConfigError, match=match):
        _load_from(tmp_path, monkeypatch, body)


def test_an_indented_apikey_line_is_not_the_top_level_key(tmp_path, monkeypatch):
    """The replacer matched `apiKey:` at any indentation and rewrote it at column 0, breaking the
    block it belonged to."""
    body, key = _persist(tmp_path, monkeypatch, "region: us-east-2\ntuning:\n  apiKey:\n  maxNumSeqs: 128\n")
    assert body == f"region: us-east-2\ntuning:\n  apiKey:\n  maxNumSeqs: 128\napiKey: {key}\n"


class _FakeEc2:
    def __init__(self, reservation):
        self.reservation = reservation

    def describe_capacity_reservations(self, CapacityReservationIds):
        return {"CapacityReservations": [self.reservation]}


def _resolve(monkeypatch, reservation, block_id="cr-0123456789abcdef0"):
    import boto3
    monkeypatch.setattr(boto3, "client", lambda *a, **k: _FakeEc2(reservation))
    cfg = {"region": "us-west-2", "capacityBlockId": block_id}
    app.resolve_capacity_block(cfg)
    return cfg


BLOCK = {"ReservationType": "capacity-block", "State": "scheduled", "AvailabilityZone": "us-west-2c",
         "InstanceType": "p5.48xlarge", "TotalInstanceCount": 1,
         "StartDate": "2026-09-30 11:30:00+00:00", "EndDate": "2026-10-01 11:30:00+00:00"}


def test_a_capacity_block_id_resolves_to_its_zone_type_and_window(monkeypatch):
    cfg = _resolve(monkeypatch, BLOCK)
    assert cfg["capacityBlock"] == {"id": "cr-0123456789abcdef0", "availabilityZone": "us-west-2c",
                                    "instanceType": "p5.48xlarge", "instanceCount": 1, "state": "scheduled",
                                    "start": "2026-09-30 11:30:00+00:00", "end": "2026-10-01 11:30:00+00:00"}


def test_an_on_demand_reservation_is_refused_as_a_capacity_block(monkeypatch):
    """An open On-Demand Capacity Reservation is used automatically; marking it capacity-block would fail the launch."""
    with pytest.raises(app.ConfigError, match="not a Capacity Block"):
        _resolve(monkeypatch, {**BLOCK, "ReservationType": "default"})


def test_an_expired_block_is_refused(monkeypatch):
    with pytest.raises(app.ConfigError, match="expired"):
        _resolve(monkeypatch, {**BLOCK, "State": "expired"})


def test_no_capacity_block_id_makes_no_aws_call(monkeypatch):
    import boto3
    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("no call expected"))
    cfg = {"region": "us-west-2", "capacityBlockId": ""}
    app.resolve_capacity_block(cfg)
    assert "capacityBlock" not in cfg
