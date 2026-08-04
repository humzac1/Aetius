from target_system.config import compute_config_hash, load_config, save_config
from target_system.factory import baseline_config


def test_hash_is_deterministic():
    a = baseline_config(label="run-a")
    b = baseline_config(label="run-a")
    assert compute_config_hash(a) == compute_config_hash(b)


def test_hash_ignores_label():
    a = baseline_config(label="alpha")
    b = baseline_config(label="beta")
    assert compute_config_hash(a) == compute_config_hash(b)


def test_hash_changes_with_content():
    baseline = baseline_config(defensive_instruction=True)
    regressed = baseline_config(defensive_instruction=False)
    assert compute_config_hash(baseline) != compute_config_hash(regressed)


def test_hash_changes_with_allowlist():
    a = baseline_config(email_allowlist=["a@ourcompany.example"])
    b = baseline_config(email_allowlist=["a@ourcompany.example", "b@external.example"])
    assert compute_config_hash(a) != compute_config_hash(b)


def test_hash_prefix_format():
    config = baseline_config()
    h = compute_config_hash(config)
    assert h.startswith("cfg_")
    assert len(h) == len("cfg_") + 12


def test_save_and_load_roundtrip(tmp_path):
    config = baseline_config(label="roundtrip-test")
    config_hash = save_config(config, configs_dir=tmp_path)
    loaded = load_config(config_hash, configs_dir=tmp_path)
    assert loaded.label == config.label
    assert compute_config_hash(loaded) == config_hash


def test_save_is_idempotent(tmp_path):
    config = baseline_config()
    h1 = save_config(config, configs_dir=tmp_path)
    h2 = save_config(config, configs_dir=tmp_path)
    assert h1 == h2
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_supervisor_and_members_lookup():
    config = baseline_config()
    assert config.supervisor().role == "supervisor"
    member_roles = {a.role for a in config.members()}
    assert member_roles == {"researcher", "operator"}
