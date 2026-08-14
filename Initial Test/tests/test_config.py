from target_system.config import compute_config_hash, list_config_hashes, load_config, save_config
from target_system.factory import baseline_config
from target_system.provenance import OtherGroupFound, ReconstructionProvenance


def test_hash_is_deterministic():
    a = baseline_config(label="run-a")
    b = baseline_config(label="run-a")
    assert compute_config_hash(a) == compute_config_hash(b)


def test_hash_ignores_non_identity_provenance_fields():
    """The parts of provenance that are notes about the pull rather than
    the pull's content stay out of the hash — extraction_date especially,
    since folding wall-clock time in would give every re-pull of identical
    data a fresh identity."""
    a = baseline_config().model_copy(
        update={
            "provenance": ReconstructionProvenance(
                project_id="proj-1", source_agent_name="Some Assistant", trace_count=10, extraction_date="2026-01-01"
            )
        }
    )
    b = a.model_copy(
        update={
            "provenance": a.provenance.model_copy(
                update={
                    "extraction_date": "2026-08-12T14:50:00+00:00",
                    "warnings": ["multiple model names observed"],
                    "other_groups_found": [OtherGroupFound(agent_name="Other", trace_count=3)],
                    "avg_cost_usd_per_trace": 0.0189,
                }
            )
        }
    )
    assert compute_config_hash(a) == compute_config_hash(b)


def test_hash_of_hand_authored_config_is_unaffected_by_provenance_support():
    """Adding a reconstructed config's provenance digest to the hash must
    not move any hand-authored (provenance is None) config's id — old
    cfg_* ids in existing run records have to stay resolvable."""
    assert compute_config_hash(baseline_config()) == "cfg_cbb9a15dafc2"


def test_agent_spec_system_prompt_source_defaults_to_observed():
    config = baseline_config()
    assert all(a.system_prompt_source == "observed" for a in config.agents)


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


def test_save_no_op_still_collapses_configs_differing_only_outside_the_hash(tmp_path):
    """Label isn't part of the experimental condition, so a second save
    under a different label is still the same config and still a no-op —
    the existence check just isn't allowed to assume that any more."""
    h1 = save_config(baseline_config(label="alpha"), configs_dir=tmp_path)
    h2 = save_config(baseline_config(label="beta"), configs_dir=tmp_path)
    assert h1 == h2
    assert load_config(h1, configs_dir=tmp_path).label == "alpha"  # first write wins
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_save_raises_rather_than_skipping_a_write_over_different_content(tmp_path):
    """A genuine id collision must be loud. Silently keeping the existing
    file is the exact failure this module was fixed to stop doing, and
    silently overwriting would discard a config earlier runs were scored
    against."""
    import json

    import pytest

    from target_system.config import ConfigHashCollisionError

    config = baseline_config(defensive_instruction=True)
    other = baseline_config(defensive_instruction=False)
    assert compute_config_hash(config) != compute_config_hash(other)

    # Force the collision: park `other` at the path `config` hashes to.
    config_hash = compute_config_hash(config)
    payload = {"config_hash": config_hash, **other.model_dump(mode="json")}
    (tmp_path / f"{config_hash}.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigHashCollisionError, match="different hashed content"):
        save_config(config, configs_dir=tmp_path)


def test_supervisor_and_members_lookup():
    config = baseline_config()
    assert config.supervisor().role == "supervisor"
    member_roles = {a.role for a in config.members()}
    assert member_roles == {"researcher", "operator"}


def test_list_config_hashes_empty_dir(tmp_path):
    assert list_config_hashes(configs_dir=tmp_path / "nope") == []


def test_list_config_hashes_returns_saved_hashes(tmp_path):
    h1 = save_config(baseline_config(label="one", defensive_instruction=True), configs_dir=tmp_path)
    h2 = save_config(baseline_config(label="two", defensive_instruction=False), configs_dir=tmp_path)
    hashes = list_config_hashes(configs_dir=tmp_path)
    assert set(hashes) == {h1, h2}
    assert all(h.startswith("cfg_") for h in hashes)
