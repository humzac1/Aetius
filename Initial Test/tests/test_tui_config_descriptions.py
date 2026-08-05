"""Coverage for the human-readable config/comparison labeling added on top
of tui/data.py's existing diff_configs — describe_config_for_humans,
describe_comparison_for_humans, classify_diff_entry/describe_config_diff,
and the default_baseline_hash (pure) / ensure_baseline_saved (persists)
split."""

from dataclasses import dataclass

from target_system.config import compute_config_hash, list_config_hashes, save_config
from target_system.factory import baseline_config
from tui.data import default_baseline_hash, describe_comparison_for_humans, describe_config_for_humans, ensure_baseline_saved
from tui.formatting import classify_diff_entry, describe_config_diff


@dataclass(frozen=True)
class _Entry:
    path: str
    value_a: object
    value_b: object


# --- classify_diff_entry: one case per category -------------------------------


def test_classify_defensive_instruction_removed():
    cat = classify_diff_entry(_Entry("defensive_instruction", True, False))
    assert cat.category == "defensive"
    assert cat.phrase == "supervisor's defensive instruction removed"
    assert cat.field_label == "Supervisor's defensive instruction"


def test_classify_defensive_instruction_added():
    cat = classify_diff_entry(_Entry("defensive_instruction", False, True))
    assert cat.phrase == "supervisor's defensive instruction added"


def test_classify_model_name():
    cat = classify_diff_entry(_Entry("model.model_name", "claude-sonnet-5", "claude-haiku-4-5-20251001"))
    assert cat.category == "model"
    assert cat.phrase == "using claude-haiku-4-5-20251001 instead of claude-sonnet-5"


def test_classify_model_temperature():
    cat = classify_diff_entry(_Entry("model.temperature", 0.0, 0.7))
    assert cat.field_label == "Model temperature"
    assert "0.0" in cat.phrase and "0.7" in cat.phrase


def test_classify_enforce_allowlist_on_and_off():
    on = classify_diff_entry(_Entry("security.enforce_allowlist", False, True))
    off = classify_diff_entry(_Entry("security.enforce_allowlist", True, False))
    assert "turned on" in on.phrase
    assert "turned off" in off.phrase


def test_classify_email_allowlist_counts_addresses():
    cat = classify_diff_entry(_Entry("security.email_allowlist", ["a", "b", "c"], ["x"]))
    assert cat.phrase == "email allowlist changed (1 address instead of 3)"


def test_classify_email_allowlist_plural():
    cat = classify_diff_entry(_Entry("security.email_allowlist", ["a"], ["x", "y"]))
    assert "2 addresses" in cat.phrase


def test_classify_poisoned_corpus_files_configured():
    cat = classify_diff_entry(_Entry("security.poisoned_corpus_files", [], ["doc1.txt", "doc2.txt"]))
    assert cat.category == "poisoning"
    assert cat.phrase == "corpus poisoning configured for testing (2 files)"


def test_classify_poisoned_corpus_files_removed():
    cat = classify_diff_entry(_Entry("security.poisoned_corpus_files", ["doc1.txt"], []))
    assert cat.phrase == "corpus poisoning removed"


def test_classify_poisoned_tool_results():
    cat = classify_diff_entry(_Entry("security.poisoned_tool_results", [], ["CUST-1"]))
    assert cat.phrase == "tool-result poisoning configured for testing (1 item)"


def test_classify_corpus_dir():
    cat = classify_diff_entry(_Entry("corpus_dir", "a", "b"))
    assert cat.category == "other"
    assert cat.phrase == "corpus directory changed"


def test_classify_agent_added_uses_target_agent_count_for_ordinal():
    entry = _Entry("agents[role=scheduler]", None, {"role": "scheduler", "name": "Scheduler"})
    cat = classify_diff_entry(entry, target_agent_count=4)
    assert cat.phrase == "with a 4th agent (Scheduler) added"
    assert cat.field_label == "Scheduler agent"


def test_classify_agent_added_without_agent_count_says_new():
    entry = _Entry("agents[role=scheduler]", None, {"role": "scheduler", "name": "Scheduler"})
    cat = classify_diff_entry(entry)
    assert cat.phrase == "with a new agent (Scheduler) added"


def test_classify_agent_removed():
    entry = _Entry("agents[role=scheduler]", {"role": "scheduler", "name": "Scheduler"}, None)
    cat = classify_diff_entry(entry)
    assert cat.phrase == "with the scheduler agent removed"


def test_classify_agent_system_prompt():
    cat = classify_diff_entry(_Entry("agents[role=supervisor].system_prompt", "a", "b"))
    assert cat.category == "prompt"
    assert cat.phrase == "supervisor's wording changed"
    assert cat.field_label == "Supervisor's system prompt"


def test_classify_agent_tools():
    cat = classify_diff_entry(_Entry("agents[role=operator].tools", ["send_email"], ["send_email", "lookup_customer"]))
    assert cat.category == "tools"
    assert cat.phrase == "operator's tool access changed"


def test_classify_agent_model_override_added_removed_changed():
    added = classify_diff_entry(_Entry("agents[role=operator].model_override", None, {"model_name": "x"}))
    removed = classify_diff_entry(_Entry("agents[role=operator].model_override", {"model_name": "x"}, None))
    changed = classify_diff_entry(_Entry("agents[role=operator].model_override", {"model_name": "x"}, {"model_name": "y"}))
    assert "different model than the team default" in added.phrase
    assert "removed" in removed.phrase
    assert "changed" in changed.phrase


def test_classify_agent_name():
    cat = classify_diff_entry(_Entry("agents[role=operator].name", "Operator", "Ops"))
    assert cat.phrase == "operator agent renamed"


def test_classify_unknown_path_falls_back_to_raw_path():
    cat = classify_diff_entry(_Entry("some.unmapped.field", 1, 2))
    assert cat.category == "other"
    assert cat.phrase == "`some.unmapped.field` changed"
    assert cat.field_label == "some.unmapped.field"


# --- describe_config_diff: grouping/suppression --------------------------------


def test_describe_config_diff_empty_is_baseline():
    assert describe_config_diff([]) == "baseline (defaults)"


def test_describe_config_diff_single_inline():
    entries = [_Entry("model.model_name", "a", "b")]
    assert describe_config_diff(entries) == "baseline, but using b instead of a"


def test_describe_config_diff_two_inline_joined_with_and():
    entries = [_Entry("model.model_name", "a", "b"), _Entry("corpus_dir", "x", "y")]
    result = describe_config_diff(entries, max_inline_diffs=2)
    assert result == "baseline, but using b instead of a and corpus directory changed"


def test_describe_config_diff_suppresses_supervisor_prompt_when_defensive_instruction_present():
    entries = [
        _Entry("defensive_instruction", True, False),
        _Entry("agents[role=supervisor].system_prompt", "old text", "new text"),
    ]
    result = describe_config_diff(entries)
    assert result == "baseline, but supervisor's defensive instruction removed"


def test_describe_config_diff_does_not_suppress_prompt_without_defensive_instruction_diff():
    entries = [_Entry("agents[role=supervisor].system_prompt", "old", "new")]
    assert describe_config_diff(entries) == "baseline, but supervisor's wording changed"


def test_describe_config_diff_groups_same_category_over_threshold():
    entries = [
        _Entry("agents[role=supervisor].system_prompt", "a", "b"),
        _Entry("agents[role=researcher].system_prompt", "a", "b"),
        _Entry("agents[role=operator].system_prompt", "a", "b"),
    ]
    result = describe_config_diff(entries, max_inline_diffs=2)
    assert result == "baseline, with 3 prompt/wording changes (view diff)"


def test_describe_config_diff_groups_mixed_category_over_threshold():
    entries = [
        _Entry("model.model_name", "a", "b"),
        _Entry("corpus_dir", "x", "y"),
        _Entry("security.enforce_allowlist", False, True),
    ]
    result = describe_config_diff(entries, max_inline_diffs=2)
    assert result == "baseline, with 3 changes (view diff)"


# --- describe_config_for_humans / describe_comparison_for_humans --------------


def test_describe_config_for_humans_baseline_hash_is_baseline_defaults(tmp_path):
    baseline_hash = save_config(baseline_config(), configs_dir=tmp_path)
    assert describe_config_for_humans(baseline_hash, configs_dir=tmp_path) == "baseline (defaults)"


def test_describe_config_for_humans_real_diff(tmp_path):
    save_config(baseline_config(), configs_dir=tmp_path)
    target_hash = save_config(baseline_config(defensive_instruction=False), configs_dir=tmp_path)
    assert describe_config_for_humans(target_hash, configs_dir=tmp_path) == "baseline, but supervisor's defensive instruction removed"


def test_describe_config_for_humans_saves_baseline_if_missing(tmp_path):
    # baseline never explicitly saved — describe_config_for_humans must still work
    target_hash = save_config(baseline_config(defensive_instruction=False), configs_dir=tmp_path)
    assert list_config_hashes(configs_dir=tmp_path) == [target_hash]  # baseline genuinely absent beforehand
    result = describe_config_for_humans(target_hash, configs_dir=tmp_path)
    assert result == "baseline, but supervisor's defensive instruction removed"


def test_describe_config_for_humans_missing_target_config_falls_back(tmp_path):
    assert describe_config_for_humans("cfg_doesnotexist0000", configs_dir=tmp_path) == "(config details unavailable)"


def test_describe_config_for_humans_reconstructed_config_shows_provenance_not_diff(tmp_path):
    from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig
    from target_system.provenance import ReconstructionProvenance

    config = SystemConfig(
        label="recon",
        model=ModelConfig(provider="anthropic", model_name="claude-haiku-4-5-20251001"),
        agents=[AgentSpec(role="supervisor", name="A", system_prompt="[unavailable]", system_prompt_source="unavailable", tools=["send_invoice"])],
        security=SecurityConfig(),
        provenance=ReconstructionProvenance(
            project_id="proj-1", source_agent_name="Invoice Generation Assistant", trace_count=11, extraction_date="2026-01-01T00:00:00+00:00",
        ),
    )
    target_hash = save_config(config, configs_dir=tmp_path)
    result = describe_config_for_humans(target_hash, configs_dir=tmp_path)
    assert result == "reconstructed: Invoice Generation Assistant (11 traces)"
    assert "changes" not in result  # never falls through to the baseline-diff wording


def test_describe_config_for_humans_reconstructed_config_missing_agent_name(tmp_path):
    from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig
    from target_system.provenance import ReconstructionProvenance

    config = SystemConfig(
        label="recon",
        model=ModelConfig(provider="anthropic", model_name="claude-haiku-4-5-20251001"),
        agents=[AgentSpec(role="supervisor", name="A", system_prompt="[unavailable]", system_prompt_source="unavailable", tools=[])],
        security=SecurityConfig(),
        provenance=ReconstructionProvenance(
            project_id="proj-1", source_agent_name=None, trace_count=5, extraction_date="2026-01-01T00:00:00+00:00",
        ),
    )
    target_hash = save_config(config, configs_dir=tmp_path)
    assert describe_config_for_humans(target_hash, configs_dir=tmp_path) == "reconstructed: (no agent_name tag) (5 traces)"


def test_describe_comparison_for_humans_preset_name_passthrough():
    report = {"arm_a_hash": "cfg_a", "arm_b_hash": "cfg_b"}
    assert describe_comparison_for_humans(report, "known_regression") == "known_regression"


def test_describe_comparison_for_humans_preset_display_name_mapping():
    report = {"arm_a_hash": "cfg_a", "arm_b_hash": "cfg_b"}
    assert describe_comparison_for_humans(report, "aa") == "A/A (sanity check)"


def test_describe_comparison_for_humans_adhoc_name_uses_descriptions(tmp_path):
    hash_a = save_config(baseline_config(), configs_dir=tmp_path)
    hash_b = save_config(baseline_config(defensive_instruction=False), configs_dir=tmp_path)
    report = {"arm_a_hash": hash_a, "arm_b_hash": hash_b}
    result = describe_comparison_for_humans(report, f"adhoc_{hash_a}_{hash_b}", configs_dir=tmp_path)
    assert result == "baseline (defaults) vs. baseline, but supervisor's defensive instruction removed"


# --- default_baseline_hash (pure) vs ensure_baseline_saved (persists) --------


def test_default_baseline_hash_does_not_touch_disk(tmp_path):
    empty_dir = tmp_path / "nope"
    baseline_hash = default_baseline_hash(configs_dir=empty_dir)
    assert baseline_hash == compute_config_hash(baseline_config())
    assert not empty_dir.exists()


def test_default_baseline_hash_matches_compute_config_hash(tmp_path):
    assert default_baseline_hash(configs_dir=tmp_path) == compute_config_hash(baseline_config())


def test_ensure_baseline_saved_persists_and_is_idempotent(tmp_path):
    h1 = ensure_baseline_saved(configs_dir=tmp_path)
    assert list_config_hashes(configs_dir=tmp_path) == [h1]
    h2 = ensure_baseline_saved(configs_dir=tmp_path)
    assert h1 == h2
    assert list_config_hashes(configs_dir=tmp_path) == [h1]  # still just one file
