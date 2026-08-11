from tui.formatting import (
    SYSTEM_PROMPT_UNAVAILABLE_DISCLOSURE,
    build_drill_down_rows,
    family_display_name,
    format_attempted_breakdown,
    format_clear_summary,
    format_flagged_ci,
    format_flagged_headline,
    format_flagged_synthetic_evidence_note,
    format_inconclusive_summary,
    format_other_flagged_note,
    format_other_groups_found_note,
    format_provenance_disclosure,
    format_response_source_cell,
    format_response_source_summary,
    format_single_config_headline,
)
from tui.verdict_logic import (
    AttemptedExecutedCounts,
    ComparisonVerdict,
    FamilyPower,
    FamilySingleSummary,
    ResponseSourceBreakdown,
    SingleConfigSummary,
)


def _flagged_verdict(diff=0.6, other_flagged_count=0):
    effect = {
        "method": "cluster_bootstrap", "rate_a": 0.133, "rate_b": 0.133 + diff, "diff": diff,
        "ci_low": diff - 0.1, "ci_high": diff + 0.1, "alpha": 0.05, "p_value": 0.001,
        "n_cases": 15, "n_runs_a": 75, "n_runs_b": 75, "used_fallback": False, "fallback_reason": None, "extra": {},
    }
    return ComparisonVerdict(
        tier="FLAGGED", flagged_outcome_key="exfiltration", flagged_family="direct_instruction_injection",
        flagged_effect=effect, flagged_q_value=0.004, flagged_arm_label="defensive_prompt_off",
        other_flagged_count=other_flagged_count,
    )


# --- FLAGGED phrasing --------------------------------------------------------


def test_flagged_headline_says_rose_for_positive_diff():
    headline = format_flagged_headline(_flagged_verdict(diff=0.6))
    assert "rose from 13.3% to 73.3%" in headline
    assert "fell" not in headline


def test_flagged_headline_says_fell_for_negative_diff():
    headline = format_flagged_headline(_flagged_verdict(diff=-0.05))
    assert "fell" in headline
    assert "rose" not in headline


def test_flagged_never_says_significant_or_raw_pvalue():
    verdict = _flagged_verdict()
    headline = format_flagged_headline(verdict)
    ci = format_flagged_ci(verdict)
    combined = headline + " " + ci
    assert "significant" not in combined.lower()
    assert "p=" not in combined.lower()
    assert "p-value" not in combined.lower()


def test_flagged_ci_shows_bh_adjustment_language():
    ci = format_flagged_ci(_flagged_verdict())
    assert "BH-adjusted" in ci
    assert "q=0.004" in ci


def test_other_flagged_note_absent_when_zero():
    assert format_other_flagged_note(_flagged_verdict(other_flagged_count=0)) is None


def test_other_flagged_note_singular_vs_plural():
    one = format_other_flagged_note(_flagged_verdict(other_flagged_count=1))
    two = format_other_flagged_note(_flagged_verdict(other_flagged_count=2))
    assert "1 other family" in one
    assert "2 other families" in two


# --- attempted/executed breakdown -------------------------------------------


def test_attempted_breakdown_none_when_no_counts():
    assert format_attempted_breakdown(None) is None


def test_attempted_breakdown_phrasing():
    counts = AttemptedExecutedCounts(tool_names=frozenset({"send_email"}), executed=2, blocked=8)
    text = format_attempted_breakdown(counts)
    assert "caught by your guardrail 8 of 10 times" in text
    assert "succeeded 2 of 10" in text


# --- CLEAR / INCONCLUSIVE phrasing -------------------------------------------


def _family_power(achieved_power=0.9, observed_effect=0.02, n_runs_per_case=8.0):
    return FamilyPower(
        outcome_key="exfiltration", family="direct_instruction_injection", n_cases=10,
        n_runs_per_case=n_runs_per_case, baseline_rate=0.15, observed_effect=observed_effect, achieved_power=achieved_power,
    )


def test_clear_summary_mentions_achieved_mde_in_points():
    verdict = ComparisonVerdict(tier="CLEAR", target_power=0.8, worst_case=_family_power(), achieved_mde=0.12)
    lines = format_clear_summary(verdict)
    assert any("12+ points" in line for line in lines)
    assert any("90.0%" in line for line in lines)
    assert any("target: 80.0%" in line for line in lines)


def test_inconclusive_summary_shows_recommended_runs():
    verdict = ComparisonVerdict(
        tier="INCONCLUSIVE", target_power=0.8, worst_case=_family_power(achieved_power=0.16), recommended_additional_runs=12
    )
    lines = format_inconclusive_summary(verdict)
    assert any("run at least 12 more run(s)/case" in line for line in lines)
    assert any("currently ~8" in line for line in lines)


def test_inconclusive_summary_heterogeneity_dominates_fallback_text():
    verdict = ComparisonVerdict(tier="INCONCLUSIVE", target_power=0.8, worst_case=_family_power(), recommended_additional_runs=None)
    lines = format_inconclusive_summary(verdict)
    assert any("more cases, not more" in line for line in lines)


def test_inconclusive_summary_no_worst_case_states_the_real_cause():
    # The real cfg_4c44f09aed30 A/A run: the homepilot-ticket-analysis
    # environment supports 2 of the 17 attack cases, one per family, and a
    # family needs >= MIN_CASES_FOR_BOOTSTRAP cases before compare_families
    # can produce an effect at all -- so both families were dropped and
    # family_results came back empty. The message used to say only "No
    # comparable family data available to assess power," which reads like
    # the run failed rather than like the case suite not covering this
    # environment densely enough.
    verdict = ComparisonVerdict(
        tier="INCONCLUSIVE", target_power=0.8, worst_case=None,
        n_cases_run=2,
        cases_per_family={"tool_result_poisoning": 1, "multi_turn_goal_hijack": 1},
    )
    lines = format_inconclusive_summary(verdict)
    joined = " ".join(lines)
    assert "2 applicable case(s) across 2 families" in joined
    assert "at least 2" in joined
    # Names which families fell short, and by how much.
    assert "(1 case)" in joined
    assert family_display_name("tool_result_poisoning") in joined
    assert family_display_name("multi_turn_goal_hijack") in joined
    # Steers away from the wrong remedy (more runs) and from reading the
    # absence of data as evidence of equivalence.
    assert "more cases" in joined
    assert "Nothing here says the two arms are the same" in joined


def test_inconclusive_summary_no_worst_case_falls_back_without_family_counts():
    # Reports saved before cases_per_family was persisted.
    verdict = ComparisonVerdict(tier="INCONCLUSIVE", target_power=0.8, worst_case=None, n_cases_run=2)
    joined = " ".join(format_inconclusive_summary(verdict))
    assert "2 applicable case(s) ran" in joined
    assert "at least 2" in joined

    bare = ComparisonVerdict(tier="INCONCLUSIVE", target_power=0.8, worst_case=None)
    assert format_inconclusive_summary(bare) == ["No comparable family data available to assess power."]


def test_inconclusive_summary_singular_family_wording():
    verdict = ComparisonVerdict(
        tier="INCONCLUSIVE", target_power=0.8, worst_case=None,
        n_cases_run=1, cases_per_family={"tool_result_poisoning": 1},
    )
    joined = " ".join(format_inconclusive_summary(verdict))
    assert "1 applicable case(s) across 1 family;" in joined


# --- single-config phrasing --------------------------------------------------


# --- drill-down table ---------------------------------------------------


def _report_with_two_families():
    def fr(family, rate_a, rate_b, q, used_fallback=False, fallback_reason=None):
        diff = rate_b - rate_a
        return {
            "family": family,
            "q_value": q,
            "significant_after_correction": q < 0.05,
            "effect": {
                "method": "cluster_bootstrap", "rate_a": rate_a, "rate_b": rate_b, "diff": diff,
                "ci_low": diff - 0.05, "ci_high": diff + 0.05, "alpha": 0.05, "p_value": 0.01,
                "n_cases": 10, "n_runs_a": 50, "n_runs_b": 50,
                "used_fallback": used_fallback, "fallback_reason": fallback_reason, "extra": {},
            },
        }

    return {
        "family_results": {
            "exfiltration": [fr("direct_instruction_injection", 0.1, 0.7, 0.01)],
            "exfiltration_attempted": [fr("direct_instruction_injection", 0.1, 0.1, 0.9)],
            "unauthorized_lookup": [fr("indirect_injection_document", 0.05, 0.06, 0.8, used_fallback=True, fallback_reason="only 3 cases")],
        }
    }


def test_build_drill_down_rows_covers_every_outcome_key():
    rows = build_drill_down_rows(_report_with_two_families())
    outcome_keys = {row[0] for row in rows}
    assert outcome_keys == {"exfiltration", "exfiltration_attempted", "unauthorized_lookup"}
    assert len(rows) == 3


def test_build_drill_down_rows_notes_fallback_method():
    rows = build_drill_down_rows(_report_with_two_families())
    fallback_row = next(r for r in rows if r[1] == "Indirect injection (document)")
    assert "fallback: only 3 cases" in fallback_row[7]  # Method column


def test_build_drill_down_rows_tool_responses_column_dash_when_no_data():
    rows = build_drill_down_rows(_report_with_two_families())
    assert all(row[8] == "—" for row in rows)  # no records/configs given -> nothing to show


def test_build_drill_down_rows_tool_responses_column_populated_when_given_data():
    from target_system.factory import baseline_config

    report = _report_with_two_families()
    records = [
        {"arm": "arm_a", "case_family": "direct_instruction_injection", "events": [{"type": "tool_call", "tool_name": "send_email", "response_source": "real"}]},
    ]
    rows = build_drill_down_rows(report, records=records, configs=[baseline_config()])
    row = next(r for r in rows if r[0] == "exfiltration" and r[1] == "Direct instruction injection")
    assert "real" in row[8]


def test_build_drill_down_rows_uses_family_display_name():
    rows = build_drill_down_rows(_report_with_two_families())
    families = {r[1] for r in rows}
    assert "direct_instruction_injection" not in families
    assert "Direct instruction injection" in families


def test_build_drill_down_rows_q_value_column_uses_fdr_adjusted_value():
    # family_results[i]["effect"]["p_value"] is also 0.01 for this row — the q-value
    # column must read from fr["q_value"] (BH-adjusted), not effect["p_value"] (raw)
    rows = build_drill_down_rows(_report_with_two_families())
    assert rows[0][6] == "0.010"  # q_value=0.01 for direct_instruction_injection/exfiltration


def test_single_config_headline_counts():
    summary = SingleConfigSummary(
        config_label="baseline", config_hash="cfg_abc", total_attacks=20, succeeded=2, blocked=3, resisted=15,
        by_family=[FamilySingleSummary(family="f", total=20, succeeded=2, blocked=3)],
    )
    headline = format_single_config_headline(summary)
    assert "20 attacks tried" in headline
    assert "2 succeeded" in headline


# --- reconstructed-environment fidelity disclosures (Part 6) -----------------


def _provenance(**overrides):
    from target_system.provenance import ReconstructionProvenance

    defaults = dict(project_id="proj-1", source_agent_name="Invoice Generation Assistant", trace_count=11, extraction_date="2026-01-15T09:00:00+00:00")
    defaults.update(overrides)
    return ReconstructionProvenance(**defaults)


def test_system_prompt_unavailable_disclosure_is_not_hedged_about_the_fact():
    # the fact itself (no system prompt observed) must be stated plainly,
    # not softened -- only the consequence is legitimately uncertain
    assert "ran with no system prompt at all" in SYSTEM_PROMPT_UNAVAILABLE_DISCLOSURE
    assert "CLEAR" in SYSTEM_PROMPT_UNAVAILABLE_DISCLOSURE
    assert "FLAGGED" in SYSTEM_PROMPT_UNAVAILABLE_DISCLOSURE


def test_format_provenance_disclosure_includes_trace_count_source_and_date():
    text = format_provenance_disclosure(_provenance())
    assert "11" in text
    assert "Invoice Generation Assistant" in text
    assert "2026-01-15" in text
    assert "09:00" not in text  # date only, not the full timestamp


def test_format_provenance_disclosure_handles_missing_agent_name():
    text = format_provenance_disclosure(_provenance(source_agent_name=None))
    assert "no agent_name tag" in text


def test_format_other_groups_found_note_absent_when_empty():
    assert format_other_groups_found_note(_provenance(other_groups_found=[])) is None


def test_format_other_groups_found_note_lists_other_groups():
    from target_system.provenance import OtherGroupFound

    provenance = _provenance(other_groups_found=[OtherGroupFound(agent_name="HR Onboarding Assistant", trace_count=33)])
    text = format_other_groups_found_note(provenance)
    assert "one of 2 systems" in text
    assert "HR Onboarding Assistant (33)" in text


def test_format_response_source_summary_only_lists_nonzero_categories():
    breakdown = ResponseSourceBreakdown(real=5, replay=2, generated=0, unavailable=0, unknown=0)
    text = format_response_source_summary(breakdown)
    assert "5 real" in text
    assert "2 replayed from history" in text
    assert "generated" not in text
    assert "unavailable" not in text
    assert "of 7 total" in text


def test_format_response_source_cell_dash_when_none():
    assert format_response_source_cell(None) == "—"


def test_format_response_source_cell_compact_form():
    breakdown = ResponseSourceBreakdown(real=3, generated=1)
    cell = format_response_source_cell(breakdown)
    assert "3 real" in cell
    assert "1 generated" in cell
    assert "replay" not in cell


def test_flagged_synthetic_evidence_note_absent_when_no_breakdown():
    assert format_flagged_synthetic_evidence_note(None) is None


def test_flagged_synthetic_evidence_note_absent_when_all_real_or_replayed():
    breakdown = ResponseSourceBreakdown(real=4, replay=2)
    assert format_flagged_synthetic_evidence_note(breakdown) is None


def test_flagged_synthetic_evidence_note_present_and_states_ratio_when_synthetic():
    breakdown = ResponseSourceBreakdown(real=2, replay=1, generated=2, unavailable=1)
    note = format_flagged_synthetic_evidence_note(breakdown)
    assert note is not None
    assert "3 of 6" in note  # generated(2) + unavailable(1) of total(6)
    assert "less weight" in note
