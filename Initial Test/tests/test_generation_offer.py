"""The generation offer must track the *kind* of refusal, not its wording.

Both refusals below are the real ones from an A/A run against the
E-Commerce Order Support reconstruction:

  exfiltration         5 cases, 80% zero diffs  -> degenerate
  unauthorized_lookup  5 cases, 20% zero diffs  -> insufficient_cases

Only the second has a remedy. Offering generation on the first would
promise a fix the measured sweep contradicts: that shape's false-positive
rate stays above 9% even at 100 cases.
"""

import numpy as np
import pytest
from textual.widgets import Label, ListView

from attacker.case_generation import estimate_generation_cost, generated_case_id, plan_variants
from attacker.cases import ATTACK_CASES
from stats.multiple_comparisons import compare_families
from stats.paired import MIN_CASES_FOR_BOOTSTRAP, bootstrap_refusal
from stats.types import CaseObservations, PairedCaseData
from tui.app import HarnessApp
from tui.formatting import format_inconclusive_summary
from tui.screens.verdict import ComparisonVerdictScreen, _variant_of
from tui.verdict_logic import compute_comparison_verdict
from tests.tui_test_support import run_async
from tests.test_case_generation import _ecommerce_config

RARE_FLOOR = [0.0390, 0.0, 0.0, 0.0, 0.0]
HIGH_CEILING = [0.4805, 1.0, 0.8831, 0.9481, 0.9870]
FAMILY = "direct_instruction_injection"


def _paired(shape, n_cases, *, n_runs=77, seed=0):
    rng = np.random.default_rng(seed)
    data = []
    for i in range(n_cases):
        rate = shape[i % len(shape)]
        cid = f"case_{i}"
        data.append(
            PairedCaseData(cid, FAMILY,
                           CaseObservations(cid, FAMILY, tuple(int(x) for x in rng.binomial(1, rate, n_runs))),
                           CaseObservations(cid, FAMILY, tuple(int(x) for x in rng.binomial(1, rate, n_runs))))
        )
    return data


# --- structured refusal ---------------------------------------------------------


def test_refusal_kinds_are_structured_not_text():
    degenerate = bootstrap_refusal(_paired(RARE_FLOOR, 5))
    shortfall = bootstrap_refusal(_paired(HIGH_CEILING, 5))

    assert degenerate.kind == "degenerate"
    assert degenerate.cases_needed is None  # no number of cases fixes it

    assert shortfall.kind == "insufficient_cases"
    assert shortfall.n_cases == 5
    assert shortfall.cases_needed == MIN_CASES_FOR_BOOTSTRAP - 5 == 75


def test_refusal_survives_into_the_persisted_effect():
    """The kind has to reach a saved report, not just the live object."""
    results = compare_families(_paired(HIGH_CEILING, 5), method="cluster_bootstrap")
    extra = results[0].effect.extra
    assert extra["refusal_kind"] == "insufficient_cases"
    assert extra["refusal_cases_needed"] == 75


def _report(data, outcome_key="exfiltration"):
    import dataclasses

    results = compare_families(data, method="cluster_bootstrap")
    rows = [
        {"family": r.family, "effect": dataclasses.asdict(r.effect),
         "q_value": r.q_value, "significant_after_correction": r.significant_after_correction}
        for r in results
    ]
    other = "unauthorized_lookup" if outcome_key == "exfiltration" else "exfiltration"
    return {
        "arm_a_label": "arm A", "arm_b_label": "arm B", "arm_a_hash": "cfg_x", "arm_b_hash": "cfg_x",
        "n_cases": len(data), "cases_per_family": {FAMILY: len(data)},
        "family_results": {outcome_key: rows, other: []},
    }


def test_verdict_carries_the_kind_and_shortfall():
    shortfall = compute_comparison_verdict(_report(_paired(HIGH_CEILING, 5)))
    assert shortfall.refused_kind == "insufficient_cases"
    assert shortfall.refused_cases_needed == 75
    assert shortfall.can_generate_more_cases is True

    degenerate = compute_comparison_verdict(_report(_paired(RARE_FLOOR, 5)))
    assert degenerate.refused_kind == "degenerate"
    assert degenerate.refused_cases_needed is None
    assert degenerate.can_generate_more_cases is False


def test_degeneracy_text_never_implies_more_cases_would_help():
    verdict = compute_comparison_verdict(_report(_paired(RARE_FLOOR, 5)))
    text = " ".join(format_inconclusive_summary(verdict))
    assert "More cases will not fix this one" in text
    # The remedy named is re-scoring under the live method, never more cases.
    assert "re-run the comparison" in text


# --- variant ids ----------------------------------------------------------------


def test_variants_extend_the_set_without_colliding_with_the_first_batch():
    """The already-approved batch used variant 0; reusing an id would
    silently merge two different cases in the paired statistics."""
    first = generated_case_id("direct_direct_email_exfil", "cfg_abc")
    later = generated_case_id("direct_direct_email_exfil", "cfg_abc", 3)
    assert first == "direct_direct_email_exfil__cfg_abc"
    assert later == "direct_direct_email_exfil__cfg_abc__v3"
    assert _variant_of(first) == 0
    assert _variant_of(later) == 3


def test_plan_cycles_templates_so_each_is_used_evenly():
    templates = [c for c in ATTACK_CASES if c.family == FAMILY]
    plan = plan_variants(templates, 75, start_variant=1)
    assert len(plan) == 75
    per_template = {}
    for template, _variant in plan:
        per_template[template.id] = per_template.get(template.id, 0) + 1
    assert set(per_template.values()) == {15}  # 75 across 5 templates
    ids = {generated_case_id(t.id, "cfg_abc", v) for t, v in plan}
    assert len(ids) == 75  # every generated id distinct
    assert all(_variant_of(i) >= 1 for i in ids)  # never reuses the first batch's variant


def test_cost_estimate_is_built_from_the_real_prompt():
    templates = [c for c in ATTACK_CASES if c.family == FAMILY]
    estimate = estimate_generation_cost(templates, _ecommerce_config(), 75)
    assert estimate.n_cases == 75
    assert estimate.input_tokens_per_case > 500  # the environment summary dominates it
    assert estimate.estimated_cost_usd > 0
    # scales linearly with the number of cases, since it's one call each
    half = estimate_generation_cost(templates, _ecommerce_config(), 150)
    assert half.estimated_cost_usd == pytest.approx(2 * estimate.estimated_cost_usd)


# --- the offer appears for exactly one of the two refusals -----------------------


def _offer_shown(report, tmp_path):
    from target_system.config import save_config

    config = _ecommerce_config()
    save_config(config, configs_dir=tmp_path)
    from target_system.config import compute_config_hash

    report = dict(report, arm_a_hash=compute_config_hash(config), arm_b_hash=compute_config_hash(config))
    verdict = compute_comparison_verdict(report)
    seen = {}

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ComparisonVerdictScreen(verdict, report, "exp", configs_dir=tmp_path))
            await pilot.pause()
            seen["menu"] = bool(app.screen.query("#generation-offer-menu"))
            seen["text"] = " ".join(str(w.render()) for w in app.screen.query(Label))

    run_async(scenario)
    return seen


def test_offer_appears_for_the_count_refused_row(tmp_path):
    seen = _offer_shown(_report(_paired(HIGH_CEILING, 5)), tmp_path)
    assert seen["menu"] is True
    assert "75 more domain-adapted case(s)" in seen["text"]
    assert "$" in seen["text"]  # real cost shown before committing


def test_offer_never_appears_for_the_degeneracy_refused_row(tmp_path):
    seen = _offer_shown(_report(_paired(RARE_FLOOR, 5)), tmp_path)
    assert seen["menu"] is False
    assert "More cases will not fix this one" in seen["text"]
    assert "Generate" not in seen["text"]
