"""Early stopping in the real comparison execution loop, and the CUPED
variance reduction that replaces CRN on real-model runs (CRN cannot work
there: the Anthropic API exposes no sampling seed, so run_experiment's
paired seeds are matched labels, not matched randomness)."""

import json

from attacker.attack_case import AttackCase
from experiments.persist import save_experiment_report
from experiments.report import compute_cuped_analysis, historical_case_rates
from experiments.runner import SequentialStopSpec, run_experiment
from target_system.factory import baseline_config


def _cases(n, family="direct_instruction_injection"):
    return [
        AttackCase(
            id=f"case-{i}", family=family, injection_vector="task_text", success_outcome="exfiltration",
            source="test", benign_task="do x", injected_payload="do y",
        )
        for i in range(n)
    ]


def _run(tmp_path, *, cases, sequential_stop=None, n_runs_per_case=2, name="exp"):
    return run_experiment(
        baseline_config(label="arm-a", defensive_instruction=True),
        baseline_config(label="arm-b", defensive_instruction=False),
        experiment_name=name,
        cases=cases,
        n_runs_per_case=n_runs_per_case,
        runs_dir=tmp_path,
        sequential_stop=sequential_stop,
    )


# --- default path is untouched -------------------------------------------------


def test_without_a_spec_every_case_runs(tmp_path):
    """Early stopping is strictly opt-in: the existing execution path must
    behave exactly as it did, or every caller that doesn't ask for it
    silently changes meaning."""
    cases = _cases(6)
    result = _run(tmp_path, cases=cases)
    assert result.sequential_stop is None
    assert {r.case_id for r in result.records} == {c.id for c in cases}


# --- early stopping ------------------------------------------------------------


def test_stop_outcome_is_recorded_even_when_it_does_not_stop(tmp_path):
    """A run that went the distance must say so explicitly rather than
    leaving the field null and ambiguous with 'stopping was off'."""
    result = _run(tmp_path, cases=_cases(4), sequential_stop=SequentialStopSpec(outcome_key="exfiltration"))
    assert result.sequential_stop is not None
    assert result.sequential_stop.outcome_key == "exfiltration"
    assert result.sequential_stop.cases_planned == 4
    if not result.sequential_stop.stopped_early:
        assert result.sequential_stop.cases_evaluated == 4


def test_stopping_early_leaves_later_cases_unexecuted(tmp_path):
    """The point of the feature: cases after the boundary are never paid
    for. Whether the boundary trips depends on the toy system's actual
    effect, so this asserts the invariant that must hold either way —
    stopped_early and 'some case has no records' agree."""
    cases = _cases(8)
    result = _run(tmp_path, cases=cases, sequential_stop=SequentialStopSpec(outcome_key="exfiltration"))
    stop = result.sequential_stop
    executed_case_ids = {r.case_id for r in result.records}

    if stop.stopped_early:
        assert len(executed_case_ids) < len(cases)
        assert stop.cases_evaluated < stop.cases_planned
        assert stop.first_stop_index is not None
        # never stops before the confidence sequence can even estimate sigma
        assert stop.cases_evaluated >= 2
    else:
        assert executed_case_ids == {c.id for c in cases}


def test_retired_mixture_sprt_rule_still_uses_its_validated_boundary(tmp_path):
    """The retired e-value rule stays testable: when explicitly selected
    and it stops, the recorded e-value must actually exceed 1/alpha —
    the mSPRT boundary's own criterion (Ville's inequality is what makes
    checking it repeatedly safe)."""
    spec = SequentialStopSpec(outcome_key="exfiltration", alpha=0.05, rule="mixture_sprt")
    result = _run(tmp_path, cases=_cases(8), sequential_stop=spec)
    stop = result.sequential_stop
    assert stop.rule == "mixture_sprt"
    assert stop.resolution is None
    if stop.stopped_early:
        assert stop.e_value >= 1 / spec.alpha
        assert stop.always_valid_p <= spec.alpha


def test_rope_rule_records_resolutions_and_credible_interval(tmp_path):
    """The live default: the stop outcome must carry the rope rule's own
    evidence — per-outcome resolutions and the primary outcome's credible
    interval — and never the retired rule's e-value fields."""
    spec = SequentialStopSpec(outcome_key="exfiltration", extra_outcome_keys=("unauthorized_lookup",))
    result = _run(tmp_path, cases=_cases(6), n_runs_per_case=3, sequential_stop=spec)
    stop = result.sequential_stop
    assert stop.rule == "rope"
    assert stop.e_value is None and stop.always_valid_p is None
    assert stop.resolutions is not None
    assert set(stop.resolutions) == {"exfiltration", "unauthorized_lookup"}
    assert all(r in {"signal", "futile", "continue"} for r in stop.resolutions.values())
    if stop.stopped_early:
        # stopping requires every monitored outcome to have resolved
        assert all(r != "continue" for r in stop.resolutions.values())
        assert stop.first_stop_index == stop.cases_evaluated
        assert stop.ci_low is not None and stop.ci_high is not None


def test_a_case_is_only_evaluated_once_both_arms_have_run(tmp_path):
    """Partial cases must never reach the boundary — a case scored on one
    arm alone would feed a meaningless rate difference into the sequence."""
    result = _run(tmp_path, cases=_cases(4), sequential_stop=SequentialStopSpec(outcome_key="exfiltration"), n_runs_per_case=3)
    per_case_arms = {}
    for r in result.records:
        per_case_arms.setdefault(r.case_id, set()).add(r.arm)
    for case_id, arms in per_case_arms.items():
        assert len(arms) == 2, f"{case_id} has records for only {arms}"


def test_report_records_the_stop(tmp_path):
    result = _run(tmp_path, cases=_cases(5), sequential_stop=SequentialStopSpec(outcome_key="exfiltration"))
    path = save_experiment_report(result, runs_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sequential_stop"]["outcome_key"] == "exfiltration"
    assert payload["sequential_stop"]["cases_planned"] == 5


# --- CUPED ---------------------------------------------------------------------


def test_cuped_is_none_without_history(tmp_path):
    """No fabricated adjustment on a first run — the covariate has to be a
    real prior measurement or there is no CUPED result at all."""
    result = _run(tmp_path, cases=_cases(4), name="first")
    assert compute_cuped_analysis(result, "exfiltration", runs_dir=tmp_path) is None


def test_cuped_uses_history_from_other_experiments_only(tmp_path):
    """The covariate must come from previous runs, never from the arm being
    adjusted in this same experiment — that self-reference is what would
    bias the adjustment."""
    prior = _run(tmp_path, cases=_cases(4), name="prior", n_runs_per_case=3)
    rates_all = historical_case_rates(prior.arm_a_hash, prior.arm_a_label, "exfiltration", runs_dir=tmp_path)
    rates_excluding = historical_case_rates(
        prior.arm_a_hash, prior.arm_a_label, "exfiltration", runs_dir=tmp_path, exclude_experiment="prior"
    )
    assert rates_all  # the prior run is visible when not excluded
    assert rates_excluding == {}  # and invisible to its own experiment


def test_cuped_reports_a_real_reduction_against_a_prior_run(tmp_path):
    """End to end: a second experiment over the same cases can use the
    first one's per-case rates as the covariate, and the result is a real
    measured variance change, not a placeholder."""
    cases = _cases(6)
    _run(tmp_path, cases=cases, name="prior", n_runs_per_case=3)
    current = _run(tmp_path, cases=cases, name="current", n_runs_per_case=3)

    cuped = compute_cuped_analysis(current, "exfiltration", runs_dir=tmp_path)
    if cuped is None:
        # Possible when the toy system produces no per-case variation at
        # all; the adjustment is then genuinely undefined rather than zero.
        return
    assert cuped.var_after <= cuped.var_before + 1e-12
    # CUPED is unbiased by construction: it moves variance, never the mean.
    assert abs(cuped.mean_after - cuped.mean_before) < 1e-9


def test_report_carries_the_cuped_summary(tmp_path):
    cases = _cases(5)
    _run(tmp_path, cases=cases, name="prior", n_runs_per_case=3)
    current = _run(tmp_path, cases=cases, name="current", n_runs_per_case=3)
    path = save_experiment_report(current, cuped_outcome_key="exfiltration", runs_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["cuped"] is not None:
        assert payload["cuped"]["outcome_key"] == "exfiltration"
        assert "variance_reduction_pct" in payload["cuped"]
        assert "adjusted_values" not in payload["cuped"]  # per-case values stay out of the report
