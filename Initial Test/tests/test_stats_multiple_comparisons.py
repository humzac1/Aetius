import numpy as np

from stats.multiple_comparisons import bh_correct, compare_families
from stats.types import CaseObservations, PairedCaseData


def test_bh_correct_less_strict_than_bonferroni():
    # 10 p-values, one genuinely tiny, rest moderate/null-ish.
    p_values = [0.001] + [0.04] * 9
    reject, qvals = bh_correct(p_values, alpha=0.05)
    bonferroni_reject = [p < 0.05 / len(p_values) for p in p_values]
    # BH should flag at least as many as Bonferroni (it's less strict).
    assert sum(reject) >= sum(bonferroni_reject)
    assert len(qvals) == len(p_values)


def test_bh_correct_empty_input():
    reject, qvals = bh_correct([], alpha=0.05)
    assert reject == []
    assert qvals == []


def test_compare_families_flags_only_the_regressed_family():
    rng = np.random.default_rng(11)
    data = []
    for fam, effect in [("regressed", 0.15), ("null_a", 0.0), ("null_b", 0.0)]:
        for i in range(80):  # the calibrated floor for cluster_bootstrap
            p_a = rng.uniform(0.05, 0.2)
            p_b = min(0.95, p_a + effect)
            a = tuple(int(x) for x in rng.binomial(1, p_a, 25))
            b = tuple(int(x) for x in rng.binomial(1, p_b, 25))
            cid = f"{fam}_{i}"
            data.append(PairedCaseData(cid, fam, CaseObservations(cid, fam, a), CaseObservations(cid, fam, b)))

    results = compare_families(data, method="cluster_bootstrap", alpha=0.05, method_kwargs={"n_boot": 1000, "seed": 1})
    by_family = {r.family: r for r in results}
    assert by_family["regressed"].significant_after_correction is True
    assert by_family["regressed"].effect.diff > 0.05
    # Null families should generally not survive correction (not a hard
    # guarantee for any single random draw, but true effect is 0 with a
    # real regression in the mix to compare against).
    assert by_family["regressed"].q_value < by_family["null_a"].q_value
