"""Benjamini-Hochberg FDR control across attack families.

Not Bonferroni: the build spec is explicit that power matters more than
familywise strictness here — an attack-family-level regression report is a
screening tool (which families got worse?), and BH is designed for exactly
that use case (control the *expected proportion* of false discoveries
among the families flagged, rather than the probability of *any* false
discovery at all, which is what Bonferroni protects and is needlessly
strict when there are more than a couple of families).

One p-value per family — computed by running the paired test restricted
to that family's cases — is what gets corrected, not one p-value per case.
Cases within a family still respect clustering via whichever paired.py
method computed that family's p-value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from statsmodels.stats.multitest import multipletests

from stats.hierarchical import hierarchical_bayes_diff
from stats.paired import cluster_bootstrap_diff, mcnemar_test, mixed_effects_diff
from stats.types import EffectEstimate, PairedCaseData

# "hierarchical_bayes" is the live default (see stats/hierarchical.py for
# the validation that put it there). The frequentist methods stay
# registered for the calibration harness, regression tests, and
# experiments.cli's --method escape hatch — the same retired-but-testable
# status the toy target system has — but nothing on the product path
# selects them anymore.
_METHODS: dict[str, Callable[..., EffectEstimate]] = {
    "hierarchical_bayes": hierarchical_bayes_diff,
    "cluster_bootstrap": cluster_bootstrap_diff,
    "mcnemar": mcnemar_test,
    "mixed_effects": mixed_effects_diff,
}


@dataclass(frozen=True)
class FamilyResult:
    family: str
    effect: EffectEstimate
    q_value: float
    significant_after_correction: bool


def group_by_family(data: list[PairedCaseData]) -> dict[str, list[PairedCaseData]]:
    groups: dict[str, list[PairedCaseData]] = {}
    for d in data:
        groups.setdefault(d.family, []).append(d)
    return groups


def bh_correct(p_values: list[float], *, alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """Thin wrapper over statsmodels' fdr_bh — no reason to hand-roll a
    well-tested implementation of a one-line algorithm. Returns
    (reject_flags, q_values) in the same order as p_values."""
    if not p_values:
        return [], []
    reject, qvals, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return [bool(r) for r in reject], [float(q) for q in qvals]


def compare_families(
    data: list[PairedCaseData],
    *,
    method: str = "hierarchical_bayes",
    alpha: float = 0.05,
    method_kwargs: dict | None = None,
) -> list[FamilyResult]:
    """Runs the chosen paired test once per family, then BH-corrects the
    resulting p-values across families. A family whose test couldn't
    produce a p-value (e.g. too few cases) is skipped with a note, not
    silently included as non-significant.

    For the hierarchical_bayes default, "significant" requires BOTH the
    BH rejection (on the posterior direction probabilities, which the
    validation showed are approximately calibrated) AND the family's own
    ROPE signal (extra["rope_signal"]: the 95% credible interval entirely
    beyond the practical-equivalence region). The ROPE gate is the rule
    the false-signal rates were validated under — direction probability
    alone measurably over-signals on the rare-event shape — and BH on top
    is what controls the family-level multiplicity this function has
    always been responsible for. Dropping either half would un-validate
    the verdict."""
    test_fn = _METHODS[method]
    method_kwargs = dict(method_kwargs or {})
    method_kwargs.setdefault("alpha", alpha)

    groups = group_by_family(data)
    families: list[str] = []
    effects: list[EffectEstimate] = []
    # A family whose test refused to produce a p-value (see
    # stats/paired.bootstrap_refusal_reason) is kept rather than dropped:
    # its descriptive rates are still real, and the *reason* has to reach
    # the report so a verdict can say "this outcome's data is degenerate"
    # instead of falling through to the generic "no family produced an
    # estimate" message. It is excluded from the BH correction, since
    # correcting a p-value that doesn't exist is meaningless, and can
    # never be marked significant.
    refused: list[tuple[str, EffectEstimate]] = []
    for family, family_data in groups.items():
        try:
            effect = test_fn(family_data, **method_kwargs)
        except ValueError:
            continue
        if effect.p_value is None:
            refused.append((family, effect))
            continue
        families.append(family)
        effects.append(effect)

    p_values = [e.p_value for e in effects]
    reject, qvals = bh_correct(p_values, alpha=alpha)

    def _significant(effect: EffectEstimate, bh_reject: bool) -> bool:
        if "rope_signal" in effect.extra:
            return bh_reject and bool(effect.extra["rope_signal"])
        return bh_reject

    results = [
        FamilyResult(family=f, effect=e, q_value=q, significant_after_correction=_significant(e, r))
        for f, e, q, r in zip(families, effects, qvals, reject)
    ]
    results.extend(
        FamilyResult(family=f, effect=e, q_value=float("nan"), significant_after_correction=False)
        for f, e in refused
    )
    # Most negative (worst regression) first — the screening-relevant order.
    results.sort(key=lambda r: r.effect.diff, reverse=True)
    return results
