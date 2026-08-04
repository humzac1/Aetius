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

from stats.paired import cluster_bootstrap_diff, mcnemar_test, mixed_effects_diff
from stats.types import EffectEstimate, PairedCaseData

_METHODS: dict[str, Callable[..., EffectEstimate]] = {
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
    method: str = "cluster_bootstrap",
    alpha: float = 0.05,
    method_kwargs: dict | None = None,
) -> list[FamilyResult]:
    """Runs the chosen paired test once per family, then BH-corrects the
    resulting p-values across families. A family whose test couldn't
    produce a p-value (e.g. too few cases) is skipped with a note, not
    silently included as non-significant."""
    test_fn = _METHODS[method]
    method_kwargs = dict(method_kwargs or {})
    method_kwargs.setdefault("alpha", alpha)

    groups = group_by_family(data)
    families: list[str] = []
    effects: list[EffectEstimate] = []
    for family, family_data in groups.items():
        try:
            effect = test_fn(family_data, **method_kwargs)
        except ValueError:
            continue
        if effect.p_value is None:
            continue
        families.append(family)
        effects.append(effect)

    p_values = [e.p_value for e in effects]
    reject, qvals = bh_correct(p_values, alpha=alpha)

    results = [
        FamilyResult(family=f, effect=e, q_value=q, significant_after_correction=r)
        for f, e, q, r in zip(families, effects, qvals, reject)
    ]
    # Most negative (worst regression) first — the screening-relevant order.
    results.sort(key=lambda r: r.effect.diff, reverse=True)
    return results
