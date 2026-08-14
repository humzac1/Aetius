"""Which case suite a given set of configs should actually be tested with.

This is the *only* place that knows domain-adapted cases
(attacker/case_generation.py) exist. Everything downstream — applicability
filtering, execution, outcome scoring, the paired statistics — receives a
plain list[AttackCase] and cannot tell a generated case from a
hand-authored one, because there is nothing to tell: a generated case is
the same schema, carrying its template's family, injection vector and
success outcome unchanged. That's deliberate. The moment "is this
generated?" leaks into the runner or the stats layer, generated cases stop
being comparable to hand-authored ones and every guarantee built on case
identity gets a special case attached to it.

The selection rule, in one place:
  - a reconstructed environment with an *approved* generated set uses it
  - anything else uses the hand-authored suite (attacker/cases.py)
  - a paired comparison uses generated cases only when both arms resolve
    to the same set, since both arms must face identical cases
"""

from __future__ import annotations

from pathlib import Path

from attacker.attack_case import AttackCase
from attacker.applicability import applicable_cases_for_configs
from attacker.case_store import DEFAULT_GENERATED_CASES_DIR, load_generated_cases
from attacker.cases import ATTACK_CASES
from target_system.config import SystemConfig, compute_config_hash


def suite_for_configs(
    configs: list[SystemConfig],
    *,
    cases_dir: Path = DEFAULT_GENERATED_CASES_DIR,
) -> tuple[list[AttackCase], str]:
    """(cases, source_label). source_label is for display only — nothing
    branches on it downstream.

    Falls back to the hand-authored suite rather than failing when a set is
    missing or unapproved, so an environment that hasn't been through
    generation still runs exactly as it did before this feature existed."""
    hashes = [compute_config_hash(c) for c in configs]
    sets = [load_generated_cases(h, cases_dir=cases_dir) for h in hashes]

    usable = [s for s in sets if s is not None and s.approved and s.coherent_cases]
    if len(usable) != len(configs):
        return list(ATTACK_CASES), "hand-authored suite"

    # Both arms must face identical cases. Two arms of the same environment
    # (an A/A run, or a model swap that leaves the tool surface untouched)
    # share a config hash only when nothing hashed differs — so compare the
    # case ids themselves rather than assuming.
    id_sets = [tuple(sorted(c.id for c in s.coherent_cases)) for s in usable]
    if len(set(id_sets)) != 1:
        return list(ATTACK_CASES), "hand-authored suite (arms have different generated sets)"

    return list(usable[0].coherent_cases), f"domain-adapted for this environment ({len(usable[0].coherent_cases)} cases)"


def applicable_suite_for_configs(
    configs: list[SystemConfig],
    *,
    cases_dir: Path = DEFAULT_GENERATED_CASES_DIR,
) -> tuple[list[AttackCase], str]:
    """suite_for_configs + the existing structural applicability filter, in
    the order every execution path needs them: pick the suite, then drop
    whatever this environment's tool roles can't support. Generated cases
    go through the identical filter — they inherit their template's
    injection vector and outcome, so applicability.py judges them by
    exactly the same rules."""
    cases, label = suite_for_configs(configs, cases_dir=cases_dir)
    return applicable_cases_for_configs(cases, configs), label
