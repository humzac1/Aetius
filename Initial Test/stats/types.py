"""The shared data model every stats/ function operates on.

The one rule everything downstream must respect: runs are Bernoulli trials
*nested within attack cases*, and cases differ enormously in difficulty
(see the build spec). PairedCaseData is deliberately structured so a case
is always the addressable unit — bootstrap resampling, the mixed model's
random intercept, and the sequential test's per-step observation all
operate case-by-case, never on a flattened pool of runs treated as iid.
Nothing in this package should ever compute e.g. a pooled binomial CI over
all runs regardless of which case they came from — that's exactly the
mistake the build spec calls out.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CaseObservations:
    """One case's runs in one arm. `outcomes` is ordered by trial index —
    outcomes[i] in arm A and outcomes[i] in arm B of the *same case* are
    the matched pair under common random numbers (CRN) when seeds were
    shared across arms; see variance_reduction.py."""

    case_id: str
    family: str
    outcomes: tuple[int, ...]  # 0/1 per run

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def successes(self) -> int:
        return sum(self.outcomes)

    @property
    def rate(self) -> float:
        if self.n == 0:
            return float("nan")
        return self.successes / self.n


@dataclass(frozen=True)
class PairedCaseData:
    """One case's observations in both arms of a paired comparison. The
    statistical design throughout stats/ is paired on case_id — every
    function that compares two arms takes a list[PairedCaseData], one
    entry per case, never two separately-indexed lists of cases that might
    silently misalign."""

    case_id: str
    family: str
    arm_a: CaseObservations
    arm_b: CaseObservations

    def __post_init__(self) -> None:
        if self.arm_a.case_id != self.case_id or self.arm_b.case_id != self.case_id:
            raise ValueError(f"case_id mismatch in PairedCaseData for {self.case_id!r}")

    @property
    def rate_diff(self) -> float:
        """arm_b rate minus arm_a rate for this case. NaN-safe only in the
        sense that CaseObservations.rate already returns NaN for n=0 —
        callers should filter zero-n cases before aggregating."""
        return self.arm_b.rate - self.arm_a.rate


@dataclass(frozen=True)
class EffectEstimate:
    """A single result: an effect size, its uncertainty, and how it was
    computed — never a bare p-value, per the build spec's reporting rule."""

    method: str
    rate_a: float
    rate_b: float
    diff: float  # rate_b - rate_a
    ci_low: float
    ci_high: float
    alpha: float
    p_value: float | None = None
    n_cases: int = 0
    n_runs_a: int = 0
    n_runs_b: int = 0
    used_fallback: bool = False
    fallback_reason: str | None = None
    extra: dict = field(default_factory=dict)


def case_rate(cases: list[CaseObservations]) -> float:
    """Unweighted mean of per-case rates — the cluster-respecting estimate
    of an arm's overall rate. NOT sum(successes)/sum(n): that would let
    high-n cases dominate and would treat every run as an independent
    Bernoulli trial, exactly the pooling the build spec forbids."""
    rates = [c.rate for c in cases if c.n > 0]
    if not rates:
        return float("nan")
    return sum(rates) / len(rates)


def paired_rate_diff(data: list[PairedCaseData]) -> float:
    diffs = [d.rate_diff for d in data if d.arm_a.n > 0 and d.arm_b.n > 0]
    if not diffs:
        return float("nan")
    return sum(diffs) / len(diffs)
