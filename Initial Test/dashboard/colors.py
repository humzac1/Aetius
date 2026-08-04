"""Color roles, from the dataviz skill's validated reference palette
(categorical hues in fixed order, a reserved status palette, chart chrome).

Light-mode values only: this is an internal analytics tool ("prioritize
legibility over polish" per the build spec), so this skips the skill's
full light/dark dual-declaration machinery (meant for hand-authored
HTML/SVG artifacts with a viewer-controlled theme toggle) and just uses
the light-surface values directly.
"""

from __future__ import annotations

# Categorical — fixed order, never cycled, never reassigned by a filter.
# Two arms in a paired comparison is exactly what slots 1/2 are for.
ARM_A = "#2a78d6"  # slot 1: blue
ARM_B = "#eb6834"  # slot 2: orange

# Status — reserved, never reused as a categorical series color. Paired
# with a text/icon label everywhere they're used, per the skill's rule
# that status color never carries meaning alone.
STATUS_GOOD = "#0ca30c"  # not significant / calibrated / a safety improvement
STATUS_CRITICAL = "#d03b3b"  # significant regression (rate rose)
STATUS_WARNING = "#fab219"  # borderline / blocked-not-executed
STATUS_NEUTRAL = "#898781"  # muted ink, used for "not significant" bars

# Chart chrome
SURFACE = "#fcfcfb"
PAGE_PLANE = "#f9f9f7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Diverging pair (for confidence-sequence / power-curve reference lines)
DIVERGING_COOL = "#2a78d6"  # blue
DIVERGING_WARM = "#e34948"  # red (categorical slot 8 — distinct from status-critical)


def significance_color(significant: bool, diff: float) -> str:
    """Not significant -> neutral. Significant + diff>0 (outcome rose,
    worse) -> critical red. Significant + diff<0 (outcome fell, safer)
    -> good green. Three states, always paired with a text label
    (SIGNIFICANT / not significant) wherever used — never color alone."""
    if not significant:
        return STATUS_NEUTRAL
    return STATUS_CRITICAL if diff > 0 else STATUS_GOOD
