"""Pure text-phrasing for verdict screens — no Textual imports, so every
wording rule (never say "significant", never print a raw p-value, always
show the attempted-vs-executed breakdown when it's available) is
pytest-testable without spinning up a screen. Screens call these and wrap
the results in Label/Static widgets; they don't format numbers inline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tui.verdict_logic import AttemptedExecutedCounts, ComparisonVerdict, ResponseSourceBreakdown, SingleConfigSummary

if TYPE_CHECKING:
    # Type-checking only: tui.data imports describe_config_diff/classify_diff_entry
    # from this module, so a runtime import here would be circular. ConfigDiffEntry
    # is only ever read via duck-typed .path/.value_a/.value_b access below.
    from tui.data import ConfigDiffEntry
    from target_system.provenance import ReconstructionProvenance

SINGLE_CONFIG_DISCLAIMER = (
    "This only reflects the attacks actually tried in this suite against this config. "
    "It is not proof of general safety — an agent that resisted every case here can still "
    "be vulnerable to an attack that isn't in this suite."
)

# Reconstructed-environment fidelity disclosures (Part 6). Shown inline and
# prominently on every verdict screen for a config with provenance is not
# None -- same "never buried, never a footnote" discipline as
# SINGLE_CONFIG_DISCLAIMER above. Wording reviewed and approved before
# being wired into any screen, since these are the two most likely to be
# misread as more or less alarming than intended.

SYSTEM_PROMPT_UNAVAILABLE_DISCLOSURE = (
    "No system prompt was observed for this agent — it ran with no system prompt at all. "
    "This result reflects this reconstruction's defenses, not necessarily the real deployed "
    "agent's actual instructions: a CLEAR result here is not evidence the real agent resists "
    "this attack, and a FLAGGED result may not reproduce against the real agent's actual prompt."
)

_NONE_GROUP_LABEL = "(no agent_name tag)"


def format_provenance_disclosure(provenance: "ReconstructionProvenance") -> str:
    source = provenance.source_agent_name or _NONE_GROUP_LABEL
    date = provenance.extraction_date.split("T", 1)[0]
    return f"Reconstructed from {provenance.trace_count} trace(s), {source}, pulled {date}."


def format_other_groups_found_note(provenance: "ReconstructionProvenance") -> str | None:
    if not provenance.other_groups_found:
        return None
    others = ", ".join(f"{g.agent_name or _NONE_GROUP_LABEL} ({g.trace_count})" for g in provenance.other_groups_found)
    n_total = len(provenance.other_groups_found) + 1
    return f"This environment reflects only one of {n_total} systems detected in the source trace batch — also found: {others}."


_RESPONSE_SOURCE_LABELS = {
    "real": "real",
    "replay": "replayed from history",
    "generated": "model-generated",
    "unavailable": "unavailable",
    "unknown": "unknown provenance",
}


def format_response_source_summary(breakdown: ResponseSourceBreakdown) -> str:
    parts = [
        f"{count} {_RESPONSE_SOURCE_LABELS[field]}"
        for field, count in (
            ("real", breakdown.real), ("replay", breakdown.replay),
            ("generated", breakdown.generated), ("unavailable", breakdown.unavailable), ("unknown", breakdown.unknown),
        )
        if count > 0
    ]
    return f"Tool responses this run — {', '.join(parts)} (of {breakdown.total} total)."


_RESPONSE_SOURCE_CELL_LABELS = {"real": "real", "replay": "replay", "generated": "generated", "unavailable": "unavail.", "unknown": "unknown"}


def format_response_source_cell(breakdown: ResponseSourceBreakdown | None) -> str:
    if breakdown is None:
        return "—"
    parts = [
        f"{count} {_RESPONSE_SOURCE_CELL_LABELS[field]}"
        for field, count in (
            ("real", breakdown.real), ("replay", breakdown.replay),
            ("generated", breakdown.generated), ("unavailable", breakdown.unavailable), ("unknown", breakdown.unknown),
        )
        if count > 0
    ]
    return " / ".join(parts) if parts else "—"


def format_flagged_synthetic_evidence_note(breakdown: ResponseSourceBreakdown | None) -> str | None:
    if breakdown is None or breakdown.synthetic == 0:
        return None
    return (
        f"{breakdown.synthetic} of {breakdown.total} tool response(s) behind this flagged result were "
        "model-generated or unavailable, not replayed from real observed behavior. A regression detected "
        "partly through a synthesized tool response carries less weight than one detected through replayed "
        "real behavior."
    )


def _pct(rate: float) -> str:
    return f"{100 * rate:.1f}%"


def _pts(value: float) -> str:
    return f"{100 * value:+.1f} points"


# --- family / preset display names -------------------------------------------
# Internal identifiers (attacker/cases.py family strings, experiments/presets.py
# preset names) stay as-is everywhere in code and data — these mappings only
# affect what a screen renders, never what's stored or looked up by.

FAMILY_DISPLAY_NAMES: dict[str, str] = {
    "direct_instruction_injection": "Direct instruction injection",
    "indirect_injection_document": "Indirect injection (document)",
    "tool_result_poisoning": "Tool-result poisoning",
    "multi_turn_goal_hijack": "Multi-turn goal hijack",
}


def family_display_name(family: str) -> str:
    return FAMILY_DISPLAY_NAMES.get(family, family)


PRESET_DISPLAY_NAMES: dict[str, str] = {"aa": "A/A (sanity check)"}


def preset_display_name(name: str) -> str:
    return PRESET_DISPLAY_NAMES.get(name, name)


PRESET_VERDICT_HINTS: dict[str, str] = {
    "aa": "expect no flag (sanity check — run this first)",
    "known_regression": "expect FLAGGED",
    "known_neutral": "expect no flag (this is the false-alarm test)",
    "model_swap": "expect no flag with the mock backend; real models: unknown",
    "added_agent": "expect no flag, or a small inconclusive shift",
}


def preset_verdict_hint(name: str) -> str | None:
    return PRESET_VERDICT_HINTS.get(name)


def format_flagged_headline(verdict: ComparisonVerdict) -> str:
    effect = verdict.flagged_effect
    verb = "rose" if effect["diff"] >= 0 else "fell"
    return (
        f"{family_display_name(verdict.flagged_family)} / {verdict.flagged_outcome_key} {verb} from "
        f"{_pct(effect['rate_a'])} to {_pct(effect['rate_b'])}"
    )


def format_flagged_ci(verdict: ComparisonVerdict) -> str:
    effect = verdict.flagged_effect
    return (
        f"95% CI for the change: [{_pts(effect['ci_low'])}, {_pts(effect['ci_high'])}] "
        f"(BH-adjusted for multiple comparisons, q={verdict.flagged_q_value:.3f})"
    )


def format_other_flagged_note(verdict: ComparisonVerdict) -> str | None:
    if verdict.other_flagged_count <= 0:
        return None
    plural = "family" if verdict.other_flagged_count == 1 else "families"
    return f"{verdict.other_flagged_count} other {plural} also flagged in this run — see the statistics drill-down."


def format_attempted_breakdown(counts: AttemptedExecutedCounts | None) -> str | None:
    if counts is None:
        return None
    return f"caught by your guardrail {counts.blocked} of {counts.total} times, succeeded {counts.executed} of {counts.total}"


def format_clear_summary(verdict: ComparisonVerdict) -> list[str]:
    wc = verdict.worst_case
    assert wc is not None and verdict.achieved_mde is not None
    return [
        f"Worst-powered family: {family_display_name(wc.family)} / {wc.outcome_key}",
        f"Achieved power at the observed effect: {_pct(wc.achieved_power)} (target: {_pct(verdict.target_power)})",
        f"This run could reliably detect a change of {100 * verdict.achieved_mde:.0f}+ points in the worst-covered family.",
    ]


def format_inconclusive_summary(verdict: ComparisonVerdict) -> list[str]:
    wc = verdict.worst_case
    if wc is None:
        return ["No comparable family data available to assess power."]
    lines = [
        f"Worst-powered family: {family_display_name(wc.family)} / {wc.outcome_key}",
        f"Achieved power at the observed effect: {_pct(wc.achieved_power)} (target: {_pct(verdict.target_power)})",
        f"Observed effect: {_pts(wc.observed_effect)} — too small to distinguish from noise at this sample size.",
    ]
    if verdict.recommended_additional_runs is not None:
        lines.append(
            f"Recommended: run at least {verdict.recommended_additional_runs} more run(s)/case "
            f"(currently ~{round(wc.n_runs_per_case)}) to reach {_pct(verdict.target_power)} power at this effect size."
        )
    else:
        lines.append(
            "Recommended: this family's between-case variability dominates — adding more cases, not more "
            "runs per case, is what would sharpen this."
        )
    return lines


DRILL_DOWN_COLUMNS = ("Outcome", "Family", "Rate A", "Rate B", "Diff", "95% CI", "q-value", "Method", "Tool responses")


def build_drill_down_rows(
    report: dict[str, Any], *, records: list[dict[str, Any]] | None = None, configs: list | None = None
) -> list[tuple[str, str, str, str, str, str, str, str, str]]:
    """One row per (outcome_key, family) already computed in the saved
    report — the statistics drill-down's full table, including the
    _attempted outcome keys and, when a family's effect used the mixed-
    effects fallback, a note saying so rather than silently showing the
    fallback method's numbers as if they were what was asked for.

    records/configs are optional (default: no "Tool responses" data,
    every row shows "—" in that column) so callers that only have a
    report (no raw records or resolved configs loaded) still work — the
    existing statistics drill-down calls always pass both now, but keeping
    them optional avoids forcing every caller to thread through data it
    might not have. When given, the response_source breakdown for that
    row (attacker.applicability role-resolved, both arms combined — see
    tui.verdict_logic.compute_response_source_breakdown_for_row) is
    the same "audit exactly which calls were real vs. synthesized"
    granularity requested for Part 6, one level finer than the whole-run
    summary shown on the verdict screen itself."""
    from tui.verdict_logic import compute_response_source_breakdown_for_row

    records = records or []
    configs = configs or []
    rows = []
    for outcome_key, family_results in report.get("family_results", {}).items():
        for fr in family_results:
            effect = fr["effect"]
            method = effect["method"]
            if effect.get("used_fallback"):
                method = f"{method} (fallback: {effect.get('fallback_reason', '?')})"
            base_outcome_key = outcome_key.removesuffix("_attempted")
            breakdown = (
                compute_response_source_breakdown_for_row(records, family=fr["family"], base_outcome_key=base_outcome_key, configs=configs)
                if configs
                else None
            )
            rows.append(
                (
                    outcome_key,
                    family_display_name(fr["family"]),
                    _pct(effect["rate_a"]),
                    _pct(effect["rate_b"]),
                    _pts(effect["diff"]),
                    f"[{_pts(effect['ci_low'])}, {_pts(effect['ci_high'])}]",
                    f"{fr['q_value']:.3f}",
                    method,
                    format_response_source_cell(breakdown),
                )
            )
    return rows


def format_single_config_headline(summary: SingleConfigSummary) -> str:
    return (
        f"{summary.total_attacks} attacks tried  •  {summary.succeeded} succeeded  •  "
        f"{summary.blocked} blocked  •  {summary.resisted} resisted"
    )


# --- config diff -> plain language --------------------------------------------
# Turns a list of tui.data.ConfigDiffEntry (produced by tui.data's diff walk —
# never recomputed here) into either a short field label (Manage Configs' diff
# table) or a full inline phrase (describe_config_for_humans' one-liner). Both
# consumers go through classify_diff_entry so the wording is defined exactly
# once, not duplicated per screen.


@dataclass(frozen=True)
class DiffCategory:
    category: str  # bucket key used only for the grouped-count case, e.g. "prompt", "model"
    field_label: str  # short name for the diff table's Field column, e.g. "Supervisor's system prompt"
    phrase: str  # lowercase clause fitting after "baseline, but ", e.g. "supervisor's wording changed"


_ROLE_RE = re.compile(r"agents\[role=([^\]]+)\]")


def _extract_role(path: str) -> str:
    match = _ROLE_RE.search(path)
    return match.group(1) if match else "agent"


_ORDINAL_SUFFIXES = {1: "st", 2: "nd", 3: "rd"}


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = _ORDINAL_SUFFIXES.get(n % 10, "th")
    return f"{n}{suffix}"


_MODEL_FIELD_LABELS = {"temperature": "Model temperature", "max_tokens": "Max output tokens", "seed": "Model seed"}


def _poisoning_category(field_label: str, name: str, unit: str, value_b: Any) -> DiffCategory:
    count = len(value_b) if isinstance(value_b, list) else 0
    if count == 0:
        phrase = f"{name} removed"
    else:
        phrase = f"{name} configured for testing ({count} {unit}{'' if count == 1 else 's'})"
    return DiffCategory("poisoning", field_label, phrase)


def classify_diff_entry(entry: "ConfigDiffEntry", *, target_agent_count: int | None = None) -> DiffCategory:
    """entry: a tui.data.ConfigDiffEntry (duck-typed here as any object with
    .path/.value_a/.value_b — see the TYPE_CHECKING import above for why this
    isn't a real import). target_agent_count, if given, is the total agent
    count of the *target* (non-baseline) config — only used to phrase "with a
    4th agent added" correctly; omit it and the phrase falls back to "a new
    agent" rather than guessing a number."""
    path, a, b = entry.path, entry.value_a, entry.value_b

    if path == "defensive_instruction":
        verb = "removed" if b is False else "added"
        return DiffCategory("defensive", "Supervisor's defensive instruction", f"supervisor's defensive instruction {verb}")

    if path == "model.model_name":
        return DiffCategory("model", "Model", f"using {b} instead of {a}")

    if path in ("model.temperature", "model.max_tokens", "model.seed"):
        field = path.split(".", 1)[1]
        label = _MODEL_FIELD_LABELS.get(field, f"Model {field}")
        return DiffCategory("model", label, f"model {field} changed from {a} to {b}")

    if path == "security.enforce_allowlist":
        verb = "turned on" if b else "turned off"
        return DiffCategory("allowlist", "Tool-call allowlist enforcement", f"tool-call allowlist enforcement {verb}")

    if path == "security.email_allowlist":
        len_a = len(a) if isinstance(a, list) else 0
        len_b = len(b) if isinstance(b, list) else 0
        unit = "address" if len_b == 1 else "addresses"
        return DiffCategory("allowlist", "Email allowlist", f"email allowlist changed ({len_b} {unit} instead of {len_a})")

    if path == "security.poisoned_corpus_files":
        return _poisoning_category("Poisoned corpus files (testing)", "corpus poisoning", "file", b)

    if path == "security.poisoned_tool_results":
        return _poisoning_category("Poisoned tool results (testing)", "tool-result poisoning", "item", b)

    if path == "corpus_dir":
        return DiffCategory("other", "Corpus directory", "corpus directory changed")

    if path.startswith("agents[role=") and path.endswith("]"):
        role = _extract_role(path)
        if a is None:
            name = b.get("name", role) if isinstance(b, dict) else role
            ordinal = _ordinal(target_agent_count) if target_agent_count else "new"
            return DiffCategory("agents", f"{role.title()} agent", f"with a {ordinal} agent ({name}) added")
        if b is None:
            return DiffCategory("agents", f"{role.title()} agent", f"with the {role} agent removed")
        return DiffCategory("agents", f"{role.title()} agent", f"{role} agent changed")

    if ".system_prompt" in path:
        role = _extract_role(path)
        return DiffCategory("prompt", f"{role.title()}'s system prompt", f"{role}'s wording changed")

    if ".tools" in path:
        role = _extract_role(path)
        return DiffCategory("tools", f"{role.title()}'s tools", f"{role}'s tool access changed")

    if ".model_override" in path:
        role = _extract_role(path)
        if a is None:
            phrase = f"{role} now uses a different model than the team default"
        elif b is None:
            phrase = f"{role}'s model override removed"
        else:
            phrase = f"{role}'s model override changed"
        return DiffCategory("model", f"{role.title()}'s model override", phrase)

    if ".name" in path:
        role = _extract_role(path)
        return DiffCategory("other", f"{role.title()} agent name", f"{role} agent renamed")

    return DiffCategory("other", path, f"`{path}` changed")


def diff_field_label(entry: "ConfigDiffEntry") -> str:
    return classify_diff_entry(entry).field_label


_CATEGORY_GROUP_LABELS = {
    "prompt": "prompt/wording",
    "model": "model",
    "allowlist": "allowlist",
    "agents": "agent",
    "tools": "tool",
    "poisoning": "poisoning-config",
    "defensive": "config",
    "other": "config",
}


def _join_phrases(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def _suppress_redundant_entries(entries: list["ConfigDiffEntry"]) -> list["ConfigDiffEntry"]:
    """The supervisor's system_prompt text is mechanically derived from
    defensive_instruction (and cosmetic_variant, which isn't a stored field) —
    when defensive_instruction is itself in the diff set, the resulting
    system_prompt diff is downstream noise, not an independent change.
    Inline-description-only: the raw diff table (Manage Configs) still shows
    both entries."""
    paths = {e.path for e in entries}
    if "defensive_instruction" in paths:
        return [e for e in entries if e.path != "agents[role=supervisor].system_prompt"]
    return list(entries)


def describe_config_diff(
    entries: list["ConfigDiffEntry"], *, max_inline_diffs: int = 2, target_agent_count: int | None = None
) -> str:
    """entries: the output of tui.data.diff_configs(baseline_hash, config_hash)
    — this function only phrases them, it never recomputes what differs."""
    effective = _suppress_redundant_entries(entries)
    if not effective:
        return "baseline (defaults)"
    categorized = [classify_diff_entry(e, target_agent_count=target_agent_count) for e in effective]
    if len(categorized) <= max_inline_diffs:
        return "baseline, but " + _join_phrases([c.phrase for c in categorized])
    categories = {c.category for c in categorized}
    if len(categories) == 1:
        label = _CATEGORY_GROUP_LABELS[next(iter(categories))]
        return f"baseline, with {len(categorized)} {label} changes (view diff)"
    return f"baseline, with {len(categorized)} changes (view diff)"


def format_config_list_label(description: str, config_hash: str) -> str:
    """Two-line label for any list row that shows a config — description
    first (primary), hash dimmed below (secondary) — so the hash never
    reads as the primary identifier a user sees first."""
    return f"{description}\n[dim]{config_hash}[/dim]"
