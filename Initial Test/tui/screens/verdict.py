"""Verdict screens: single-config check results and the three comparison
tiers (FLAGGED/CLEAR/INCONCLUSIVE), plus the statistics drill-down every
comparison verdict can open with 's'. All numbers come from
tui/verdict_logic.py (which calls stats/hierarchical.py's ROPE power
model live) and are phrased by tui/formatting.py — this module only lays
out widgets, it never computes a tier or formats a number itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Label, ListItem, ListView, Static

from attacker.case_generation import estimate_generation_cost, plan_variants
from attacker.case_store import load_generated_cases
from attacker.cases import ATTACK_CASES
from target_system.config import DEFAULT_CONFIGS_DIR, SystemConfig, compute_config_hash, load_config
from tui.app import BaseScreen
from tui.dashboard_link import open_dashboard_for_run
from tui.data import describe_comparison_for_humans, describe_config_for_humans
from tui.formatting import (
    DRILL_DOWN_COLUMNS,
    SINGLE_CONFIG_DISCLAIMER,
    SYSTEM_PROMPT_UNAVAILABLE_DISCLOSURE,
    build_drill_down_rows,
    family_display_name,
    format_attempted_breakdown,
    format_clear_summary,
    format_flagged_ci,
    format_flagged_headline,
    format_flagged_synthetic_evidence_note,
    format_generation_offer,
    format_inconclusive_summary,
    format_other_flagged_note,
    format_other_groups_found_note,
    format_provenance_disclosure,
    format_response_source_summary,
    format_single_config_headline,
)
from tui.verdict_logic import (
    ComparisonVerdict,
    SingleConfigSummary,
    compute_attempted_executed_counts,
    compute_overall_response_source_breakdown,
    compute_response_source_breakdown,
)


def _variant_of(case_id: str) -> int:
    """Variant index encoded in a generated case id ({template}__{hash} or
    {template}__{hash}__v{k}). 0 for the first batch, which carries no
    suffix. Used to continue numbering rather than colliding with ids
    already on disk — a reused id would silently merge two different
    cases in the paired statistics."""
    tail = case_id.rsplit("__", 1)[-1]
    if tail.startswith("v") and tail[1:].isdigit():
        return int(tail[1:])
    return 0


def _try_load_config(config_hash: str, *, configs_dir: Path) -> SystemConfig | None:
    """Best-effort: verdict screens must still render (and every existing
    non-fidelity element must still work) when a config_hash doesn't
    resolve to anything on disk -- ad hoc test fixtures, or a config that
    was since deleted. None here just means "skip the fidelity block",
    never a crash."""
    try:
        return load_config(config_hash, configs_dir=configs_dir)
    except FileNotFoundError:
        return None


def _compose_fidelity_block(configs: list[SystemConfig]) -> ComposeResult:
    """Reconstructed-environment fidelity disclosure (Part 6) — inline and
    prominent on any verdict screen that involves a config with
    provenance is not None, same discipline as SINGLE_CONFIG_DISCLAIMER.
    Shared between SingleConfigVerdictScreen (one config) and
    ComparisonVerdictScreen (two arms, either or both possibly
    reconstructed) so the wording/ordering can't drift between them.
    response_source breakdown is NOT included here — it needs the run's
    raw records, which callers pass separately via
    compute_overall_response_source_breakdown and yield themselves."""
    reconstructed = [c for c in configs if c.provenance is not None]
    if not reconstructed:
        return
    for config in reconstructed:
        yield Label(format_provenance_disclosure(config.provenance), classes="disclaimer")
        other_note = format_other_groups_found_note(config.provenance)
        if other_note is not None:
            yield Label(other_note, classes="disclaimer")
    unavailable_agents = [a.name for c in reconstructed for a in c.agents if a.system_prompt_source == "unavailable"]
    if unavailable_agents:
        yield Label(SYSTEM_PROMPT_UNAVAILABLE_DISCLOSURE, classes="disclaimer")


class SingleConfigVerdictScreen(BaseScreen):
    """"Test my agent" with no comparison arm — a raw tally, no
    statistical language, and the disclaimer below is a plain Static
    always in the layout (never inside a Collapsible), so it can't be
    dismissed or hidden."""

    def __init__(
        self,
        summary: SingleConfigSummary,
        *,
        records: list[dict[str, Any]] | None = None,
        configs_dir: Path = DEFAULT_CONFIGS_DIR,
    ) -> None:
        super().__init__()
        self.summary = summary
        self.records = records or []
        self.configs_dir = configs_dir

    def compose(self) -> ComposeResult:
        yield Header()
        config = _try_load_config(self.summary.config_hash, configs_dir=self.configs_dir)
        with Vertical(classes="verdict-body"):
            yield Label(f"Single-config check — {self.summary.config_label}", classes="title")
            yield Label(self.summary.config_hash, classes="hint")
            yield Label(format_single_config_headline(self.summary), classes="subtitle")
            table = DataTable(id="family-table")
            table.add_columns("Family", "Total", "Succeeded", "Blocked", "Resisted")
            for fam in self.summary.by_family:
                table.add_row(family_display_name(fam.family), str(fam.total), str(fam.succeeded), str(fam.blocked), str(fam.resisted))
            yield table
            if config is not None and config.provenance is not None:
                yield from _compose_fidelity_block([config])
                overall_breakdown = compute_overall_response_source_breakdown(self.records)
                if overall_breakdown is not None:
                    yield Label(format_response_source_summary(overall_breakdown), classes="hint")
        yield Static(SINGLE_CONFIG_DISCLAIMER, classes="disclaimer", id="disclaimer")
        yield Footer()


class ComparisonVerdictScreen(BaseScreen):
    BINDINGS = BaseScreen.BINDINGS + [
        Binding("s", "show_statistics", "Statistics"),
        Binding("d", "open_dashboard", "Open dashboard"),
    ]

    def __init__(
        self,
        verdict: ComparisonVerdict,
        report: dict[str, Any],
        experiment_name: str,
        records: list[dict[str, Any]] | None = None,
        *,
        configs_dir: Path = DEFAULT_CONFIGS_DIR,
    ) -> None:
        super().__init__()
        self.verdict = verdict
        self.report = report
        self.experiment_name = experiment_name
        self.records = records or []
        self.configs_dir = configs_dir

    def compose(self) -> ComposeResult:
        yield Header()
        config_a = _try_load_config(self.report.get("arm_a_hash", ""), configs_dir=self.configs_dir)
        config_b = _try_load_config(self.report.get("arm_b_hash", ""), configs_dir=self.configs_dir)
        configs = [c for c in (config_a, config_b) if c is not None]
        with Vertical(classes="verdict-body"):
            name = describe_comparison_for_humans(self.report, self.experiment_name, configs_dir=self.configs_dir)
            yield Label(f"Comparison check — {name}", classes="title")
            yield Label(f"run: {self.experiment_name}", classes="hint")
            yield from _compose_fidelity_block(configs)
            overall_breakdown = compute_overall_response_source_breakdown(self.records)
            if any(c.provenance is not None for c in configs) and overall_breakdown is not None:
                yield Label(format_response_source_summary(overall_breakdown), classes="hint")
            if self.verdict.tier == "FLAGGED":
                yield from self._compose_flagged(config_a, config_b)
            elif self.verdict.tier == "CLEAR":
                yield from self._compose_clear()
            else:
                yield from self._compose_inconclusive()
        yield Footer()

    def _compose_flagged(self, config_a: SystemConfig | None, config_b: SystemConfig | None) -> ComposeResult:
        yield Label("🚩 FLAGGED — a real effect was found", classes="tier-flagged")
        yield Label(format_flagged_headline(self.verdict))
        yield Label(format_flagged_ci(self.verdict))

        flagged_config = config_a if self.verdict.flagged_arm_label == self.report.get("arm_a_label") else config_b
        if flagged_config is not None:
            # Hash the resolved config object directly rather than picking
            # report["arm_a_hash"]/["arm_b_hash"] through the same
            # flagged_arm_label comparison used just above -- that
            # comparison is ambiguous whenever arm_a_label == arm_b_label
            # (a separate, pre-existing issue from this fix), so deriving
            # the hash from flagged_config itself is correct regardless.
            flagged_arm_hash = compute_config_hash(flagged_config)
            counts = compute_attempted_executed_counts(
                self.records,
                arm_hash=flagged_arm_hash,
                arm_label=self.verdict.flagged_arm_label,
                family=self.verdict.flagged_family,
                base_outcome_key=self.verdict.flagged_outcome_key,
                config=flagged_config,
            )
            breakdown = format_attempted_breakdown(counts)
            if breakdown is not None:
                yield Label(breakdown, classes="breakdown")

            if flagged_config.provenance is not None:
                source_breakdown = compute_response_source_breakdown(
                    self.records,
                    arm_hash=flagged_arm_hash,
                    arm_label=self.verdict.flagged_arm_label,
                    family=self.verdict.flagged_family,
                    base_outcome_key=self.verdict.flagged_outcome_key,
                    config=flagged_config,
                )
                synthetic_note = format_flagged_synthetic_evidence_note(source_breakdown)
                if synthetic_note is not None:
                    yield Label(synthetic_note, classes="disclaimer")

        other_note = format_other_flagged_note(self.verdict)
        if other_note is not None:
            yield Label(other_note, classes="subtitle")

    def _compose_clear(self) -> ComposeResult:
        # Internally the tier stays "CLEAR"; every place a person reads it
        # says what was actually established: no difference was detected,
        # by a run with the power to have detected one.
        yield Label("✅ No difference detected — and this run had the power to see one", classes="tier-clear")
        for line in format_clear_summary(self.verdict):
            yield Label(line)

    def _compose_inconclusive(self) -> ComposeResult:
        yield Label("❓ INCONCLUSIVE — not enough data to tell", classes="tier-inconclusive")
        for line in format_inconclusive_summary(self.verdict):
            yield Label(line)
        yield from self._compose_generation_offer()

    def _compose_generation_offer(self) -> ComposeResult:
        """Offered only when generating cases would actually resolve the
        refusal. `can_generate_more_cases` is a structured check on the
        refusal kind, never a match on the message text — a degenerate
        refusal must never reach this, because no case count fixes it and
        the offer would be a promise the measured sweep contradicts."""
        if not self.verdict.can_generate_more_cases:
            return
        config = _try_load_config(self.report.get("arm_a_hash", ""), configs_dir=self.configs_dir)
        if config is None or config.provenance is None:
            # Generation is domain-adaptation for a reconstructed
            # environment; there's nothing to adapt to for a toy config.
            return

        templates = [c for c in ATTACK_CASES if c.family == self.verdict.refused_family]
        if not templates:
            return
        estimate = estimate_generation_cost(templates, config, self.verdict.refused_cases_needed)
        yield Label(format_generation_offer(self.verdict, estimate), classes="hint")
        yield ListView(
            ListItem(Label(f"Generate {self.verdict.refused_cases_needed} more case(s)"), id="generate-more"),
            ListItem(Label("Not now"), id="skip-generation"),
            id="generation-offer-menu",
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id != "generate-more":
            return
        config = _try_load_config(self.report.get("arm_a_hash", ""), configs_dir=self.configs_dir)
        if config is None:
            return
        from tui.screens.generated_cases import GenerateCasesScreen

        templates = [c for c in ATTACK_CASES if c.family == self.verdict.refused_family]
        existing = load_generated_cases(compute_config_hash(config))
        start_variant = 1 + max(
            (_variant_of(e.case.id) for e in existing.entries), default=0
        ) if existing else 0
        self.app.push_screen(
            GenerateCasesScreen(
                config=config,
                plan=plan_variants(templates, self.verdict.refused_cases_needed, start_variant=start_variant),
                append_to_existing=True,
            )
        )

    def action_show_statistics(self) -> None:
        self.app.push_screen(
            StatisticsDrillDownScreen(self.report, self.experiment_name, records=self.records, configs_dir=self.configs_dir)
        )

    def action_open_dashboard(self) -> None:
        opened, url = open_dashboard_for_run(self.experiment_name)
        if opened:
            self.notify(f"Opened {url}")
        else:
            self.notify(
                f"Dashboard isn't running. Start it with `streamlit run dashboard/app.py`, then open {url}",
                severity="warning",
                timeout=10,
            )


class StatisticsDrillDownScreen(BaseScreen):
    """Every family's effect size, interval (credible for the live
    method, frequentist CI for older saved reports), BH-adjusted q, and
    the per-row flag decision, straight from the saved report — the
    non-default 's' destination every verdict screen can reach, and a
    hotkey to open the real dashboard instead of reimplementing any of
    its charts here."""

    BINDINGS = BaseScreen.BINDINGS + [Binding("d", "open_dashboard", "Open dashboard")]

    def __init__(
        self,
        report: dict[str, Any],
        experiment_name: str,
        *,
        records: list[dict[str, Any]] | None = None,
        configs_dir: Path = DEFAULT_CONFIGS_DIR,
    ) -> None:
        super().__init__()
        self.report = report
        self.experiment_name = experiment_name
        self.records = records or []
        self.configs_dir = configs_dir

    def compose(self) -> ComposeResult:
        yield Header()
        desc_a = describe_config_for_humans(self.report.get("arm_a_hash", ""), configs_dir=self.configs_dir)
        desc_b = describe_config_for_humans(self.report.get("arm_b_hash", ""), configs_dir=self.configs_dir)
        config_a = _try_load_config(self.report.get("arm_a_hash", ""), configs_dir=self.configs_dir)
        config_b = _try_load_config(self.report.get("arm_b_hash", ""), configs_dir=self.configs_dir)
        configs = [c for c in (config_a, config_b) if c is not None]
        with Vertical(classes="verdict-body"):
            yield Label(f"Statistics — {self.experiment_name}", classes="title")
            yield Label(
                f"arm A = {desc_a}  •  arm B = {desc_b}  •  "
                f"{self.report.get('n_cases')} cases x {self.report.get('n_runs_per_case')} runs/case/arm",
                classes="subtitle",
            )
            table = DataTable(id="drill-down-table")
            table.add_columns(*DRILL_DOWN_COLUMNS)
            for row in build_drill_down_rows(self.report, records=self.records, configs=configs):
                table.add_row(*row)
            yield table
        yield Footer()

    def action_open_dashboard(self) -> None:
        opened, url = open_dashboard_for_run(self.experiment_name)
        if opened:
            self.notify(f"Opened {url}")
        else:
            self.notify(
                f"Dashboard isn't running. Start it with `streamlit run dashboard/app.py`, then open {url}",
                severity="warning",
                timeout=10,
            )
