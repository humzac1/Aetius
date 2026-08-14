"""Review and approve domain-adapted attack cases before they can ever run
against a real system.

Same spirit as CostConfirmScreen (tui/screens/wizard.py): the thing that's
about to happen is shown in full, and it does not happen until someone
explicitly says so. There the stake is money; here it's what gets sent to a
real agent under the name "security test" — a generated case that reads
plausibly but tests nothing is worse than no case at all, because it
produces a confident number. So the generated set starts unapproved, the
run path ignores unapproved sets entirely (attacker/case_selection.py), and
this screen is the only thing that flips that bit.

Every row shows the generated task text and the hand-authored template it
came from, so a reviewer can see what changed and judge whether the
adaptation kept the attack's intent. Cases that failed the coherence guard
(attacker/case_generation.check_case_coherence) are shown too, marked, and
excluded from what approval enables — surfacing them is the point: a
silently dropped case would hide that generation is going wrong.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, Header, Label, ListItem, ListView

from attacker.case_generation import DEFAULT_GENERATION_MODEL, generate_case_batch
from attacker.case_store import (
    DEFAULT_GENERATED_CASES_DIR,
    GeneratedCaseSet,
    approve_generated_cases,
    load_generated_cases,
    save_generated_cases,
)
from attacker.cases import ATTACK_CASES
from target_system.config import DEFAULT_CONFIGS_DIR, SystemConfig, compute_config_hash
from tui.app import BaseScreen
from tui.execution import build_anthropic_client
from tui.screens.progress import WorkerProgressScreen


def format_generated_case(entry, index: int) -> str:
    """One reviewable row: what will be sent, and what it was adapted from."""
    case = entry.case
    mark = "" if entry.coherent else "[!] FAILED COHERENCE CHECK — will not run\n"
    turns = f"\n  turns: {len(case.turns)}" if case.turns else ""
    return (
        f"{mark}{index}. {case.family} / {case.injection_vector} -> {case.success_outcome}\n"
        f"  task: {case.benign_task}\n"
        f"  payload: {case.injected_payload}{turns}\n"
        f"[dim]  adapted from {entry.template_id}: {entry.template_benign_task}\n"
        f"  check: {entry.coherence_reason}[/dim]"
    )


class GenerateCasesScreen(WorkerProgressScreen):
    """Runs generation in a worker thread — one model call per template, so
    a full seed set is a real wait."""

    title_text = "Generating domain-adapted attack cases..."

    def __init__(
        self,
        *,
        config: SystemConfig,
        templates=None,
        plan: list[tuple] | None = None,
        append_to_existing: bool = False,
        cases_dir: Path = DEFAULT_GENERATED_CASES_DIR,
        model: str = DEFAULT_GENERATION_MODEL,
    ) -> None:
        super().__init__()
        self.config = config
        self.templates = list(templates) if templates is not None else list(ATTACK_CASES)
        # plan: (template, variant) pairs, for topping an existing set up
        # toward the calibrated case floor rather than generating one case
        # per template.
        self.plan = list(plan) if plan is not None else [(t, 0) for t in self.templates]
        self.append_to_existing = append_to_existing
        self.cases_dir = cases_dir
        self.model = model

    def _execute(self) -> None:
        client = build_anthropic_client()
        total = len(self.plan)
        self._on_progress(0, total)
        batch = generate_case_batch(
            self.templates, self.config, anthropic_client=client, model=self.model,
            plan=self.plan, on_progress=self._on_progress,
        )
        entries = list(batch.entries)
        self.failures = list(batch.failures)

        config_hash = compute_config_hash(self.config)
        if self.append_to_existing:
            # Topping up must not discard the batch already reviewed and
            # approved — those ids are what previous runs were recorded
            # against. The combined set goes back to unapproved, so the
            # additions get reviewed before anything runs against them.
            existing = load_generated_cases(config_hash, cases_dir=self.cases_dir)
            if existing is not None:
                entries = list(existing.entries) + entries
        save_generated_cases(
            config_hash,
            entries,
            model=self.model,
            generated_at=datetime.now(timezone.utc).isoformat(),
            approved=False,  # never self-approving; ReviewGeneratedCasesScreen does that
            cases_dir=self.cases_dir,
        )
        case_set = load_generated_cases(config_hash, cases_dir=self.cases_dir)
        self.app.call_from_thread(self._land, case_set)

    def _land(self, case_set: GeneratedCaseSet) -> None:
        self.app.switch_screen(ReviewGeneratedCasesScreen(case_set=case_set, cases_dir=self.cases_dir))


class ReviewGeneratedCasesScreen(BaseScreen):
    """Shows every generated case, then asks. Approval is the only path by
    which these cases become runnable."""

    def __init__(
        self,
        *,
        case_set: GeneratedCaseSet,
        cases_dir: Path = DEFAULT_GENERATED_CASES_DIR,
        configs_dir: Path = DEFAULT_CONFIGS_DIR,
    ) -> None:
        super().__init__()
        self.case_set = case_set
        self.cases_dir = cases_dir
        self.configs_dir = configs_dir

    def compose(self) -> ComposeResult:
        yield Header()
        entries = self.case_set.entries
        n_coherent = sum(1 for e in entries if e.coherent)
        n_failed = len(entries) - n_coherent

        summary = f"{len(entries)} case(s) generated for {self.case_set.config_hash} using {self.case_set.model}."
        if n_failed:
            summary += f" {n_failed} failed the coherence check and will not run."

        yield Vertical(
            Label("Review generated attack cases", classes="title"),
            Label(summary, classes="subtitle"),
            Label(
                "These have not run against anything yet. Nothing executes until you approve them.",
                classes="hint",
            ),
            VerticalScroll(
                *(Label(format_generated_case(e, i), classes="hint") for i, e in enumerate(entries, start=1)),
                id="generated-case-list",
            ),
            ListView(
                ListItem(Label(f"Approve these {n_coherent} case(s) for use"), id="approve"),
                ListItem(Label("Discard — do not use them"), id="discard"),
                id="generated-cases-menu",
            ),
            classes="wizard-body",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id == "approve":
            approve_generated_cases(self.case_set.config_hash, cases_dir=self.cases_dir)
        self.app.pop_screen()
