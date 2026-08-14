"""The "Test my agent" wizard: pick a mode, pick config(s), watch progress,
land on the verdict screen. Execution goes through tui/execution.py
exactly as any other caller would use it — the only things this module
adds are the guided config-selection flow and threading run_experiment /
run_single_config_check's on_progress callback onto the UI thread via
Textual's call_from_thread, per run_experiment's documented contract that
the callback always fires on the caller's thread.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView

from attacker.attack_case import AttackCase
from attacker.case_selection import applicable_suite_for_configs
from experiments.cost_estimate import CostEstimate, estimate_batch_cost, format_cost_estimate
from experiments.persist import save_experiment_report
from experiments.runner import DEFAULT_RUNS_DIR
from target_system.config import DEFAULT_CONFIGS_DIR, SystemConfig, load_config
from tui.app import BaseScreen
from tui.data import describe_config_for_humans, list_configs
from experiments.runner import SequentialStopSpec
from tui.execution import build_anthropic_client, enforce_reconstructed_provider, peek_n_cached, run_comparison_check, run_single_config_check
from tui.formatting import (
    family_display_name,
    format_baseline_assumption,
    format_budget_option,
    format_config_list_label,
    format_run_count_option,
    format_run_count_recommendation,
)
from tui.run_sizing import (
    BudgetSizedOption,
    RunCountRecommendation,
    detectable_effect_at,
    estimated_wall_seconds,
    observed_wall_seconds_per_run,
    recommend_runs_per_case,
    size_for_budget,
)
from tui.screens.progress import WorkerProgressScreen
from tui.screens.verdict import ComparisonVerdictScreen, SingleConfigVerdictScreen
from tui.verdict_logic import compute_comparison_verdict, compute_single_config_summary

DEFAULT_N_RUNS_PER_CASE = 5
_ORDINAL_LETTERS = ["A", "B", "C", "D"]

# The outcome the wizard's early-stopping boundary and CUPED adjustment key
# off. "exfiltration" (not its _attempted variant) is the strict, actually-
# succeeded outcome — the one a user is asking about when they compare two
# configs — and it's the same key experiments/presets.py already nominates
# for sequential analysis, so a preset run and a wizard run are watching
# the same quantity rather than two different definitions of "worse".
PRIMARY_OUTCOME_KEY = "exfiltration"


class WizardModeScreen(BaseScreen):
    def __init__(self, *, runs_dir: Path = DEFAULT_RUNS_DIR, configs_dir: Path = DEFAULT_CONFIGS_DIR) -> None:
        super().__init__()
        self.runs_dir = runs_dir
        self.configs_dir = configs_dir

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label("Test my agent", classes="title"),
            Label("Choose what you want to check.", classes="subtitle"),
            ListView(
                ListItem(Label("Test a single config (no comparison)"), id="single"),
                ListItem(Label("Compare two configs"), id="comparison"),
                id="wizard-mode-menu",
            ),
            classes="wizard-body",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id == "single":
            self.app.push_screen(ConfigPickerScreen(n_needed=1, on_chosen=self._start, configs_dir=self.configs_dir))
        elif event.item.id == "comparison":
            self.app.push_screen(ConfigPickerScreen(n_needed=2, on_chosen=self._start, configs_dir=self.configs_dir))

    def _start(self, configs: list[SystemConfig]) -> None:
        # Reconstructed environments are real-model-only (Part 5) — force
        # provider="anthropic" here unconditionally, no matter what's
        # saved on disk, so this dispatch point can never hand one to the
        # mock backend (see enforce_reconstructed_provider's docstring).
        configs = [enforce_reconstructed_provider(c) for c in configs]
        mode = "single" if len(configs) == 1 else "comparison"
        # Filters out cases whose delivery vector or outcome needs a tool
        # role this environment doesn't have (attacker/applicability.py) —
        # a no-op for the toy system (its tools cover every role), load-
        # bearing for a reconstructed environment (e.g. no untrusted-
        # content-entry-point tool means corpus_document cases would
        # otherwise fail outright — see execute_case's ValueError).
        # Picks the domain-adapted suite when this environment has an
        # approved one, else the hand-authored 17 — then applies the same
        # structural applicability filter to whichever it got. The single
        # place that distinction exists; nothing downstream of here knows.
        cases, _suite_label = applicable_suite_for_configs(configs)

        def _confirm_cost(
            scoped_cases: list[AttackCase],
            n_runs_per_case: int,
            recommendation: RunCountRecommendation | None = None,
        ) -> None:
            n_cached = peek_n_cached(configs, cases=scoped_cases, runs_dir=self.runs_dir)
            estimate = estimate_batch_cost(scoped_cases, configs, n_runs_per_case=n_runs_per_case, n_cached=n_cached)
            # State plainly, at the moment money is confirmed, what this
            # spend can and cannot resolve — same ROPE power model the
            # verdict will grade the run with.
            detection_line = None
            if recommendation is not None:
                mde = detectable_effect_at(n_runs_per_case, recommendation)
                detection_line = (
                    f"At this budget, you can reliably catch differences of {100 * mde:.1f} points or "
                    "larger; anything smaller will read as INCONCLUSIVE."
                )

            def _proceed() -> None:
                self.app.push_screen(
                    WizardProgressScreen(
                        mode=mode,
                        configs=configs,
                        cases=scoped_cases,
                        n_runs_per_case=n_runs_per_case,
                        runs_dir=self.runs_dir,
                        configs_dir=self.configs_dir,
                    )
                )

            if estimate.any_real_model:
                # Never a free/instant mock option for a run that touches a
                # real model — the estimate must be shown and explicitly
                # confirmed before anything executes.
                self.app.push_screen(
                    CostConfirmScreen(estimate=estimate, on_confirm=_proceed, detection_line=detection_line)
                )
            else:
                _proceed()

        def _size(scoped_cases: list[AttackCase]) -> None:
            # Size the run before pricing it. Skipped only when there's
            # nothing to size (no applicable cases) — the cost screen still
            # runs, so a real-model batch is never reachable without an
            # explicit money confirmation either way.
            recommendation = recommend_runs_per_case(scoped_cases, configs, runs_dir=self.runs_dir)
            if recommendation is None:
                _confirm_cost(scoped_cases, DEFAULT_N_RUNS_PER_CASE)
                return
            self.app.push_screen(
                RunCountScreen(
                    recommendation=recommendation,
                    cases=scoped_cases,
                    configs=configs,
                    runs_dir=self.runs_dir,
                    on_chosen=lambda n: _confirm_cost(scoped_cases, n, recommendation=recommendation),
                )
            )

        # Scope first, then size: which families are in play determines the
        # weakest family, which is what the run count is derived from.
        # Skipped when there's no choice to make (a single applicable
        # family is not a decision, just a screen in the way).
        if len({c.family for c in cases}) > 1:
            self.app.push_screen(FamilyScopeScreen(cases=cases, on_chosen=_size))
        else:
            _size(cases)


class FamilyScopeScreen(BaseScreen):
    """Picks which attack families this particular test needs to cover,
    before the run is sized.

    Why it's a screen and not a default: sizing is driven by the *weakest*
    applicable family (see tui/run_sizing.py), so an environment that
    happens to have one 3-case family pays for that family's statistical
    requirement across the whole run — even for a user who only cares
    about direct injection. Narrowing scope is therefore the single
    biggest lever on cost, and it would be wrong to apply it silently in
    either direction: dropping a family without being asked would hide
    attacks the user assumed were covered, and this screen states plainly
    what each choice removes.

    The math is unchanged — the same required_runs_for_rope_signal over
    the same weakest-family rule, just over a narrower set."""

    def __init__(
        self,
        *,
        cases: list[AttackCase],
        on_chosen: Callable[[list[AttackCase]], None],
    ) -> None:
        super().__init__()
        self.cases = cases
        self.on_chosen = on_chosen
        self.families = sorted({c.family for c in cases})
        self.selected: set[str] = set(self.families)  # everything, until narrowed

    def _case_count(self, family: str) -> int:
        return sum(1 for c in self.cases if c.family == family)

    def _row_label(self, family: str) -> str:
        mark = "[x]" if family in self.selected else "[ ]"
        return f"{mark} {family_display_name(family)} — {self._case_count(family)} applicable case(s)"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label("Which attacks does this test need to cover?", classes="title"),
            Label("Enter toggles a family. Narrower scope costs less to run — anything you turn off is not tested at all.", classes="subtitle"),
            ListView(
                *(ListItem(Label(self._row_label(f)), id=f"family-{i}") for i, f in enumerate(self.families)),
                ListItem(Label("Continue"), id="continue"),
                id="family-scope-menu",
            ),
            classes="wizard-body",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id != "continue":
            index = int(event.item.id.removeprefix("family-"))
            family = self.families[index]
            # Never allow an empty scope: there'd be nothing to size, and
            # the run would be silently meaningless rather than obviously
            # refused.
            if family in self.selected and len(self.selected) > 1:
                self.selected.discard(family)
            else:
                self.selected.add(family)
            event.item.query_one(Label).update(self._row_label(family))
            return

        chosen = [c for c in self.cases if c.family in self.selected]
        on_chosen = self.on_chosen
        self.app.pop_screen()
        on_chosen(chosen)


@dataclass(frozen=True)
class RunCountOption:
    """One selectable run count, with everything needed to judge it: what
    it costs, how long it takes, and what it can actually detect."""

    n_runs_per_case: int
    kind: str  # "recommended" | "smaller" | "larger"
    estimate: CostEstimate
    wall_seconds: float
    detectable_effect: float
    meets_target_power: bool


class RunCountScreen(BaseScreen):
    """Chooses how many runs per case to execute, before any money is
    spent. Replaces the previous behaviour of silently running a
    hardcoded 5 (DEFAULT_N_RUNS_PER_CASE was never exposed anywhere in
    the UI) and only revealing the achieved power afterwards, on the
    verdict screen, once the run was already paid for.

    Every option states the effect it could actually detect, computed by
    the same ROPE power model (stats/hierarchical.py) the verdict screen
    grades the finished run with — so picking fewer runs is an informed
    trade ("I accept that anything under N points will read as
    INCONCLUSIVE") rather than an invisible one. 'b' cancels back without
    running, as everywhere else."""

    def __init__(
        self,
        *,
        recommendation: RunCountRecommendation,
        cases: list[AttackCase],
        configs: list[SystemConfig],
        on_chosen: Callable[[int], None],
        runs_dir: Path = DEFAULT_RUNS_DIR,
    ) -> None:
        super().__init__()
        self.recommendation = recommendation
        self.cases = cases
        self.configs = configs
        self.on_chosen = on_chosen
        self.runs_dir = runs_dir
        self.options = self._build_options()

    def _build_options(self) -> list[RunCountOption]:
        recommended = self.recommendation.recommended_runs_per_case
        n_cached = peek_n_cached(self.configs, cases=self.cases, runs_dir=self.runs_dir)
        self.n_cached = n_cached
        wall_per_run, _grounded = observed_wall_seconds_per_run(self.configs, runs_dir=self.runs_dir)

        candidates: list[tuple[int, str]] = [(recommended, "recommended")]
        # The old hardcoded default, offered as the explicit cheap option
        # rather than applied silently — only when it really is smaller.
        if DEFAULT_N_RUNS_PER_CASE < recommended:
            candidates.append((DEFAULT_N_RUNS_PER_CASE, "smaller"))
        candidates.append((recommended * 2, "larger"))

        options: list[RunCountOption] = []
        for n, kind in candidates:
            estimate = estimate_batch_cost(self.cases, self.configs, n_runs_per_case=n, n_cached=n_cached)
            options.append(
                RunCountOption(
                    n_runs_per_case=n,
                    kind=kind,
                    estimate=estimate,
                    # Jobs run concurrently (tui.execution's worker pool) —
                    # multiplying jobs by per-run seconds overstated this
                    # ~8x ("~1.5 hr" for a ~10-minute run).
                    wall_seconds=estimated_wall_seconds(estimate.n_jobs_remaining, wall_per_run),
                    detectable_effect=detectable_effect_at(n, self.recommendation),
                    meets_target_power=n >= recommended,
                )
            )
        return options

    def compose(self) -> ComposeResult:
        yield Header()
        rec = self.recommendation
        body = [
            Label("How many runs per case?", classes="title"),
            Label(format_run_count_recommendation(rec), classes="subtitle"),
            Label(format_baseline_assumption(rec), classes="hint"),
        ]
        if getattr(self, "n_cached", 0):
            body.append(
                Label(
                    f"{self.n_cached} run(s) for this comparison are already cached — the verdict always "
                    "uses every cached run, and only missing runs are executed and charged.",
                    classes="hint",
                )
            )
        yield Vertical(
            *body,
            ListView(
                *(
                    ListItem(Label(format_run_count_option(o, rec)), id=f"runs-{o.n_runs_per_case}")
                    for o in self.options
                ),
                ListItem(
                    Label("Fit a budget: state a $ or time ceiling, see what it can actually detect"),
                    id="budget",
                ),
                ListItem(Label("Cancel"), id="cancel"),
                id="run-count-menu",
            ),
            classes="wizard-body",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id == "cancel":
            self.app.pop_screen()
            return
        if event.item.id == "budget":
            self.app.push_screen(
                BudgetSizingScreen(
                    recommendation=self.recommendation,
                    cases=self.cases,
                    configs=self.configs,
                    runs_dir=self.runs_dir,
                    on_chosen=self.on_chosen,
                )
            )
            return
        n = int(event.item.id.removeprefix("runs-"))
        on_chosen = self.on_chosen
        self.app.pop_screen()
        on_chosen(n)


class BudgetSizingScreen(BaseScreen):
    """Budget-first sizing: the user states a dollar and/or minute
    ceiling, and the screen answers with the one fact that matters —
    the smallest difference that budget can reliably catch — computed by
    tui.run_sizing.size_for_budget from the real cost estimator, the
    measured per-run wall time, and the same ROPE power model the verdict
    grades with. The run count is offered only alongside that fact, never
    as a bare number, and an infeasible budget is said plainly rather
    than clamped up to a count the money cannot pay for."""

    def __init__(
        self,
        *,
        recommendation: RunCountRecommendation,
        cases: list[AttackCase],
        configs: list[SystemConfig],
        on_chosen: Callable[[int], None],
        runs_dir: Path = DEFAULT_RUNS_DIR,
    ) -> None:
        super().__init__()
        self.recommendation = recommendation
        self.cases = cases
        self.configs = configs
        self.on_chosen = on_chosen
        self.runs_dir = runs_dir
        self.option: BudgetSizedOption | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label("Size to a budget", classes="title"),
            Label(
                "Enter a cost ceiling, a time ceiling, or both — leave a field empty for no limit.",
                classes="subtitle",
            ),
            Input(placeholder="max dollars, e.g. 2.00", id="budget-usd"),
            Input(placeholder="max minutes, e.g. 5", id="budget-minutes"),
            Label("", id="budget-result", classes="hint"),
            ListView(
                ListItem(Label("Compute what this budget can detect"), id="compute"),
                ListItem(Label("Run at this budget"), id="accept"),
                ListItem(Label("Back"), id="back"),
                id="budget-menu",
            ),
            classes="wizard-body",
        )
        yield Footer()

    def _parse(self, input_id: str) -> float | None:
        raw = self.query_one(f"#{input_id}", Input).value.strip().lstrip("$")
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if value > 0 else None

    def _compute(self) -> None:
        max_usd = self._parse("budget-usd")
        max_minutes = self._parse("budget-minutes")
        result = self.query_one("#budget-result", Label)
        if max_usd is None and max_minutes is None:
            self.option = None
            result.update("Enter at least one ceiling (a positive number).")
            return
        self.option = size_for_budget(
            self.cases, self.configs, self.recommendation,
            max_usd=max_usd, max_minutes=max_minutes, runs_dir=self.runs_dir,
        )
        result.update(format_budget_option(self.option))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id == "back":
            self.app.pop_screen()
            return
        if event.item.id == "compute":
            self._compute()
            return
        # accept: compute from the current fields if not done yet, then run
        if self.option is None:
            self._compute()
        option = self.option
        if option is None or not option.feasible:
            return  # the result label already states why this can't run
        on_chosen = self.on_chosen
        self.app.pop_screen()  # this screen
        self.app.pop_screen()  # the run-count screen beneath it
        on_chosen(option.n_runs_per_case)


class CostConfirmScreen(BaseScreen):
    """Blocks on an explicit choice before any run that touches a real
    model — reconstructed environments (Part 5) have no free/instant mock
    path, so this is the one place that stands between "configs picked"
    and "money and time spent." 'b' (inherited from BaseScreen) cancels
    back to the mode screen without proceeding; only picking "Proceed"
    calls on_confirm."""

    def __init__(
        self, *, estimate: CostEstimate, on_confirm: Callable[[], None], detection_line: str | None = None
    ) -> None:
        super().__init__()
        self.estimate = estimate
        self.on_confirm = on_confirm
        # What this spend can actually detect (ROPE power model) — shown
        # beside the price so the budget and its resolving power are one
        # decision, never two screens apart. None when no sizing context
        # exists (e.g. no applicable cases were sized).
        self.detection_line = detection_line

    def compose(self) -> ComposeResult:
        yield Header()
        body = [
            Label("Confirm before running", classes="title"),
            Label(format_cost_estimate(self.estimate), classes="subtitle"),
        ]
        if self.detection_line:
            body.append(Label(self.detection_line, classes="hint"))
        yield Vertical(
            *body,
            Label("This run calls a real model and spends real money.", classes="hint"),
            ListView(
                ListItem(Label("Proceed"), id="proceed"),
                ListItem(Label("Cancel"), id="cancel"),
                id="cost-confirm-menu",
            ),
            classes="wizard-body",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id == "proceed":
            on_confirm = self.on_confirm
            self.app.pop_screen()
            on_confirm()
        else:
            self.app.pop_screen()


def _ordinal_label(index: int, total: int) -> str:
    if total == 1:
        return "a config"
    return f"config {_ORDINAL_LETTERS[index]}"


class ConfigPickerScreen(BaseScreen):
    """Picks n_needed configs one at a time from what's already saved
    (tui.data.list_configs) — every real config comes from "Add
    environment" (a reconstructed SystemConfig); there's no baseline/toy
    option here (see tui/app.py's module docstring on why). Calls
    on_chosen(configs) and pops itself once enough have been picked.

    Every row's primary label is tui.data's auto-generated description
    (never the bare SystemConfig.label a human never named for this
    purpose)."""

    def __init__(self, *, n_needed: int, on_chosen: Callable[[list[SystemConfig]], None], configs_dir: Path = DEFAULT_CONFIGS_DIR) -> None:
        super().__init__()
        self.n_needed = n_needed
        self.on_chosen = on_chosen
        self.configs_dir = configs_dir
        self.chosen: list[SystemConfig] = []

    def compose(self) -> ComposeResult:
        yield Header()
        # Fetched once, not once per call site: describe_config_for_humans
        # (via ensure_baseline_saved) persists the baseline config as a
        # side effect on its first call for a given configs_dir — calling
        # list_configs() a second time in the same render would then see
        # that freshly-saved row and (being newest by mtime) sort it
        # first, ahead of the config actually being described.
        summaries = list_configs(configs_dir=self.configs_dir)
        if not summaries:
            yield Vertical(
                Label(f"Pick {_ordinal_label(0, self.n_needed)}", classes="title"),
                Label("No environments yet.", classes="subtitle"),
                Label("Add one from the home menu (Add environment) first, then come back here.", classes="hint"),
                classes="wizard-body",
            )
        else:
            yield Vertical(
                Label(f"Pick {_ordinal_label(0, self.n_needed)}", id="picker-title", classes="title"),
                ListView(*self._build_items(summaries), id="config-picker-list"),
                classes="wizard-body",
            )
        yield Footer()

    def _build_items(self, summaries=None) -> list[ListItem]:
        if summaries is None:
            summaries = list_configs(configs_dir=self.configs_dir)
        return [
            ListItem(Label(format_config_list_label(summary.description, summary.config_hash)), id=summary.config_hash)
            for summary in summaries
        ]

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        key = event.item.id
        config = load_config(key, configs_dir=self.configs_dir)
        self.chosen.append(config)
        if len(self.chosen) >= self.n_needed:
            on_chosen, chosen = self.on_chosen, self.chosen
            self.app.pop_screen()
            on_chosen(chosen)
        else:
            self.query_one("#picker-title", Label).update(f"Pick {_ordinal_label(len(self.chosen), self.n_needed)}")
            list_view = self.query_one("#config-picker-list", ListView)
            await list_view.clear()  # clear() returns an awaitable — appending before it lands races and can duplicate IDs
            for item in self._build_items():
                await list_view.append(item)
            list_view.index = 0
            list_view.focus()


class WizardProgressScreen(WorkerProgressScreen):
    """Runs the check in a background thread so the UI stays responsive,
    then lands on the appropriate verdict screen. mode is "single" (one
    config, tui.execution.run_single_config_check) or "comparison" (two
    configs, tui.execution.run_comparison_check — a thin pass-through to
    experiments.runner.run_experiment)."""

    title_text = "Running attack suite..."

    def __init__(
        self,
        *,
        mode: str,
        configs: list[SystemConfig],
        cases: list[AttackCase] | None = None,
        n_runs_per_case: int = DEFAULT_N_RUNS_PER_CASE,
        runs_dir: Path = DEFAULT_RUNS_DIR,
        configs_dir: Path = DEFAULT_CONFIGS_DIR,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.configs = configs
        self.cases = cases
        self.n_runs_per_case = n_runs_per_case
        self.runs_dir = runs_dir
        self.configs_dir = configs_dir

    def _execute(self) -> None:
        # Only reconstructed/real-model configs need a live client — build
        # it here (not earlier) so a toy-only run never requires
        # ANTHROPIC_API_KEY to be set at all.
        anthropic_client = build_anthropic_client() if any(c.model.provider == "anthropic" for c in self.configs) else None
        if self.mode == "single":
            result = run_single_config_check(
                self.configs[0], cases=self.cases, n_runs_per_case=self.n_runs_per_case, runs_dir=self.runs_dir,
                on_progress=self._on_progress, anthropic_client=anthropic_client,
            )
            records = [r.model_dump(mode="json") for r in result.records]
            summary = compute_single_config_summary(
                records,
                config_label=describe_config_for_humans(result.config_hash, configs_dir=self.configs_dir),
                config_hash=result.config_hash,
            )
            self.app.call_from_thread(self._land_single, summary, records)
        else:
            config_a, config_b = self.configs
            result = run_comparison_check(
                config_a, config_b, cases=self.cases, n_runs_per_case=self.n_runs_per_case, runs_dir=self.runs_dir,
                on_progress=self._on_progress, anthropic_client=anthropic_client,
                # ROPE rule (the live default), watching both base
                # outcomes: the run stops as soon as each has confidently
                # resolved either beyond or inside the practical-
                # equivalence band. Monitoring only the primary key would
                # have missed the one real finding to date (unauthorized_
                # lookup moving while exfiltration stayed flat).
                sequential_stop=SequentialStopSpec(
                    outcome_key=PRIMARY_OUTCOME_KEY, extra_outcome_keys=("unauthorized_lookup",)
                ),
            )
            report_path = save_experiment_report(
                result,
                sequential_outcome_key=PRIMARY_OUTCOME_KEY,
                cuped_outcome_key=PRIMARY_OUTCOME_KEY,
                runs_dir=self.runs_dir,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            verdict = compute_comparison_verdict(report)
            records = [r.model_dump(mode="json") for r in result.records]
            self.app.call_from_thread(self._land_comparison, verdict, report, result.name, records)

    def _land_single(self, summary, records) -> None:
        self.app.pop_screen()
        self.app.push_screen(SingleConfigVerdictScreen(summary, records=records, configs_dir=self.configs_dir))

    def _land_comparison(self, verdict, report, name, records) -> None:
        self.app.pop_screen()
        self.app.push_screen(ComparisonVerdictScreen(verdict, report, name, records, configs_dir=self.configs_dir))
