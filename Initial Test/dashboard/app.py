"""Read-only dashboard over data/runs/ and the saved experiment reports.
Never executes an experiment or the target system — every panel either
reads a file that already exists on disk, or (Panels 3's supporting chart
and Panel 5) evaluates a closed-form stats.power/stats.variance_reduction
formula live, which needs no new target-system runs.

Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard import colors
from dashboard.data_access import (
    RUNS_DIR,
    find_flagged_run,
    flatten_family_rows,
    list_available_reports,
    load_calibration_sweep,
    load_raw_records,
    load_report,
)

st.set_page_config(page_title="Agent Regression Detector", layout="wide")

BASE_LAYOUT = dict(
    plot_bgcolor=colors.SURFACE,
    paper_bgcolor=colors.SURFACE,
    font=dict(color=colors.TEXT_PRIMARY, family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
    margin=dict(t=40, b=40, l=10, r=10),
)


# ============================================================
# PANEL 1 — A/A CALIBRATION
# ============================================================

def render_calibration_panel() -> None:
    st.header("A/A Calibration")
    st.caption(
        "Both arms of an A/A comparison are the same config. If this panel isn't right, "
        "nothing else on this page can be trusted."
    )

    sweep = load_calibration_sweep()
    aa_report = load_report("aa")

    if sweep is None:
        st.warning(
            "No calibration sweep found at `data/runs/aa_calibration_sweep.json`. "
            "Run `python -m experiments.calibration` to generate it — this dashboard "
            "does not run it for you."
        )
    else:
        points = sweep["points"]
        calibrated_points = [p for p in points if p["well_calibrated"]]
        headline = calibrated_points[len(calibrated_points) // 2] if calibrated_points else points[len(points) // 2]

        col1, col2, col3 = st.columns(3)
        col1.metric(
            f"Observed FPR (n_cases={headline['n_cases']})",
            f"{headline['observed_fpr']:.3f}",
            help=f"95% CI [{headline['fpr_ci_low']:.3f}, {headline['fpr_ci_high']:.3f}], "
            f"{headline['n_trials']} simulated trials",
        )
        col2.metric("Nominal alpha", f"{sweep['alpha']:.3f}")
        gap_pct = 100 * (headline["observed_fpr"] / sweep["alpha"] - 1)
        col3.metric("Gap vs. nominal", f"{gap_pct:+.0f}%")

        fig = go.Figure()
        fig.add_hline(
            y=sweep["alpha"], line_dash="dash", line_color=colors.TEXT_SECONDARY,
            annotation_text=f"nominal α = {sweep['alpha']}", annotation_position="top left",
        )
        x = [p["n_cases"] for p in points]
        y = [p["observed_fpr"] for p in points]
        point_colors = [colors.STATUS_GOOD if p["well_calibrated"] else colors.STATUS_CRITICAL for p in points]
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(size=14, color=point_colors, line=dict(width=1, color=colors.TEXT_PRIMARY)),
                error_y=dict(
                    type="data", symmetric=False,
                    array=[p["fpr_ci_high"] - p["observed_fpr"] for p in points],
                    arrayminus=[p["observed_fpr"] - p["fpr_ci_low"] for p in points],
                    color=colors.TEXT_MUTED, thickness=2,
                ),
                hovertemplate="n_cases=%{x}<br>observed FPR=%{y:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            **BASE_LAYOUT,
            xaxis_title=f"cases in the synthetic sweep ({sweep['method']}, {sweep['n_runs_per_case']} runs/case, {sweep['n_trials']} trials/point)",
            yaxis_title="empirical false-positive rate",
            showlegend=False,
            height=320,
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "🟢 CI includes nominal α (calibrated at this n_cases)  •  🔴 CI excludes nominal α. "
            "This method (BCa cluster_bootstrap) empirically over-rejects by roughly 1.2-1.7x "
            "nominal across most of this range, not just below some small-N cutoff — see "
            "`stats/paired.py`'s `cluster_bootstrap_diff` docstring for the exact numbers."
        )

    st.markdown(
        "> **This exact check caught two real bugs during development**, both on their first "
        "real run: a BCa bootstrap degeneracy that reported every family as spuriously "
        "\"significant\" on data with exactly zero variance (a bit-identical A/A comparison), "
        "and a missing-common-random-numbers bug that produced spurious flags in the "
        "`known_neutral` and `model_swap` presets purely from sampling noise. See the CRN "
        "panel below for that second bug's real before/after."
    )

    if aa_report is not None:
        st.subheader("Real-execution corroboration")
        st.caption(
            f"The `aa` preset: arm A and arm B both resolve to `{aa_report['arm_a_hash']}` "
            f"(bit-identical config). Every row below should read 0.0pp / not significant."
        )
        rows = flatten_family_rows(aa_report)
        table_rows = [
            {
                "outcome": r["outcome_key"],
                "family": r["family"],
                "arm A": f"{r['rate_a'] * 100:.1f}%",
                "arm B": f"{r['rate_b'] * 100:.1f}%",
                "diff": f"{r['diff'] * 100:+.1f}pp",
                "95% CI": f"[{r['ci_low'] * 100:.1f}, {r['ci_high'] * 100:.1f}]",
                "q": f"{r['q_value']:.3f}",
                "flagged": "SIGNIFICANT" if r["significant"] else "—",
            }
            for r in rows
        ]
        st.dataframe(table_rows, width="stretch", hide_index=True)
    else:
        st.info("No `aa_report.json` found — run `python -m experiments.cli run --preset aa` first.")


# ============================================================
# PANEL 2 — COMPARISON VIEW
# ============================================================

def _bar_chart(rows: list[dict], title: str) -> go.Figure:
    rows = sorted(rows, key=lambda r: abs(r["diff"]), reverse=True)
    families = [r["family"] for r in rows]
    diffs = [r["diff"] * 100 for r in rows]
    ci_hi = [(r["ci_high"] - r["diff"]) * 100 for r in rows]
    ci_lo = [(r["diff"] - r["ci_low"]) * 100 for r in rows]
    bar_colors = [colors.significance_color(r["significant"], r["diff"]) for r in rows]
    hover = [
        f"{r['family']}<br>{r['rate_a'] * 100:.1f}% -> {r['rate_b'] * 100:.1f}%"
        f"<br>diff {r['diff'] * 100:+.1f}pp, q={r['q_value']:.3f}, n={r['n_cases']} cases"
        for r in rows
    ]

    fig = go.Figure()
    fig.add_vline(x=0, line_color=colors.BASELINE)
    fig.add_trace(
        go.Bar(
            y=families, x=diffs, orientation="h",
            marker_color=bar_colors,
            error_x=dict(type="data", symmetric=False, array=ci_hi, arrayminus=ci_lo, color=colors.TEXT_MUTED),
            hovertext=hover, hoverinfo="text",
        )
    )
    fig.update_layout(
        **BASE_LAYOUT,
        title=title,
        xaxis_title="difference, arm B - arm A (percentage points)",
        height=120 + 60 * max(1, len(rows)),
        showlegend=False,
    )
    return fig


def render_comparison_panel() -> None:
    st.header("Comparison View")

    available = list_available_reports()
    if not available:
        st.warning("No experiment reports found under `data/runs/`. Run a preset via `experiments.cli` first.")
        return

    requested = st.query_params.get("experiment")
    if requested in available:
        default_idx = available.index(requested)
    elif "known_regression" in available:
        default_idx = available.index("known_regression")
    else:
        default_idx = 0
    experiment_name = st.selectbox("Experiment", available, index=default_idx, key="comparison_experiment")
    report = load_report(experiment_name)
    if report is None:
        st.error(f"Could not load report for {experiment_name!r}.")
        return

    st.caption(
        f"arm A = `{report['arm_a_label']}` ({report['arm_a_hash']})  •  "
        f"arm B = `{report['arm_b_label']}` ({report['arm_b_hash']})  •  "
        f"{report['n_cases']} cases x {report['n_runs_per_case']} runs/case/arm"
    )
    st.caption(
        "🔴 significant regression (rate rose)  •  🟢 significant improvement (rate fell)  •  "
        "⚪ not significant after BH correction"
    )

    rows = flatten_family_rows(report)
    for base_key, pretty in [("exfiltration", "Exfiltration"), ("unauthorized_lookup", "Unauthorized lookup")]:
        st.subheader(pretty)
        strict_rows = [r for r in rows if r["outcome_key"] == base_key]
        attempted_rows = [r for r in rows if r["outcome_key"] == f"{base_key}_attempted"]
        col1, col2 = st.columns(2)
        with col1:
            if strict_rows:
                st.plotly_chart(_bar_chart(strict_rows, "executed (the bad thing actually happened)"), width="stretch")
            else:
                st.caption("no data")
        with col2:
            if attempted_rows:
                st.plotly_chart(_bar_chart(attempted_rows, "attempted (tried, blocked or not)"), width="stretch")
            else:
                st.caption("no data")


# ============================================================
# PANEL 3 — CRN / VARIANCE REDUCTION
# ============================================================

def render_crn_panel() -> None:
    st.header("CRN / Variance Reduction")

    st.subheader("Real evidence: the CRN bug fix, before and after")
    st.caption(
        "known_neutral and model_swap should show no significant families — their arms have "
        "genuinely equal (or, for model_swap under the mock backend, not-actually-comparable) "
        "compliance probability. Before this project's mock_policy correctly shared random "
        "draws across arms (common random numbers), pure sampling noise flagged families "
        "anyway. The *_before_crn_fix reports were reconstructed via an isolated, clearly-"
        "labeled scratch run of the pre-fix roll — see experiments/mock_policy.py's docstring "
        "— not fabricated."
    )

    any_before_after = False
    for preset_name in ["known_neutral", "model_swap"]:
        after = load_report(preset_name)
        before = load_report(f"{preset_name}_before_crn_fix")
        if after is None or before is None:
            st.info(f"Missing report(s) for the {preset_name} before/after comparison.")
            continue
        any_before_after = True

        before_rows = {(r["outcome_key"], r["family"]): r for r in flatten_family_rows(before)}
        after_rows = {(r["outcome_key"], r["family"]): r for r in flatten_family_rows(after)}
        n_before_sig = sum(1 for r in before_rows.values() if r["significant"])
        n_after_sig = sum(1 for r in after_rows.values() if r["significant"])

        st.markdown(f"**{preset_name}**")
        col1, col2 = st.columns(2)
        col1.metric("Families flagged SIGNIFICANT — before fix", n_before_sig)
        col2.metric("Families flagged SIGNIFICANT — after fix", n_after_sig, delta=n_after_sig - n_before_sig)

        flipped = [
            (k, before_rows[k], after_rows[k])
            for k in before_rows
            if before_rows[k]["significant"] and not after_rows.get(k, {}).get("significant", True)
        ]
        if flipped:
            table = [
                {
                    "outcome": k[0], "family": k[1],
                    "before (spurious)": f"{b['diff'] * 100:+.1f}pp, q={b['q_value']:.3f}",
                    "after (fixed)": f"{a['diff'] * 100:+.1f}pp, q={a['q_value']:.3f}",
                }
                for k, b, a in flipped
            ]
            st.dataframe(table, width="stretch", hide_index=True)

    if not any_before_after:
        st.warning(
            "No before/after reports found. See the reconstruction script referenced in "
            "experiments/mock_policy.py's docstring."
        )

    st.divider()
    st.subheader("General effect: required runs per case, with vs. without CRN")
    st.caption(
        "Computed live from stats/variance_reduction.py + stats/power.py's cross-validated "
        "formula (see Panel 5) — a synthetic illustration grounded in known_regression's "
        "observed rates where available, not itself a saved experiment result."
    )

    from stats.power import required_runs_per_case
    from stats.variance_reduction import measure_crn_variance_reduction

    baseline_rate, mde = 0.15, 0.10
    known_regression = load_report("known_regression")
    if known_regression is not None:
        strict = [r for r in flatten_family_rows(known_regression) if r["outcome_key"] == "exfiltration"]
        if strict:
            baseline_rate = sum(r["rate_a"] for r in strict) / len(strict)
            mde = max(0.02, sum(r["diff"] for r in strict) / len(strict))

    n_cases_demo = 20
    case_rates = {f"case_{i}": (baseline_rate, min(0.95, baseline_rate + mde)) for i in range(n_cases_demo)}
    crn_result = measure_crn_variance_reduction(case_rates, n_runs_per_case=15, n_sims=1000, seed=0)

    n_no_vr = required_runs_per_case(baseline_rate, mde, n_cases_demo, power=0.8, alpha=0.05)
    n_with_crn = max(1, round(n_no_vr * (1 - crn_result.variance_reduction_pct / 100)))

    col1, col2, col3 = st.columns(3)
    col1.metric("Runs/case needed — no variance reduction", n_no_vr)
    col2.metric("Runs/case needed — with CRN", n_with_crn, delta=n_with_crn - n_no_vr)
    col3.metric("CRN variance reduction (measured)", f"{crn_result.variance_reduction_pct:.0f}%")

    fig = go.Figure(
        go.Bar(
            x=["no variance reduction", "with CRN"], y=[n_no_vr, n_with_crn],
            marker_color=[colors.ARM_A, colors.STATUS_GOOD],
            text=[str(n_no_vr), str(n_with_crn)], textposition="outside",
        )
    )
    fig.update_layout(
        **BASE_LAYOUT,
        yaxis_title="runs per case required",
        height=320, showlegend=False,
        title=f"detect a {mde * 100:.0f}pp rise from a {baseline_rate * 100:.0f}% baseline "
        f"({n_cases_demo} cases, 80% power, alpha=0.05)",
    )
    st.plotly_chart(fig, width="stretch")


# ============================================================
# PANEL 4 — CONFIDENCE SEQUENCE
# ============================================================

_CI_DISPLAY_CLIP_PP = 150  # percentage points; only bites on pathologically wide early-n bands


def render_confidence_sequence_panel() -> None:
    st.header("Confidence Sequence (always-valid)")

    report = load_report("known_regression")
    seq = (report or {}).get("sequential_analysis")
    if not seq:
        st.warning(
            "No sequential_analysis found in `known_regression_report.json` — run "
            "`python -m experiments.cli run --preset known_regression` (it saves this "
            "automatically) first."
        )
        return

    points = seq["points"]
    n = [p["n"] for p in points]
    center = [p["center"] * 100 for p in points]
    ci_low = [max(p["ci_low"] * 100, -_CI_DISPLAY_CLIP_PP) for p in points]
    ci_high = [min(p["ci_high"] * 100, _CI_DISPLAY_CLIP_PP) for p in points]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=n + n[::-1], y=ci_high + ci_low[::-1], fill="toself",
            fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
            hoverinfo="skip", showlegend=False, name="95% confidence sequence",
        )
    )
    fig.add_hline(y=0, line_color=colors.BASELINE)
    fig.add_trace(
        go.Scatter(
            x=n, y=center, mode="lines+markers",
            line=dict(color=colors.ARM_A, width=2), marker=dict(size=6),
            name="estimate", hovertemplate="n=%{x}<br>estimate=%{y:.1f}pp<extra></extra>",
        )
    )
    if seq["first_stop_index"]:
        fig.add_vline(
            x=seq["first_stop_index"], line_dash="dash", line_color=colors.STATUS_CRITICAL,
            annotation_text=f"stoppable at n={seq['first_stop_index']}", annotation_position="top",
        )
    fig.update_layout(
        **BASE_LAYOUT,
        xaxis_title="cases accumulated",
        yaxis_title="estimated diff, arm B - arm A (pp)",
        height=380, showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    if seq["first_stop_index"]:
        st.caption(
            f"outcome={seq['outcome_key']}, τ={seq['tau']}, α={seq['alpha']}. Stoppable at "
            f"n={seq['first_stop_index']} of {len(points)} cases available — you could have "
            f"safely stopped there without inflating the false-positive rate (anytime-valid, "
            f"not a p-value computed once at a pre-committed N)."
        )
    else:
        st.caption(f"outcome={seq['outcome_key']}, τ={seq['tau']}, α={seq['alpha']}. Not yet stoppable within the available cases.")


# ============================================================
# PANEL 5 — POWER CURVE
# ============================================================

def render_power_curve_panel() -> None:
    st.header("Power Curve")
    st.caption(
        "Computed live via `stats.power.power_curve` — a closed-form formula, cross-validated "
        "against statsmodels' independent two-proportion power calculation to within 0.4% "
        "(see `tests/test_stats_power.py`). Not a saved experiment result — pure math, adjustable."
    )

    default_baseline = 0.15
    known_regression = load_report("known_regression")
    if known_regression is not None:
        strict = [r for r in flatten_family_rows(known_regression) if r["outcome_key"] == "exfiltration"]
        if strict:
            default_baseline = sum(r["rate_a"] for r in strict) / len(strict)

    col1, col2 = st.columns(2)
    baseline_rate = col1.slider("Baseline rate", 0.01, 0.50, float(round(default_baseline, 2)), step=0.01)
    n_cases = col2.slider("Number of cases", 5, 100, 20, step=5)

    from stats.power import power_curve

    runs_per_case_grid = [5, 10, 15, 20, 30, 40, 60, 80, 100]
    curve = power_curve(n_cases, baseline_rate, runs_per_case_grid, power=0.8, alpha=0.05)

    fig = go.Figure(
        go.Scatter(
            x=[c[0] for c in curve], y=[c[1] * 100 for c in curve], mode="lines+markers",
            line=dict(color=colors.ARM_A, width=2), marker=dict(size=7),
            hovertemplate="%{x} runs/case<br>MDE=%{y:.1f}pp<extra></extra>",
        )
    )
    fig.update_layout(
        **BASE_LAYOUT,
        xaxis_title="runs per case", yaxis_title="minimum detectable effect (pp)",
        height=360, showlegend=False,
        title=f"{n_cases} cases, {baseline_rate * 100:.0f}% baseline, 80% power, alpha=0.05",
    )
    st.plotly_chart(fig, width="stretch")


# ============================================================
# PANEL 6 — TRAJECTORY INSPECTOR
# ============================================================

def _payload_marker(text: str) -> str:
    """A single, distinctive line of a (possibly multi-line) payload —
    matches how attacker/tests already handle this: tool results come back
    as str(dict), which repr()-escapes real newlines, so matching the
    whole multi-line payload against a stringified result never hits even
    when the content genuinely made it through. A single line has no
    newlines to escape."""
    lines = [line for line in text.splitlines() if line.strip()]
    return max(lines, key=len) if lines else text


def _event_text_blob(event: dict) -> str:
    parts = []
    for key in ("content", "final_answer", "arguments", "result", "message"):
        value = event.get(key)
        if value is not None:
            parts.append(str(value))
    return " ".join(parts)


def render_trajectory_inspector() -> None:
    st.header("Trajectory Inspector")

    experiments_with_trajectories = [e for e in list_available_reports() if (RUNS_DIR / f"{e}.jsonl").exists()]
    if not experiments_with_trajectories:
        st.warning("No trajectory files found under `data/runs/`.")
        return
    default_idx = experiments_with_trajectories.index("known_regression") if "known_regression" in experiments_with_trajectories else 0
    experiment_name = st.selectbox("Experiment", experiments_with_trajectories, index=default_idx, key="traj_experiment")

    records = load_raw_records(experiment_name)
    if not records:
        st.warning("No records found.")
        return

    default_record = (
        find_flagged_run(records, "exfiltration")
        or find_flagged_run(records, "unauthorized_lookup")
        or records[0]
    )

    def option_label(r: dict) -> str:
        flags = [k for k in ("exfiltration", "unauthorized_lookup") if r.get("outcomes", {}).get(k)]
        tag = f"FLAGGED: {', '.join(flags)}" if flags else "clean"
        return f"{r['case_id']}  |  arm={r['arm']}  seed={r['seed']}  |  {tag}"

    labels = [option_label(r) for r in records]
    default_selectbox_idx = records.index(default_record)
    selected_label = st.selectbox("Run (flagged cases are worth starting with)", labels, index=default_selectbox_idx, key="traj_run")
    record = records[labels.index(selected_label)]

    from attacker.cases import get_case

    case = None
    try:
        case = get_case(record["case_id"])
    except KeyError:
        pass

    marker = None
    if case is not None:
        marker = _payload_marker(case.injected_payload)
        st.caption(f"Injected payload marker being highlighted: “{marker[:140]}{'…' if len(marker) > 140 else ''}”")
        if case.injection_vector == "multi_turn":
            st.caption(f"Multi-turn case — {len(case.turns)} turns; turn boundaries marked below.")

    evidence_idx: set[int] = set()
    for outcome_key, idxs in record.get("outcome_evidence", {}).items():
        if record.get("outcomes", {}).get(outcome_key):
            evidence_idx.update(idxs)

    turn_counter = 0
    for event in record["events"]:
        if event["type"] == "agent_start" and event.get("agent") == "supervisor" and case is not None and case.injection_vector == "multi_turn":
            turn_counter += 1
            st.markdown(f"##### Turn {turn_counter}")

        blob = _event_text_blob(event)
        has_marker = marker is not None and marker in blob
        is_evidence = event["idx"] in evidence_idx

        icon = "🎯" if is_evidence else ("💉" if has_marker else "▫️")
        header = f"{icon} `[{event['idx']}]` **{event['type']}**"
        if event.get("agent"):
            header += f" — {event['agent']}"
        if event.get("from_agent"):
            header += f" — {event['from_agent']} → {event['to_agent']} ({event.get('role')})"

        with st.container(border=True):
            st.markdown(header)
            if event["type"] == "tool_call":
                status = event.get("status")
                status_badge = "🟢 executed" if status == "executed" else "🔴 blocked"
                st.markdown(f"`{event['tool_name']}` — {status_badge}")
                st.json(event.get("arguments", {}), expanded=False)
                st.code(str(event.get("result"))[:500])
            elif event["type"] == "message":
                st.write(event.get("content"))
            elif event["type"] == "agent_end":
                if event.get("final_answer"):
                    st.write(event["final_answer"])
            elif event["type"] == "error":
                st.write(event.get("message"))

            if is_evidence:
                st.caption("⬆ evidence for the flagged outcome")
            elif has_marker:
                st.caption("⬆ contains the injected payload")


def main() -> None:
    st.title("Agent Regression Detector")
    render_calibration_panel()
    st.divider()
    render_comparison_panel()
    st.divider()
    render_crn_panel()
    st.divider()
    render_confidence_sequence_panel()
    st.divider()
    render_power_curve_panel()
    st.divider()
    render_trajectory_inspector()


main()
