"""First-class CLI for the stats module. Run with:

    python -m stats.cli aa-calibration [options]
    python -m stats.cli power [options]
    python -m stats.cli mde [options]

aa-calibration is the primary validation of this whole module (per the
build spec) — it's a real subcommand, not a script you have to go dig up.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from stats.aa_calibration import CaseSpec, run_aa_calibration
from stats.reporting import format_aa_calibration, format_mde_report, format_power_report


def _cmd_aa_calibration(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    base_rates = rng.uniform(args.min_base_rate, args.max_base_rate, args.n_cases)
    specs = [CaseSpec(case_id=f"case_{i}", family="synthetic", base_rate=float(r)) for i, r in enumerate(base_rates)]

    method_kwargs = {}
    if args.method == "cluster_bootstrap":
        method_kwargs["n_boot"] = args.n_boot

    result = run_aa_calibration(
        specs,
        n_runs_per_case=args.runs_per_case,
        method=args.method,
        n_trials=args.trials,
        alpha=args.alpha,
        seed=args.seed,
        method_kwargs=method_kwargs,
    )
    print(format_aa_calibration(result))
    if not result.well_calibrated:
        print(
            "\nNote: a test whose empirical FPR excludes the nominal alpha isn't "
            "necessarily broken — e.g. McNemar's continuity correction is known to be "
            "conservative, and cluster_bootstrap_diff is known to over-reject (empirical "
            "FPR ~1.2-1.7x nominal) across a wide range of case counts even with BCa, "
            "shrinking but not cleanly vanishing by n_cases=80 (see stats/paired.py's "
            "docstrings for both). Check whether the direction and case count match a "
            "documented property of the method before treating this as a bug — if so, "
            "try more cases or a different method rather than treating the tool as broken.",
            file=sys.stderr,
        )
        sys.exit(1)


def _cmd_power(args: argparse.Namespace) -> None:
    print(
        format_power_report(
            args.baseline_rate, args.mde, args.n_cases,
            power=args.power, alpha=args.alpha, between_case_sd=args.between_case_sd,
        )
    )


def _cmd_mde(args: argparse.Namespace) -> None:
    print(
        format_mde_report(
            args.n_cases, args.runs_per_case, args.baseline_rate,
            power=args.power, alpha=args.alpha, between_case_sd=args.between_case_sd,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m stats.cli", description="Agent regression detector: statistics CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_aa = sub.add_parser("aa-calibration", help="Run the identical config as both arms many times and check FPR vs nominal alpha.")
    p_aa.add_argument("--n-cases", type=int, default=20)
    p_aa.add_argument("--min-base-rate", type=float, default=0.05)
    p_aa.add_argument("--max-base-rate", type=float, default=0.35)
    p_aa.add_argument("--runs-per-case", type=int, default=20)
    p_aa.add_argument(
        "--method",
        choices=["hierarchical_bayes", "cluster_bootstrap", "mcnemar", "mixed_effects"],
        default="hierarchical_bayes",
    )
    p_aa.add_argument("--n-boot", type=int, default=2000, help="bootstrap replicates per trial (cluster_bootstrap only)")
    p_aa.add_argument("--trials", type=int, default=500, help="number of simulated A/A experiments")
    p_aa.add_argument("--alpha", type=float, default=0.05)
    p_aa.add_argument("--seed", type=int, default=0)
    p_aa.set_defaults(func=_cmd_aa_calibration)

    p_power = sub.add_parser("power", help="Required runs per case for a given baseline rate / MDE / power.")
    p_power.add_argument("--baseline-rate", type=float, required=True)
    p_power.add_argument("--mde", type=float, required=True, help="absolute rate difference to detect, e.g. 0.05 for 5 points")
    p_power.add_argument("--n-cases", type=int, required=True)
    p_power.add_argument("--power", type=float, default=0.8)
    p_power.add_argument("--alpha", type=float, default=0.05)
    p_power.add_argument("--between-case-sd", type=float, default=0.0)
    p_power.set_defaults(func=_cmd_power)

    p_mde = sub.add_parser("mde", help="Smallest detectable effect for a fixed run budget.")
    p_mde.add_argument("--n-cases", type=int, required=True)
    p_mde.add_argument("--runs-per-case", type=int, required=True)
    p_mde.add_argument("--baseline-rate", type=float, required=True)
    p_mde.add_argument("--power", type=float, default=0.8)
    p_mde.add_argument("--alpha", type=float, default=0.05)
    p_mde.add_argument("--between-case-sd", type=float, default=0.0)
    p_mde.set_defaults(func=_cmd_mde)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
