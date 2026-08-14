"""CLI for running preset (or ad hoc) paired experiments. Run with:

    python -m experiments.cli run --preset aa
    python -m experiments.cli run --config-a <hash> --config-b <hash> --experiment-name my_check
    python -m experiments.cli list-presets

Run aa first — it's the pipeline sanity check every other preset's result
depends on being trustworthy.
"""

from __future__ import annotations

import argparse
import logging
import sys

from experiments.persist import save_experiment_report
from experiments.presets import PRESETS
from experiments.report import format_experiment_report, format_sequential_analysis
from experiments.runner import run_experiment
from target_system.config import DEFAULT_CONFIGS_DIR, load_config


def _cmd_run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sequential_outcome_key = None
    if args.preset:
        preset = PRESETS[args.preset]
        arm_a, arm_b = preset.arm_a, preset.arm_b
        experiment_name = args.experiment_name or preset.name
        sequential_outcome_key = preset.sequential_outcome_key
        print(f"Preset: {preset.name} — {preset.description}")
        print(f"Expectation: {preset.expectation}\n")
    elif args.config_a and args.config_b:
        arm_a = load_config(args.config_a, configs_dir=DEFAULT_CONFIGS_DIR)
        arm_b = load_config(args.config_b, configs_dir=DEFAULT_CONFIGS_DIR)
        if not args.experiment_name:
            print("error: --experiment-name is required with --config-a/--config-b", file=sys.stderr)
            sys.exit(2)
        experiment_name = args.experiment_name
    else:
        print("error: pass --preset, or both --config-a and --config-b", file=sys.stderr)
        sys.exit(2)

    result = run_experiment(
        arm_a, arm_b,
        experiment_name=experiment_name,
        n_runs_per_case=args.runs_per_case,
        max_workers=args.max_workers,
        stats_method=args.method,
        alpha=args.alpha,
    )
    print(format_experiment_report(result))
    if sequential_outcome_key:
        print()
        print(format_sequential_analysis(result, sequential_outcome_key, alpha=args.alpha))

    report_path = save_experiment_report(result, sequential_outcome_key=sequential_outcome_key)
    print(f"\nSaved report: {report_path}")


def _cmd_list_presets(_args: argparse.Namespace) -> None:
    for name, preset in PRESETS.items():
        print(f"{name}: {preset.description}\n  expectation: {preset.expectation}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.cli", description="Agent regression detector: experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a preset or ad hoc paired experiment.")
    p_run.add_argument("--preset", choices=sorted(PRESETS.keys()), help="Name of a built-in preset experiment.")
    p_run.add_argument("--config-a", help="config_hash of an already-saved config (ad hoc mode).")
    p_run.add_argument("--config-b", help="config_hash of an already-saved config (ad hoc mode).")
    p_run.add_argument("--experiment-name", help="Output file stem under data/runs/. Defaults to the preset name.")
    p_run.add_argument("--runs-per-case", type=int, default=5)
    p_run.add_argument("--max-workers", type=int, default=8)
    # hierarchical_bayes is the only method the product path uses; the
    # frequentist choices remain here as this harness's escape hatch for
    # regression comparisons, same retired-but-testable status as the toy
    # presets this same subcommand can still run.
    p_run.add_argument(
        "--method",
        choices=["hierarchical_bayes", "cluster_bootstrap", "mcnemar", "mixed_effects"],
        default="hierarchical_bayes",
    )
    p_run.add_argument("--alpha", type=float, default=0.05)
    p_run.add_argument("--verbose", action="store_true")
    p_run.set_defaults(func=_cmd_run)

    p_list = sub.add_parser("list-presets", help="List available preset experiments.")
    p_list.set_defaults(func=_cmd_list_presets)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
