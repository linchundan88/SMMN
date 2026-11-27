"""CLI entry point for the Python analysis pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from analysis import run_analyses
from preprocessing import run_preprocessing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the multi-lab musician memory data and reproduce its analyses."
    )
    parser.add_argument(
        "--rebuild-data", action="store_true",
        help="Rebuild corrected, selected, scored datasets from data/raw-before-fixing.",
    )
    parser.add_argument(
        "--input-clean", type=Path, default=None,
        help="Clean CSV to analyse (defaults to data/clean/dat-clean.csv).",
    )
    parser.add_argument(
        "--input-full", type=Path, default=None,
        help="Optional full clean CSV (defaults to data/clean/dat-clean-full.csv).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "output",
        help="Directory for Python-produced data, tables, plots, and reports.",
    )
    parser.add_argument(
        "--simulations", type=int, default=5000,
        help="Number of post-hoc power simulations per scenario; use 0 to skip.",
    )
    parser.add_argument("--seed", type=int, default=20261126, help="Random seed.")
    parser.add_argument(
        "--skip-mixed-models", action="store_true",
        help="Skip the statsmodels mixed-effects regression models.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    workspace = Path(__file__).resolve().parents[1]
    project_root = workspace / "multilab-musicians-memory"
    output_root = args.output.expanduser().resolve()

    if args.rebuild_data:
        clean_path, full_path = run_preprocessing(project_root, output_root)
    else:
        clean_path = args.input_clean or project_root / "data" / "clean" / "dat-clean.csv"
        full_path = args.input_full or project_root / "data" / "clean" / "dat-clean-full.csv"
        clean_path = clean_path.expanduser().resolve()
        full_path = full_path.expanduser().resolve() if full_path else None
        if not clean_path.exists():
            raise FileNotFoundError(
                f"{clean_path} does not exist. Use --rebuild-data or set --input-clean."
            )

    run_analyses(
        clean_path=clean_path,
        full_path=full_path,
        project_root=project_root,
        output_root=output_root,
        n_simulations=max(0, args.simulations),
        include_models=not args.skip_mixed_models,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

