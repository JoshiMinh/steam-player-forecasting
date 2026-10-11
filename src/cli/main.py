"""Unified CLI entrypoint for Steam Player Forecasting framework.

Provides console commands to execute pipeline stages, benchmarks, figures,
and launch the interactive Streamlit demonstration application.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.data.loader import get_project_root


def run_app() -> None:
    """Launch the interactive Streamlit demonstration dashboard."""
    root = get_project_root()
    app_path = root / "app" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print(f"Launching Streamlit application from: {app_path}")
    subprocess.run(cmd)


def main(argv: list[str] | None = None) -> None:
    """Main CLI parser and subcommand dispatcher."""
    parser = argparse.ArgumentParser(
        prog="steam-forecast",
        description="Steam Player Population Forecasting — Research & Benchmarking Suite",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. Pipeline (Phase 1)
    p_pipeline = subparsers.add_parser("pipeline", help="Acquire, clean, and validate player datasets (Phase 1)")
    p_pipeline.add_argument("--force-download", action="store_true", help="Force dataset download/generation")

    # 2. EDA Figures (Phase 2)
    subparsers.add_parser("eda-figures", help="Generate all 9 Phase 2 EDA figures (Phase 2)")

    # 3. Phase 3 (Statistical Baselines)
    subparsers.add_parser("phase3", help="Execute Phase 3 statistical forecasting benchmarks (SARIMA, ARIMA, Holt-Winters)")

    # 4. Phase 4 (ML & DL Baselines)
    subparsers.add_parser("phase4", help="Execute Phase 4 ML & Deep Learning benchmarks (XGBoost, RF, Ridge, LSTM, GRU)")

    # 5. Phase 5 (Final Test Evaluation)
    subparsers.add_parser("phase5", help="Execute Phase 5 final out-of-sample holdout test evaluation")

    # 6. Phase 5 Figures
    subparsers.add_parser("phase5-figures", help="Generate all 4 publication test figures (Figures 17, 18, 19, 20)")

    # 7. Interactive Streamlit App
    subparsers.add_parser("app", help="Launch the interactive Steam Store Streamlit dashboard")

    # 8. Notebook execution
    p_nb = subparsers.add_parser("notebook", help="Execute and populate outputs for a specified Jupyter notebook")
    p_nb.add_argument("notebook_name", nargs="?", default="01_eda_and_preprocessing.ipynb", help="Target notebook filename")

    # 9. Run all
    subparsers.add_parser("run-all", help="Execute end-to-end benchmark pipeline across all phases")

    args = parser.parse_args(argv)

    if args.subcommand == "pipeline":
        from src.data.pipeline import run_phase1_pipeline
        run_phase1_pipeline(force_download=args.force_download)

    elif args.subcommand == "eda-figures":
        from src.cli.eda_figures import generate_all_figures
        generate_all_figures()

    elif args.subcommand == "phase3":
        from src.cli.phase3 import run_phase3_pipeline
        run_phase3_pipeline()

    elif args.subcommand == "phase4":
        from src.cli.phase4 import run_phase4_pipeline
        run_phase4_pipeline()

    elif args.subcommand == "phase5":
        from src.cli.phase5 import run_phase5_evaluation
        run_phase5_evaluation()

    elif args.subcommand == "phase5-figures":
        from src.cli.phase5_figures import main as run_figures
        run_figures()

    elif args.subcommand == "app":
        run_app()

    elif args.subcommand == "notebook":
        from src.cli.notebooks import execute_and_populate_notebook
        execute_and_populate_notebook(args.notebook_name)

    elif args.subcommand == "run-all":
        print("Running end-to-end benchmarking pipeline...\n")
        from src.data.pipeline import run_phase1_pipeline
        from src.cli.eda_figures import generate_all_figures
        from src.cli.phase3 import run_phase3_pipeline
        from src.cli.phase4 import run_phase4_pipeline
        from src.cli.phase5 import run_phase5_evaluation
        from src.cli.phase5_figures import main as run_figures

        run_phase1_pipeline()
        generate_all_figures()
        run_phase3_pipeline()
        run_phase4_pipeline()
        run_phase5_evaluation()
        run_figures()
        print("\nAll pipeline benchmarks and figures successfully updated!")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
