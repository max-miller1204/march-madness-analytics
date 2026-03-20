#!/usr/bin/env python3
"""
CLI entry point for the March Madness tournament refresh pipeline.

Usage examples:
    python scripts/refresh.py all                          # Full pipeline
    python scripts/refresh.py all --refresh                # Force re-scrape
    python scripts/refresh.py scrape                       # Re-scrape metrics
    python scripts/refresh.py results                      # Scrape scores only
    python scripts/refresh.py simulate                     # Re-fit + simulate
    python scripts/refresh.py accuracy                     # Accuracy dashboard
    python scripts/refresh.py archive                      # Archive outputs
    python scripts/refresh.py watch --interval 30m         # Scheduled mode
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Lazy helpers — each wraps a module that may not exist yet.  On ImportError
# we print a warning and return gracefully so the rest of the pipeline can
# continue.
# ---------------------------------------------------------------------------


def _try_import(module_path: str, func_name: str):
    """Return *func_name* from *module_path*, or ``None`` with a warning."""
    try:
        mod = __import__(module_path, fromlist=[func_name])
        return getattr(mod, func_name)
    except (ImportError, AttributeError) as exc:
        print(f"[WARN] Could not import {func_name} from {module_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_scrape(**kwargs):
    """Re-run the existing metric scrapers (KenPom, NET, injuries)."""
    scrapers = [
        ("scrape_net_teamsheets", "main"),
        ("scrape_injuries", "main"),
        ("scrape_player_stats", "main"),
    ]

    for module_name, func_name in scrapers:
        fn = _try_import(module_name, func_name)
        if fn is not None:
            print(f"\n--- Running {module_name}.{func_name}() ---")
            try:
                fn()
            except Exception as exc:
                print(f"[WARN] {module_name} failed: {exc}")
        else:
            print(f"[SKIP] {module_name} not available")


def step_results(**kwargs):
    """Scrape tournament game results."""
    fn = _try_import("refresh.scrape_results", "scrape_tournament_results")
    if fn is not None:
        print("\n--- Scraping tournament results ---")
        try:
            fn()
        except Exception as exc:
            print(f"[WARN] scrape_results failed: {exc}")
    else:
        print("[SKIP] refresh.scrape_results not available")


def step_simulate(**kwargs):
    """Re-fit models and re-run simulation (notebook or quant_models)."""
    tournament_weight = kwargs.get("tournament_weight", 2.0)
    no_notebook = kwargs.get("no_notebook", False)

    # 1. Re-fit models with tournament data
    refit = _try_import("refresh.refit_models", "refit_models")
    if refit is not None:
        print("\n--- Re-fitting models ---")
        try:
            refit_result = refit(tournament_weight=tournament_weight)
        except Exception as exc:
            refit_result = None
            print(f"[WARN] refit_models failed: {exc}")
    else:
        refit_result = None
        print("[SKIP] refresh.refit_models not available")

    # 2. Re-execute the notebook (unless --no-notebook)
    if not no_notebook:
        notebook_path = PROJECT_ROOT / "final_four_analysis.ipynb"
        if notebook_path.exists():
            print("\n--- Re-executing notebook ---")
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "jupyter",
                        "nbconvert",
                        "--execute",
                        str(notebook_path),
                        "--to",
                        "notebook",
                        "--inplace",
                    ],
                    check=True,
                    cwd=str(PROJECT_ROOT),
                )
                print("Notebook re-execution complete.")
            except subprocess.CalledProcessError as exc:
                print(f"[WARN] Notebook execution failed (exit {exc.returncode})")
            except FileNotFoundError:
                print(
                    "[WARN] jupyter nbconvert not found — skipping notebook execution"
                )
        else:
            print("[SKIP] Notebook not found")
    else:
        if refit_result is not None:
            print(
                "[NOTE] Refitted models available but --no-notebook skips simulation. "
                "Re-run without --no-notebook to apply refitted models."
            )
        print("[SKIP] Notebook execution disabled (--no-notebook)")


def step_accuracy(**kwargs):
    """Generate the accuracy dashboard."""
    fn = _try_import("refresh.accuracy", "track_accuracy")
    if fn is not None:
        print("\n--- Tracking accuracy ---")
        try:
            fn()
        except Exception as exc:
            print(f"[WARN] accuracy tracking failed: {exc}")
    else:
        print("[SKIP] refresh.accuracy not available")


def step_changelog(**kwargs):
    """Generate and save the changelog."""
    gen = _try_import("refresh.changelog", "generate_changelog")
    save = _try_import("refresh.changelog", "save_changelog")
    if gen is not None and save is not None:
        print("\n--- Generating changelog ---")
        try:
            # Load completed games from tournament state
            state_path = PROJECT_ROOT / "scraped_data" / "tournament_state.json"
            completed_games = []
            if state_path.exists():
                import json

                with open(state_path) as f:
                    state = json.load(f)
                for g in state.get("completed_games", []):
                    completed_games.append(
                        {
                            "game_id": g.get("game_id", "?"),
                            "result": (
                                f"({g.get('seed_a', '?')}) {g.get('team_a', '?')} "
                                f"{g.get('score_a', '?')}-{g.get('score_b', '?')} "
                                f"{g.get('team_b', '?')} ({g.get('seed_b', '?')})"
                            ),
                            "predicted_winner": g.get("predicted_winner", "N/A"),
                            "predicted_prob": g.get("predicted_prob", 0.0),
                            "correct": g.get("correct", False),
                            "upset": g.get("upset", False),
                        }
                    )

            # Load accuracy summary if available
            accuracy_path = PROJECT_ROOT / "scraped_data" / "accuracy_log.json"
            accuracy_summary = None
            if accuracy_path.exists():
                with open(accuracy_path) as f:
                    acc_data = json.load(f)
                runs = acc_data.get("runs", [])
                if runs:
                    accuracy_summary = runs[-1].get("summary")

            content = gen(
                completed_games=completed_games,
                accuracy_summary=accuracy_summary,
            )
            save(content)
        except Exception as exc:
            print(f"[WARN] changelog failed: {exc}")
    else:
        print("[SKIP] refresh.changelog not available")


def step_archive(**kwargs):
    """Archive the current run's outputs."""
    fn = _try_import("refresh.archive", "archive_run")
    if fn is not None:
        print("\n--- Archiving run ---")
        try:
            fn()
        except Exception as exc:
            print(f"[WARN] archive failed: {exc}")
    else:
        print("[SKIP] refresh.archive not available")


def run_full_pipeline(**kwargs):
    """Execute the complete refresh pipeline in order."""
    print("=" * 60)
    print("FULL PIPELINE")
    print("=" * 60)

    # 1. Scrape metrics (if --refresh)
    if kwargs.get("refresh", False):
        step_scrape(**kwargs)

    # 2. Scrape tournament results
    step_results(**kwargs)

    # 3. Re-fit models + re-simulate
    step_simulate(**kwargs)

    # 4. Track accuracy
    step_accuracy(**kwargs)

    # 5. Generate changelog
    step_changelog(**kwargs)

    # 6. Archive
    step_archive(**kwargs)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refresh",
        description="March Madness tournament refresh pipeline",
    )

    # Global flags
    parser.add_argument(
        "--tournament-weight",
        type=float,
        default=2.0,
        help="Weight multiplier for tournament games (default: 2.0)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-scrape ignoring TTL cache",
    )
    parser.add_argument(
        "--no-notebook",
        action="store_true",
        help="Skip notebook re-execution (just update data)",
    )

    sub = parser.add_subparsers(dest="command")

    # all (default)
    sub.add_parser("all", help="Run the full pipeline (default)")

    # Individual steps
    sub.add_parser("scrape", help="Re-scrape metrics (KenPom, NET, injuries)")
    sub.add_parser("results", help="Scrape tournament scores only")
    sub.add_parser("simulate", help="Re-fit models + re-simulate")
    sub.add_parser("accuracy", help="Generate accuracy dashboard")
    sub.add_parser("archive", help="Archive current outputs")

    # Scheduled mode
    watch_p = sub.add_parser("watch", help="Scheduled polling mode")
    watch_p.add_argument(
        "--interval",
        type=str,
        default="30m",
        help="Polling interval (e.g. 15m, 30m, 1h). Default: 30m",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Collect kwargs that get forwarded to pipeline functions
    kwargs = dict(
        tournament_weight=args.tournament_weight,
        refresh=args.refresh,
        no_notebook=args.no_notebook,
    )

    command = args.command or "all"

    dispatch = {
        "all": run_full_pipeline,
        "scrape": step_scrape,
        "results": step_results,
        "simulate": step_simulate,
        "accuracy": step_accuracy,
        "archive": step_archive,
    }

    if command == "watch":
        from refresh.scheduler import parse_interval, run_watch

        interval = parse_interval(args.interval)
        run_watch(interval, run_full_pipeline, **kwargs)
    elif command in dispatch:
        dispatch[command](**kwargs)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
