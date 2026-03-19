#!/usr/bin/env python3
"""
Archive module for tournament refresh pipeline.

Copies images/, tournament state, accuracy log, and changelog into a
timestamped directory under reports/.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def _generate_summary(state_path, accuracy_path):
    """Generate a one-line summary string from state and accuracy data.

    Returns:
        str: e.g. "R64 16/32 complete, 75.0% accuracy"
    """
    parts = []

    # Try to read round and completion info from state
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
            current_round = state.get("current_round", "Unknown")
            completed = state.get("completed_games", [])
            total_games = state.get("total_games")

            if isinstance(completed, list):
                n_completed = len(completed)
            elif isinstance(completed, int):
                n_completed = completed
            else:
                n_completed = 0

            if total_games:
                parts.append(f"{current_round} {n_completed}/{total_games} complete")
            else:
                parts.append(f"{current_round} {n_completed} games complete")
        except (json.JSONDecodeError, KeyError):
            parts.append("State: parse error")

    # Try to read accuracy info
    if os.path.exists(accuracy_path):
        try:
            with open(accuracy_path, "r") as f:
                accuracy_data = json.load(f)

            if isinstance(accuracy_data, dict):
                summary = accuracy_data.get("summary", accuracy_data)
                pct = summary.get("pct")
                if pct is not None:
                    parts.append(f"{pct:.1f}% accuracy")
                else:
                    correct = summary.get("correct", 0)
                    total = summary.get("total", 0)
                    if total > 0:
                        parts.append(f"{correct / total * 100:.1f}% accuracy")
            elif isinstance(accuracy_data, list) and accuracy_data:
                total = len(accuracy_data)
                correct = sum(1 for r in accuracy_data if r.get("correct", False))
                if total > 0:
                    parts.append(f"{correct / total * 100:.1f}% accuracy")
        except (json.JSONDecodeError, KeyError):
            pass

    if not parts:
        return f"Archive created {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    return ", ".join(parts)


def archive_run(
    state_path="scraped_data/tournament_state.json",
    accuracy_path="scraped_data/accuracy_log.json",
    images_dir="images",
    changelog_path=None,
    output_base="reports",
):
    """Archive current run data to a timestamped directory.

    Copies:
        - images/ directory (all visualizations)
        - tournament_state.json snapshot
        - accuracy_log.json snapshot
        - changelog.md (if provided)
        - summary.txt (auto-generated one-liner)

    Args:
        state_path: path to tournament_state.json
        accuracy_path: path to accuracy_log.json
        images_dir: path to images directory
        changelog_path: path to changelog markdown file (optional)
        output_base: base directory for archive output

    Returns:
        str: path to the created archive directory
    """
    now = datetime.now()
    run_dir_name = f"run_{now.strftime('%Y-%m-%d_%H-%M')}"
    archive_dir = Path(output_base) / run_dir_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    copied = []

    # Copy images directory
    if os.path.isdir(images_dir):
        dest_images = archive_dir / "images"
        if dest_images.exists():
            shutil.rmtree(dest_images)
        shutil.copytree(images_dir, dest_images)
        n_files = sum(1 for _ in dest_images.rglob("*") if _.is_file())
        copied.append(f"images/ ({n_files} files)")
    else:
        print(f"Warning: images directory '{images_dir}' not found, skipping")

    # Copy tournament state
    if os.path.exists(state_path):
        shutil.copy2(state_path, archive_dir / "tournament_state.json")
        copied.append("tournament_state.json")
    else:
        print(f"Warning: state file '{state_path}' not found, skipping")

    # Copy accuracy log
    if os.path.exists(accuracy_path):
        shutil.copy2(accuracy_path, archive_dir / "accuracy_log.json")
        copied.append("accuracy_log.json")
    else:
        print(f"Warning: accuracy file '{accuracy_path}' not found, skipping")

    # Copy changelog
    if changelog_path and os.path.exists(changelog_path):
        shutil.copy2(changelog_path, archive_dir / "changelog.md")
        copied.append("changelog.md")
    elif changelog_path:
        print(f"Warning: changelog file '{changelog_path}' not found, skipping")

    # Generate and write summary
    summary_text = _generate_summary(state_path, accuracy_path)
    summary_path = archive_dir / "summary.txt"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    copied.append("summary.txt")

    archive_path = str(archive_dir)
    print(f"Archived run to: {archive_path}")
    print(f"  Contents: {', '.join(copied)}")
    print(f"  Summary: {summary_text}")

    return archive_path


def main():
    """Standalone entry point: archive current state from repo root defaults."""
    repo_root = Path(__file__).resolve().parent.parent.parent

    state_path = str(repo_root / "scraped_data" / "tournament_state.json")
    accuracy_path = str(repo_root / "scraped_data" / "accuracy_log.json")
    images_dir = str(repo_root / "images")
    output_base = str(repo_root / "reports")

    # Look for the most recent changelog in reports/
    reports_dir = Path(output_base)
    changelog_path = None
    if reports_dir.exists():
        changelogs = sorted(reports_dir.glob("changelog_*.md"), reverse=True)
        if changelogs:
            changelog_path = str(changelogs[0])
            print(f"Found changelog: {changelog_path}")

    path = archive_run(
        state_path=state_path,
        accuracy_path=accuracy_path,
        images_dir=images_dir,
        changelog_path=changelog_path,
        output_base=output_base,
    )
    print(f"\nArchive complete: {path}")


if __name__ == "__main__":
    main()
