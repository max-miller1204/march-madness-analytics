#!/usr/bin/env python3
"""
Changelog generator for tournament refresh pipeline.

Diffs previous vs current predictions, generates timestamped markdown reports.
"""

import json
import os
from datetime import datetime
from pathlib import Path


def generate_changelog(
    completed_games,
    predictions=None,
    previous_composites=None,
    current_composites=None,
    accuracy_summary=None,
):
    """Generate a markdown changelog string from tournament refresh data.

    Args:
        completed_games: list of dicts, each with keys like:
            - game_id (str): e.g. "E1"
            - result (str): e.g. "(1) Duke 82-55 Siena (16)"
            - predicted_winner (str): team name
            - predicted_prob (float): e.g. 0.987
            - correct (bool): whether prediction was correct
            - upset (bool, optional): whether the result was an upset
        predictions: dict, optional — current model predictions (for shift detection)
            Keys like "final_four", "champion", etc.
        previous_composites: dict, optional — {team: score} from previous run
        current_composites: dict, optional — {team: score} from current run
        accuracy_summary: dict, optional — with keys like:
            - correct (int)
            - total (int)
            - pct (float)
            - brier_score (float, optional)
            - ev_actual (float, optional)
            - ev_predicted (float, optional)

    Returns:
        str: markdown-formatted changelog
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M")
    lines = [f"# Refresh Changelog — {timestamp}", ""]

    # --- Section 1: New Results ---
    if completed_games:
        lines.append(f"## New Results ({len(completed_games)} games completed)")
        lines.append("| Game | Result | Predicted | Correct |")
        lines.append("|------|--------|-----------|---------|")
        for game in completed_games:
            game_id = game.get("game_id", "?")
            result = game.get("result", "N/A")
            winner = game.get("predicted_winner", "N/A")
            prob = game.get("predicted_prob", 0.0)
            correct = game.get("correct", False)
            upset = game.get("upset", False)

            prob_str = f"{winner} ({prob * 100:.1f}%)"
            if correct:
                correct_str = "Yes"
            else:
                correct_str = "No — UPSET" if upset else "No"

            lines.append(f"| {game_id} | {result} | {prob_str} | {correct_str} |")
        lines.append("")

    # --- Section 2: Composite Score Changes ---
    if previous_composites and current_composites:
        changes = []
        all_teams = set(list(previous_composites.keys()) + list(current_composites.keys()))
        for team in all_teams:
            prev = previous_composites.get(team)
            curr = current_composites.get(team)
            if prev is not None and curr is not None:
                delta = curr - prev
                if abs(delta) > 0.1:
                    changes.append((team, prev, curr, delta))
            elif prev is not None and curr is None:
                # Team eliminated
                changes.append((team, prev, None, None))
            elif prev is None and curr is not None:
                # New team appeared (unlikely but handle)
                changes.append((team, None, curr, None))

        if changes:
            # Sort by absolute change magnitude, eliminated teams last
            changes.sort(
                key=lambda x: (
                    0 if x[3] is not None else 1,
                    -(abs(x[3]) if x[3] is not None else 0),
                )
            )
            lines.append("## Composite Score Changes (top movers)")
            lines.append("| Team | Previous | Current | Change |")
            lines.append("|------|----------|---------|--------|")
            for team, prev, curr, delta in changes[:20]:
                prev_str = f"{prev:.1f}" if prev is not None else "N/A"
                if curr is None:
                    curr_str = "ELIMINATED"
                    delta_str = "—"
                elif prev is None:
                    curr_str = f"{curr:.1f}"
                    delta_str = "NEW"
                else:
                    curr_str = f"{curr:.1f}"
                    delta_str = f"{delta:+.1f}"
                lines.append(f"| {team} | {prev_str} | {curr_str} | {delta_str} |")
            lines.append("")

    # --- Section 3: Prediction Shifts ---
    if predictions:
        shifts = []
        ff_changes = predictions.get("final_four_changes")
        if ff_changes:
            for change in ff_changes:
                old = change.get("old", "?")
                new = change.get("new", "?")
                model = change.get("model", "model")
                shifts.append(f"**Final Four**: {new} replaced by {old} in {model}")

        champion_info = predictions.get("champion")
        if champion_info:
            name = champion_info.get("name", "?")
            prev_pct = champion_info.get("previous_pct")
            curr_pct = champion_info.get("current_pct")
            if prev_pct is not None and curr_pct is not None:
                if name == champion_info.get("previous_name", name):
                    shifts.append(
                        f"**Champion**: {name} (unchanged, "
                        f"{prev_pct:.1f}% -> {curr_pct:.1f}%)"
                    )
                else:
                    prev_name = champion_info.get("previous_name", "?")
                    shifts.append(
                        f"**Champion**: {name} ({curr_pct:.1f}%) "
                        f"replaced {prev_name} ({prev_pct:.1f}%)"
                    )

        if shifts:
            lines.append("## Prediction Shifts")
            for shift in shifts:
                lines.append(f"- {shift}")
            lines.append("")

    # --- Section 4: Model Accuracy ---
    if accuracy_summary:
        correct = accuracy_summary.get("correct", 0)
        total = accuracy_summary.get("total", 0)
        pct = accuracy_summary.get("pct", 0.0)
        lines.append("## Model Accuracy So Far")
        lines.append(f"- Overall: {correct}/{total} correct ({pct:.1f}%)")

        brier = accuracy_summary.get("brier_score")
        if brier is not None:
            lines.append(f"- Brier Score: {brier:.3f}")

        ev_actual = accuracy_summary.get("ev_actual")
        ev_predicted = accuracy_summary.get("ev_predicted")
        if ev_actual is not None:
            ev_line = f"- EV: {ev_actual:+.1f}"
            if ev_predicted is not None:
                ev_line += f" (predicted {ev_predicted:.1f})"
            lines.append(ev_line)
        lines.append("")

    # If nothing was generated beyond the header, add a note
    if len(lines) <= 2:
        lines.append("No changes detected in this refresh cycle.")
        lines.append("")

    return "\n".join(lines)


def save_changelog(content, output_dir="reports"):
    """Write changelog content to a timestamped markdown file.

    Args:
        content: str — the markdown changelog content
        output_dir: str — directory to write into (created if needed)

    Returns:
        str: path to the written file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    filename = f"changelog_{now.strftime('%Y-%m-%d_%H-%M')}.md"
    filepath = output_path / filename

    filepath.write_text(content, encoding="utf-8")
    print(f"Changelog saved to {filepath}")
    return str(filepath)


def main():
    """Standalone entry point: load state files and generate a sample changelog."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    state_path = repo_root / "scraped_data" / "tournament_state.json"
    accuracy_path = repo_root / "scraped_data" / "accuracy_log.json"

    completed_games = []
    accuracy_summary = None

    # Try loading tournament state for completed games
    if state_path.exists():
        with open(state_path, "r") as f:
            state = json.load(f)
        # Extract completed games if available
        raw_games = state.get("completed_games", [])
        for g in raw_games:
            completed_games.append({
                "game_id": g.get("game_id", "?"),
                "result": g.get("result", "N/A"),
                "predicted_winner": g.get("predicted_winner", "N/A"),
                "predicted_prob": g.get("predicted_prob", 0.0),
                "correct": g.get("correct", False),
                "upset": g.get("upset", False),
            })

    # Try loading accuracy log
    if accuracy_path.exists():
        with open(accuracy_path, "r") as f:
            accuracy_data = json.load(f)
        if isinstance(accuracy_data, dict):
            accuracy_summary = accuracy_data.get("summary", accuracy_data)
        elif isinstance(accuracy_data, list) and accuracy_data:
            # Compute summary from list of results
            total = len(accuracy_data)
            correct = sum(1 for r in accuracy_data if r.get("correct", False))
            pct = (correct / total * 100) if total > 0 else 0.0
            accuracy_summary = {"correct": correct, "total": total, "pct": pct}

    content = generate_changelog(
        completed_games=completed_games,
        accuracy_summary=accuracy_summary,
    )
    output_dir = str(repo_root / "reports")
    path = save_changelog(content, output_dir=output_dir)
    print(f"\nGenerated changelog at: {path}")
    print("\n--- Preview ---")
    print(content)


if __name__ == "__main__":
    main()
