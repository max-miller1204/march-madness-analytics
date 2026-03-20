#!/usr/bin/env python3
"""Update the accuracy section in README.md from accuracy_log.json."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README = PROJECT_ROOT / "README.md"
ACCURACY_LOG = PROJECT_ROOT / "scraped_data" / "accuracy_log.json"
TOURNAMENT_STATE = PROJECT_ROOT / "scraped_data" / "tournament_state.json"

START_MARKER = "<!-- ACCURACY_START -->"
END_MARKER = "<!-- ACCURACY_END -->"


def _load_game_details():
    """Load completed game details (seeds, scores) from tournament state."""
    if not TOURNAMENT_STATE.exists():
        return {}
    try:
        state = json.loads(TOURNAMENT_STATE.read_text())
        return {g["game_id"]: g for g in state.get("completed_games", [])}
    except (json.JSONDecodeError, KeyError):
        return {}


def build_section() -> str:
    """Build the markdown accuracy section from the latest log entry."""
    if not ACCURACY_LOG.exists():
        return ""

    data = json.loads(ACCURACY_LOG.read_text())
    runs = data.get("runs", [])
    if not runs:
        return ""

    latest = runs[-1]
    summary = latest.get("summary", {})
    timestamp = latest.get("timestamp", "unknown")
    predictions = latest.get("predictions", [])

    total = latest.get("games_completed", 0)
    if total == 0:
        return ""

    accuracy = summary.get("accuracy", 0)
    correct = round(accuracy * total)
    upsets_caught = summary.get("upsets_predicted_correctly", 0)
    upsets_missed = summary.get("upsets_missed", 0)
    upsets_total = upsets_caught + upsets_missed

    pct = f"{accuracy * 100:.1f}%" if isinstance(accuracy, (int, float)) else "N/A"

    lines = [
        "## Tournament Accuracy Tracker",
        "",
        f"*Last updated: {timestamp}*",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Games tracked | {total} |",
        f"| Correct picks | {correct}/{total} |",
        f"| Accuracy | {pct} |",
        f"| Upsets caught | {upsets_caught}/{upsets_total} |",
        "",
        "![Accuracy Dashboard](images/accuracy_dashboard.png)",
    ]

    # Game-by-game results table
    if predictions:
        game_details = _load_game_details()
        lines.append("")
        lines.append("### Game-by-Game Results")
        lines.append("")
        lines.append(
            "| Game | Matchup | Score | Winner | Our Pick | Confidence | Result |"
        )
        lines.append(
            "|------|---------|-------|--------|----------|------------|--------|"
        )

        for p in predictions:
            gid = p["game_id"]
            g = game_details.get(gid, {})
            sa = g.get("seed_a", "?")
            sb = g.get("seed_b", "?")
            ta = g.get("team_a", "?")
            tb = g.get("team_b", "?")
            score_a = g.get("score_a", "?")
            score_b = g.get("score_b", "?")
            prob = p.get("predicted_prob", 0)
            result = "Correct" if p.get("correct") else "Wrong"

            lines.append(
                f"| {gid} "
                f"| ({sa}) {ta} vs ({sb}) {tb} "
                f"| {score_a}-{score_b} "
                f"| {p['actual_winner']} "
                f"| {p['predicted_winner']} "
                f"| {prob:.1%} "
                f"| {result} |"
            )

        lines.append("")
        lines.append(
            f"**Record: {correct}/{total} ({pct}) "
            f"| Upsets caught: {upsets_caught}/{upsets_total} "
            f"| All misses were upsets**"
        )

    return "\n".join(lines)


def main():
    readme_text = README.read_text()

    start_idx = readme_text.find(START_MARKER)
    end_idx = readme_text.find(END_MARKER)
    if start_idx == -1 or end_idx == -1:
        print("[WARN] Accuracy markers not found in README.md")
        return

    section = build_section()
    if not section:
        print("[SKIP] No accuracy data available")
        return

    new_readme = (
        readme_text[: start_idx + len(START_MARKER)]
        + "\n"
        + section
        + "\n"
        + readme_text[end_idx:]
    )

    README.write_text(new_readme)
    print(
        f"[OK] Updated README.md accuracy section ({end_idx - start_idx} chars replaced)"
    )


if __name__ == "__main__":
    main()
