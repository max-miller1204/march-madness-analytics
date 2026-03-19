#!/usr/bin/env python3
"""Update the accuracy section in README.md from accuracy_log.json."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README = PROJECT_ROOT / "README.md"
ACCURACY_LOG = PROJECT_ROOT / "scraped_data" / "accuracy_log.json"

START_MARKER = "<!-- ACCURACY_START -->"
END_MARKER = "<!-- ACCURACY_END -->"


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

    total = summary.get("total_games", 0)
    if total == 0:
        return ""

    correct = summary.get("correct", 0)
    accuracy = summary.get("accuracy", 0)
    upsets_caught = summary.get("upsets_caught", "N/A")
    upsets_total = summary.get("upsets_total", "N/A")

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
