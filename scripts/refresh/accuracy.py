#!/usr/bin/env python3
"""
Accuracy Tracker for March Madness bracket predictions.

Logs predictions vs outcomes to accuracy_log.json, computes per-round accuracy,
Brier score, upset detection rate, and EV predicted vs actual. Generates a
4-subplot accuracy dashboard visualization.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_POINTS = {
    "Round of 64": 10,
    "Round of 32": 20,
    "Sweet 16": 40,
    "Elite 8": 80,
    "Final Four": 160,
    "Championship": 320,
}

ROUND_ORDER = [
    "Round of 64",
    "Round of 32",
    "Sweet 16",
    "Elite 8",
    "Final Four",
    "Championship",
]

# Repo root: two levels up from scripts/refresh/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# AccuracyTracker
# ---------------------------------------------------------------------------


class AccuracyTracker:
    """Track prediction accuracy across tournament runs."""

    def __init__(self, log_path=None):
        self.log_path = log_path or str(
            REPO_ROOT / "scraped_data" / "accuracy_log.json"
        )
        self.data = self._load()

    # -- persistence --------------------------------------------------------

    def _load(self):
        """Load existing accuracy log or create empty structure."""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r") as f:
                    data = json.load(f)
                if "runs" in data:
                    return data
            except (json.JSONDecodeError, KeyError):
                pass
        return {"runs": []}

    def _save(self):
        """Persist the accuracy log to disk."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2)

    # -- core API -----------------------------------------------------------

    def log_run(self, predictions, completed_games, round_name=None):
        """Compare predictions to completed game outcomes, compute metrics, append run.

        Args:
            predictions: list of dicts, each with keys:
                - game_id: str
                - predicted_winner: str
                - predicted_prob: float (probability assigned to predicted winner)
                - round: str (round name)
            completed_games: list of dicts, each with keys:
                - game_id: str
                - winner: str
                - higher_seed_won: bool (True if the higher-seeded team won)
            round_name: optional override for the round label

        Returns:
            The run summary dict.
        """
        # Index completed games by game_id for fast lookup
        outcomes = {g["game_id"]: g for g in completed_games}

        prediction_results = []
        for pred in predictions:
            gid = pred["game_id"]
            if gid not in outcomes:
                continue  # game not yet completed

            outcome = outcomes[gid]
            correct = pred["predicted_winner"] == outcome["winner"]
            rd = pred.get("round", round_name or "Unknown")
            pts = ROUND_POINTS.get(rd, 10)

            ev_predicted = pred["predicted_prob"] * pts
            ev_actual = pts if correct else 0.0

            # An upset is when the higher seed did NOT win
            upset = not outcome.get("higher_seed_won", True)

            prediction_results.append(
                {
                    "game_id": gid,
                    "predicted_winner": pred["predicted_winner"],
                    "predicted_prob": pred["predicted_prob"],
                    "actual_winner": outcome["winner"],
                    "correct": correct,
                    "ev_predicted": round(ev_predicted, 2),
                    "ev_actual": ev_actual,
                    "upset": upset,
                }
            )

        if not prediction_results:
            return None

        # Determine round name from predictions if not given
        if round_name is None:
            rounds_seen = [
                p.get("round", "Unknown")
                for p in predictions
                if p["game_id"] in outcomes
            ]
            round_name = rounds_seen[0] if rounds_seen else "Unknown"

        summary = self._compute_summary(prediction_results)

        run = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "round": round_name,
            "games_completed": len(prediction_results),
            "predictions": prediction_results,
            "summary": summary,
        }

        self.data["runs"].append(run)
        self._save()
        return summary

    def _compute_summary(self, prediction_results):
        """Compute accuracy, Brier score, upset stats, and EV totals."""
        n = len(prediction_results)
        correct = sum(1 for p in prediction_results if p["correct"])
        accuracy = correct / n if n > 0 else 0.0

        brier = self.compute_brier_score(prediction_results)

        upsets = [p for p in prediction_results if p["upset"]]
        upsets_correct = sum(1 for p in upsets if p["correct"])
        upsets_missed = len(upsets) - upsets_correct

        total_ev_predicted = sum(p["ev_predicted"] for p in prediction_results)
        total_ev_actual = sum(p["ev_actual"] for p in prediction_results)

        return {
            "accuracy": round(accuracy, 4),
            "brier_score": round(brier, 4),
            "upsets_predicted_correctly": upsets_correct,
            "upsets_missed": upsets_missed,
            "total_ev_predicted": round(total_ev_predicted, 2),
            "total_ev_actual": round(total_ev_actual, 2),
        }

    @staticmethod
    def compute_brier_score(predictions):
        """Compute Brier score for a list of prediction results.

        Brier score = (1/N) * sum((predicted_prob - outcome)^2)
        where outcome = 1 if the predicted winner actually won, else 0.
        Lower is better (0 = perfect, 0.25 = coin flip on 50/50).

        Args:
            predictions: list of dicts with 'predicted_prob' and 'correct' keys.

        Returns:
            float: Brier score.
        """
        if not predictions:
            return 0.0
        n = len(predictions)
        total = 0.0
        for p in predictions:
            outcome = 1.0 if p["correct"] else 0.0
            prob = p["predicted_prob"]
            total += (prob - outcome) ** 2
        return total / n

    # -- dashboard ----------------------------------------------------------

    def generate_dashboard(self, output_path=None):
        """Generate 4-subplot accuracy dashboard and save to PNG.

        Subplots:
        1. Cumulative accuracy by round (bar chart)
        2. Calibration curve (predicted prob vs actual win rate)
        3. Brier score over time (line chart)
        4. EV tracking (predicted vs actual, cumulative)
        """
        if output_path is None:
            output_path = str(REPO_ROOT / "images" / "accuracy_dashboard.png")

        runs = self.data.get("runs", [])
        if not runs:
            print(
                "[AccuracyTracker] No runs to plot. Generating placeholder dashboard."
            )
            self._generate_empty_dashboard(output_path)
            return output_path

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Prediction Accuracy Dashboard", fontsize=16, fontweight="bold")

        self._plot_accuracy_by_round(axes[0, 0], runs)
        self._plot_calibration_curve(axes[0, 1], runs)
        self._plot_brier_over_time(axes[1, 0], runs)
        self._plot_ev_tracking(axes[1, 1], runs)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[AccuracyTracker] Dashboard saved to {output_path}")
        return output_path

    def _generate_empty_dashboard(self, output_path):
        """Generate a placeholder dashboard when no data is available."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Prediction Accuracy Dashboard (No Data Yet)",
            fontsize=16,
            fontweight="bold",
        )
        for ax in axes.flat:
            ax.text(
                0.5,
                0.5,
                "No prediction data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[AccuracyTracker] Empty dashboard saved to {output_path}")

    def _plot_accuracy_by_round(self, ax, runs):
        """Subplot 1: Cumulative accuracy by round — stacked bar of correct/incorrect."""
        round_correct = {}
        round_incorrect = {}

        for run in runs:
            for pred in run.get("predictions", []):
                rd = run.get("round", "Unknown")
                if rd not in round_correct:
                    round_correct[rd] = 0
                    round_incorrect[rd] = 0
                if pred["correct"]:
                    round_correct[rd] += 1
                else:
                    round_incorrect[rd] += 1

        # Order rounds canonically
        ordered = [r for r in ROUND_ORDER if r in round_correct]
        if not ordered:
            ordered = sorted(round_correct.keys())

        correct_vals = [round_correct.get(r, 0) for r in ordered]
        incorrect_vals = [round_incorrect.get(r, 0) for r in ordered]

        # Shorten labels for display
        short_labels = [_shorten_round(r) for r in ordered]

        x = np.arange(len(ordered))
        width = 0.6
        ax.bar(x, correct_vals, width, label="Correct", color="#2ecc71")
        ax.bar(
            x,
            incorrect_vals,
            width,
            bottom=correct_vals,
            label="Incorrect",
            color="#e74c3c",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Number of Picks")
        ax.set_title("Accuracy by Round")
        ax.legend(loc="upper right", fontsize=8)

    def _plot_calibration_curve(self, ax, runs):
        """Subplot 2: Calibration curve — predicted probability vs actual win rate."""
        all_preds = []
        for run in runs:
            for pred in run.get("predictions", []):
                all_preds.append(
                    (pred["predicted_prob"], 1.0 if pred["correct"] else 0.0)
                )

        if not all_preds:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("Calibration Curve")
            return

        probs, outcomes = zip(*all_preds)
        probs = np.array(probs)
        outcomes = np.array(outcomes)

        # Bin into ~10 bins
        n_bins = min(10, len(all_preds))
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_actuals = []
        bin_counts = []

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (
                (probs >= lo) & (probs < hi)
                if i < n_bins - 1
                else (probs >= lo) & (probs <= hi)
            )
            if mask.sum() > 0:
                bin_centers.append((lo + hi) / 2)
                bin_actuals.append(outcomes[mask].mean())
                bin_counts.append(mask.sum())

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(
            bin_centers, bin_actuals, "o-", color="#3498db", markersize=6, label="Model"
        )

        # Annotate bin counts
        for x_val, y_val, count in zip(bin_centers, bin_actuals, bin_counts):
            ax.annotate(
                f"n={count}",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=7,
                ha="center",
                color="gray",
            )

        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Actual Win Rate")
        ax.set_title("Calibration Curve")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    def _plot_brier_over_time(self, ax, runs):
        """Subplot 3: Brier score over time — line chart per run."""
        timestamps = []
        brier_scores = []

        for i, run in enumerate(runs):
            label = run.get("round", f"Run {i + 1}")
            brier = run.get("summary", {}).get("brier_score")
            if brier is not None:
                timestamps.append(label)
                brier_scores.append(brier)

        if not brier_scores:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("Brier Score Over Time")
            return

        x = np.arange(len(brier_scores))
        ax.plot(x, brier_scores, "o-", color="#9b59b6", markersize=6)
        ax.fill_between(x, brier_scores, alpha=0.15, color="#9b59b6")
        ax.axhline(
            y=0.25, color="red", linestyle=":", alpha=0.5, label="Coin flip (0.25)"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(timestamps, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Brier Score")
        ax.set_title("Brier Score Over Time (lower is better)")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    def _plot_ev_tracking(self, ax, runs):
        """Subplot 4: EV tracking — cumulative predicted EV vs actual points."""
        labels = []
        cum_predicted = []
        cum_actual = []
        running_predicted = 0.0
        running_actual = 0.0

        for i, run in enumerate(runs):
            summary = run.get("summary", {})
            ev_pred = summary.get("total_ev_predicted", 0)
            ev_act = summary.get("total_ev_actual", 0)
            running_predicted += ev_pred
            running_actual += ev_act
            labels.append(run.get("round", f"Run {i + 1}"))
            cum_predicted.append(running_predicted)
            cum_actual.append(running_actual)

        if not labels:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("EV Tracking")
            return

        x = np.arange(len(labels))
        ax.plot(
            x, cum_predicted, "s-", color="#e67e22", markersize=6, label="Predicted EV"
        )
        ax.plot(
            x, cum_actual, "D-", color="#27ae60", markersize=6, label="Actual Points"
        )
        ax.fill_between(x, cum_predicted, cum_actual, alpha=0.1, color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Cumulative Points")
        ax.set_title("EV Predicted vs Actual (Cumulative)")
        ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shorten_round(name):
    """Shorten round names for chart labels."""
    mapping = {
        "Round of 64": "R64",
        "Round of 32": "R32",
        "Sweet 16": "S16",
        "Elite 8": "E8",
        "Final Four": "FF",
        "Championship": "Champ",
    }
    return mapping.get(name, name)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def track_accuracy(state_path=None, log_path=None):
    """Main function: load tournament state, compare predictions to outcomes.

    This function attempts to load tournament state from the state_path JSON.
    If the state file does not exist yet (tournament hasn't started / state
    tracker not yet integrated), it initializes the accuracy log and generates
    an empty dashboard.

    Args:
        state_path: path to tournament_state.json
        log_path: path to accuracy_log.json

    Returns:
        dict with the run summary, or empty summary if no data available.
    """
    if state_path is None:
        state_path = str(REPO_ROOT / "scraped_data" / "tournament_state.json")
    if log_path is None:
        log_path = str(REPO_ROOT / "scraped_data" / "accuracy_log.json")

    tracker = AccuracyTracker(log_path=log_path)

    # Try to load tournament state for live comparison
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                state = json.load(f)

            predictions = state.get("predictions", [])
            completed = state.get("completed_games", [])

            if predictions and completed:
                summary = tracker.log_run(predictions, completed)
                tracker.generate_dashboard()
                return summary or _empty_summary()
            else:
                print(
                    "[AccuracyTracker] Tournament state loaded but no predictions/completed games yet."
                )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[AccuracyTracker] Could not parse tournament state: {e}")
    else:
        print(
            f"[AccuracyTracker] No tournament state file at {state_path}. "
            "Generating dashboard from existing log data."
        )

    # Generate dashboard from whatever log data exists (may be empty)
    tracker.generate_dashboard()
    return _empty_summary()


def _empty_summary():
    """Return an empty summary dict when no data is available."""
    return {
        "accuracy": 0.0,
        "brier_score": 0.0,
        "upsets_predicted_correctly": 0,
        "upsets_missed": 0,
        "total_ev_predicted": 0.0,
        "total_ev_actual": 0.0,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("March Madness Accuracy Tracker")
    print("=" * 60)
    result = track_accuracy()
    print(f"\nSummary: {json.dumps(result, indent=2)}")
    print("Done.")
