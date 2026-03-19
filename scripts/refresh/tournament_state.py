"""Tournament state management — tracks completed games, predictions, and bracket progression."""

import json
from datetime import datetime, timezone
from pathlib import Path


# Round detection thresholds
ROUND_THRESHOLDS = [
    (0, "Pre-tournament"),
    (32, "Round of 64"),
    (48, "Round of 32"),
    (56, "Sweet 16"),
    (60, "Elite 8"),
    (62, "Final Four"),
    (63, "Championship"),
]

# Game ID derivation: R64 games are E1-E8, S1-S8, W1-W8, M1-M8
REGION_PREFIXES = {"East": "E", "South": "S", "West": "W", "Midwest": "M"}

# R32 game IDs: winner of game i vs winner of game i+1 → game prefix + (8 + ceil(i/2))
# S16: prefix + 13/14, E8: prefix + 15, FF: F1/F2, Championship: C1


class TournamentState:
    """Manages tournament_state.json — load, save, query, and update bracket state."""

    def __init__(self, state_path="scraped_data/tournament_state.json"):
        self.state_path = Path(state_path)
        self.state = self._default_state()
        if self.state_path.exists():
            self.load()

    @staticmethod
    def _default_state():
        return {
            "last_updated": None,
            "current_round": "Pre-tournament",
            "completed_games": [],
            "predictions_at_time": {},
            "eliminated_teams": [],
            "advancing_teams": {},
        }

    def load(self):
        """Load state from JSON file."""
        with open(self.state_path) as f:
            self.state = json.load(f)

    def save(self):
        """Persist state to JSON file."""
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.state["current_round"] = self.detect_round()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def detect_round(self):
        """Determine current round based on number of completed games."""
        n = len(self.state["completed_games"])
        current = "Pre-tournament"
        for threshold, round_name in ROUND_THRESHOLDS:
            if n > threshold:
                current = round_name
            elif n == 0:
                break
        # Edge case: exactly at a threshold boundary
        for threshold, round_name in ROUND_THRESHOLDS:
            if n <= threshold:
                return round_name
        return "Championship"

    def add_result(self, game_id, round_name, region, seed_a, team_a,
                   seed_b, team_b, score_a, score_b):
        """Add a completed game result. Returns True if new, False if duplicate."""
        # Check for duplicate
        existing_ids = {g["game_id"] for g in self.state["completed_games"]}
        if game_id in existing_ids:
            return False

        winner = team_a if score_a > score_b else team_b
        loser = team_b if winner == team_a else team_a
        winner_seed = seed_a if winner == team_a else seed_b

        game = {
            "game_id": game_id,
            "round": round_name,
            "region": region,
            "seed_a": seed_a,
            "team_a": team_a,
            "seed_b": seed_b,
            "team_b": team_b,
            "score_a": score_a,
            "score_b": score_b,
            "winner": winner,
            "margin": abs(score_a - score_b),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.state["completed_games"].append(game)

        # Update eliminated / advancing
        if loser not in self.state["eliminated_teams"]:
            self.state["eliminated_teams"].append(loser)
        self.state["advancing_teams"][game_id] = {
            "team": winner,
            "seed": winner_seed,
        }
        return True

    def record_prediction(self, round_name, game_id, predicted_winner, win_prob):
        """Record a prediction for a game at the current point in time."""
        if round_name not in self.state["predictions_at_time"]:
            self.state["predictions_at_time"][round_name] = {}
        self.state["predictions_at_time"][round_name][game_id] = {
            "predicted_winner": predicted_winner,
            "win_prob": win_prob,
            "predicted_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_locked_results(self):
        """Return dict of locked results for simulator consumption.

        Format: {game_id: {"winner": team_name, "seed": seed_int}}
        """
        return dict(self.state["advancing_teams"])

    def get_completed_games(self):
        """Return list of completed game dicts."""
        return list(self.state["completed_games"])

    def get_eliminated_teams(self):
        """Return set of eliminated team names."""
        return set(self.state["eliminated_teams"])

    def is_team_eliminated(self, team):
        """Check if a team has been eliminated."""
        return team in self.state["eliminated_teams"]

    def games_completed_count(self):
        """Return number of completed games."""
        return len(self.state["completed_games"])

    @staticmethod
    def derive_game_id(region, round_name, game_index):
        """Derive a game ID from region, round, and positional index.

        R64: E1-E8 (game_index 0-7)
        R32: E9-E12 (game_index 0-3)
        S16: E13-E14 (game_index 0-1)
        E8:  E15 (game_index 0)
        FF:  F1, F2
        Championship: C1
        """
        if round_name == "Final Four":
            return f"F{game_index + 1}"
        if round_name == "Championship":
            return "C1"

        prefix = REGION_PREFIXES.get(region, "X")
        offsets = {
            "Round of 64": 1,      # E1-E8
            "Round of 32": 9,      # E9-E12
            "Sweet 16": 13,        # E13-E14
            "Elite 8": 15,         # E15
        }
        offset = offsets.get(round_name, 1)
        return f"{prefix}{offset + game_index}"
