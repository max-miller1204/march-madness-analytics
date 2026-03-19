"""Tests for foundation phase: tournament state, validator, and locked_results in simulators."""

import numpy as np
import pandas as pd
import pytest

from scripts.quant_models import (
    EVOptimizedSimulator,
    HierarchicalGARCH,
    HistoricalPrior,
    QuantEnhancedSimulator,
    _build_lookups,
    _load_kenpom,
    _parse_bracket,
    _simulate_region,
)
from scripts.refresh.tournament_state import TournamentState
from scripts.refresh.validator import DataValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_state_path(tmp_path):
    return str(tmp_path / "tournament_state.json")


@pytest.fixture
def bracket_df():
    return pd.read_csv("scraped_data/bracket.csv")


@pytest.fixture
def kenpom_df():
    return _load_kenpom("scraped_data/kenpom.csv")


@pytest.fixture
def games_df():
    return pd.read_csv("scraped_data/tournament_games.csv")


@pytest.fixture
def east_matchups(bracket_df):
    regions = _parse_bracket(bracket_df)
    return regions["East"]


@pytest.fixture
def net_lookup(kenpom_df):
    net_lookup, _ = _build_lookups(kenpom_df)
    return net_lookup


# ---------------------------------------------------------------------------
# TournamentState
# ---------------------------------------------------------------------------


class TestTournamentState:
    def test_default_state(self, tmp_state_path):
        ts = TournamentState(tmp_state_path)
        assert ts.games_completed_count() == 0
        assert ts.detect_round() == "Pre-tournament"

    def test_add_result(self, tmp_state_path):
        ts = TournamentState(tmp_state_path)
        added = ts.add_result(
            "E1", "Round of 64", "East", 1, "Duke", 16, "Siena", 82, 55
        )
        assert added is True
        assert ts.games_completed_count() == 1
        assert ts.detect_round() == "Round of 64"

    def test_duplicate_result_rejected(self, tmp_state_path):
        ts = TournamentState(tmp_state_path)
        ts.add_result("E1", "Round of 64", "East", 1, "Duke", 16, "Siena", 82, 55)
        added = ts.add_result(
            "E1", "Round of 64", "East", 1, "Duke", 16, "Siena", 82, 55
        )
        assert added is False
        assert ts.games_completed_count() == 1

    def test_locked_results_format(self, tmp_state_path):
        ts = TournamentState(tmp_state_path)
        ts.add_result("E1", "Round of 64", "East", 1, "Duke", 16, "Siena", 82, 55)
        locked = ts.get_locked_results()
        assert locked == {"E1": {"team": "Duke", "seed": 1}}

    def test_elimination_tracking(self, tmp_state_path):
        ts = TournamentState(tmp_state_path)
        ts.add_result("E1", "Round of 64", "East", 1, "Duke", 16, "Siena", 82, 55)
        assert ts.is_team_eliminated("Siena")
        assert not ts.is_team_eliminated("Duke")

    def test_save_and_reload(self, tmp_state_path):
        ts = TournamentState(tmp_state_path)
        ts.add_result("E1", "Round of 64", "East", 1, "Duke", 16, "Siena", 82, 55)
        ts.save()
        ts2 = TournamentState(tmp_state_path)
        assert ts2.games_completed_count() == 1
        assert ts2.get_locked_results() == {"E1": {"team": "Duke", "seed": 1}}

    def test_round_detection_boundaries(self, tmp_state_path):
        ts = TournamentState(tmp_state_path)
        assert ts.detect_round() == "Pre-tournament"  # 0 games

        for i in range(32):
            ts.add_result(
                f"G{i}", "Round of 64", "East", 1, f"W{i}", 16, f"L{i}", 70, 60
            )
        assert ts.detect_round() == "Round of 64"  # 32 games

        ts.add_result("G32", "Round of 32", "East", 1, "W32", 8, "L32", 70, 60)
        assert ts.detect_round() == "Round of 32"  # 33 games

    def test_game_id_derivation(self):
        assert TournamentState.derive_game_id("East", "Round of 64", 0) == "E1"
        assert TournamentState.derive_game_id("East", "Round of 64", 7) == "E8"
        assert TournamentState.derive_game_id("South", "Round of 32", 0) == "S9"
        assert TournamentState.derive_game_id("West", "Sweet 16", 1) == "W14"
        assert TournamentState.derive_game_id("Midwest", "Elite 8", 0) == "M15"
        assert TournamentState.derive_game_id("East", "Final Four", 0) == "F1"
        assert TournamentState.derive_game_id("East", "Final Four", 1) == "F2"
        assert TournamentState.derive_game_id("East", "Championship", 0) == "C1"


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------


class TestDataValidator:
    def test_valid_game(self):
        v = DataValidator()
        assert v.validate_tournament_results(
            [
                {
                    "game_id": "E1",
                    "team_a": "Duke",
                    "team_b": "Siena",
                    "score_a": 82,
                    "score_b": 55,
                    "winner": "Duke",
                }
            ]
        )

    def test_empty_list_valid(self):
        v = DataValidator()
        assert v.validate_tournament_results([])

    def test_zero_score_rejected(self):
        v = DataValidator()
        assert not v.validate_tournament_results(
            [
                {
                    "game_id": "E1",
                    "team_a": "A",
                    "team_b": "B",
                    "score_a": 0,
                    "score_b": 55,
                    "winner": "B",
                }
            ]
        )

    def test_duplicate_game_ids_rejected(self):
        v = DataValidator()
        game = {
            "game_id": "E1",
            "team_a": "Duke",
            "team_b": "Siena",
            "score_a": 82,
            "score_b": 55,
            "winner": "Duke",
        }
        assert not v.validate_tournament_results([game, game])

    def test_wrong_winner_rejected(self):
        v = DataValidator()
        assert not v.validate_tournament_results(
            [
                {
                    "game_id": "E1",
                    "team_a": "Duke",
                    "team_b": "Siena",
                    "score_a": 82,
                    "score_b": 55,
                    "winner": "Siena",
                }
            ]
        )

    def test_bracket_csv_validation(self, bracket_df):
        v = DataValidator()
        assert v.validate_bracket_csv(bracket_df)


# ---------------------------------------------------------------------------
# Locked Results in Simulators
# ---------------------------------------------------------------------------


class TestLockedResults:
    @staticmethod
    def _make_wp(net_lookup):
        def wp(ta, sa, tb, sb, rng):
            na = net_lookup.get(ta, 0)
            nb = net_lookup.get(tb, 0)
            return 1 / (1 + 10 ** (-(na - nb) / 22))

        return wp

    def test_locked_game_deterministic(self, east_matchups, net_lookup):
        team_a, seed_a, team_b, seed_b = east_matchups[0]
        locked = {"E1": {"winner": team_b, "seed": seed_b}}
        wp = self._make_wp(net_lookup)

        for i in range(100):
            rng = np.random.default_rng(i)
            _, _, results = _simulate_region(
                east_matchups, wp, rng, locked_results=locked, region="East"
            )
            assert results[0][1] == team_b, (
                f"Sim {i}: expected {team_b}, got {results[0][1]}"
            )

    def test_empty_locked_results_unchanged(self, east_matchups, net_lookup):
        wp = self._make_wp(net_lookup)
        rng1 = np.random.default_rng(42)
        w1, _, _ = _simulate_region(
            east_matchups, wp, rng1, locked_results={}, region="East"
        )
        rng2 = np.random.default_rng(42)
        w2, _, _ = _simulate_region(east_matchups, wp, rng2)
        assert w1 == w2

    def test_quant_simulator_with_locked(self, bracket_df, kenpom_df, games_df):
        garch = HierarchicalGARCH(games_df)
        prior = HistoricalPrior("scraped_data")
        locked = {"E1": {"winner": "Siena", "seed": 16}}
        sim = QuantEnhancedSimulator(
            bracket_df=bracket_df,
            kenpom_df=kenpom_df,
            garch=garch,
            prior=prior,
            locked_results=locked,
            n_sims=50,
            seed=42,
        )
        result = sim.run()
        assert result["champion"] is not None

    def test_ev_simulator_with_locked(self, bracket_df, kenpom_df, games_df):
        garch = HierarchicalGARCH(games_df)
        locked = {"E1": {"winner": "Siena", "seed": 16}}
        sim = EVOptimizedSimulator(
            bracket_df=bracket_df,
            kenpom_df=kenpom_df,
            garch=garch,
            locked_results=locked,
            n_sims=50,
            seed=42,
        )
        result = sim.run()
        assert result["champion"] is not None
