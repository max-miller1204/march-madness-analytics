#!/usr/bin/env python3
"""
Re-fit GARCH/HMM/Kalman models with tournament games appended.

Re-computes composite scores with tournament margin adjustments.
Outputs updated model objects and adjusted composites.
"""

import json
from pathlib import Path

import pandas as pd

from scripts.quant_models import (
    HierarchicalGARCH,
    KalmanMomentum,
    TeamHMM,
    _load_kenpom,
    _resolve_name,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "scraped_data"
DEFAULT_STATE_PATH = str(DATA_DIR / "tournament_state.json")
GAMES_CSV = DATA_DIR / "tournament_games.csv"
KENPOM_CSV = DATA_DIR / "kenpom.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_tournament_state(state_path: str) -> dict:
    """Load tournament state JSON. Returns empty dict if file missing."""
    p = Path(state_path)
    if not p.exists():
        print(f"  [refit] No tournament state at {state_path} — nothing to append")
        return {}
    with open(p) as f:
        return json.load(f)


def _completed_games_from_state(state: dict) -> list[dict]:
    """Extract completed game records from tournament state.

    Expects state to have a 'completed_games' key containing a list of dicts
    with keys: team, opponent, date, team_score, opp_score, quadrant (optional).
    Falls back to inspecting 'rounds' structure if present.
    """
    if "completed_games" in state:
        return state["completed_games"]

    # Alternate: walk rounds -> games looking for completed entries
    games = []
    for round_info in state.get("rounds", []):
        for game in round_info.get("games", []):
            if game.get("completed", False) and "winner" in game:
                # Build two records (one per team perspective)
                winner = game["winner"]
                loser = game["loser"]
                w_score = game.get("winner_score", game.get("team_score"))
                l_score = game.get("loser_score", game.get("opp_score"))
                date = game.get("date", "03-20")
                if w_score is not None and l_score is not None:
                    games.append(
                        {
                            "team": winner,
                            "opponent": loser,
                            "date": date,
                            "team_score": int(w_score),
                            "opp_score": int(l_score),
                            "quadrant": "Quadrant 1",
                        }
                    )
                    games.append(
                        {
                            "team": loser,
                            "opponent": winner,
                            "date": date,
                            "team_score": int(l_score),
                            "opp_score": int(w_score),
                            "quadrant": "Quadrant 1",
                        }
                    )
    return games


def _build_tournament_rows(
    completed_games: list[dict], tournament_weight: float
) -> pd.DataFrame:
    """Create a DataFrame of tournament game rows, duplicated by weight.

    Each tournament game is duplicated ``round(tournament_weight)`` times so the
    model sees it as multiple observations.
    """
    if not completed_games:
        return pd.DataFrame()

    rows = []
    n_copies = max(1, round(tournament_weight))
    for game in completed_games:
        row = {
            "team": _resolve_name(game.get("team", "")),
            "opponent": _resolve_name(game.get("opponent", "")),
            "date": game.get("date", "03-20"),
            "team_score": int(game.get("team_score", 0)),
            "opp_score": int(game.get("opp_score", 0)),
            "quadrant": game.get("quadrant", "Quadrant 1"),
            # Preserve columns expected by quant_models loaders
            "team_net_rank": "",
            "sub_quadrant": "",
            "opp_net_rank": "",
            "location": "N",
            "is_conference": False,
            "result": "W"
            if int(game.get("team_score", 0)) > int(game.get("opp_score", 0))
            else "L",
            "overtime": False,
        }
        for _ in range(n_copies):
            rows.append(row.copy())

    return pd.DataFrame(rows)


def _compute_expected_margins(
    kenpom_df: pd.DataFrame, completed_games: list[dict]
) -> dict:
    """Compute expected margin for each tournament game from pre-tournament KenPom.

    Returns dict keyed by (team, opponent, date) -> expected_margin.
    """
    name_col = "StdName" if "StdName" in kenpom_df.columns else "Team"
    net_col = "adjusted_NetRtg" if "adjusted_NetRtg" in kenpom_df.columns else "NetRtg"
    net_lookup = {}
    for _, row in kenpom_df.iterrows():
        net_lookup[row[name_col]] = row[net_col]

    expected = {}
    for game in completed_games:
        team = _resolve_name(game.get("team", ""))
        opp = _resolve_name(game.get("opponent", ""))
        team_net = net_lookup.get(team)
        opp_net = net_lookup.get(opp)
        if team_net is not None and opp_net is not None:
            expected[(team, opp, game.get("date", ""))] = (team_net - opp_net) / 2
        else:
            expected[(team, opp, game.get("date", ""))] = 0.0
    return expected


def _compute_composite_adjustments(
    completed_games: list[dict],
    expected_margins: dict,
    tournament_weight: float,
) -> dict:
    """Compute per-team composite score adjustments from tournament performance.

    For each team, averages (actual_margin - expected_margin) across its tournament
    games and scales by tournament_weight / 10.  The result is a small adjustment
    (positive = outperformed, negative = underperformed) that can be added to the
    team's 0-100 composite score.
    """
    team_deltas: dict[str, list[float]] = {}
    for game in completed_games:
        team = _resolve_name(game.get("team", ""))
        opp = _resolve_name(game.get("opponent", ""))
        date = game.get("date", "")
        actual_margin = int(game.get("team_score", 0)) - int(game.get("opp_score", 0))
        exp = expected_margins.get((team, opp, date), 0.0)
        delta = actual_margin - exp
        team_deltas.setdefault(team, []).append(delta)

    from scripts.quant_models import ADJUSTMENT_SCALE_DIVISOR
    scale = tournament_weight / ADJUSTMENT_SCALE_DIVISOR
    adjustments = {}
    for team, deltas in team_deltas.items():
        avg_delta = sum(deltas) / len(deltas)
        adjustments[team] = avg_delta * scale
    return adjustments


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def refit_models(
    state_path: str = DEFAULT_STATE_PATH,
    tournament_weight: float = 2.0,
) -> dict:
    """Re-fit GARCH/HMM/Kalman with tournament games and compute composite adjustments.

    Parameters
    ----------
    state_path : str
        Path to tournament_state.json (or any JSON with completed game records).
    tournament_weight : float
        How many times each tournament game is duplicated when appended to the
        regular-season data.  E.g. 2.0 means each tournament game counts as 2
        observations.

    Returns
    -------
    dict with keys:
        garch : HierarchicalGARCH — re-fit model object
        hmm   : TeamHMM            — re-fit model object
        kalman: KalmanMomentum     — re-fit model object
        composite_adjustments : dict[str, float] — per-team score adjustments
    """
    print("=" * 60)
    print("Re-fitting models with tournament data")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load base data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading base season data...")
    games_df = pd.read_csv(GAMES_CSV)
    kenpom_df = _load_kenpom(KENPOM_CSV)
    print(f"  Base games: {len(games_df)}, KenPom teams: {len(kenpom_df)}")

    # ------------------------------------------------------------------
    # 2. Load tournament state and build synthetic rows
    # ------------------------------------------------------------------
    print("\n[2/5] Loading tournament state...")
    state = _load_tournament_state(state_path)
    completed_games = _completed_games_from_state(state)
    print(f"  Completed tournament games: {len(completed_games)}")

    tourn_rows = _build_tournament_rows(completed_games, tournament_weight)
    if not tourn_rows.empty:
        augmented_df = pd.concat([games_df, tourn_rows], ignore_index=True)
        print(
            f"  Augmented dataset: {len(augmented_df)} rows "
            f"(+{len(tourn_rows)} tournament copies, weight={tournament_weight})"
        )
    else:
        augmented_df = games_df
        print("  No tournament games to append — using base data only")

    # ------------------------------------------------------------------
    # 3. Re-fit models
    # ------------------------------------------------------------------
    print("\n[3/5] Re-fitting GARCH...")
    garch = HierarchicalGARCH(augmented_df)
    print(
        f"  Alpha={garch.alpha:.4f}, Beta={garch.beta:.4f}, "
        f"Teams={len(garch.team_sigma)}"
    )

    print("\n[4/5] Re-fitting HMM...")
    hmm = TeamHMM(augmented_df)
    print(f"  Teams with HMM models: {len(hmm.team_models)}")

    print("\n[5/5] Re-fitting Kalman...")
    kalman = KalmanMomentum(augmented_df, kenpom_df)
    print(f"  Teams with momentum: {len(kalman.team_momentum)}")

    # ------------------------------------------------------------------
    # 4. Composite adjustments
    # ------------------------------------------------------------------
    expected_margins = _compute_expected_margins(kenpom_df, completed_games)
    composite_adjustments = _compute_composite_adjustments(
        completed_games, expected_margins, tournament_weight
    )

    if composite_adjustments:
        sorted_adj = sorted(composite_adjustments.items(), key=lambda x: -x[1])
        print("\n  Top composite adjustments:")
        for team, adj in sorted_adj[:5]:
            print(f"    {team}: {adj:+.2f}")
        print("  Bottom composite adjustments:")
        for team, adj in sorted_adj[-5:]:
            print(f"    {team}: {adj:+.2f}")

    print("\n" + "=" * 60)
    print("Model re-fit complete.")
    print("=" * 60)

    return {
        "garch": garch,
        "hmm": hmm,
        "kalman": kalman,
        "tournament_adjustments": composite_adjustments,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-fit quant models with tournament data",
    )
    parser.add_argument(
        "--state-path",
        default=DEFAULT_STATE_PATH,
        help="Path to tournament_state.json",
    )
    parser.add_argument(
        "--tournament-weight",
        type=float,
        default=2.0,
        help="Weight multiplier for tournament games (default: 2.0)",
    )
    args = parser.parse_args()

    results = refit_models(
        state_path=args.state_path,
        tournament_weight=args.tournament_weight,
    )

    # Summary
    print(f"\nGARCH teams: {len(results['garch'].team_sigma)}")
    print(f"HMM teams:   {len(results['hmm'].team_models)}")
    print(f"Kalman teams: {len(results['kalman'].team_momentum)}")
    print(f"Composite adjustments: {len(results['composite_adjustments'])} teams")
