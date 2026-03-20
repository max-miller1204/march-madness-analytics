#!/usr/bin/env python3
"""
Quantitative Models for March Madness bracket prediction.

Unit 2: GARCH volatility, HMM regime detection, Kalman momentum,
historical priors, and Monte Carlo bracket simulation.
"""

import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAME_ALIASES = {
    "Michigan St.": "Michigan State",
    "Iowa St.": "Iowa State",
    "Ohio St.": "Ohio State",
    "Connecticut": "UConn",
    "Utah St.": "Utah State",
    "Miami FL": "Miami (FL)",
    "N.C. State": "NC State",
    "North Dakota St.": "North Dakota State",
    "Wright St.": "Wright State",
    "Kennesaw St.": "Kennesaw State",
    "Miami OH": "Miami (OH)",
    "Tennessee St.": "Tennessee State",
    "LIU": "Long Island University",
    "Queens": "Queens (NC)",
    "South Fla.": "South Florida",
    "UNI": "Northern Iowa",
    "California Baptist": "Cal Baptist",
    "Saint Mary's (CA)": "Saint Mary's",
    "St. John's (NY)": "St. John's",
    "Long Island": "Long Island University",
    "Saint Mary's College": "Saint Mary's",
    "Saint John's": "St. John's",
    "North Carolina State": "NC State",
}

REGION_SEED_MAP = {"East": 0, "South": 1, "West": 2, "Midwest": 3}

REGION_PREFIXES = {"East": "E", "South": "S", "West": "W", "Midwest": "M"}

QUAD_WEIGHTS = {
    "Quadrant 1": 1.5,
    "Quadrant 2": 1.2,
    "Quadrant 3": 1.0,
    "Quadrant 4": 0.8,
}

DEFAULT_RATES = {
    (1, 16): 0.994,
    (2, 15): 0.944,
    (3, 14): 0.850,
    (4, 13): 0.793,
    (5, 12): 0.644,
    (6, 11): 0.623,
    (7, 10): 0.609,
    (8, 9): 0.515,
}


def _resolve_name(name):
    """Resolve a team name through aliases."""
    return NAME_ALIASES.get(name, name)


def _sort_date_key(date_str):
    """Return a sortable key for MM-DD dates (months 10-12 before 01-04)."""
    month, day = date_str.split("-")
    month, day = int(month), int(day)
    # Months 10-12 are previous year → offset so they sort before 01-04
    if month >= 10:
        return (0, month, day)
    return (1, month, day)


def _load_kenpom(path):
    """Load kenpom.csv and return DataFrame with clean team names and NetRtg."""
    df = pd.read_csv(path, skiprows=2, header=None)
    df.columns = [
        "Rk",
        "Team",
        "Conf",
        "W_L",
        "NetRtg",
        "ORtg",
        "DRtg",
        "AdjT",
        "Luck",
        "SOS_NetRtg",
        "SOS_ORtg",
        "SOS_DRtg",
        "NCSOS_NetRtg",
    ]
    # Extract seed from team name (e.g. "Duke 1" → "Duke", seed 1)
    df["Seed"] = df["Team"].str.extract(r"\s+(\d+)$").astype(float)
    df["Team"] = df["Team"].str.replace(r"\s+\d+$", "", regex=True).str.strip()
    df["Team"] = df["Team"].map(lambda x: _resolve_name(x))
    df["NetRtg"] = pd.to_numeric(df["NetRtg"], errors="coerce")
    return df


def _build_team_margins(games_df):
    """Build per-team margin series sorted by date."""
    team_margins = {}
    for team, group in games_df.groupby("team"):
        g = group.copy()
        g["_sort"] = g["date"].apply(_sort_date_key)
        g = g.sort_values("_sort")
        margins = (g["team_score"].astype(float) - g["opp_score"].astype(float)).values
        team_margins[_resolve_name(team)] = margins
    return team_margins


def _build_lookups(kenpom_df):
    """Build net_lookup and seed_lookup dicts from a KenPom DataFrame."""
    name_col = "StdName" if "StdName" in kenpom_df.columns else "Team"
    net_col = "adjusted_NetRtg" if "adjusted_NetRtg" in kenpom_df.columns else "NetRtg"
    net_lookup = {}
    seed_lookup = {}
    for _, row in kenpom_df.iterrows():
        net_lookup[row[name_col]] = row[net_col]
        if pd.notna(row.get("Seed")):
            seed_lookup[row[name_col]] = int(row["Seed"])
    return net_lookup, seed_lookup


def _parse_bracket(bracket_df):
    """Parse bracket.csv into region -> list of R64 matchups."""
    regions = {}
    for _, row in bracket_df.iterrows():
        if row["Round"] != "R64":
            continue
        region = row["Region"]
        if region not in regions:
            regions[region] = []
        team_a = _resolve_name(row["TeamA"].split("/")[0].strip())
        team_b = _resolve_name(row["TeamB"].split("/")[0].strip())
        seed_a = int(row["SeedA"])
        seed_b = int(row["SeedB"])
        regions[region].append((team_a, seed_a, team_b, seed_b))
    return regions


def _resolve_matchup(
    game_id, team_a, seed_a, team_b, seed_b, win_prob_fn, rng, locked_results=None
):
    """Resolve a single matchup, checking locked results first.

    Returns (winner, w_seed, loser, l_seed).
    """
    if locked_results and game_id in locked_results:
        locked_winner = locked_results[game_id]["winner"]
        if locked_winner == team_a:
            return (team_a, seed_a, team_b, seed_b)
        elif locked_winner == team_b:
            return (team_b, seed_b, team_a, seed_a)
        print(
            f"  [WARN] Locked result for {game_id} names '{locked_winner}' "
            f"but participants are '{team_a}' vs '{team_b}' — ignoring lock"
        )
    p = win_prob_fn(team_a, seed_a, team_b, seed_b, rng)
    if rng.random() < p:
        return (team_a, seed_a, team_b, seed_b)
    return (team_b, seed_b, team_a, seed_a)


def _simulate_region(matchups, win_prob_fn, rng, locked_results=None, region=None):
    """Simulate a single region (4 rounds, 8->4->2->1).

    Args:
        matchups: list of (team_a, seed_a, team_b, seed_b) tuples
        win_prob_fn: callable(team_a, seed_a, team_b, seed_b, rng) -> float
        rng: numpy random generator
        locked_results: optional dict of {game_id: {"winner": name, "seed": int}}
        region: region name (needed for game ID derivation when using locked_results)

    Returns (winner_team, winner_seed, game_results).
    game_results: list of (round_name, winner, w_seed, loser, l_seed, win_p)
    """
    round_names = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]
    # Game ID offsets per round within a region: R64=1-8, R32=9-12, S16=13-14, E8=15
    round_offsets = [1, 9, 13, 15]
    region_prefix = REGION_PREFIXES.get(region, "") if region else ""
    game_results = []

    def _check_locked(game_id, team_a, seed_a, team_b, seed_b):
        """Check if a game has a locked result. Returns (winner, w_seed, loser, l_seed, p) or None."""
        if not locked_results or not game_id or game_id not in locked_results:
            return None
        locked = locked_results[game_id]
        locked_winner = locked["winner"]
        if locked_winner == team_a:
            return (team_a, seed_a, team_b, seed_b, 1.0)
        elif locked_winner == team_b:
            return (team_b, seed_b, team_a, seed_a, 1.0)
        print(
            f"  [WARN] Locked result for {game_id} names '{locked_winner}' "
            f"but participants are '{team_a}' vs '{team_b}' — ignoring lock"
        )
        return None

    round_winners = []
    for idx, (team_a, seed_a, team_b, seed_b) in enumerate(matchups):
        game_id = f"{region_prefix}{round_offsets[0] + idx}" if region_prefix else None
        locked = _check_locked(game_id, team_a, seed_a, team_b, seed_b)
        if locked:
            w, ws, l, ls, p = locked
            round_winners.append((w, ws))
            game_results.append((round_names[0], w, ws, l, ls, p))
        else:
            p = win_prob_fn(team_a, seed_a, team_b, seed_b, rng)
            if rng.random() < p:
                round_winners.append((team_a, seed_a))
                game_results.append((round_names[0], team_a, seed_a, team_b, seed_b, p))
            else:
                round_winners.append((team_b, seed_b))
                game_results.append(
                    (round_names[0], team_b, seed_b, team_a, seed_a, 1 - p)
                )

    for rd in range(1, 4):
        next_round = []
        game_idx = 0
        for i in range(0, len(round_winners), 2):
            if i + 1 < len(round_winners):
                ta, sa = round_winners[i]
                tb, sb = round_winners[i + 1]
                game_id = (
                    f"{region_prefix}{round_offsets[rd] + game_idx}"
                    if region_prefix
                    else None
                )
                game_idx += 1
                locked = _check_locked(game_id, ta, sa, tb, sb)
                if locked:
                    w, ws, l, ls, p = locked
                    next_round.append((w, ws))
                    game_results.append((round_names[rd], w, ws, l, ls, p))
                else:
                    p = win_prob_fn(ta, sa, tb, sb, rng)
                    if rng.random() < p:
                        next_round.append((ta, sa))
                        game_results.append((round_names[rd], ta, sa, tb, sb, p))
                    else:
                        next_round.append((tb, sb))
                        game_results.append((round_names[rd], tb, sb, ta, sa, 1 - p))
            else:
                next_round.append(round_winners[i])
        round_winners = next_round

    winner = round_winners[0] if round_winners else (None, None)
    return winner[0], winner[1], game_results


# ===========================================================================
# Class 1: HierarchicalGARCH
# ===========================================================================


class HierarchicalGARCH:
    """Pooled GARCH(1,1) on game margins → per-team terminal volatility."""

    def __init__(self, games_df):
        self.team_margins = _build_team_margins(games_df)
        self.alpha = None
        self.beta = None
        self.team_omega = {}
        self.team_sigma = {}
        self._fit()

    def _fit(self):
        try:
            from arch import arch_model
        except ImportError:
            print("  [GARCH] arch library not available — using fallback σ=11.0")
            self._fallback()
            return

        # Concatenate demeaned margins for pooled fit
        all_margins = []
        for team, margins in self.team_margins.items():
            demeaned = margins - margins.mean()
            all_margins.append(demeaned)
        pooled = np.concatenate(all_margins)

        try:
            am = arch_model(
                pooled, vol="Garch", p=1, o=0, q=1, dist="Normal", mean="Zero"
            )
            res = am.fit(disp="off")
            params = res.params
            self.alpha = params.get("alpha[1]", 0.1)
            self.beta = params.get("beta[1]", 0.85)
        except Exception as e:
            print(f"  [GARCH] Fit failed ({e}) — using fallback σ=11.0")
            self._fallback()
            return

        if self.alpha + self.beta >= 1.0:
            print(f"  [GARCH] Non-stationary fit (α+β={self.alpha + self.beta:.3f}) — using fallback")
            self._fallback()
            return

        # Per-team omega from team's own margin variance
        for team, margins in self.team_margins.items():
            var_t = np.var(margins, ddof=1) if len(margins) > 1 else np.var(margins)
            self.team_omega[team] = var_t * (1 - self.alpha - self.beta)
            # Terminal volatility via recursive GARCH
            sigma2 = var_t  # start from unconditional variance
            demeaned = margins - margins.mean()
            for r in demeaned:
                sigma2 = self.team_omega[team] + self.alpha * r**2 + self.beta * sigma2
            self.team_sigma[team] = np.sqrt(max(sigma2, 1.0))

    def _fallback(self):
        self.alpha = 0.1
        self.beta = 0.85
        for team, margins in self.team_margins.items():
            self.team_sigma[team] = 11.0
            self.team_omega[team] = 0.0

    def combined_volatility(self, team_a, team_b):
        """Combined volatility for a matchup: sqrt(σ_A² + σ_B²)."""
        sa = self.team_sigma.get(team_a, 11.0)
        sb = self.team_sigma.get(team_b, 11.0)
        return np.sqrt(sa**2 + sb**2)


# ===========================================================================
# Class 2: TeamHMM
# ===========================================================================


class TeamHMM:
    """Per-team HMM on quadrant-adjusted margins with BIC model selection."""

    def __init__(self, games_df):
        self.team_data = {}
        self.team_models = {}
        self.team_posteriors = {}
        self._build_data(games_df)
        self._fit_all()

    def _build_data(self, games_df):
        for team, group in games_df.groupby("team"):
            g = group.copy()
            g["_sort"] = g["date"].apply(_sort_date_key)
            g = g.sort_values("_sort")
            margins = g["team_score"].astype(float) - g["opp_score"].astype(float)
            weights = g["quadrant"].map(QUAD_WEIGHTS).fillna(1.0)
            adj_margins = (margins * weights).values
            resolved = _resolve_name(team)
            if len(adj_margins) >= 8:
                self.team_data[resolved] = adj_margins

    def _fit_all(self):
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            print("  [HMM] hmmlearn not available — skipping")
            return

        for team, data in self.team_data.items():
            X = data.reshape(-1, 1)
            best_model = None
            best_bic = np.inf

            for n in [2, 3, 4]:
                try:
                    model = GaussianHMM(
                        n_components=n,
                        covariance_type="full",
                        n_iter=100,
                        random_state=42,
                    )
                    model.fit(X)
                    log_likelihood = model.score(X)
                    n_params = (
                        n * n + n - 1 + n + n
                    )  # transition + start + means + covars
                    bic = -2 * log_likelihood + n_params * np.log(len(X))
                    if bic < best_bic:
                        best_bic = bic
                        best_model = model
                except Exception:
                    continue

            if best_model is not None:
                self.team_models[team] = best_model
                posteriors = best_model.predict_proba(X)
                self.team_posteriors[team] = posteriors[-1]  # last time step

    def sample_state_adjustment(self, team, rng=None):
        """Sample a hidden state from posterior and return emission mean offset."""
        if team not in self.team_models:
            return 0.0
        if rng is None:
            rng = np.random.default_rng()

        model = self.team_models[team]
        posterior = self.team_posteriors[team]

        # Sample state from posterior
        state = rng.choice(len(posterior), p=posterior)
        # Return emission mean for that state as the offset
        return float(model.means_[state, 0])


# ===========================================================================
# Class 3: KalmanMomentum
# ===========================================================================


class KalmanMomentum:
    """Kalman filter on per-game residuals → team momentum score."""

    def __init__(self, games_df, kenpom_df):
        self.team_momentum = {}
        self._fit_all(games_df, kenpom_df)

    def _fit_all(self, games_df, kenpom_df):
        try:
            from filterpy.kalman import KalmanFilter
        except ImportError:
            print("  [Kalman] filterpy not available — skipping")
            return

        # Build NetRtg lookup
        name_col = "StdName" if "StdName" in kenpom_df.columns else "Team"
        net_col = (
            "adjusted_NetRtg" if "adjusted_NetRtg" in kenpom_df.columns else "NetRtg"
        )
        net_lookup = {}
        for _, row in kenpom_df.iterrows():
            net_lookup[row[name_col]] = row[net_col]

        raw_momentum = {}

        for team, group in games_df.groupby("team"):
            resolved = _resolve_name(team)
            g = group.copy()
            g["_sort"] = g["date"].apply(_sort_date_key)
            g = g.sort_values("_sort")

            team_net = net_lookup.get(resolved, None)
            if team_net is None:
                continue

            residuals = []
            for _, game in g.iterrows():
                opp_name = _resolve_name(game["opponent"])
                opp_net = net_lookup.get(opp_name, None)
                if opp_net is None:
                    # Try looking up with the raw name
                    opp_net = net_lookup.get(game["opponent"], None)
                if opp_net is None:
                    continue
                expected_margin = (team_net - opp_net) / 2
                actual_margin = float(game["team_score"]) - float(game["opp_score"])
                residuals.append(actual_margin - expected_margin)

            if len(residuals) < 3:
                continue

            # Kalman filter: 1D random walk
            kf = KalmanFilter(dim_x=1, dim_z=1)
            kf.x = np.array([[0.0]])  # initial state
            kf.F = np.array([[1.0]])  # state transition
            kf.H = np.array([[1.0]])  # measurement function
            kf.P = np.array([[10.0]])  # initial covariance
            kf.Q = np.array([[2.0]])  # process noise
            kf.R = np.array([[10.0]])  # measurement noise

            for r in residuals:
                kf.predict()
                kf.update(np.array([[r]]))

            raw_momentum[resolved] = float(kf.x[0, 0])

        # Normalize to 0-100 scale
        if raw_momentum:
            vals = np.array(list(raw_momentum.values()))
            mn, mx = vals.min(), vals.max()
            rng = mx - mn if mx != mn else 1.0
            for team, v in raw_momentum.items():
                self.team_momentum[team] = ((v - mn) / rng) * 100.0

    def get_normalized_momentum(self, team):
        """Return 0-100 normalized momentum for the given team."""
        return self.team_momentum.get(team, 50.0)


# ===========================================================================
# Class 4: HistoricalPrior
# ===========================================================================


class HistoricalPrior:
    """Historical seed matchup win rates for Bayesian blending."""

    def __init__(self, data_dir="scraped_data"):
        self.rates = {}
        self.sample_sizes = {}
        self._load(data_dir)

    def _load(self, data_dir):
        csv_path = os.path.join(data_dir, "historical_seed_rates.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    key = (int(row.iloc[0]), int(row.iloc[1]))
                    self.rates[key] = float(row.iloc[2])
                    self.sample_sizes[key] = int(row.iloc[3]) if len(row) > 3 else 50
                return
            except Exception:
                pass
        # Fall back to defaults
        for key, rate in DEFAULT_RATES.items():
            self.rates[key] = rate
            self.sample_sizes[key] = 50  # reasonable default

    def blend(self, p_model, seed_a, seed_b, combined_vol):
        """Bayesian blend of model probability with historical prior."""
        seed_a, seed_b = int(seed_a), int(seed_b)
        key = (min(seed_a, seed_b), max(seed_a, seed_b))
        if key not in self.rates:
            return p_model

        p_hist = self.rates[key]
        # If the higher seed is seed_a, p_hist is already the right direction
        # If seed_a is the higher seed number (lower seed), flip
        if seed_a > seed_b:
            p_hist = 1.0 - p_hist

        n = self.sample_sizes.get(key, 50)
        w_hist = np.sqrt(n)
        base_weight = 25.0
        w_model = base_weight / max(combined_vol, 1.0)

        blended = (w_hist * p_hist + w_model * p_model) / (w_hist + w_model)
        return blended


# ===========================================================================
# Class 5: PublicOwnership
# ===========================================================================


class PublicOwnership:
    """Estimate public pick percentages by seed for contrarian leverage.

    Higher-seeded (lower number) teams get higher public ownership.
    These estimates approximate typical ESPN bracket challenge distributions.
    """

    # Base public ownership by seed (approximate ESPN bracket challenge rates)
    SEED_OWNERSHIP = {
        1: 0.92,
        2: 0.85,
        3: 0.78,
        4: 0.68,
        5: 0.55,
        6: 0.52,
        7: 0.48,
        8: 0.46,
        9: 0.44,
        10: 0.42,
        11: 0.38,
        12: 0.35,
        13: 0.15,
        14: 0.10,
        15: 0.06,
        16: 0.02,
    }

    # Round-level decay: public concentrates on chalk more in later rounds
    ROUND_DECAY = {
        "Round of 64": 1.0,
        "Round of 32": 1.05,
        "Sweet 16": 1.10,
        "Elite 8": 1.15,
        "Final Four": 1.20,
        "Championship": 1.25,
    }

    def ownership(self, seed, round_name="Round of 64"):
        """Return estimated public ownership % for a team given seed and round.

        Asymmetric scaling: chalk (>=50% base) gets amplified in later rounds
        because the public concentrates on favorites; underdogs get suppressed
        because casual bettors abandon upset picks in deeper rounds.
        """
        base = self.SEED_OWNERSHIP.get(int(seed), 0.30)
        decay = self.ROUND_DECAY.get(round_name, 1.0)
        if base >= 0.5:
            return min(base * decay, 0.99)
        return max(base / decay, 0.01)

    def leverage(self, ev, ownership_pct, weight=1.0):
        """Compute leverage-adjusted EV: reward picks the public is fading.

        leverage_ev = ev * (1 + weight * (1 - ownership_pct))
        High EV + low ownership = high leverage score.
        """
        return ev * (1.0 + weight * (1.0 - ownership_pct))


# ===========================================================================
# Class 6: EVOptimizedSimulator
# ===========================================================================


class EVOptimizedSimulator:
    """Unconstrained bracket optimizer maximizing expected ESPN fantasy points.

    Unlike QuantEnhancedSimulator, this class:
    - Does NOT blend with historical seed priors (pure model probability)
    - Scores each game slot by expected fantasy points
    - Applies contrarian leverage to find high-EV, low-ownership picks

    ESPN scoring: R64=10, R32=20, S16=40, E8=80, FF=160, Championship=320
    """

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

    def __init__(
        self,
        bracket_df=None,
        kenpom_df=None,
        garch=None,
        hmm=None,
        kalman=None,
        n_sims=10000,
        seed=42,
        leverage_weight=1.5,
        locked_results=None,
        tournament_adjustments=None,
        # Aliases for notebook compatibility
        df=None,
        bracket=None,
    ):
        self.bracket_df = bracket_df if bracket_df is not None else bracket
        self.kenpom_df = kenpom_df if kenpom_df is not None else df
        self.garch = garch
        self.hmm = hmm
        self.kalman = kalman
        self.n_sims = n_sims
        self.seed = seed
        self.leverage_weight = leverage_weight
        self.locked_results = locked_results or {}
        self.tournament_adjustments = tournament_adjustments or {}
        self.pub = PublicOwnership()

        # Build lookups
        self.net_lookup, self.seed_lookup = _build_lookups(self.kenpom_df)
        self.region_games = _parse_bracket(self.bracket_df)

        # Results
        self.ev_bracket = {}
        self.leverage_bracket = {}
        self.slot_evs = {}
        self.champion = None
        self.total_ev = 0.0
        self.total_leverage_ev = 0.0

    def _win_prob(self, team_a, seed_a, team_b, seed_b, rng):
        """Win probability WITHOUT historical prior blending (pure model)."""
        net_a = self.net_lookup.get(team_a, 0.0)
        net_b = self.net_lookup.get(team_b, 0.0)

        # Tournament performance adjustment (compounding across games)
        if self.tournament_adjustments:
            net_a = net_a + self.tournament_adjustments.get(team_a, 0.0)
            net_b = net_b + self.tournament_adjustments.get(team_b, 0.0)

        # HMM state adjustment
        if self.hmm is not None:
            adj_a = self.hmm.sample_state_adjustment(team_a, rng)
            adj_b = self.hmm.sample_state_adjustment(team_b, rng)
            net_a = net_a + adj_a * 0.3
            net_b = net_b + adj_b * 0.3

        # Kalman momentum adjustment
        if self.kalman is not None:
            mom_a = self.kalman.get_normalized_momentum(team_a)
            mom_b = self.kalman.get_normalized_momentum(team_b)
            net_a = net_a + (mom_a - 50) * 0.15
            net_b = net_b + (mom_b - 50) * 0.15

        margin = (net_a - net_b) / 2

        # GARCH combined volatility
        if self.garch is not None:
            combined_vol = self.garch.combined_volatility(team_a, team_b)
        else:
            combined_vol = 11.0

        if combined_vol > 0:
            p_model = 1 / (1 + 10 ** (-margin / combined_vol))
        else:
            p_model = 1 / (1 + 10 ** (-margin / 11))

        # NO prior blending — pure model probability
        return np.clip(p_model, 0.01, 0.99)

    def run(self):
        """3-phase run: simulate, compute per-slot EV, apply leverage.

        Returns dict with ev_bracket, leverage_bracket, slot_evs, champion,
        total_ev, total_leverage_ev.
        """
        rng = np.random.default_rng(self.seed)
        region_order = ["East", "South", "West", "Midwest"]

        # Phase 1: Monte Carlo simulation — count wins per slot
        # slot key: (round_name, region_or_label, game_idx)
        # value: {(winner, w_seed, loser, l_seed): count}
        game_slot_wins = defaultdict(lambda: defaultdict(int))
        champ_counts = defaultdict(int)

        for _sim in range(self.n_sims):
            ff_teams = []

            for region in region_order:
                if region not in self.region_games:
                    continue
                winner, seed, game_results = _simulate_region(
                    self.region_games[region],
                    self._win_prob,
                    rng,
                    locked_results=self.locked_results,
                    region=region,
                )
                if winner:
                    ff_teams.append((winner, seed, region))

                round_counters = defaultdict(int)
                for rd_name, w, ws, l, ls, wp in game_results:
                    idx = round_counters[rd_name]
                    round_counters[rd_name] += 1
                    game_slot_wins[(rd_name, region, idx)][(w, ws, l, ls)] += 1

            # Final Four semis + Championship
            if len(ff_teams) >= 4:
                ta, sa, _ra = ff_teams[0]
                tb, sb, _rb = ff_teams[1]
                w, ws, l, ls = _resolve_matchup(
                    "F1", ta, sa, tb, sb, self._win_prob, rng, self.locked_results
                )
                finalist_1 = (w, ws)
                game_slot_wins[("Final Four", "National", 0)][(w, ws, l, ls)] += 1

                ta, sa, _ra = ff_teams[2]
                tb, sb, _rb = ff_teams[3]
                w, ws, l, ls = _resolve_matchup(
                    "F2", ta, sa, tb, sb, self._win_prob, rng, self.locked_results
                )
                finalist_2 = (w, ws)
                game_slot_wins[("Final Four", "National", 1)][(w, ws, l, ls)] += 1

                ta, sa = finalist_1
                tb, sb = finalist_2
                w, ws, l, ls = _resolve_matchup(
                    "C1", ta, sa, tb, sb, self._win_prob, rng, self.locked_results
                )
                champ_counts[w] += 1
                game_slot_wins[("Championship", "National", 0)][(w, ws, l, ls)] += 1

        n = self.n_sims

        # Phase 2 + 3: Per-slot EV and leverage-weighted picks (single pass)
        self.ev_bracket = defaultdict(list)
        self.leverage_bracket = defaultdict(list)
        self.slot_evs = {}

        for (rd_name, region, idx), outcomes in game_slot_wins.items():
            points = self.ROUND_POINTS.get(rd_name, 10)

            # Aggregate wins per team (shared by EV and leverage)
            team_wins = {}
            for (w, ws, l, ls), count in outcomes.items():
                if w not in team_wins:
                    team_wins[w] = {"count": 0, "seed": ws, "opponents": []}
                team_wins[w]["count"] += count
                team_wins[w]["opponents"].append((l, ls, count))

            # Pre-compute per-team EV and most common opponent
            best_ev_team = None
            best_ev = -1
            best_lev_team = None
            best_lev = -1

            team_ev_info = {}
            for team, info in team_wins.items():
                ev = (info["count"] / n) * points
                modal_opp, modal_opp_seed, _ = max(
                    info["opponents"], key=lambda x: x[2]
                )
                own = self.pub.ownership(info["seed"], rd_name)
                lev = self.pub.leverage(ev, own, self.leverage_weight)
                team_ev_info[team] = {
                    "ev": ev,
                    "lev": lev,
                    "own": own,
                    "seed": info["seed"],
                    "count": info["count"],
                    "loser": modal_opp,
                    "loser_seed": modal_opp_seed,
                }
                if ev > best_ev:
                    best_ev = ev
                    best_ev_team = team
                if lev > best_lev:
                    best_lev = lev
                    best_lev_team = team

            # EV bracket entry
            ev_info = team_ev_info[best_ev_team]
            self.slot_evs[(rd_name, region, idx)] = {
                "team": best_ev_team,
                "seed": ev_info["seed"],
                "ev": ev_info["ev"],
                "win_pct": ev_info["count"] / n,
                "points": points,
            }
            self.ev_bracket[rd_name].append(
                {
                    "winner": best_ev_team,
                    "winner_seed": ev_info["seed"],
                    "loser": ev_info["loser"],
                    "loser_seed": ev_info["loser_seed"],
                    "win_prob": ev_info["count"] / n,
                    "ev": ev_info["ev"],
                    "region": region,
                    "upset": ev_info["seed"] > ev_info["loser_seed"]
                    if ev_info["loser_seed"]
                    else False,
                }
            )

            # Leverage bracket entry
            lev_info = team_ev_info[best_lev_team]
            self.leverage_bracket[rd_name].append(
                {
                    "winner": best_lev_team,
                    "winner_seed": lev_info["seed"],
                    "loser": lev_info["loser"],
                    "loser_seed": lev_info["loser_seed"],
                    "win_prob": lev_info["count"] / n,
                    "ev": lev_info["ev"],
                    "leverage_ev": lev_info["lev"],
                    "ownership": lev_info["own"],
                    "region": region,
                    "upset": lev_info["seed"] > lev_info["loser_seed"]
                    if lev_info["loser_seed"]
                    else False,
                }
            )

        # Sort each round
        for rd in self.ev_bracket:
            self.ev_bracket[rd].sort(key=lambda g: (g["region"], -g["ev"]))
        for rd in self.leverage_bracket:
            self.leverage_bracket[rd].sort(
                key=lambda g: (g["region"], -g["leverage_ev"])
            )

        # Champion and totals
        if champ_counts:
            self.champion = max(champ_counts, key=champ_counts.get)
        else:
            self.champion = "Unknown"

        self.total_ev = sum(s["ev"] for s in self.slot_evs.values())
        self.total_leverage_ev = sum(
            g["leverage_ev"] for games in self.leverage_bracket.values() for g in games
        )

        return {
            "ev_bracket": self.ev_bracket,
            "leverage_bracket": self.leverage_bracket,
            "slot_evs": self.slot_evs,
            "champion": self.champion,
            "total_ev": self.total_ev,
            "total_leverage_ev": self.total_leverage_ev,
        }


# ===========================================================================
# Class 7: QuantEnhancedSimulator
# ===========================================================================


class QuantEnhancedSimulator:
    """Full bracket Monte Carlo simulation with quant enhancements."""

    def __init__(
        self,
        bracket_df=None,
        kenpom_df=None,
        garch=None,
        hmm=None,
        kalman=None,
        prior=None,
        n_sims=10000,
        seed=42,
        locked_results=None,
        tournament_adjustments=None,
        # Aliases for notebook compatibility
        df=None,
        bracket=None,
        games_df=None,
    ):
        self.bracket_df = bracket_df if bracket_df is not None else bracket
        self.kenpom_df = kenpom_df if kenpom_df is not None else df
        self.garch = garch
        self.hmm = hmm
        self.kalman = kalman
        self.prior = prior
        self.n_sims = n_sims
        self.seed = seed
        self.locked_results = locked_results or {}
        self.tournament_adjustments = tournament_adjustments or {}

        # Build lookups
        self.net_lookup, self.seed_lookup = _build_lookups(self.kenpom_df)
        self.region_games = _parse_bracket(self.bracket_df)

        # Results
        self.bracket_picks = {}
        self.round_probs = {}
        self.ff_probs = {}
        self.champion = None

    def _win_prob(self, team_a, seed_a, team_b, seed_b, rng):
        """Compute win probability for team_a vs team_b with all enhancements."""
        net_a = self.net_lookup.get(team_a, 0.0)
        net_b = self.net_lookup.get(team_b, 0.0)

        # Tournament performance adjustment (compounding across games)
        if self.tournament_adjustments:
            net_a = net_a + self.tournament_adjustments.get(team_a, 0.0)
            net_b = net_b + self.tournament_adjustments.get(team_b, 0.0)

        # HMM state adjustment
        if self.hmm is not None:
            adj_a = self.hmm.sample_state_adjustment(team_a, rng)
            adj_b = self.hmm.sample_state_adjustment(team_b, rng)
            # Scale down HMM adjustments to avoid dominating
            net_a = net_a + adj_a * 0.3
            net_b = net_b + adj_b * 0.3

        # Kalman momentum adjustment
        if self.kalman is not None:
            mom_a = self.kalman.get_normalized_momentum(team_a)
            mom_b = self.kalman.get_normalized_momentum(team_b)
            net_a = net_a + (mom_a - 50) * 0.15
            net_b = net_b + (mom_b - 50) * 0.15

        # Base win probability
        margin = (net_a - net_b) / 2
        p_model = 1 / (1 + 10 ** (-margin / 11))

        # GARCH combined volatility
        if self.garch is not None:
            combined_vol = self.garch.combined_volatility(team_a, team_b)
        else:
            combined_vol = 11.0

        # Adjust probability with volatility (wider vol → regress to 0.5)
        if combined_vol > 0:
            p_model = 1 / (1 + 10 ** (-margin / combined_vol))

        # Historical prior blending
        if self.prior is not None:
            p_model = self.prior.blend(p_model, seed_a, seed_b, combined_vol)

        return np.clip(p_model, 0.01, 0.99)

    def run(self):
        """Run full Monte Carlo bracket simulation. Returns dict with bracket_picks, ff_probs, champion."""
        rng = np.random.default_rng(self.seed)
        region_order = ["East", "South", "West", "Midwest"]

        ff_counts = defaultdict(int)
        champ_counts = defaultdict(int)
        game_slot_wins = defaultdict(lambda: defaultdict(int))

        for sim in range(self.n_sims):
            ff_teams = []

            for region in region_order:
                if region not in self.region_games:
                    continue
                winner, seed, game_results = _simulate_region(
                    self.region_games[region],
                    self._win_prob,
                    rng,
                    locked_results=self.locked_results,
                    region=region,
                )
                if winner:
                    ff_counts[winner] += 1
                    ff_teams.append((winner, seed, region))

                round_counters = defaultdict(int)
                for rd_name, w, ws, l, ls, wp in game_results:
                    idx = round_counters[rd_name]
                    round_counters[rd_name] += 1
                    game_slot_wins[(rd_name, region, idx)][(w, ws, l, ls)] += 1

            # Final Four semis + Championship
            if len(ff_teams) >= 4:
                ta, sa, _ra = ff_teams[0]
                tb, sb, _rb = ff_teams[1]
                w, ws, l, ls = _resolve_matchup(
                    "F1", ta, sa, tb, sb, self._win_prob, rng, self.locked_results
                )
                finalist_1 = (w, ws)
                game_slot_wins[("Final Four", "National", 0)][(w, ws, l, ls)] += 1

                ta, sa, _ra = ff_teams[2]
                tb, sb, _rb = ff_teams[3]
                w, ws, l, ls = _resolve_matchup(
                    "F2", ta, sa, tb, sb, self._win_prob, rng, self.locked_results
                )
                finalist_2 = (w, ws)
                game_slot_wins[("Final Four", "National", 1)][(w, ws, l, ls)] += 1

                ta, sa = finalist_1
                tb, sb = finalist_2
                w, ws, l, ls = _resolve_matchup(
                    "C1", ta, sa, tb, sb, self._win_prob, rng, self.locked_results
                )
                champ_counts[w] += 1
                game_slot_wins[("Championship", "National", 0)][(w, ws, l, ls)] += 1

        # Compute probabilities
        n = self.n_sims
        self.ff_probs = {
            t: c / n for t, c in sorted(ff_counts.items(), key=lambda x: -x[1])
        }

        # Modal champion
        if champ_counts:
            self.champion = max(champ_counts, key=champ_counts.get)
        else:
            self.champion = "Unknown"

        # Build bracket_picks as round_name → list of game dicts (modal outcomes)
        self.bracket_picks = defaultdict(list)
        for (rd_name, region, idx), outcomes in game_slot_wins.items():
            modal = max(outcomes, key=outcomes.get)
            w, ws, l, ls = modal
            win_count = outcomes[modal]
            self.bracket_picks[rd_name].append(
                {
                    "winner": w,
                    "winner_seed": ws,
                    "loser": l,
                    "loser_seed": ls,
                    "win_prob": win_count / n,
                    "region": region,
                    "upset": ws > ls,
                }
            )

        # Sort each round's games by region then index
        for rd in self.bracket_picks:
            self.bracket_picks[rd].sort(key=lambda g: (g["region"], -g["win_prob"]))

        return {
            "bracket_picks": self.bracket_picks,
            "ff_probs": self.ff_probs,
            "champion": self.champion,
        }

    def compare_with_baseline(self, baseline_ff_probs):
        """Compare quant FF probabilities with baseline side by side."""
        all_teams = sorted(
            set(list(self.ff_probs.keys()) + list(baseline_ff_probs.keys())),
            key=lambda t: -self.ff_probs.get(t, 0),
        )
        rows = []
        for team in all_teams:
            rows.append(
                {
                    "Team": team,
                    "Baseline_FF%": baseline_ff_probs.get(team, 0.0),
                    "Enhanced_FF%": self.ff_probs.get(team, 0.0),
                    "Diff": self.ff_probs.get(team, 0.0)
                    - baseline_ff_probs.get(team, 0.0),
                }
            )
        return pd.DataFrame(rows)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    # Determine data directory (handle running from repo root or scripts/)
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "scraped_data"
    bracket_path = repo_root / "scraped_data" / "bracket.csv"

    # If bracket.csv is at repo root instead
    if not bracket_path.exists():
        bracket_path = repo_root / "bracket.csv"

    print("=" * 60)
    print("Quantitative Models for March Madness")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    games_df = pd.read_csv(data_dir / "tournament_games.csv")
    kenpom_df = _load_kenpom(data_dir / "kenpom.csv")
    bracket_df = pd.read_csv(bracket_path)
    print(
        f"  Games: {len(games_df)}, KenPom teams: {len(kenpom_df)}, "
        f"Bracket matchups: {len(bracket_df)}"
    )

    # Fit GARCH
    print("\n[2/5] Fitting Hierarchical GARCH...")
    garch = HierarchicalGARCH(games_df)
    print(f"  Alpha={garch.alpha:.4f}, Beta={garch.beta:.4f}")
    print("  Sample volatilities:")
    for team in list(garch.team_sigma.keys())[:8]:
        print(f"    {team}: σ = {garch.team_sigma[team]:.2f}")

    # Fit HMM
    print("\n[3/5] Fitting Team HMMs...")
    hmm_model = TeamHMM(games_df)
    print(f"  Teams with HMM models: {len(hmm_model.team_models)}")
    for team in list(hmm_model.team_models.keys())[:5]:
        n_states = hmm_model.team_models[team].n_components
        print(f"    {team}: {n_states} states")

    # Fit Kalman
    print("\n[4/5] Fitting Kalman Momentum...")
    kalman = KalmanMomentum(games_df, kenpom_df)
    print(f"  Teams with momentum: {len(kalman.team_momentum)}")
    sorted_mom = sorted(kalman.team_momentum.items(), key=lambda x: -x[1])
    print("  Top 5 momentum:")
    for team, mom in sorted_mom[:5]:
        print(f"    {team}: {mom:.1f}")
    print("  Bottom 5 momentum:")
    for team, mom in sorted_mom[-5:]:
        print(f"    {team}: {mom:.1f}")

    # Historical Prior
    print("\n[5/5] Loading Historical Prior...")
    prior = HistoricalPrior(str(data_dir))
    print(f"  Seed matchup rates loaded: {len(prior.rates)}")

    # Run simulator
    print("\n" + "=" * 60)
    print("Running Monte Carlo Bracket Simulation (10,000 sims)...")
    print("=" * 60)
    sim = QuantEnhancedSimulator(
        bracket_df=bracket_df,
        kenpom_df=kenpom_df,
        garch=garch,
        hmm=hmm_model,
        kalman=kalman,
        prior=prior,
        n_sims=10000,
        seed=42,
    )
    results = sim.run()

    print(f"\nChampion (modal): {results['champion']}")
    print("\nFinal Four Probabilities:")
    for team, prob in list(results["ff_probs"].items())[:10]:
        print(f"  {team}: {prob:.1%}")

    print("\nBracket Picks:")
    for rd_name in [
        "Round of 64",
        "Round of 32",
        "Sweet 16",
        "Elite 8",
        "Final Four",
        "Championship",
    ]:
        games = results["bracket_picks"].get(rd_name, [])
        if games:
            print(f"\n  --- {rd_name} ---")
            for g in games:
                upset = " UPSET" if g["upset"] else ""
                print(
                    f"    ({g['winner_seed']}) {g['winner']:<20s} over "
                    f"({g['loser_seed']}) {g['loser']:<20s} [{g['win_prob']:.1%}]{upset}"
                )

    total_games = sum(len(g) for g in results["bracket_picks"].values())
    print(f"\nTotal game slots in bracket: {total_games}")
    print("Done.")
