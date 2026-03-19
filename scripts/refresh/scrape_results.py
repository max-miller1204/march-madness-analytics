#!/usr/bin/env python3
"""Scrape NCAA tournament scores from ESPN and update tournament state.

Fetches completed game cards from ESPN's tournament scoreboard, matches teams
to bracket.csv entries via NAME_ALIASES, validates results, and returns the
count of newly completed games.

Resilience:
  - 3 retry attempts with exponential backoff (1s, 3s, 9s)
  - Post-scrape validation (no duplicate IDs, positive scores, winner in bracket)
  - Graceful degradation: failures log warnings and return 0
"""

import argparse
import csv
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Imports from project foundation
# ---------------------------------------------------------------------------
from scripts.quant_models import NAME_ALIASES, _resolve_name

# Optional imports — foundation files may not be committed yet
try:
    from scripts.refresh.tournament_state import TournamentState
except ImportError:
    TournamentState = None

try:
    from scripts.refresh.validator import DataValidator
except ImportError:
    DataValidator = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

ESPN_SCOREBOARD_URL = (
    "https://www.espn.com/mens-college-basketball/scoreboard/_/group/100"
    "/date/{date}/seasontype/3"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

MAX_RETRIES = 3
BACKOFF_BASE = 1  # seconds; retry delays: 1, 3, 9

# Map bracket regions to game-ID prefixes used in bracket.csv
REGION_PREFIXES = {
    "East": "E",
    "South": "S",
    "West": "W",
    "Midwest": "M",
}

ROUND_ORDER = [
    "Round of 64",
    "Round of 32",
    "Sweet 16",
    "Elite Eight",
    "Final Four",
    "Championship",
]

# ESPN-specific aliases (ESPN display name -> bracket name)
ESPN_ALIASES = {
    **NAME_ALIASES,
    "UConn": "UConn",
    "Connecticut": "UConn",
    "Michigan St": "Michigan State",
    "Michigan St.": "Michigan State",
    "Iowa St": "Iowa State",
    "Iowa St.": "Iowa State",
    "Ohio St": "Ohio State",
    "Ohio St.": "Ohio State",
    "Utah St": "Utah State",
    "Utah St.": "Utah State",
    "N.C. State": "NC State",
    "NC State": "NC State",
    "North Carolina State": "NC State",
    "North Dakota St": "North Dakota State",
    "Wright St": "Wright State",
    "Kennesaw St": "Kennesaw State",
    "Tennessee St": "Tennessee State",
    "South Fla": "South Florida",
    "South Fla.": "South Florida",
    "Cal Baptist": "Cal Baptist",
    "California Baptist": "Cal Baptist",
    "Saint Mary's": "Saint Mary's",
    "Saint Mary's (CA)": "Saint Mary's",
    "St. John's": "St. John's",
    "St. John's (NY)": "St. John's",
    "Saint John's": "St. John's",
    "Pitt": "Pittsburgh",
    "Prairie View": "Prairie View A&M",
    "Prairie View AM": "Prairie View A&M",
    "Texas AM": "Texas A&M",
    "LIU": "Long Island University",
    "Long Island": "Long Island University",
    "Queens": "Queens (NC)",
    "UNI": "Northern Iowa",
    "Miami": "Miami (FL)",
    "Miami FL": "Miami (FL)",
    "Miami (FL)": "Miami (FL)",
    "Miami OH": "Miami (OH)",
    "Miami (OH)": "Miami (OH)",
}

# Play-in placeholders in bracket.csv that expand to two teams
PLAYIN_TEAMS = {
    "NC State/Texas": ["NC State", "Texas"],
    "SMU/Miami (OH)": ["SMU", "Miami (OH)"],
    "Howard/UMBC": ["Howard", "UMBC"],
    "Lehigh/Prairie View A&M": ["Lehigh", "Prairie View A&M"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_bracket_teams(bracket_path):
    """Load all team names from bracket.csv, expanding play-in slots."""
    teams = set()
    with open(bracket_path, newline="") as fh:
        for row in csv.DictReader(fh):
            for col in ("TeamA", "TeamB"):
                name = row[col].strip()
                if name in PLAYIN_TEAMS:
                    teams.update(PLAYIN_TEAMS[name])
                else:
                    teams.add(name)
    return teams


def _load_bracket_games(bracket_path):
    """Load bracket.csv rows as a list of dicts for game-ID matching."""
    games = []
    with open(bracket_path, newline="") as fh:
        for row in csv.DictReader(fh):
            games.append(row)
    return games


def _resolve_espn_name(espn_name, bracket_teams):
    """Match an ESPN team name to a bracket team name.

    Tries, in order:
      1. Direct match
      2. ESPN_ALIASES lookup
      3. NAME_ALIASES lookup (via _resolve_name)
      4. Case-insensitive match against bracket teams
    """
    name = espn_name.strip()

    # 1. Direct
    if name in bracket_teams:
        return name

    # 2. ESPN aliases
    alias = ESPN_ALIASES.get(name)
    if alias and alias in bracket_teams:
        return alias

    # 3. quant_models _resolve_name
    resolved = _resolve_name(name)
    if resolved in bracket_teams:
        return resolved

    # 4. Case-insensitive
    name_lower = name.lower()
    for t in bracket_teams:
        if t.lower() == name_lower:
            return t

    # 5. Partial / mascot stripping: remove trailing words one-by-one
    words = name.split()
    for i in range(len(words) - 1, 0, -1):
        prefix = " ".join(words[:i])
        if prefix in bracket_teams:
            return prefix
        alias = ESPN_ALIASES.get(prefix)
        if alias and alias in bracket_teams:
            return alias

    return None


def _find_game_id(team_a, team_b, bracket_games, existing_ids):
    """Derive a game ID from bracket.csv for two competing teams.

    Returns the GameID column from bracket.csv if the matchup matches
    (order-independent).  Falls back to generating a synthetic ID.
    """
    for row in bracket_games:
        row_teams = set()
        for col in ("TeamA", "TeamB"):
            raw = row[col].strip()
            if raw in PLAYIN_TEAMS:
                row_teams.update(PLAYIN_TEAMS[raw])
            else:
                row_teams.add(raw)
        if team_a in row_teams and team_b in row_teams:
            return row.get("GameID", "").strip()

    # Synthetic fallback for later rounds not pre-listed in bracket.csv
    region = None
    for row in bracket_games:
        for col in ("TeamA", "TeamB"):
            raw = row[col].strip()
            expanded = PLAYIN_TEAMS.get(raw, [raw])
            if team_a in expanded or team_b in expanded:
                region = row.get("Region", "").strip()
                break
        if region:
            break

    prefix = REGION_PREFIXES.get(region, "X")
    idx = 1
    while f"{prefix}{idx}" in existing_ids:
        idx += 1
    return f"{prefix}{idx}"


def _detect_round(game_status_text):
    """Infer tournament round from ESPN game status / context text."""
    text = (game_status_text or "").lower()
    if "championship" in text or "final" in text:
        return "Championship"
    if "final four" in text or "semifinal" in text:
        return "Final Four"
    if "elite" in text:
        return "Elite Eight"
    if "sweet" in text:
        return "Sweet 16"
    if "32" in text or "second round" in text:
        return "Round of 32"
    # Default
    return "Round of 64"


def _fetch_with_retry(url, retries=MAX_RETRIES, backoff=BACKOFF_BASE):
    """GET *url* with retry logic (exponential backoff).

    Returns a requests.Response on success or raises after all attempts.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            wait = backoff * (3 ** attempt)  # 1, 3, 9
            logger.warning(
                "Attempt %d/%d failed for %s: %s — retrying in %ss",
                attempt + 1,
                retries,
                url,
                exc,
                wait,
            )
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ESPN HTML parsing
# ---------------------------------------------------------------------------


def _parse_espn_scoreboard(html, bracket_teams, bracket_games, existing_ids):
    """Parse ESPN scoreboard HTML and return list of completed-game dicts.

    Each dict follows the tournament_state.json completed_games schema.
    """
    soup = BeautifulSoup(html, "lxml")
    completed = []
    seen_ids = set(existing_ids)

    # ESPN embeds game data in a JSON blob inside a script tag
    # Try to extract the __espnfitt__ or window.espn.scoreboardData JSON
    games_data = _extract_espn_json(soup)
    if games_data:
        completed = _parse_from_json(
            games_data, bracket_teams, bracket_games, seen_ids
        )
        if completed:
            return completed

    # Fallback: parse HTML game cards directly
    completed = _parse_from_html(soup, bracket_teams, bracket_games, seen_ids)
    return completed


def _extract_espn_json(soup):
    """Try to extract structured game data from ESPN's embedded JSON."""
    for script in soup.find_all("script"):
        text = script.string or ""
        # window.espn.scoreboardData pattern
        if "window.espn.scoreboardData" in text or "scoreboardData" in text:
            match = re.search(
                r"scoreboardData\s*=\s*(\{.*?\});", text, re.DOTALL
            )
            if match:
                try:
                    data = json.loads(match.group(1))
                    return data.get("events", [])
                except (json.JSONDecodeError, TypeError):
                    pass
        # window['__espnfitt__'] pattern
        if "__espnfitt__" in text:
            match = re.search(r"__espnfitt__['\"]]\s*=\s*(\{.*?\});", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    events = (
                        data.get("page", {})
                        .get("content", {})
                        .get("scoreboard", {})
                        .get("evts", [])
                    )
                    return events if events else None
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
    return None


def _parse_from_json(events, bracket_teams, bracket_games, seen_ids):
    """Parse completed games from ESPN's embedded JSON event list."""
    completed = []
    for event in events:
        competitions = event.get("competitions", [])
        for comp in competitions:
            status_obj = comp.get("status", {})
            status_type = status_obj.get("type", {})
            if not status_type.get("completed", False):
                continue

            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            teams_info = []
            for c in competitors:
                team_obj = c.get("team", {})
                name = (
                    team_obj.get("shortDisplayName")
                    or team_obj.get("displayName")
                    or team_obj.get("name", "")
                )
                seed_str = c.get("curatedRank", {}).get("current", "")
                try:
                    seed = int(seed_str)
                except (ValueError, TypeError):
                    seed = 0
                try:
                    score = int(c.get("score", "0"))
                except (ValueError, TypeError):
                    score = 0
                teams_info.append(
                    {"name": name, "seed": seed, "score": score}
                )

            if len(teams_info) < 2:
                continue

            t1, t2 = teams_info[0], teams_info[1]
            name_a = _resolve_espn_name(t1["name"], bracket_teams)
            name_b = _resolve_espn_name(t2["name"], bracket_teams)

            if not name_a or not name_b:
                logger.debug(
                    "Skipping game: could not resolve %s vs %s",
                    t1["name"],
                    t2["name"],
                )
                continue

            score_a, score_b = t1["score"], t2["score"]
            if score_a <= 0 or score_b <= 0:
                continue

            winner = name_a if score_a > score_b else name_b

            game_id = _find_game_id(name_a, name_b, bracket_games, seen_ids)
            if game_id in seen_ids:
                continue
            seen_ids.add(game_id)

            round_text = status_obj.get("type", {}).get("description", "")
            game_round = _detect_round(round_text)

            # Determine region from bracket games
            region = ""
            for row in bracket_games:
                for col in ("TeamA", "TeamB"):
                    raw = row[col].strip()
                    expanded = PLAYIN_TEAMS.get(raw, [raw])
                    if name_a in expanded or name_b in expanded:
                        region = row.get("Region", "").strip()
                        break
                if region:
                    break

            completed.append(
                {
                    "game_id": game_id,
                    "round": game_round,
                    "region": region,
                    "seed_a": t1["seed"],
                    "team_a": name_a,
                    "seed_b": t2["seed"],
                    "team_b": name_b,
                    "score_a": score_a,
                    "score_b": score_b,
                    "winner": winner,
                    "margin": abs(score_a - score_b),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    return completed


def _parse_from_html(soup, bracket_teams, bracket_games, seen_ids):
    """Fallback: parse completed games from ESPN HTML game cards."""
    completed = []

    # ESPN uses section.Scoreboard elements or div with class containing
    # "ScoreboardScoreCell"
    game_cards = soup.select("section.Scoreboard") or soup.select(
        "div.ScoreboardScoreCell"
    )
    if not game_cards:
        # Broader fallback
        game_cards = soup.find_all("div", class_=re.compile(r"scoreboard", re.I))

    for card in game_cards:
        # Check if game is final
        status_el = card.find(
            string=re.compile(r"Final", re.I)
        ) or card.find(class_=re.compile(r"status", re.I))
        if not status_el:
            continue
        status_text = status_el.get_text(strip=True) if hasattr(status_el, "get_text") else str(status_el)
        if "final" not in status_text.lower():
            continue

        # Extract team names and scores from competitor rows
        team_rows = card.select("li.ScoreboardScoreCell__Item") or card.select(
            "tr"
        )
        if len(team_rows) < 2:
            team_rows = card.find_all("div", class_=re.compile(r"competitor", re.I))
        if len(team_rows) < 2:
            continue

        teams_info = []
        for row in team_rows[:2]:
            # Team name
            name_el = (
                row.select_one("div.ScoreCell__TeamName")
                or row.select_one("span.sb-team-short")
                or row.find("span", class_=re.compile(r"team", re.I))
            )
            name = name_el.get_text(strip=True) if name_el else ""

            # Seed
            seed_el = row.find("span", class_=re.compile(r"seed", re.I))
            try:
                seed = int(seed_el.get_text(strip=True)) if seed_el else 0
            except ValueError:
                seed = 0

            # Score
            score_el = (
                row.select_one("div.ScoreCell__Score")
                or row.find("span", class_=re.compile(r"score", re.I))
            )
            try:
                score = int(score_el.get_text(strip=True)) if score_el else 0
            except ValueError:
                score = 0

            teams_info.append({"name": name, "seed": seed, "score": score})

        if len(teams_info) < 2:
            continue

        t1, t2 = teams_info[0], teams_info[1]
        name_a = _resolve_espn_name(t1["name"], bracket_teams)
        name_b = _resolve_espn_name(t2["name"], bracket_teams)

        if not name_a or not name_b:
            continue

        score_a, score_b = t1["score"], t2["score"]
        if score_a <= 0 or score_b <= 0:
            continue

        winner = name_a if score_a > score_b else name_b
        game_id = _find_game_id(name_a, name_b, bracket_games, seen_ids)
        if game_id in seen_ids:
            continue
        seen_ids.add(game_id)

        game_round = _detect_round(status_text)

        region = ""
        for row in bracket_games:
            for col in ("TeamA", "TeamB"):
                raw = row[col].strip()
                expanded = PLAYIN_TEAMS.get(raw, [raw])
                if name_a in expanded or name_b in expanded:
                    region = row.get("Region", "").strip()
                    break
            if region:
                break

        completed.append(
            {
                "game_id": game_id,
                "round": game_round,
                "region": region,
                "seed_a": t1["seed"],
                "team_a": name_a,
                "seed_b": t2["seed"],
                "team_b": name_b,
                "score_a": score_a,
                "score_b": score_b,
                "winner": winner,
                "margin": abs(score_a - score_b),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return completed


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_results(new_games, existing_games, bracket_teams):
    """Run post-scrape quality checks on newly scraped games.

    Returns (valid_games, warnings) where valid_games is the filtered list
    and warnings is a list of human-readable issue strings.
    """
    warnings_list = []
    valid = []

    existing_ids = {g["game_id"] for g in existing_games}

    for g in new_games:
        issues = []

        # No duplicate game IDs
        if g["game_id"] in existing_ids:
            issues.append(f"Duplicate game_id {g['game_id']}")
            continue  # skip entirely — already recorded

        # Scores are positive integers
        if not isinstance(g["score_a"], int) or g["score_a"] <= 0:
            issues.append(
                f"Invalid score_a ({g['score_a']}) in {g['game_id']}"
            )
        if not isinstance(g["score_b"], int) or g["score_b"] <= 0:
            issues.append(
                f"Invalid score_b ({g['score_b']}) in {g['game_id']}"
            )

        # Winner must be in bracket
        if g["winner"] not in bracket_teams:
            issues.append(
                f"Winner '{g['winner']}' not found in bracket teams"
            )

        # Winner must be one of the two teams
        if g["winner"] not in (g["team_a"], g["team_b"]):
            issues.append(
                f"Winner '{g['winner']}' not in matchup "
                f"({g['team_a']} vs {g['team_b']})"
            )

        # No contradictory results: team cannot appear twice as loser
        # (checked at aggregate level below)

        if issues:
            warnings_list.extend(issues)
        else:
            valid.append(g)
            existing_ids.add(g["game_id"])

    # Contradiction check: a team that lost should not also be winning
    # another game in the same batch
    losers = set()
    for g in valid:
        loser = g["team_a"] if g["winner"] != g["team_a"] else g["team_b"]
        losers.add(loser)
    contradictions = []
    for g in valid:
        if g["winner"] in losers:
            # Could be valid across rounds, so only flag within same round
            same_round_loss = any(
                (v["winner"] != g["winner"]
                 and g["winner"] in (v["team_a"], v["team_b"])
                 and v["round"] == g["round"])
                for v in valid
            )
            if same_round_loss:
                contradictions.append(
                    f"Contradiction: {g['winner']} wins {g['game_id']} "
                    f"but also loses in round {g['round']}"
                )
    if contradictions:
        warnings_list.extend(contradictions)

    return valid, warnings_list


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def scrape_tournament_results(
    state_path="scraped_data/tournament_state.json",
    bracket_path="scraped_data/bracket.csv",
):
    """Scrape ESPN for completed tournament games and update state.

    Parameters
    ----------
    state_path : str
        Path to tournament_state.json.
    bracket_path : str
        Path to bracket.csv.

    Returns
    -------
    int
        Number of newly added completed games.
    """
    logger.info("Starting tournament results scrape")

    # Load bracket data
    try:
        bracket_teams = _load_bracket_teams(bracket_path)
        bracket_games = _load_bracket_games(bracket_path)
    except FileNotFoundError:
        logger.error("Bracket file not found at %s", bracket_path)
        return 0

    # Load existing state
    existing_games = []
    state_data = {}
    if TournamentState is not None and os.path.exists(state_path):
        try:
            ts = TournamentState(state_path)
            state_data = ts.data if hasattr(ts, "data") else {}
            existing_games = state_data.get("completed_games", [])
        except Exception as exc:
            logger.warning("Could not load tournament state: %s", exc)
    elif os.path.exists(state_path):
        try:
            with open(state_path) as fh:
                state_data = json.load(fh)
            existing_games = state_data.get("completed_games", [])
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Could not read state file: %s", exc)

    existing_ids = {g["game_id"] for g in existing_games}

    # Fetch ESPN scoreboard — try today and recent tournament dates
    today = datetime.now()
    dates_to_try = []
    for offset in range(0, 7):
        d = today.replace(
            day=today.day - offset
        ) if today.day - offset > 0 else today
        try:
            dt = today.replace(day=today.day - offset)
            dates_to_try.append(dt.strftime("%Y%m%d"))
        except ValueError:
            pass

    all_new_games = []

    for date_str in dates_to_try:
        url = ESPN_SCOREBOARD_URL.format(date=date_str)
        logger.info("Fetching %s", url)
        try:
            resp = _fetch_with_retry(url)
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s after retries: %s", url, exc)
            continue

        new_games = _parse_espn_scoreboard(
            resp.text, bracket_teams, bracket_games, existing_ids
        )
        if new_games:
            all_new_games.extend(new_games)
            # Update existing_ids to avoid duplicates across dates
            existing_ids.update(g["game_id"] for g in new_games)

    if not all_new_games:
        logger.info("No new completed games found")
        return 0

    # Validate
    valid_games, warnings_list = _validate_results(
        all_new_games, existing_games, bracket_teams
    )

    for w in warnings_list:
        logger.warning("Validation: %s", w)

    # Run DataValidator if available
    if DataValidator is not None:
        try:
            validator = DataValidator()
            all_games = existing_games + valid_games
            validator.validate(all_games)
        except Exception as exc:
            logger.warning("DataValidator raised: %s", exc)

    if not valid_games:
        logger.info("No valid new games after validation")
        return 0

    # Persist results
    added = 0
    if TournamentState is not None:
        try:
            ts = TournamentState(state_path)
            for g in valid_games:
                ts.add_result(g)
                added += 1
            ts.save()
        except Exception as exc:
            logger.warning(
                "TournamentState save failed, falling back to direct write: %s",
                exc,
            )
            added = _direct_save(state_path, state_data, existing_games, valid_games)
    else:
        added = _direct_save(state_path, state_data, existing_games, valid_games)

    logger.info("Added %d new completed games", added)
    return added


def _direct_save(state_path, state_data, existing_games, valid_games):
    """Fallback: write state JSON directly when TournamentState unavailable."""
    all_games = existing_games + valid_games
    state_data["completed_games"] = all_games
    state_data["last_updated"] = datetime.now(timezone.utc).isoformat()

    # Detect current round
    rounds_seen = {g["round"] for g in all_games}
    for r in reversed(ROUND_ORDER):
        if r in rounds_seen:
            state_data["current_round"] = r
            break

    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    with open(state_path, "w") as fh:
        json.dump(state_data, fh, indent=2)

    return len(valid_games)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Scrape NCAA tournament results from ESPN"
    )
    parser.add_argument(
        "--state-path",
        default="scraped_data/tournament_state.json",
        help="Path to tournament_state.json",
    )
    parser.add_argument(
        "--bracket-path",
        default="scraped_data/bracket.csv",
        help="Path to bracket.csv",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    count = scrape_tournament_results(
        state_path=args.state_path,
        bracket_path=args.bracket_path,
    )
    print(f"Scrape complete: {count} new game(s) added")


if __name__ == "__main__":
    main()
