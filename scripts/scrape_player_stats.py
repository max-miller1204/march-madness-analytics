#!/usr/bin/env python3
"""Scrape player stats from ESPN for tournament teams,
compute injury-adjusted NetRtg values."""

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = "scraped_data"
PLAYER_STATS_CSV = os.path.join(OUTPUT_DIR, "player_stats.csv")
INJURY_ADJ_CSV = os.path.join(OUTPUT_DIR, "injury_adjustments.csv")
META_FILE = os.path.join(OUTPUT_DIR, "player_stats.meta")
CACHE_DIR = os.path.join(OUTPUT_DIR, ".player_cache")
ESPN_TEAM_IDS_CACHE = os.path.join(OUTPUT_DIR, ".espn_team_ids.json")
BRACKET_CSV = os.path.join(OUTPUT_DIR, "bracket.csv")
KENPOM_CSV = os.path.join(OUTPUT_DIR, "kenpom.csv")
INJURIES_CSV = os.path.join(OUTPUT_DIR, "injuries.csv")
TTL_HOURS = 6
SF = -10  # Scaling factor for penalty
DECAY = 0.85  # Geometric decay factor

# Bracket name -> ESPN location (only for names that don't match ESPN's location field)
ESPN_NAME_OVERRIDES = {
    "Cal Baptist": "California Baptist",
    "Hawaii": "Hawai'i",
    "Miami (FL)": "Miami",
}

# Bracket name -> ESPN team ID (for teams not in the 362-team API list)
ESPN_ID_OVERRIDES = {
    "Queens (NC)": 2511,
}

# KenPom name -> bracket-standard name (for KenPom normalization)
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

# Play-in resolutions
PLAYIN = {
    "NC State/Texas": ["NC State", "Texas"],
    "SMU/Miami (OH)": ["SMU", "Miami (OH)"],
    "Howard/UMBC": ["Howard", "UMBC"],
    "Lehigh/Prairie View A&M": ["Lehigh", "Prairie View A&M"],
}


def get_tournament_teams():
    """Load tournament team names from bracket.csv."""
    teams = set()
    with open(BRACKET_CSV) as f:
        for row in csv.DictReader(f):
            for col in ("TeamA", "TeamB"):
                name = row[col].strip()
                if name in PLAYIN:
                    for t in PLAYIN[name]:
                        teams.add(t)
                else:
                    teams.add(name)
    return sorted(teams)


def resolve_team_ids(bracket_teams, refresh=False):
    """Resolve bracket team names to ESPN team IDs.

    Returns dict of bracket_name -> espn_team_id.
    Halts if any team cannot be resolved.
    """
    # Check cache first
    if not refresh and os.path.exists(ESPN_TEAM_IDS_CACHE):
        with open(ESPN_TEAM_IDS_CACHE) as f:
            cached = json.load(f)
        # Verify all bracket teams are in cache
        if all(t in cached for t in bracket_teams):
            print(f"ESPN team IDs loaded from cache ({len(cached)} teams)")
            return {t: int(cached[t]) for t in bracket_teams}

    print("Fetching ESPN team list...")
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams?limit=400"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    espn_teams = data["sports"][0]["leagues"][0]["teams"]

    # Build lookup by ESPN location field -> (id, displayName)
    location_lookup = {}
    for t in espn_teams:
        team = t["team"]
        location_lookup[team.get("location", "")] = int(team["id"])

    # Also build by shortDisplayName for fallback
    short_lookup = {}
    for t in espn_teams:
        team = t["team"]
        short_lookup[team.get("shortDisplayName", "")] = int(team["id"])

    # Resolve each bracket team
    result = {}
    unmatched = []
    for bt in bracket_teams:
        # 1. Direct ID override (for teams not in API list)
        if bt in ESPN_ID_OVERRIDES:
            result[bt] = ESPN_ID_OVERRIDES[bt]
            continue

        # 2. Name override -> look up by ESPN location
        espn_location = ESPN_NAME_OVERRIDES.get(bt, bt)
        if espn_location in location_lookup:
            result[bt] = location_lookup[espn_location]
            continue

        # 3. Try shortDisplayName
        if bt in short_lookup:
            result[bt] = short_lookup[bt]
            continue

        unmatched.append(bt)

    if unmatched:
        print(f"ERROR: Could not resolve ESPN team IDs for: {unmatched}")
        print("Add entries to ESPN_NAME_OVERRIDES or ESPN_ID_OVERRIDES to fix.")
        sys.exit(1)

    # Cache the mapping
    with open(ESPN_TEAM_IDS_CACHE, "w") as f:
        json.dump({k: v for k, v in result.items()}, f, indent=2)
    print(f"Resolved {len(result)} teams to ESPN IDs (cached to {ESPN_TEAM_IDS_CACHE})")

    return result


def check_ttl():
    """Return True if cached data is still fresh."""
    if not os.path.exists(META_FILE) or not os.path.exists(PLAYER_STATS_CSV):
        return False
    try:
        with open(META_FILE) as f:
            meta = json.load(f)
        ts = datetime.fromisoformat(meta["timestamp"])
        ttl = meta.get("ttl_hours", TTL_HOURS)
        return datetime.now() - ts < timedelta(hours=ttl)
    except (json.JSONDecodeError, KeyError, ValueError):
        return False


def load_kenpom():
    """Load KenPom data and return dict of team -> NetRtg."""
    col_names = [
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
    import pandas as pd

    kp = pd.read_csv(KENPOM_CSV, skiprows=2, header=None, names=col_names)

    result = {}
    for _, row in kp.iterrows():
        m = re.match(r"(.+?)\s+(\d+)$", str(row["Team"]))
        if m:
            name = m.group(1).strip()
            name = NAME_ALIASES.get(name, name)
            result[name] = float(row["NetRtg"])
    return result


def load_injuries():
    """Load injuries CSV into dict of team -> list of injured players."""
    if not os.path.exists(INJURIES_CSV):
        return {}
    injuries = {}
    with open(INJURIES_CSV) as f:
        for row in csv.DictReader(f):
            team = row["team"]
            if team not in injuries:
                injuries[team] = []
            injuries[team].append(row)
    return injuries


def team_slug(team_name):
    """Convert a bracket team name to a safe filename slug."""
    slug = team_name.lower()
    slug = re.sub(r"[.'()&]", "", slug)
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def scrape_team_stats(team, espn_id):
    """Scrape per-game stats for a team from ESPN.

    ESPN stats pages have 4 tables:
      Table 0: Player names (for per-game stats) — Name column with <a> links and <span class='font10'> position
      Table 1: Per-game stats — GP, MIN, PTS, REB, AST, STL, BLK, TO, FG%, FT%, 3P%
      Table 2: Player names (for season totals)
      Table 3: Season totals
    """
    url = f"https://www.espn.com/mens-college-basketball/team/stats/_/id/{espn_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for attempt in range(4):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 429:
                wait = [10, 30, 60][min(attempt, 2)]
                print(f"    Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            if attempt < 3:
                wait = [10, 30, 60][min(attempt, 2)]
                print(f"    HTTP {resp.status_code}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    ERROR: Failed to fetch {team} (ESPN ID {espn_id}): {e}")
                return None
    else:
        print(f"    ERROR: All retries exhausted for {team}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table")

    if len(tables) < 2:
        print(f"    ERROR: Expected at least 2 tables for {team}, found {len(tables)}")
        return None

    # Table 0: player names + positions
    name_table = tables[0]
    # Table 1: per-game stats
    stats_table = tables[1]

    # Parse names and positions from Table 0
    name_rows = name_table.find_all("tr")
    stats_rows = stats_table.find_all("tr")

    # First row is header in both tables
    if not name_rows or not stats_rows:
        print(f"    ERROR: Empty tables for {team}")
        return None

    # Verify stats header
    header_cells = stats_rows[0].find_all(["th", "td"])
    header_labels = [c.get_text(strip=True) for c in header_cells]
    try:
        min_idx = header_labels.index("MIN")
        pts_idx = header_labels.index("PTS")
    except ValueError:
        print(
            f"    ERROR: Could not find MIN/PTS columns for {team}. Headers: {header_labels}"
        )
        return None

    players = []
    # Skip header row (index 0)
    for i in range(1, min(len(name_rows), len(stats_rows))):
        name_cell = name_rows[i].find("td")
        if not name_cell:
            continue

        # Get player name from <a> tag
        link = name_cell.find("a")
        if link:
            player_name = link.get_text(strip=True)
        else:
            player_name = name_cell.get_text(strip=True)

        if not player_name or player_name.lower() in (
            "player",
            "team",
            "totals",
            "total",
        ):
            continue

        # Get position from <span class="font10">
        pos_span = name_cell.find("span", class_="font10")
        position = pos_span.get_text(strip=True) if pos_span else ""

        # Get stats from the corresponding row
        stat_cells = stats_rows[i].find_all(["th", "td"])
        stat_values = [c.get_text(strip=True) for c in stat_cells]

        try:
            mpg = float(stat_values[min_idx])
            ppg = float(stat_values[pts_idx])
        except (ValueError, IndexError):
            continue

        if mpg > 0:
            players.append(
                {
                    "team": team,
                    "player": player_name,
                    "position": position,
                    "mpg": round(mpg, 1),
                    "ppg": round(ppg, 1),
                }
            )

    return players


def compute_injury_adjustments(all_player_stats, injuries, kenpom_ratings):
    """Compute injury penalties and adjusted NetRtg for each team."""
    # Group stats by team
    team_stats = {}
    for p in all_player_stats:
        team = p["team"]
        if team not in team_stats:
            team_stats[team] = []
        team_stats[team].append(p)

    # Compute team PPG for each team
    team_ppg = {}
    for team, players in team_stats.items():
        total_ppg = sum(p["ppg"] for p in players)
        team_ppg[team] = total_ppg if total_ppg > 0 else 1.0

    # Add team_ppg to player stats
    for p in all_player_stats:
        p["team_ppg"] = round(team_ppg.get(p["team"], 1.0), 1)

    adjustments = []
    all_teams = set(p["team"] for p in all_player_stats)
    # Include teams from kenpom even if we didn't get stats
    all_teams.update(kenpom_ratings.keys())

    for team in sorted(all_teams):
        net_rtg = kenpom_ratings.get(team)
        if net_rtg is None:
            continue

        team_injuries = injuries.get(team, [])
        if not team_injuries:
            adjustments.append(
                {
                    "team": team,
                    "num_injuries": 0,
                    "total_penalty": 0.0,
                    "adjusted_NetRtg": net_rtg,
                    "injury_health_raw": 0.0,
                    "key_injuries_summary": "",
                }
            )
            continue

        # Match injured players to roster stats
        penalties = []
        key_injuries = []
        tppg = team_ppg.get(team, 1.0)

        for inj in team_injuries:
            inj_name = inj["player"].strip()
            status_weight = float(inj.get("status_weight", 0.25))

            # Find matching player in stats
            matched_player = None
            team_players = team_stats.get(team, [])
            for p in team_players:
                # Try exact match first, then partial
                if p["player"].lower() == inj_name.lower():
                    matched_player = p
                    break
                # Last name match
                inj_last = inj_name.split()[-1].lower() if inj_name else ""
                p_last = p["player"].split()[-1].lower() if p["player"] else ""
                if inj_last and p_last and inj_last == p_last:
                    matched_player = p
                    break

            if matched_player:
                mpg = matched_player["mpg"]
                ppg = matched_player["ppg"]
            else:
                # Default for unmatched players: assume bench player
                mpg = 10.0
                ppg = 3.0

            # Penalty formula: (MPG/40) * (PPG/TeamPPG) * SF * status_weight
            penalty = (mpg / 40.0) * (ppg / tppg) * SF * status_weight
            penalties.append(penalty)
            key_injuries.append(
                f"{inj_name} ({inj.get('status', '?')}, {penalty:+.2f})"
            )

        # Geometric decay: sort by |penalty| desc, apply 0.85^(i-1)
        penalties.sort(key=lambda x: abs(x), reverse=True)
        decayed_penalties = [p * (DECAY**i) for i, p in enumerate(penalties)]
        total_penalty = sum(decayed_penalties)

        adjusted_net = net_rtg + total_penalty

        adjustments.append(
            {
                "team": team,
                "num_injuries": len(team_injuries),
                "total_penalty": round(total_penalty, 3),
                "adjusted_NetRtg": round(adjusted_net, 3),
                "injury_health_raw": round(total_penalty, 3),
                "key_injuries_summary": "; ".join(key_injuries),
            }
        )

    return adjustments


def write_output(all_player_stats, adjustments):
    """Write player stats and injury adjustments CSVs."""
    # Player stats
    ps_fields = ["team", "player", "position", "mpg", "ppg", "team_ppg"]
    with open(PLAYER_STATS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ps_fields)
        writer.writeheader()
        for p in all_player_stats:
            writer.writerow({k: p[k] for k in ps_fields})
    print(f"  Wrote {len(all_player_stats)} player records to {PLAYER_STATS_CSV}")

    # Injury adjustments
    adj_fields = [
        "team",
        "num_injuries",
        "total_penalty",
        "adjusted_NetRtg",
        "injury_health_raw",
        "key_injuries_summary",
    ]
    with open(INJURY_ADJ_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=adj_fields)
        writer.writeheader()
        writer.writerows(adjustments)
    print(f"  Wrote {len(adjustments)} team adjustments to {INJURY_ADJ_CSV}")

    # Meta file
    with open(META_FILE, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "ttl_hours": TTL_HOURS}, f)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape player stats from ESPN and compute injury adjustments"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Force re-scrape ignoring TTL cache"
    )
    args = parser.parse_args()

    if not args.refresh and check_ttl():
        print(
            f"Cached data is fresh (TTL: {TTL_HOURS}h). Use --refresh to force update."
        )
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Delete old sports-reference cache if it exists (one-time migration)
    if os.path.exists(CACHE_DIR):
        # Check if cache contains old sports-reference JSON files
        old_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
        if old_files:
            # Peek at first file to see if it's sports-reference format
            sample = os.path.join(CACHE_DIR, old_files[0])
            try:
                with open(sample) as f:
                    data = json.load(f)
                # Old format won't have espn-sourced data marker
                if data and isinstance(data, list) and data[0].get("source") != "espn":
                    print(f"Clearing old cache ({len(old_files)} files)...")
                    shutil.rmtree(CACHE_DIR)
            except (json.JSONDecodeError, KeyError, IndexError):
                shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    teams = get_tournament_teams()
    print(f"Tournament teams: {len(teams)}")

    # Resolve ESPN team IDs
    team_ids = resolve_team_ids(teams, refresh=args.refresh)

    kenpom_ratings = load_kenpom()
    print(f"KenPom ratings loaded: {len(kenpom_ratings)} teams")

    injuries = load_injuries()
    injured_teams = [t for t in teams if t in injuries]
    print(f"Injured teams: {len(injured_teams)}")
    if injured_teams:
        for t in injured_teams:
            print(f"  {t}: {len(injuries[t])} injuries")

    # Scrape player stats for all tournament teams
    all_player_stats = []
    cached_count = 0
    failed_teams = []

    for i, team in enumerate(teams):
        slug = team_slug(team)
        cache_file = os.path.join(CACHE_DIR, f"{slug}.json")

        # Check per-team cache
        if not args.refresh and os.path.exists(cache_file):
            with open(cache_file) as cf:
                players = json.load(cf)
            print(f"[{i + 1}/{len(teams)}] {team} (cached, {len(players)} players)")
            all_player_stats.extend(players)
            cached_count += 1
            continue

        espn_id = team_ids[team]
        print(f"[{i + 1}/{len(teams)}] Scraping {team} (ESPN ID {espn_id})...")
        players = scrape_team_stats(team, espn_id)

        if players is None or len(players) == 0:
            failed_teams.append(team)
            print(f"    FAILED: No players scraped for {team}")
            continue

        # Mark as ESPN-sourced for cache migration detection
        for p in players:
            p["source"] = "espn"

        all_player_stats.extend(players)
        print(f"    Found {len(players)} players")

        # Save to per-team cache
        with open(cache_file, "w") as cf:
            json.dump(players, cf)

        # Courtesy delay
        time.sleep(0.7)

    if cached_count:
        print(f"  ({cached_count} teams loaded from cache)")

    if failed_teams:
        print(f"\nERROR: Failed to scrape {len(failed_teams)} teams: {failed_teams}")
        sys.exit(1)

    print(f"\nTotal player records: {len(all_player_stats)}")

    # Compute adjustments
    adjustments = compute_injury_adjustments(all_player_stats, injuries, kenpom_ratings)

    # Verify all tournament teams have adjustments
    adjustment_teams = {a["team"] for a in adjustments}
    missing = [t for t in teams if t not in adjustment_teams]
    if missing:
        print(
            f"\nERROR: {len(missing)} tournament teams missing from adjustments: {missing}"
        )
        print("These teams may be missing from KenPom data. Check NAME_ALIASES.")
        sys.exit(1)

    # Show injury impacts
    impacted = [a for a in adjustments if a["total_penalty"] != 0]
    if impacted:
        print(f"\nInjury impacts ({len(impacted)} teams):")
        for a in sorted(impacted, key=lambda x: x["total_penalty"]):
            print(
                f"  {a['team']}: penalty={a['total_penalty']:+.3f}, "
                f"NetRtg {a['adjusted_NetRtg']:.1f} (was {a['adjusted_NetRtg'] - a['total_penalty']:.1f})"
            )
            if a["key_injuries_summary"]:
                print(f"    {a['key_injuries_summary']}")

    write_output(all_player_stats, adjustments)
    print("\nDone!")


if __name__ == "__main__":
    main()
