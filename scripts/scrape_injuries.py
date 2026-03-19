#!/usr/bin/env python3
"""Scrape injury reports from boydsbets.com, filter to tournament teams."""

import argparse
import csv
import json
import os
import random
import re
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = "scraped_data"
INJURIES_CSV = os.path.join(OUTPUT_DIR, "injuries.csv")
META_FILE = os.path.join(OUTPUT_DIR, "injuries.meta")
BRACKET_CSV = os.path.join(OUTPUT_DIR, "bracket.csv")
TTL_HOURS = 6

# Bracket name -> boydsbets name (reverse of notebook's NAME_ALIASES where needed)
NAME_ALIASES = {
    'Michigan St.': 'Michigan State', 'Iowa St.': 'Iowa State',
    'Ohio St.': 'Ohio State', 'Connecticut': 'UConn',
    'Utah St.': 'Utah State', 'Miami FL': 'Miami (FL)',
    'N.C. State': 'NC State', 'North Dakota St.': 'North Dakota State',
    'Wright St.': 'Wright State', 'Kennesaw St.': 'Kennesaw State',
    'Miami OH': 'Miami (OH)', 'Tennessee St.': 'Tennessee State',
    'LIU': 'Long Island University', 'Queens': 'Queens (NC)',
    'South Fla.': 'South Florida', 'UNI': 'Northern Iowa',
    'California Baptist': 'Cal Baptist',
    "Saint Mary's (CA)": "Saint Mary's", "St. John's (NY)": "St. John's",
    'Long Island': 'Long Island University',
    "Saint Mary's College": "Saint Mary's", "Saint John's": "St. John's",
    'North Carolina State': 'NC State',
}

# Boydsbets team name variants -> standard bracket name
BOYDS_ALIASES = {
    'Michigan St': 'Michigan State', 'Michigan St.': 'Michigan State',
    'Iowa St': 'Iowa State', 'Iowa St.': 'Iowa State',
    'Ohio St': 'Ohio State', 'Ohio St.': 'Ohio State',
    'Utah St': 'Utah State', 'Utah St.': 'Utah State',
    'UConn': 'UConn', 'Connecticut': 'UConn',
    'Miami (FL)': 'Miami (FL)', 'Miami FL': 'Miami (FL)', 'Miami': 'Miami (FL)',
    'NC State': 'NC State', 'N.C. State': 'NC State', 'North Carolina State': 'NC State',
    'North Dakota St': 'North Dakota State', 'North Dakota St.': 'North Dakota State',
    'Wright St': 'Wright State', 'Wright St.': 'Wright State',
    'Kennesaw St': 'Kennesaw State', 'Kennesaw St.': 'Kennesaw State',
    'Tennessee St': 'Tennessee State', 'Tennessee St.': 'Tennessee State',
    'Long Island': 'Long Island University', 'LIU': 'Long Island University',
    'Queens': 'Queens (NC)',
    'South Fla': 'South Florida', 'South Fla.': 'South Florida', 'USF': 'South Florida',
    'Northern Iowa': 'Northern Iowa', 'UNI': 'Northern Iowa',
    'Cal Baptist': 'Cal Baptist', 'California Baptist': 'Cal Baptist',
    "Saint Mary's": "Saint Mary's", "Saint Mary's (CA)": "Saint Mary's",
    "St. John's": "St. John's", "Saint John's": "St. John's", "St. John's (NY)": "St. John's",
    "St Johns": "St. John's", "Saint Johns": "St. John's",
    'Prairie View': 'Prairie View A&M', 'Prairie View AM': 'Prairie View A&M',
    'Texas AM': 'Texas A&M', 'Texas A&M': 'Texas A&M',
    'Pitt': 'Pittsburgh',
    'SMU': 'SMU',
    'Miami (OH)': 'Miami (OH)', 'Miami OH': 'Miami (OH)',
    'UMBC': 'UMBC',
}

# Status weight mapping
STATUS_WEIGHTS = {
    'out': 1.0, 'out for season': 1.0, 'ofs': 1.0, 'out for the season': 1.0,
    'gtd': 0.5, 'game-time decision': 0.5, 'doubtful': 0.5,
    'day-to-day': 0.25, 'questionable': 0.25, 'probable': 0.25, 'dtd': 0.25,
}

# Injury types to filter out (not real game injuries)
EXCLUDED_TYPES = {'redshirt', 'transfer', 'suspension', 'personal', 'academic', 'dismissed',
                  'ineligible', 'left team', 'disciplinary'}

# Play-in resolutions matching the notebook
PLAYIN = {
    'NC State/Texas': ['NC State', 'Texas'],
    'SMU/Miami (OH)': ['SMU', 'Miami (OH)'],
    'Howard/UMBC': ['Howard', 'UMBC'],
    'Lehigh/Prairie View A&M': ['Lehigh', 'Prairie View A&M'],
}


def get_tournament_teams():
    """Load tournament team names from bracket.csv."""
    teams = set()
    with open(BRACKET_CSV) as f:
        for row in csv.DictReader(f):
            for col in ('TeamA', 'TeamB'):
                name = row[col].strip()
                if name in PLAYIN:
                    for t in PLAYIN[name]:
                        teams.add(t)
                else:
                    teams.add(name)
    return teams


def normalize_team(name, tournament_teams):
    """Try to match a scraped team name to a tournament team."""
    name = name.strip()
    if name in tournament_teams:
        return name
    if name in BOYDS_ALIASES and BOYDS_ALIASES[name] in tournament_teams:
        return BOYDS_ALIASES[name]
    # Try case-insensitive match
    name_lower = name.lower()
    for t in tournament_teams:
        if t.lower() == name_lower:
            return t
    return None


def check_ttl():
    """Return True if cached data is still fresh."""
    if not os.path.exists(META_FILE) or not os.path.exists(INJURIES_CSV):
        return False
    try:
        with open(META_FILE) as f:
            meta = json.load(f)
        ts = datetime.fromisoformat(meta['timestamp'])
        ttl = meta.get('ttl_hours', TTL_HOURS)
        return datetime.now() - ts < timedelta(hours=ttl)
    except (json.JSONDecodeError, KeyError, ValueError):
        return False


def parse_date(date_str):
    """Parse various date formats from injury reports."""
    date_str = date_str.strip()
    now = datetime.now()
    # Try common formats
    for fmt in ('%b %d, %Y', '%B %d, %Y', '%m/%d/%Y', '%m/%d/%y', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    # Try "Mar 7" style (no year)
    for fmt in ('%b %d', '%B %d'):
        try:
            d = datetime.strptime(date_str, fmt)
            d = d.replace(year=now.year)
            if d > now + timedelta(days=30):
                d = d.replace(year=now.year - 1)
            return d
        except ValueError:
            continue
    return None


def is_recent(date_str, days=14):
    """Check if a date is within the recency window."""
    d = parse_date(date_str)
    if d is None:
        return True  # If we can't parse, include it (conservative)
    return (datetime.now() - d).days <= days


def strip_mascot(team_raw, tournament_teams):
    """Strip mascot suffix from team name (e.g., 'Alabama Crimson Tide' -> 'Alabama').
    Try progressively shorter prefixes until we find a match."""
    team_raw = team_raw.strip()

    # Direct match first
    matched = normalize_team(team_raw, tournament_teams)
    if matched:
        return matched

    # Try removing words from the end (mascot stripping)
    words = team_raw.split()
    for i in range(len(words) - 1, 0, -1):
        prefix = ' '.join(words[:i])
        matched = normalize_team(prefix, tournament_teams)
        if matched:
            return matched

    return None


def parse_status_field(status_text):
    """Parse combined status field like 'Out – Knee' into (status, injury_type)."""
    # Split on common delimiters: ' – ', ' - ', ' — '
    for delim in [' – ', ' - ', ' — ', '–', '—']:
        if delim in status_text:
            parts = status_text.split(delim, 1)
            return parts[0].strip(), parts[1].strip()
    # No delimiter: entire text is status, no specific injury type
    return status_text.strip(), ''


def scrape_injuries(tournament_teams):
    """Scrape injury data from boydsbets.com."""
    url = "https://www.boydsbets.com/ncaa-basketball-injuries/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    print(f"Fetching {url}...")
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    injuries = []

    # Page uses <table id="props-table"> with columns:
    # Team | Player | POS | Status | Date Reported | Notes
    # Team names include mascots (e.g., "Alabama Crimson Tide")
    # Status combines level + injury (e.g., "Out – Knee")
    table = soup.find('table', id='props-table')
    if not table:
        # Fallback: find any table
        tables = soup.find_all('table')
        table = tables[0] if tables else None

    if not table:
        print("  WARNING: No injury table found on page")
        return injuries

    tbody = table.find('tbody')
    rows = tbody.find_all('tr') if tbody else table.find_all('tr')
    print(f"  Found {len(rows)} total injury rows")

    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 4:
            continue

        texts = [c.get_text(strip=True) for c in cells]

        # Columns: Team | Player | POS | Status | Date Reported | Notes
        team_raw = texts[0]
        player = texts[1]
        position = texts[2] if len(cells) >= 3 else ''
        status_raw = texts[3] if len(cells) >= 4 else ''
        date_reported = texts[4] if len(cells) >= 5 else ''
        # notes = texts[5] if len(cells) >= 6 else ''  # Not used in output

        # Parse status field ("Out – Knee" -> status="Out", injury_type="Knee")
        status, injury_type = parse_status_field(status_raw)

        # Match team name (strip mascot)
        team = strip_mascot(team_raw, tournament_teams)
        if team is None:
            continue

        # Filter excluded injury types
        injury_lower = injury_type.lower()
        if any(excl in injury_lower for excl in EXCLUDED_TYPES):
            continue

        status_lower = status.lower().strip()
        if any(excl in status_lower for excl in EXCLUDED_TYPES):
            continue

        # Check recency
        if date_reported and not is_recent(date_reported):
            continue

        # Compute status weight
        sw = 0.25  # default for unknown statuses
        for key, weight in STATUS_WEIGHTS.items():
            if key in status_lower:
                sw = weight
                break

        injuries.append({
            'player': player,
            'team': team,
            'position': position,
            'injury_type': injury_type,
            'status': status,
            'date_reported': date_reported,
            'status_weight': sw,
        })

    time.sleep(random.uniform(2, 3))
    return injuries


def write_output(injuries):
    """Write injuries CSV and meta file."""
    fieldnames = ['player', 'team', 'position', 'injury_type', 'status', 'date_reported', 'status_weight']

    with open(INJURIES_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(injuries)
    print(f"  Wrote {len(injuries)} injury records to {INJURIES_CSV}")

    with open(META_FILE, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'ttl_hours': TTL_HOURS}, f)
    print(f"  Wrote cache metadata to {META_FILE}")


def main():
    parser = argparse.ArgumentParser(description='Scrape NCAA basketball injury reports')
    parser.add_argument('--refresh', action='store_true', help='Force re-scrape ignoring TTL cache')
    args = parser.parse_args()

    if not args.refresh and check_ttl():
        print(f"Cached data is fresh (TTL: {TTL_HOURS}h). Use --refresh to force update.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tournament_teams = get_tournament_teams()
    print(f"Tournament teams: {len(tournament_teams)}")

    injuries = scrape_injuries(tournament_teams)
    print(f"Found {len(injuries)} injuries for tournament teams")

    # Show summary
    teams_with_injuries = set(i['team'] for i in injuries)
    for team in sorted(teams_with_injuries):
        team_injuries = [i for i in injuries if i['team'] == team]
        print(f"  {team}: {len(team_injuries)} injuries")
        for i in team_injuries:
            print(f"    - {i['player']} ({i['status']}, weight={i['status_weight']})")

    write_output(injuries)
    print("Done!")


if __name__ == "__main__":
    main()
