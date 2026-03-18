#!/usr/bin/env python3
"""Filter NET team sheets data to only include NCAA tournament teams."""

import csv

BRACKET = "bracket.csv"
TEAMS_IN = "scraped_data/net_teamsheets_teams.csv"
GAMES_IN = "scraped_data/net_teamsheets_games.csv"
TEAMS_OUT = "scraped_data/tournament_teams.csv"
GAMES_OUT = "scraped_data/tournament_games.csv"


def get_tournament_teams():
    teams = set()
    with open(BRACKET) as f:
        for row in csv.DictReader(f):
            for col in ("TeamA", "TeamB"):
                # Handle play-in games like "Lehigh/Prairie View A&M"
                for name in row[col].split("/"):
                    teams.add(name.strip())
    return teams


def fuzzy_match(team_name, tournament_teams):
    """Match scraped team names to bracket names."""
    if team_name in tournament_teams:
        return True
    # Handle common mismatches
    aliases = {
        "Saint John's": "St. John's",
        "California Baptist": "Cal Baptist",
        "Connecticut": "UConn",
        "Long Island": "Long Island University",
        "Miami": "Miami (FL)",
        "Miami (OH)": "Miami (OH)",
        "North Carolina State": "NC State",
        "Queens": "Queens (NC)",
        "Saint Mary's College": "Saint Mary's",
        "UMBC": "UMBC",
    }
    return aliases.get(team_name, None) in tournament_teams


def main():
    tournament_teams = get_tournament_teams()
    print(f"Tournament teams from bracket: {len(tournament_teams)}")

    # Read and filter teams
    with open(TEAMS_IN) as f:
        reader = csv.DictReader(f)
        all_teams = list(reader)
        fieldnames = reader.fieldnames

    matched = [t for t in all_teams if fuzzy_match(t["team"], tournament_teams)]
    print(f"Matched teams in teamsheets: {len(matched)}")

    # Show which bracket teams weren't matched
    matched_names = set()
    for t in all_teams:
        if fuzzy_match(t["team"], tournament_teams):
            matched_names.add(t["team"])

    # Build reverse lookup for reporting
    aliases = {
        "Saint John's": "St. John's",
        "California Baptist": "Cal Baptist",
        "Connecticut": "UConn",
        "Long Island": "Long Island University",
        "Miami": "Miami (FL)",
        "North Carolina State": "NC State",
        "Queens": "Queens (NC)",
        "Saint Mary's College": "Saint Mary's",
    }
    resolved = set()
    for name in matched_names:
        resolved.add(aliases.get(name, name))

    missing = tournament_teams - resolved
    if missing:
        print(f"Bracket teams not found in teamsheets: {missing}")

    with open(TEAMS_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matched)
    print(f"Wrote {len(matched)} teams to {TEAMS_OUT}")

    # Read and filter games
    with open(GAMES_IN) as f:
        reader = csv.DictReader(f)
        all_games = list(reader)
        game_fields = reader.fieldnames

    matched_games = [g for g in all_games if fuzzy_match(g["team"], tournament_teams)]

    with open(GAMES_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=game_fields)
        writer.writeheader()
        writer.writerows(matched_games)
    print(f"Wrote {len(matched_games)} games to {GAMES_OUT}")


if __name__ == "__main__":
    main()
