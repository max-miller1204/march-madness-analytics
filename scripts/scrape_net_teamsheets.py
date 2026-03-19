#!/usr/bin/env python3
"""Scrape NET Team Sheets Plus from warrennolan.com."""

import csv
import sys
import requests
from bs4 import BeautifulSoup

URL = "https://www.warrennolan.com/basketball/2026/net-teamsheets-plus"
OUTPUT_DIR = "scraped_data"


def scrape_teamsheets():
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    teams = []
    games = []

    wrappers = soup.select("div.ts-wrapper")
    for wrapper in wrappers:
        container = wrapper.select_one("div.ts-container")
        if not container:
            continue

        # --- Team-level info ---
        rank = container.select_one("div.ts-rank")
        rank = rank.get_text(strip=True) if rank else ""

        teamname_div = container.select_one("div.ts-teamname")
        if not teamname_div:
            continue
        # Team name is first text node; conference info is in <span>
        team_name = teamname_div.contents[0].strip().rstrip(" \n")
        conf_span = teamname_div.select_one("span")
        conf_text = conf_span.get_text(strip=True) if conf_span else ""
        # conf_text looks like "ACC (17-1)"
        conference = conf_text.split("(")[0].strip() if conf_text else ""
        conf_record = conf_text.split("(")[1].rstrip(")") if "(" in conf_text else ""

        # Records from ts-data-center blocks
        data_centers = container.select("div.ts-flex-size-1 div.ts-data-center")
        div1_record = ""
        nonconf_record = ""
        road_record = ""
        for dc in data_centers:
            text = dc.get_text("\n", strip=True)
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if "RECORD" in lines[0]:
                div1_record = lines[1] if len(lines) > 1 else ""
                nonconf_record = lines[2] if len(lines) > 2 else ""
            elif "ROAD" in lines[0]:
                road_record = lines[1] if len(lines) > 1 else ""

        # SOS
        sos_section = container.select("div.ts-flex-size-1")
        net_sos = ""
        net_nonconf_sos = ""
        rpi_sos = ""
        rpi_nonconf_sos = ""
        for section in sos_section:
            title = section.select_one("div.ts-title-full-width")
            if title and "STRENGTH" in title.get_text():
                rights = section.select("div.ts-title-right")
                centers = section.select("div.ts-data-center")
                for r, c in zip(rights, centers):
                    label = r.get_text("\n", strip=True)
                    vals = [
                        l.strip()
                        for l in c.get_text("\n", strip=True).split("\n")
                        if l.strip()
                    ]
                    if "NET SOS" in label:
                        net_sos = vals[0] if vals else ""
                        net_nonconf_sos = vals[1] if len(vals) > 1 else ""
                    elif "RPI SOS" in label:
                        rpi_sos = vals[0] if vals else ""
                        rpi_nonconf_sos = vals[1] if len(vals) > 1 else ""

        # Average NET wins/losses
        avg_net_section = container.select_one("div.ts-flex-size-0 div.ts-data-center")
        avg_net_wins = ""
        avg_net_losses = ""
        if avg_net_section and "Average NET" in avg_net_section.get_text():
            text = avg_net_section.get_text("\n", strip=True)
            for line in text.split("\n"):
                if "Wins:" in line:
                    avg_net_wins = line.split(":")[1].strip()
                elif "Losses:" in line:
                    avg_net_losses = line.split(":")[1].strip()

        # Metrics (KPI, SOR, WAB, BPI, POM, T-Rank)
        metrics = {}
        half_widths = container.select("div.ts-half-width")
        for hw in half_widths:
            right = hw.select_one("div.ts-data-right")
            left = hw.select_one("div.ts-data-left")
            if right and left:
                labels = [
                    l.strip().rstrip(":")
                    for l in right.get_text("\n", strip=True).split("\n")
                    if l.strip()
                ]
                values = [
                    v.strip()
                    for v in left.get_text("\n", strip=True).split("\n")
                    if v.strip()
                ]
                for lbl, val in zip(labels, values):
                    metrics[lbl] = val

        # Quadrant records
        quad_records = {}
        flex2 = container.select_one("div.ts-flex-size-2")
        if flex2:
            centers = flex2.select("div.ts-data-center")
            for c in centers:
                title_el = c.select_one("div.ts-wide-title")
                if title_el:
                    q_name = title_el.get_text(strip=True).replace("QUADRANT ", "Q")
                    vals = [
                        l.strip()
                        for l in c.get_text("\n", strip=True).split("\n")
                        if l.strip() and l.strip() != title_el.get_text(strip=True)
                    ]
                    # Remove the narrow title duplicate
                    vals = [v for v in vals if v not in ("Q1", "Q2", "Q3", "Q4")]
                    if vals:
                        quad_records[q_name] = vals[0]  # overall record

        team_row = {
            "net_rank": rank,
            "team": team_name,
            "conference": conference,
            "conf_record": conf_record,
            "div1_record": div1_record,
            "nonconf_record": nonconf_record,
            "road_record": road_record,
            "net_sos": net_sos,
            "net_nonconf_sos": net_nonconf_sos,
            "rpi_sos": rpi_sos,
            "rpi_nonconf_sos": rpi_nonconf_sos,
            "avg_net_wins": avg_net_wins,
            "avg_net_losses": avg_net_losses,
            "kpi": metrics.get("KPI", ""),
            "sor": metrics.get("SOR", ""),
            "wab": metrics.get("WAB", ""),
            "bpi": metrics.get("BPI", ""),
            "pom": metrics.get("POM", ""),
            "t_rank": metrics.get("T-Rank", ""),
            "q1_record": quad_records.get("Q1", ""),
            "q2_record": quad_records.get("Q2", ""),
            "q3_record": quad_records.get("Q3", ""),
            "q4_record": quad_records.get("Q4", ""),
        }
        teams.append(team_row)

        # --- Game-level data from quadrant sections ---
        quad_container = wrapper.select_one("div.ts-quad-container")
        if not quad_container:
            continue

        subdivisions = quad_container.select("div.ts-quad-subdivision")
        for subdiv in subdivisions:
            q_title_el = subdiv.select_one("div.ts-quad-top-title")
            quadrant = q_title_el.get_text(strip=True) if q_title_el else ""

            # Check for sub-quadrant subtitle (e.g. "H: 1-15 | N: 1-25 | A: 1-40")
            sub_q = subdiv.select_one("div.ts-quad-subtitle")
            sub_quadrant = sub_q.get_text(strip=True) if sub_q else ""

            nitty_containers = subdiv.select("div.ts-nitty-container")
            for nc in nitty_containers:
                rows = nc.select("div.ts-nitty-row")
                for row in rows:
                    rank_cell = row.select_one("div.ts-nitty-rank")
                    if not rank_cell or rank_cell.get_text(strip=True) == "NET":
                        continue  # skip header row

                    loc_cell = row.select_one("div.ts-nitty-location")
                    opp_cell = row.select_one("div.ts-nitty-opponent")
                    score_cells = row.select("div.ts-nitty-score")
                    date_cell = row.select_one("div.ts-nitty-date")

                    opp_net = rank_cell.get_text(strip=True)
                    location = loc_cell.get_text(strip=True) if loc_cell else ""
                    opponent = opp_cell.get_text(strip=True) if opp_cell else ""
                    is_nonconf = "ts-nitty-nonconf" in (
                        opp_cell.get("class", []) if opp_cell else []
                    )

                    team_score = (
                        score_cells[0].get_text(strip=True)
                        if len(score_cells) > 0
                        else ""
                    )
                    opp_score = (
                        score_cells[1].get_text(strip=True)
                        if len(score_cells) > 1
                        else ""
                    )
                    is_loss = any(
                        "ts-nitty-loss" in (sc.get("class", []) or [])
                        for sc in score_cells
                    )
                    result = "L" if is_loss else "W"

                    is_ot = False
                    if date_cell and "ts-nitty-ot" in (
                        date_cell.get("class", []) or []
                    ):
                        is_ot = True
                    date = date_cell.get_text(strip=True) if date_cell else ""

                    games.append(
                        {
                            "team": team_name,
                            "team_net_rank": rank,
                            "quadrant": quadrant,
                            "sub_quadrant": sub_quadrant,
                            "opp_net_rank": opp_net,
                            "location": location,
                            "opponent": opponent,
                            "is_conference": not is_nonconf,
                            "team_score": team_score,
                            "opp_score": opp_score,
                            "result": result,
                            "overtime": is_ot,
                            "date": date,
                        }
                    )

    return teams, games


def write_csv(rows, filename, fieldnames):
    path = f"{OUTPUT_DIR}/{filename}"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {path}")


def main():
    print("Scraping NET Team Sheets Plus...")
    teams, games = scrape_teamsheets()

    if not teams:
        print(
            "ERROR: No teams scraped. The page structure may have changed.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Scraped {len(teams)} teams and {len(games)} games")

    write_csv(teams, "net_teamsheets_teams.csv", list(teams[0].keys()))
    write_csv(games, "net_teamsheets_games.csv", list(games[0].keys()))

    print("Done!")


if __name__ == "__main__":
    main()
