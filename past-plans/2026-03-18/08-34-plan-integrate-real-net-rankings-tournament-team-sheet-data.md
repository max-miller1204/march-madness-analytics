# Archived Plan

**Source:** `staged-finding-pond.md`
**Session:** `fdb7c08f-ccf6-46ce-abef-ae0b85e0225c`
**Trigger:** `resume`
**Archived:** 2026-03-18 08:34:08

---

# Plan: Integrate Real NET Rankings & Tournament Team Sheet Data into Model

## Context

The current model in `_build_notebook.py` scrapes NET rankings from Warren Nolan, but when scraping fails (common), it falls back to using KenPom rank as a proxy for NET rank. This means the `n_NET` feature (0.15 weight) just duplicates KenPom, double-counting it. The user has added three new CSV files with rich data that can fix this and improve the model.

## New Data Available

### 1. `ncaa-net-rankings.csv` (365 teams)
Headers: `Rank, School, Record, Conf, Road, Neutral, Home, Non D-1, Prev, Quad 1, Quad 2, Quad 3, Quad 4`
- **Real NET rank** for all D1 teams (not just tournament)
- Quad records (Q1-Q4) as W-L strings like "17-2"
- Road/Neutral/Home records

### 2. `scraped_data/tournament_teams.csv` (68 tournament teams)
Headers: `net_rank, team, conference, conf_record, div1_record, nonconf_record, road_record, net_sos, net_nonconf_sos, rpi_sos, rpi_nonconf_sos, avg_net_wins, avg_net_losses, kpi, sor, wab, bpi, pom, t_rank, q1_record, q2_record, q3_record, q4_record`
- **Goldmine of independent metrics**: KPI, SOR (Strength of Record), WAB (Wins Above Bubble), BPI, T-Rank
- NET-specific SOS (separate from KenPom SOS)
- Average NET of wins/losses (quality measures)
- Quad records already parsed

### 3. `scraped_data/tournament_games.csv` (game-by-game)
Headers: `team, team_net_rank, quadrant, sub_quadrant, opp_net_rank, location, opponent, is_conference, team_score, opp_score, result, overtime, date`
- Not needed for the composite model directly (tournament_teams.csv already aggregates this)

## Why This Doesn't Decompose into Parallel Units

This is a **single-file change** (`_build_notebook.py`) where the data loading, feature engineering, weight scheme, markdown docs, and visualizations are all deeply interconnected within sequential notebook cells. Splitting into parallel worktrees would create merge conflicts on every section. This should be implemented as **one focused change**.

## Implementation Plan

### File to modify: `/workspace/_build_notebook.py`

### Step 1: Replace NET scraping with CSV loading (lines 179-228)
- Load `ncaa-net-rankings.csv` as primary NET source (all 365 teams)
- Load `scraped_data/tournament_teams.csv` for rich tournament-team metrics
- Parse team name matching: NET CSV has trailing spaces (e.g., "Duke "), tournament_teams uses clean names
- Remove Warren Nolan scraping code for NET rankings
- Add name mapping for NET CSV (similar to KENPOM_TO_STD)

### Step 2: Replace quad record scraping with CSV data (lines 230-311)
- Parse Q1-Q4 records from `ncaa-net-rankings.csv` (format: "17-2") or from `tournament_teams.csv` (format: "17-2")
- Remove Warren Nolan team sheet scraping (WN_SLUGS dict, URL fetching, regex parsing)
- This eliminates 80+ lines of fragile scraping code

### Step 3: Merge new metrics from tournament_teams.csv (lines 314-340)
- Add columns: `NET_Rank`, `KPI`, `SOR`, `WAB`, `BPI`, `NET_SOS`, `Avg_NET_Wins`, `Avg_NET_Losses`
- Name matching: tournament_teams uses names like "Connecticut" (model uses "UConn"), "Saint Mary's College" (model uses "Saint Mary's"), etc.

### Step 4: Add new normalized features & rebalance weights (lines 364-403)
New feature set with real NET data and additional metrics:

**New weight scheme (full):**
| Metric | Weight | Source | Notes |
|--------|--------|--------|-------|
| KenPom NetRtg | 0.20 | kenpom.csv | Reduced from 0.25 (less double-counting) |
| KenPom ORtg | 0.10 | kenpom.csv | Reduced from 0.15 |
| KenPom DRtg (inv) | 0.10 | kenpom.csv | Same |
| NET Rank (inv) | 0.15 | ncaa-net-rankings.csv | Now truly independent from KenPom! |
| SOR (inv) | 0.10 | tournament_teams.csv | Strength of Record - committee metric |
| WAB | 0.10 | tournament_teams.csv | Wins Above Bubble - resume quality |
| SOS (NET) | 0.05 | tournament_teams.csv | NET-based SOS replaces KenPom SOS |
| Q1 Win% | 0.10 | ncaa-net-rankings.csv | Same concept, better source |
| Q1+Q2 Win% | 0.10 | ncaa-net-rankings.csv | Same concept, better source |

**Fallback scheme** (if tournament_teams.csv missing):
Redistribute SOR/WAB weights to NET and KenPom metrics.

### Step 5: Update markdown descriptions (lines 345-358)
- Update weight table in section 5 header
- Update section 4 header (no longer scraping, loading CSVs)
- Note the new independent metrics in the title cell

### Step 6: Update visualizations (lines 620-695)
- Add SOR/WAB to regional tables
- Add new metrics to radar charts (replace or augment)
- Essay section: reference new metrics in the narrative

## Name Mapping Needed

tournament_teams.csv → StdName:
- "Connecticut" → "UConn"
- "Michigan State" → "Michigan State" (OK)
- "Saint Mary's College" → "Saint Mary's"
- "North Carolina State" → "NC State"
- "Long Island" → "Long Island University"
- "Queens" → "Queens (NC)"
- "Northern Iowa" → "Northern Iowa" (OK)
- "California Baptist" → "Cal Baptist"
- "Prairie View A&M" → "Prairie View A&M" (need to check)

ncaa-net-rankings.csv → StdName:
- Names have trailing spaces (need .strip())
- "Michigan St." → "Michigan State"
- "Iowa St." → "Iowa State"
- "St. John's (NY)" → "St. John's"
- Etc. (similar to KENPOM_TO_STD mapping)

## Verification
1. Run `python _build_notebook.py` to generate the notebook
2. Run the notebook: `jupyter nbconvert --to notebook --execute final_four_analysis.ipynb`
3. Check that NET ranks differ from KenPom ranks (the whole point)
4. Verify all 68 teams match successfully
5. Check composite scores produce reasonable rankings
