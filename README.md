# March Madness 2026: Data-Driven Final Four Analysis

A composite analytical pipeline that predicts NCAA Tournament Final Four teams using KenPom efficiency metrics, real NCAA NET rankings, committee evaluation metrics (SOR, WAB, KPI, BPI), Monte Carlo bracket simulation, and seed-constraint optimization.

## How It Works

1. **Composite Scoring** -- Each team gets a weighted score from 9 normalized (0-100) metrics
2. **Monte Carlo Simulation** -- 10,000 bracket simulations per region using efficiency differentials
3. **Seed-Constrained Optimization** -- Brute-force search for the highest-scoring Final Four with seed sum >= 15

## Data Sources

| Data | File | Source |
|------|------|--------|
| KenPom efficiency ratings | `kenpom.csv` | [kenpom.com](https://kenpom.com) |
| NCAA NET rankings + quad records | `ncaa-net-rankings.csv` | [ncaa.com](https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings) |
| Team sheet metrics (SOR, WAB, KPI, BPI) | `scraped_data/tournament_teams.csv` | [warrennolan.com](https://www.warrennolan.com/basketball/2026/net-teamsheets-plus) |
| Game-by-game results | `scraped_data/tournament_games.csv` | [warrennolan.com](https://www.warrennolan.com/basketball/2026/net-teamsheets-plus) |
| Tournament bracket | `bracket.csv` | Manual entry |

## Quickstart

```bash
pip install -r requirements.txt
python _build_notebook.py
jupyter notebook final_four_analysis.ipynb
```

To refresh scraped data from Warren Nolan:

```bash
python scrape_net_teamsheets.py
python filter_tournament_teams.py
```

## Composite Model Weights

The full scheme uses 9 independent features (no KenPom/NET double-counting):

| Metric | Weight | Source |
|--------|--------|--------|
| KenPom Net Rating | 0.20 | kenpom.csv |
| KenPom Offensive Rating | 0.10 | kenpom.csv |
| KenPom Defensive Rating (inv) | 0.10 | kenpom.csv |
| NCAA NET Rank (inv) | 0.15 | ncaa-net-rankings.csv |
| Strength of Record (inv) | 0.10 | tournament_teams.csv |
| Wins Above Bubble | 0.10 | tournament_teams.csv |
| NET Strength of Schedule (inv) | 0.05 | tournament_teams.csv |
| Q1 Win % | 0.10 | ncaa-net-rankings.csv |
| Q1+Q2 Win % | 0.10 | ncaa-net-rankings.csv |

Fallback weight schemes activate when team sheet or quad data is unavailable.

## Monte Carlo Simulation

Win probability uses a logistic function on KenPom net efficiency differentials:

```
P(A wins) = 1 / (1 + 10^(-(NetRtg_A - NetRtg_B) / 22))
```

Each region is simulated 10,000 times (R64 -> R32 -> Sweet 16 -> Elite 8) to produce Final Four probabilities.

## Notebook Output

- Composite score rankings for all 68 tournament teams
- Regional tables with full metrics (KP#, NET#, SOR, WAB, Q1 record, FF%)
- Efficiency scatter plot (ORtg vs DRtg)
- Composite score bar chart (top 25)
- Radar charts comparing top 3 contenders per region
- Seed constraint configuration comparison
- Dark horse identification (ORtg vs seed)
- Final Four picks with detailed statistical breakdown
- Two 200-word essay drafts (per-team paragraphs and cohesive narrative)

## Project Structure

```
_build_notebook.py              # Generates the notebook programmatically
final_four_analysis.ipynb       # Output notebook (run all cells)
kenpom.csv                      # KenPom efficiency data (68 tournament teams)
ncaa-net-rankings.csv           # NCAA NET rankings (365 D1 teams)
bracket.csv                     # Tournament bracket (32 first-round games)
requirements.txt                # Python dependencies
scrape_net_teamsheets.py        # Scrapes Warren Nolan team sheets
filter_tournament_teams.py      # Filters scraped data to tournament teams
assignment.md                   # Assignment specification
scraped_data/
  tournament_teams.csv          # Team-level metrics (68 teams)
  tournament_games.csv          # Game-by-game results (~2200 games)
  quad_records.csv              # Q1/Q2 win-loss records
```

## Key Metrics Glossary

| Metric | Source | Description |
|--------|--------|-------------|
| NetRtg | [KenPom](https://kenpom.com) | Adjusted efficiency margin (points per 100 possessions) |
| ORtg / DRtg | [KenPom](https://kenpom.com) | Adjusted offensive / defensive efficiency |
| NET Rank | [NCAA](https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings) | Official NCAA evaluation tool used by the selection committee |
| SOR | [Warren Nolan](https://www.warrennolan.com/basketball/2026/net-teamsheets-plus) | Strength of Record -- resume quality ranking |
| WAB | [Warren Nolan](https://www.warrennolan.com/basketball/2026/net-teamsheets-plus) | Wins Above Bubble -- wins beyond the tournament bubble threshold |
| KPI | [Warren Nolan](https://www.warrennolan.com/basketball/2026/net-teamsheets-plus) | Kevin Pauga Index -- predictive metric used by the committee |
| BPI | [ESPN](https://www.espn.com/mens-college-basketball/bpi) | Basketball Power Index |
| Q1 / Q2 | [NCAA NET](https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings) | Quadrant records (Q1: vs top 75, Q2: vs 76-150) |
