# Archived Plan

**Source:** `abstract-fluttering-crescent.md`
**Session:** `2e60f3b4-532d-4af4-a3c0-3ad23504c568`
**Trigger:** `resume`
**Archived:** 2026-03-18 00:45:24

---

# March Madness Final Four Analytics Pipeline — Spec

## Context

Class assignment due **March 18 by 5pm**: pick a data-driven Final Four (seed sum ≥ 15), produce a visual report, and write a 200-word conversational essay. The user wants a full analytical pipeline — scrape data, build a composite model, optimize picks under the seed constraint, generate charts, and output essay bullet points. No full bracket needed.

---

## Data Sources

| Source | What we get | How |
|--------|-------------|-----|
| `kenpom.csv` (local) | Efficiency metrics, SOS, seeds for 70 teams | Already have it |
| Warren Nolan NET page | NET rankings for all D1 teams | Scrape HTML table |
| Warren Nolan team sheets | Quad 1-4 win-loss records per team | Scrape per-team pages (~70 requests, 0.5s delay) |
| Sports Reference | SRS ratings (backup) | Scrape if Warren Nolan fails |
| **Fallback**: user enters CSV manually if all scraping fails | | |

---

## Dev Environment

- Dev container built via `.devcontainer/Dockerfile` from `ghcr.io/astral-sh/uv:python3.12-trixie-slim`
- Deps installed at build time with `uv pip install --system` for layer caching
- `.devcontainer/devcontainer.json` references Dockerfile with `context: ".."` to access project-root `requirements.txt`
- VS Code extensions: Jupyter + Python
- **Free to add any libraries/frameworks** during implementation — not limited to what's in `requirements.txt`

## Deliverables

1. **`final_four_analysis.ipynb`** — Jupyter notebook with the full pipeline
2. **`requirements.txt`** — Python dependencies
3. **`scraped_data/`** — Cached CSVs from scraping (so notebook reruns don't re-scrape)

---

## Notebook Structure (9 sections)

### 1. Setup & Imports
- pandas, matplotlib, seaborn, requests, BeautifulSoup
- Color palette per region (East=blue, South=red, West=green, Midwest=orange)

### 2. Load KenPom Data
- Read `kenpom.csv` with `skiprows=1`, manually rename duplicate columns (NetRtg/ORtg/DRtg appear for both team and SOS)
- Parse seed from team name via regex (`r'(\D+)\s+(\d+)$'` → team + seed)
- Convert all numeric columns to float

### 3. Load Bracket Data
- Read `bracket.csv`
- Handle play-in teams (split on "/", project winner by KenPom rank): NC State > Texas, SMU > Miami (OH), others are 16-seeds so irrelevant
- **Name normalization dictionary** for merging across sources:
  - `Michigan St.` → `Michigan State`, `N.C. State` → `NC State`, `Miami FL` → `Miami (FL)`, `Connecticut` → `UConn`, `Utah St.` → `Utah State`, etc.
- Merge bracket regions/seeds onto KenPom data

### 4. Scrape NET Rankings & Quad Records (robust)
- **NET rankings**: scrape `warrennolan.com/basketball/2026/net`, parse HTML table
- **Quad records**: scrape `warrennolan.com/basketball/2026/team-sheet/{Team}` for each of the 70 tournament teams
  - Build a **complete, hand-verified URL slug dictionary** for all 70 teams (handle St./Saint, parenthetical locations, apostrophes, hyphens, etc.)
  - Retry with alternate URL encodings on 404
  - Parse Q1/Q2/Q3/Q4 W-L from team sheet tables
  - Compute `Q1_WinPct` and `Q1Q2_WinPct`
- **Cache all scraped data** to `scraped_data/*.csv` — skip scraping on re-run if cache exists
- **Fallback chain**: Warren Nolan → Sports Reference → print CSV template for manual entry

### 5. Composite Scoring Model

Normalize each metric to 0–100 (min-max across 70 tournament teams), then weighted sum:

| Metric | Weight | Notes |
|--------|--------|-------|
| KenPom NetRtg | 0.25 | Primary efficiency |
| KenPom ORtg | 0.15 | Offense (professor's hint) |
| KenPom DRtg | 0.10 | Defense (inverted, lower=better) |
| KenPom SOS NetRtg | 0.10 | Schedule strength |
| NET Rank | 0.15 | Committee ranking (inverted) |
| Q1 Win % | 0.15 | Elite opponent performance |
| Q1+Q2 Win % | 0.10 | Broader quality wins |

**No defense floor penalty** — let the composite math handle bad defense naturally through DRtg and NetRtg weights.

**Tiebreaker**: When composite scores are within 1 point, rank by ORtg (soft offensive tiebreaker per professor's hint).

**If quad data scraping fails entirely**: redistribute to NetRtg=0.35, ORtg=0.20, NET=0.25, SOS=0.10, DRtg=0.10.

### 6. Round-by-Round Monte Carlo Bracket Simulation

Simulate each round (R64 → R32 → Sweet 16 → Elite 8) per region using KenPom efficiency differentials, **10,000 Monte Carlo iterations**:
- **Win probability formula**: Use log5 method or KenPom-style expected margin = (TeamA_NetRtg - TeamB_NetRtg) / 2, then convert to win probability via logistic function
- Each sim randomly draws winners weighted by win probability
- Across 10K sims, compute each team's **Final Four appearance rate** (e.g., Duke reaches Final Four in 68% of sims)
- This naturally accounts for bracket path difficulty and variance (a 12-seed *can* beat a 5-seed in some sims)
- Output: probability distribution per region showing how often each team reaches the Final Four

### 7. Seed Constraint Optimization

- After simulation, each region produces a projected Elite 8 winner with a Final Four probability
- **Brute-force** all 4-tuples (one team per region, considering top ~3 candidates who could plausibly win through), filter to seed sum ≥ 15, pick max total composite score weighted by path probability
- Output: optimal Final Four + 2–3 alternative configurations for comparison
- **Flag suspicious picks**: any team with SOS in the bottom quartile of the field or from a non-power conference gets a warning label — user decides whether to keep or swap
- Strategy: expect a mix of 1–2 chalk picks + 2–3 data-backed upsets (good for class debate differentiation)

### 8. Visualizations (full report)

1. **Efficiency Scatter Plot** — ORtg (x) vs DRtg (y, inverted). Color by region, size by composite. Label seeds 1–4 and Final Four picks with arrows.
2. **Composite Ranking Bar Chart** — Top 25 teams, horizontal bars colored by region. Final Four picks highlighted.
3. **Regional Analysis Tables** — Styled table per region: top 6 teams with seed, composite, KenPom rank, NET rank, Q1 record, ORtg. Final Four pick row highlighted.
4. **Radar Charts (presentation-quality)** — 2×2 grid, one per region. Compare top 2–3 contenders across 6 dimensions (NetRtg, ORtg, DRtg⁻¹, SOS, NET⁻¹, Q1%). Clean labels, good sizing, suitable for screenshotting into class debate slides.
5. **Seed Constraint Visualization** — Top 3–4 configurations meeting seed ≥ 15, showing composite totals.
6. **ORtg vs Seed** — Scatter showing offensive efficiency by seed, highlighting dark horse candidates.

### 9. Essay Bullet Points & Summary

Per Final Four pick, output:
- KenPom rank, NET rank, seed, region
- ORtg (rank among field), DRtg (rank)
- Q1 record and win %
- SOS rank
- Path probability from simulation
- Most impressive distinguishing stat
- Whether this is a chalk or dark horse pick

Generate **two essay draft structures** (user picks which to use):
1. **Per-team paragraphs**: 4 short paragraphs (~50 words each), each covering one team with key stats
2. **Cohesive narrative**: One flowing 200-word essay connecting picks thematically (e.g., "my bracket is built around offensive efficiency and elite scheduling...")

Plus a one-line thesis statement tying the four picks together.

---

## Key Implementation Details

- **Play-in projections**: NC State (KP#34) over Texas (KP#37), SMU (KP#42) over Miami OH (KP#93)
- **KenPom rank gaps** (39→41, etc.) are real KenPom ranks — preserve as-is
- **Seed constraint math**: All four 1-seeds = sum of 4 (fails badly). The optimizer will find configurations like Duke(1) + Vanderbilt(5) + Arkansas(4) + Texas Tech(5) = 15, balancing composite strength against the constraint.

---

## Verification

1. Run all notebook cells top-to-bottom — no errors
2. Confirm scraped data matches spot-checks against Warren Nolan website
3. Confirm seed sum of final picks ≥ 15
4. Confirm one pick per region (East, South, West, Midwest)
5. Visually inspect all 6+ charts render correctly
6. Confirm essay bullet points contain accurate stats matching the data
