#!/usr/bin/env python3
"""Build the Final Four analysis notebook."""
import json


def src(text):
    """Convert triple-quoted text to .ipynb source array."""
    lines = text.split('\n')
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return []
    return [line + '\n' for line in lines[:-1]] + [lines[-1]]


def md(cell_id, text):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src(text)}


def code(cell_id, text):
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": src(text)}


cells = []

# ==============================================================
# CELL: Title
# ==============================================================
cells.append(md("title", r"""# March Madness 2026: Data-Driven Final Four Analysis

A composite analytical pipeline using KenPom efficiency metrics, real NCAA NET rankings,
committee metrics (SOR, WAB, KPI, BPI), quad records, Monte Carlo bracket simulation, and seed-constraint optimization.

**Seed constraint:** Sum of seeds >= 15 | **Method:** Composite model -> 10K Monte Carlo sims -> Brute-force optimization"""))

# ==============================================================
# CELL: Setup & Imports
# ==============================================================
cells.append(code("setup", r"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import re, os, warnings
from itertools import product as iterproduct
from collections import defaultdict

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = [12, 8]

REGION_COLORS = {'East': '#1f77b4', 'South': '#d62728', 'West': '#2ca02c', 'Midwest': '#ff7f0e'}
SCRAPED_DIR = 'scraped_data'
os.makedirs(SCRAPED_DIR, exist_ok=True)
print('Setup complete.')"""))

# ==============================================================
# CELL: Section 2 Header
# ==============================================================
cells.append(md("s2-hdr", r"""## 2. Load KenPom Data

Parse team efficiency ratings, offensive/defensive metrics, and strength of schedule."""))

# ==============================================================
# CELL: Load KenPom
# ==============================================================
cells.append(code("load-kenpom", r"""col_names = ['Rk', 'Team', 'Conf', 'W_L', 'NetRtg', 'ORtg', 'DRtg', 'AdjT', 'Luck',
             'SOS_NetRtg', 'SOS_ORtg', 'SOS_DRtg', 'NCSOS_NetRtg']
kenpom = pd.read_csv('kenpom.csv', skiprows=2, header=None, names=col_names)

# Parse seed from team name (e.g., "Duke 1" -> ("Duke", 1))
def parse_team_seed(team_str):
    m = re.match(r'(.+?)\s+(\d+)$', str(team_str))
    if m:
        return m.group(1).strip(), int(m.group(2))
    return str(team_str).strip(), None

kenpom[['TeamName', 'KP_Seed']] = kenpom['Team'].apply(
    lambda x: pd.Series(parse_team_seed(x))
)

for col in ['Rk', 'NetRtg', 'ORtg', 'DRtg', 'AdjT', 'Luck',
            'SOS_NetRtg', 'SOS_ORtg', 'SOS_DRtg', 'NCSOS_NetRtg']:
    kenpom[col] = pd.to_numeric(kenpom[col], errors='coerce')
kenpom['Rk'] = kenpom['Rk'].astype(int)
kenpom['KP_Seed'] = kenpom['KP_Seed'].astype(int)

print(f'Loaded {len(kenpom)} teams from KenPom')
kenpom[['Rk', 'TeamName', 'KP_Seed', 'Conf', 'W_L', 'NetRtg', 'ORtg', 'DRtg', 'SOS_NetRtg']].head(10)"""))

# ==============================================================
# CELL: Section 3 Header
# ==============================================================
cells.append(md("s3-hdr", r"""## 3. Load Bracket Data & Merge

Resolve play-in games, normalize team names across sources, and merge bracket structure with KenPom data."""))

# ==============================================================
# CELL: Load Bracket + Merge
# ==============================================================
cells.append(code("load-bracket", r"""bracket = pd.read_csv('bracket.csv')

# KenPom name -> standard/bracket name
# Canonical name mapping: variant -> standard bracket name (shared across all data sources)
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
kenpom['StdName'] = kenpom['TeamName'].replace(NAME_ALIASES)

# Play-in resolutions (winner projected by KenPom rank)
PLAYIN = {
    'NC State/Texas': ('NC State', 11),       # KP #34 > #37
    'SMU/Miami (OH)': ('SMU', 11),            # KP #42 > #93
    'Howard/UMBC': ('UMBC', 16),              # KP #188 > #204
    'Lehigh/Prairie View A&M': ('Lehigh', 16) # KP #284 > #288
}

rows = []
for _, row in bracket.iterrows():
    r = row.to_dict()
    for tcol, scol in [('TeamA', 'SeedA'), ('TeamB', 'SeedB')]:
        if r[tcol] in PLAYIN:
            r[tcol], r[scol] = PLAYIN[r[tcol]]
    rows.append(r)
bracket = pd.DataFrame(rows)

print('Play-in projections:')
for matchup, (winner, seed) in PLAYIN.items():
    print(f'  {matchup} -> {winner} (seed {seed})')

# Build flat team list from bracket
team_rows = []
for _, row in bracket.iterrows():
    team_rows.append({'StdName': row['TeamA'], 'Region': row['Region'],
                      'Seed': int(row['SeedA']), 'GameID': row['GameID']})
    team_rows.append({'StdName': row['TeamB'], 'Region': row['Region'],
                      'Seed': int(row['SeedB']), 'GameID': row['GameID']})
bracket_teams = pd.DataFrame(team_rows).drop_duplicates(subset='StdName')

# Merge with KenPom
df = bracket_teams.merge(
    kenpom.drop(columns=['Team', 'TeamName', 'KP_Seed']),
    on='StdName', how='left'
)

missing = df[df['NetRtg'].isna()]['StdName'].tolist()
if missing:
    print(f'\nWARNING: No KenPom data for: {missing}')
else:
    print(f'\nAll {len(df)} tournament teams matched successfully.')

df.sort_values('Rk')[['StdName', 'Region', 'Seed', 'Rk', 'NetRtg', 'ORtg', 'DRtg', 'SOS_NetRtg']].head(16)"""))

# ==============================================================
# CELL: Section 4 Header
# ==============================================================
cells.append(md("s4-hdr", r"""## 4. Load NET Rankings & Tournament Team Metrics

Load real NCAA NET rankings and rich team-sheet metrics (KPI, SOR, WAB, BPI) from CSV files.
This replaces web scraping with reliable local data and adds truly independent metrics beyond KenPom."""))

# ==============================================================
# CELL: Scrape + Cache
# ==============================================================
cells.append(code("load-metrics", r"""# --- Load NET Rankings from CSV (all 365 D1 teams) ---
NET_CSV = 'ncaa-net-rankings.csv'
TEAM_SHEET_CSV = os.path.join(SCRAPED_DIR, 'tournament_teams.csv')

has_net = False
has_teamsheet = False


if os.path.exists(NET_CSV):
    net_df = pd.read_csv(NET_CSV)
    net_df['School'] = net_df['School'].str.strip()
    net_df['StdName'] = net_df['School'].replace(NAME_ALIASES)
    has_net = True
    print(f'Loaded NET rankings: {len(net_df)} teams from {NET_CSV}')

    # Parse quad records from NET CSV (format: "17-2") using vectorized str.extract
    def parse_wl(series):
        parsed = series.str.extract(r'(\d+)-(\d+)')
        w = pd.to_numeric(parsed[0], errors='coerce').fillna(0).astype(int)
        l = pd.to_numeric(parsed[1], errors='coerce').fillna(0).astype(int)
        return w, l
    net_df['Q1_W'], net_df['Q1_L'] = parse_wl(net_df['Quad 1'])
    net_df['Q2_W'], net_df['Q2_L'] = parse_wl(net_df['Quad 2'])
    net_df['Q1_WinPct'] = net_df['Q1_W'] / (net_df['Q1_W'] + net_df['Q1_L']).replace(0, np.nan)
    net_df['Q1_WinPct'] = net_df['Q1_WinPct'].fillna(0)
    q1q2_total = net_df['Q1_W'] + net_df['Q2_W'] + net_df['Q1_L'] + net_df['Q2_L']
    net_df['Q1Q2_WinPct'] = (net_df['Q1_W'] + net_df['Q2_W']) / q1q2_total.replace(0, np.nan)
    net_df['Q1Q2_WinPct'] = net_df['Q1Q2_WinPct'].fillna(0)
else:
    print(f'WARNING: {NET_CSV} not found - using KenPom rank as NET proxy')

# --- Load tournament team sheet metrics (KPI, SOR, WAB, BPI, etc.) ---
if os.path.exists(TEAM_SHEET_CSV):
    ts_df = pd.read_csv(TEAM_SHEET_CSV)
    ts_df['StdName'] = ts_df['team'].replace(NAME_ALIASES)
    has_teamsheet = True
    print(f'Loaded tournament team sheets: {len(ts_df)} teams from {TEAM_SHEET_CSV}')
else:
    print(f'WARNING: {TEAM_SHEET_CSV} not found - SOR/WAB/KPI metrics unavailable')

# --- Merge NET rankings ---
if has_net:
    net_merge = net_df[['StdName', 'Rank', 'Q1_W', 'Q1_L', 'Q1_WinPct', 'Q2_W', 'Q2_L', 'Q1Q2_WinPct']].copy()
    net_merge = net_merge.rename(columns={'Rank': 'NET_Rank'})
    df = df.merge(net_merge, on='StdName', how='left')
    matched = df['NET_Rank'].notna().sum()
    print(f'Matched {matched}/{len(df)} teams with NET rankings')
    df['NET_Rank'] = df['NET_Rank'].fillna(df['Rk'])
    df['Q1_WinPct'] = df['Q1_WinPct'].fillna(0)
    df['Q1Q2_WinPct'] = df['Q1Q2_WinPct'].fillna(0)
else:
    df['NET_Rank'] = df['Rk'].astype(float)
    df['Q1_WinPct'] = 0.0
    df['Q1Q2_WinPct'] = 0.0
    df['Q1_W'] = np.nan
    df['Q1_L'] = np.nan

# --- Merge team sheet metrics (SOR, WAB, KPI, BPI, NET SOS) ---
if has_teamsheet:
    ts_cols = ts_df[['StdName', 'net_sos', 'kpi', 'sor', 'wab', 'bpi', 'avg_net_wins', 'avg_net_losses']].copy()
    ts_cols = ts_cols.rename(columns={
        'net_sos': 'NET_SOS', 'kpi': 'KPI', 'sor': 'SOR',
        'wab': 'WAB', 'bpi': 'BPI',
        'avg_net_wins': 'Avg_NET_Wins', 'avg_net_losses': 'Avg_NET_Losses',
    })
    for col in ['NET_SOS', 'KPI', 'SOR', 'WAB', 'BPI', 'Avg_NET_Wins', 'Avg_NET_Losses']:
        ts_cols[col] = pd.to_numeric(ts_cols[col], errors='coerce')
    df = df.merge(ts_cols, on='StdName', how='left')
    ts_matched = df['SOR'].notna().sum()
    print(f'Matched {ts_matched}/{len(df)} teams with team sheet metrics (SOR/WAB/KPI/BPI)')
    if ts_matched < len(df):
        missing_ts = df[df['SOR'].isna()]['StdName'].tolist()
        print(f'  Missing: {missing_ts}')
else:
    for col in ['NET_SOS', 'KPI', 'SOR', 'WAB', 'BPI', 'Avg_NET_Wins', 'Avg_NET_Losses']:
        df[col] = np.nan

has_quad = has_net and df['Q1_WinPct'].sum() > 0

# Show NET vs KenPom rank comparison
print(f'\nNET vs KenPom rank divergence (top 10 by |diff|):')
_diff = (df['NET_Rank'] - df['Rk']).abs()
for idx in _diff.nlargest(10).index:
    row = df.loc[idx]
    print(f'  {row["StdName"]}: NET #{int(row["NET_Rank"])} vs KP #{int(row["Rk"])} (diff: {int(row["NET_Rank"] - row["Rk"])})')

print(f'\nData ready: {len(df)} teams')
print(f'  NET rankings: {"real" if has_net else "KenPom proxy"}')
print(f'  Quad records: {"yes" if has_quad else "no"}')
print(f'  Team sheet metrics (SOR/WAB/KPI/BPI): {"yes" if has_teamsheet else "no"}')"""))

# ==============================================================
# CELL: Section 5 Header
# ==============================================================
cells.append(md("s5-hdr", r"""## 5. Composite Scoring Model

Normalize each metric to 0-100 (min-max across tournament field), then compute a weighted composite score.
Now uses **real NET rankings** (not KenPom proxy) and adds committee-used metrics (SOR, WAB) from team sheets.

| Metric | Weight (full) | Weight (fallback) | Source |
|--------|:---:|:---:|--------|
| KenPom NetRtg | 0.20 | 0.30 | kenpom.csv |
| KenPom ORtg | 0.10 | 0.15 | kenpom.csv |
| KenPom DRtg (inverted) | 0.10 | 0.10 | kenpom.csv |
| NET Rank (inverted) | 0.15 | 0.20 | ncaa-net-rankings.csv |
| SOR (inverted) | 0.10 | -- | tournament_teams.csv |
| WAB | 0.10 | -- | tournament_teams.csv |
| NET SOS (inverted) | 0.05 | 0.10 | tournament_teams.csv |
| Q1 Win % | 0.10 | 0.15 | ncaa-net-rankings.csv |
| Q1+Q2 Win % | 0.10 | -- | ncaa-net-rankings.csv |

**Tiebreaker:** When composite scores are within 1 point, rank by ORtg."""))

# ==============================================================
# CELL: Composite Model
# ==============================================================
cells.append(code("composite", r"""def normalize_0_100(series, invert=False):
    s = series.copy()
    if invert:
        s = -s
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(50.0, index=series.index)
    return ((s - s.min()) / rng * 100)

# Core KenPom metrics
df['n_NetRtg'] = normalize_0_100(df['NetRtg'])
df['n_ORtg'] = normalize_0_100(df['ORtg'])
df['n_DRtg'] = normalize_0_100(df['DRtg'], invert=True)

# Real NET rank (now independent from KenPom!)
df['n_NET'] = normalize_0_100(df['NET_Rank'], invert=True)

# Team sheet metrics (SOR, WAB, NET SOS)
has_sor = has_teamsheet and df['SOR'].notna().any()
if has_sor:
    df['n_SOR'] = normalize_0_100(df['SOR'].fillna(df['SOR'].max()), invert=True)
    df['n_WAB'] = normalize_0_100(df['WAB'].fillna(df['WAB'].min()))
    df['n_NET_SOS'] = normalize_0_100(df['NET_SOS'].fillna(df['NET_SOS'].max()), invert=True)

# Shared normalizations (computed once, used by multiple branches)
has_q1 = has_quad
if has_q1:
    df['n_Q1'] = normalize_0_100(df['Q1_WinPct'].fillna(0))
    df['n_Q1Q2'] = normalize_0_100(df['Q1Q2_WinPct'].fillna(0))
if not has_sor:
    df['n_SOS'] = normalize_0_100(df['SOS_NetRtg'])

if has_sor and has_q1:
    # Full scheme: 9 truly independent features
    df['Composite'] = (
        0.20 * df['n_NetRtg'] + 0.10 * df['n_ORtg'] + 0.10 * df['n_DRtg'] +
        0.15 * df['n_NET'] +
        0.10 * df['n_SOR'] + 0.10 * df['n_WAB'] + 0.05 * df['n_NET_SOS'] +
        0.10 * df['n_Q1'] + 0.10 * df['n_Q1Q2']
    )
    weight_scheme = 'Full (NET + SOR/WAB + quad records)'
elif has_q1:
    # Mid scheme: no team sheet but have quad records
    df['Composite'] = (
        0.25 * df['n_NetRtg'] + 0.10 * df['n_ORtg'] + 0.10 * df['n_DRtg'] +
        0.10 * df['n_SOS'] + 0.15 * df['n_NET'] +
        0.15 * df['n_Q1'] + 0.15 * df['n_Q1Q2']
    )
    weight_scheme = 'Mid (NET + quad records, no SOR/WAB)'
else:
    # Fallback: minimal data
    df['Composite'] = (
        0.30 * df['n_NetRtg'] + 0.15 * df['n_ORtg'] + 0.10 * df['n_DRtg'] +
        0.10 * df['n_SOS'] + 0.35 * df['n_NET']
    )
    weight_scheme = 'Fallback (no quad records or team sheets)'

# Sort with ORtg tiebreaker
df = df.sort_values(['Composite', 'ORtg'], ascending=[False, False]).reset_index(drop=True)
df['CompRank'] = range(1, len(df) + 1)
df['ORtg_Rank'] = df['ORtg'].rank(ascending=False).astype(int)
df['DRtg_Rank'] = df['DRtg'].rank(ascending=True).astype(int)

print(f'Composite scoring: {weight_scheme}\n')
print('Top 20 by Composite Score:')
show_cols = ['CompRank', 'StdName', 'Region', 'Seed', 'Composite', 'NetRtg', 'NET_Rank']
if has_sor:
    show_cols += ['SOR', 'WAB']
df[show_cols].head(20)"""))

# ==============================================================
# CELL: Section 6 Header
# ==============================================================
cells.append(md("s6-hdr", r"""## 6. Round-by-Round Monte Carlo Bracket Simulation

Simulate each regional bracket **10,000 times** using KenPom efficiency differentials converted to win probabilities via logistic function.

- **Win probability:** `P(A) = 1 / (1 + 10^(-(NetRtg_A - NetRtg_B) / 22))`
- Naturally accounts for bracket path difficulty and variance"""))

# ==============================================================
# CELL: Monte Carlo Simulation
# ==============================================================
cells.append(code("montecarlo", r"""N_SIMS = 10_000

def win_prob(net_a, net_b):
    margin = (net_a - net_b) / 2
    return 1 / (1 + 10 ** (-margin / 11))

REGION_SEED = {'East': 0, 'South': 1, 'West': 2, 'Midwest': 3}

def simulate_region(region, df, bracket, n_sims=N_SIMS, seed=42):
    rng = np.random.default_rng(seed + REGION_SEED[region])
    games = bracket[bracket['Region'] == region].sort_values('GameID')

    # Build R64 matchups as lightweight dicts
    matchups = []
    for _, g in games.iterrows():
        a = df[df['StdName'] == g['TeamA']].iloc[0]
        b = df[df['StdName'] == g['TeamB']].iloc[0]
        matchups.append((
            {'name': a['StdName'], 'net': float(a['NetRtg']), 'seed': int(a['Seed'])},
            {'name': b['StdName'], 'net': float(b['NetRtg']), 'seed': int(b['Seed'])}
        ))

    counts = defaultdict(int)

    for _ in range(n_sims):
        # R64
        w = []
        for a, b in matchups:
            p = win_prob(a['net'], b['net'])
            w.append(a if rng.random() < p else b)
        # R32
        r32 = []
        for i in range(0, 8, 2):
            p = win_prob(w[i]['net'], w[i+1]['net'])
            r32.append(w[i] if rng.random() < p else w[i+1])
        # Sweet 16
        s16 = []
        for i in range(0, 4, 2):
            p = win_prob(r32[i]['net'], r32[i+1]['net'])
            s16.append(r32[i] if rng.random() < p else r32[i+1])
        # Elite 8
        p = win_prob(s16[0]['net'], s16[1]['net'])
        champ = s16[0] if rng.random() < p else s16[1]
        counts[champ['name']] += 1

    return {t: c / n_sims for t, c in sorted(counts.items(), key=lambda x: -x[1])}

print(f'Running {N_SIMS:,} Monte Carlo simulations per region...\n')
sim_results = {}
for region in REGION_COLORS:
    sim_results[region] = simulate_region(region, df, bracket)
    top = list(sim_results[region].items())[:5]
    print(f'{region} Region:')
    for team, prob in top:
        seed = df[df['StdName'] == team]['Seed'].values[0]
        print(f'  ({seed}) {team}: {prob:.1%}')
    print()

# Add FF probability to main dataframe
ff_map = {team: prob for region_res in sim_results.values() for team, prob in region_res.items()}
df['FF_Prob'] = df['StdName'].map(ff_map).fillna(0)"""))

# ==============================================================
# CELL: Section 7 Header
# ==============================================================
cells.append(md("s7-hdr", r"""## 7. Seed Constraint Optimization

Brute-force all 4-tuples (one per region, top 5 candidates each) to find the configuration
that maximizes probability-weighted composite score while satisfying seed sum >= 15."""))

# ==============================================================
# CELL: Optimization
# ==============================================================
cells.append(code("optimize", r"""# Top 5 candidates per region by FF probability
candidates = {}
for region in REGION_COLORS:
    region_df = df[df['Region'] == region].nlargest(5, 'FF_Prob')
    candidates[region] = region_df[['StdName', 'Seed', 'Composite', 'FF_Prob']].to_dict('records')

# Brute-force all valid combos
configs = []
for e, s, w, m in iterproduct(candidates['East'], candidates['South'],
                               candidates['West'], candidates['Midwest']):
    seed_sum = e['Seed'] + s['Seed'] + w['Seed'] + m['Seed']
    if seed_sum >= 15:
        total_comp = e['Composite'] + s['Composite'] + w['Composite'] + m['Composite']
        weighted = (e['Composite'] * e['FF_Prob'] + s['Composite'] * s['FF_Prob'] +
                    w['Composite'] * w['FF_Prob'] + m['Composite'] * m['FF_Prob'])
        configs.append({
            'East': e['StdName'], 'South': s['StdName'],
            'West': w['StdName'], 'Midwest': m['StdName'],
            'Seeds': f"{e['Seed']}+{s['Seed']}+{w['Seed']}+{m['Seed']}={seed_sum}",
            'Seed_Sum': seed_sum, 'Total_Composite': round(total_comp, 1),
            'Weighted_Score': round(weighted, 1),
            'E_Seed': e['Seed'], 'S_Seed': s['Seed'],
            'W_Seed': w['Seed'], 'M_Seed': m['Seed'],
        })

configs_df = pd.DataFrame(configs).sort_values('Weighted_Score', ascending=False)
print(f'Found {len(configs_df)} valid configurations (seed sum >= 15)\n')

print('Top 5 Configurations:')
print('=' * 85)
for idx, (_, row) in enumerate(configs_df.head(5).iterrows()):
    marker = ' <<<' if idx == 0 else ''
    print(f"#{idx+1}: {row['East']:>16} | {row['South']:>12} | {row['West']:>12} | {row['Midwest']:>12}")
    print(f"   Seeds: {row['Seeds']}   Composite: {row['Total_Composite']}   Weighted: {row['Weighted_Score']}{marker}")
    print()

# Select optimal
optimal = configs_df.iloc[0]
FINAL_FOUR = [optimal['East'], optimal['South'], optimal['West'], optimal['Midwest']]
FINAL_FOUR_SEEDS = {
    optimal['East']: int(optimal['E_Seed']), optimal['South']: int(optimal['S_Seed']),
    optimal['West']: int(optimal['W_Seed']), optimal['Midwest']: int(optimal['M_Seed']),
}

print('=' * 85)
print(f'OPTIMAL FINAL FOUR:  {" | ".join(FINAL_FOUR)}')
print(f'Seed Sum: {optimal["Seed_Sum"]}')

# Flag suspicious picks
print('\nPick Analysis:')
# Normalize SOS to percentile (higher = harder schedule) regardless of source
if has_sor:
    _sos_pctile = df['NET_SOS'].rank(ascending=True, pct=True)  # rank is inverted: low NET_SOS = hard
else:
    _sos_pctile = df['SOS_NetRtg'].rank(ascending=False, pct=True)  # rating: high = hard
for team in FINAL_FOUR:
    _match = df[df['StdName'] == team]
    row = _match.iloc[0]
    tidx = _match.index[0]
    flags = []
    if _sos_pctile.loc[tidx] < 0.25:
        flags.append('LOW SOS')
    if row['Conf'] not in ['ACC', 'B10', 'B12', 'SEC', 'BE']:
        flags.append('NON-POWER')
    pick_type = 'CHALK' if row['Seed'] <= 2 else ('DARK HORSE' if row['Seed'] >= 5 else 'MODERATE')
    flag_str = f'  [{", ".join(flags)}]' if flags else ''
    print(f'  ({int(row["Seed"])}) {team} - Composite: {row["Composite"]:.1f}, '
          f'FF Prob: {row["FF_Prob"]:.1%}, Type: {pick_type}{flag_str}')"""))

# ==============================================================
# CELL: Section 8 Header
# ==============================================================
cells.append(md("s8-hdr", r"""## 8. Visualizations"""))

# ==============================================================
# CELL: Efficiency Scatter Plot
# ==============================================================
cells.append(code("viz-scatter", r"""fig, ax = plt.subplots(figsize=(14, 10))

for _, row in df.iterrows():
    color = REGION_COLORS[row['Region']]
    is_ff = row['StdName'] in FINAL_FOUR
    size = 400 if is_ff else max(20, row['Composite'] * 2.5)
    marker = '*' if is_ff else 'o'
    ax.scatter(row['ORtg'], row['DRtg'], c=color, s=size, alpha=1.0 if is_ff else 0.65,
              marker=marker, edgecolors='black' if is_ff else 'none',
              linewidths=2 if is_ff else 0, zorder=10 if is_ff else 3)

# Label seeds 1-4 and Final Four picks
for _, row in df.iterrows():
    if row['Seed'] <= 4 or row['StdName'] in FINAL_FOUR:
        weight = 'bold' if row['StdName'] in FINAL_FOUR else 'normal'
        ax.annotate(f"({int(row['Seed'])}) {row['StdName']}", (row['ORtg'], row['DRtg']),
                   textcoords='offset points', xytext=(8, 5), fontsize=7, fontweight=weight)

ax.set_xlabel('Offensive Rating (ORtg) -->', fontsize=12)
ax.set_ylabel('<-- Defensive Rating (DRtg) -- lower is better', fontsize=12)
ax.invert_yaxis()
ax.set_title('2026 NCAA Tournament: Offensive vs Defensive Efficiency', fontsize=14, fontweight='bold')

handles = [mpatches.Patch(color=c, label=r) for r, c in REGION_COLORS.items()]
handles.append(Line2D([0], [0], marker='*', color='gray', markersize=15, linestyle='None', label='Final Four Pick'))
ax.legend(handles=handles, loc='lower left', fontsize=9)
plt.tight_layout()
plt.show()"""))

# ==============================================================
# CELL: Composite Bar Chart
# ==============================================================
cells.append(code("viz-bars", r"""top25 = df.nlargest(25, 'Composite').sort_values('Composite')
fig, ax = plt.subplots(figsize=(12, 10))

colors = [REGION_COLORS[r] for r in top25['Region']]
bars = ax.barh(range(len(top25)), top25['Composite'], color=colors, edgecolor='white')

for i, (_, row) in enumerate(top25.iterrows()):
    if row['StdName'] in FINAL_FOUR:
        bars[i].set_edgecolor('gold')
        bars[i].set_linewidth(3)

ax.set_yticks(range(len(top25)))
ax.set_yticklabels([f"({int(r['Seed'])}) {r['StdName']}" for _, r in top25.iterrows()], fontsize=9)
ax.set_xlabel('Composite Score', fontsize=12)
ax.set_title('Top 25 Teams by Composite Score', fontsize=14, fontweight='bold')

handles = [mpatches.Patch(color=c, label=r) for r, c in REGION_COLORS.items()]
handles.append(mpatches.Patch(facecolor='gray', edgecolor='gold', linewidth=3, label='Final Four Pick'))
ax.legend(handles=handles, loc='lower right', fontsize=9)
plt.tight_layout()
plt.show()"""))

# ==============================================================
# CELL: Regional Analysis Tables
# ==============================================================
cells.append(code("viz-tables", r"""from IPython.display import display, HTML

for region in REGION_COLORS:
    region_df = df[df['Region'] == region].nlargest(6, 'Composite')

    cols = ['Seed', 'StdName', 'Composite', 'Rk', 'NetRtg', 'ORtg', 'DRtg', 'NET_Rank', 'FF_Prob']
    if has_sor:
        cols.insert(8, 'SOR')
        cols.insert(9, 'WAB')
    if has_quad and 'Q1_W' in df.columns:
        cols.insert(-1, 'Q1_W')
        cols.insert(-1, 'Q1_L')

    tbl = region_df[cols].copy()
    rename_map = {'StdName': 'Team', 'Rk': 'KP#', 'NET_Rank': 'NET#', 'FF_Prob': 'FF%'}
    tbl = tbl.rename(columns=rename_map)

    tbl['FF%'] = tbl['FF%'].apply(lambda x: f'{x:.1%}')
    tbl['Composite'] = tbl['Composite'].round(1)
    tbl['NET#'] = tbl['NET#'].astype(int)
    if 'Q1_W' in tbl.columns and 'Q1_L' in tbl.columns:
        tbl['Q1'] = tbl.apply(lambda r: f"{int(r['Q1_W'])}-{int(r['Q1_L'])}" if pd.notna(r['Q1_W']) else '-', axis=1)
        tbl = tbl.drop(columns=['Q1_W', 'Q1_L'])
    tbl = tbl.reset_index(drop=True)

    def highlight_ff(row):
        if row['Team'] in FINAL_FOUR:
            return ['background-color: #ffffcc; font-weight: bold'] * len(row)
        return [''] * len(row)

    color = REGION_COLORS[region]
    styled = (tbl.style
              .apply(highlight_ff, axis=1)
              .set_caption(f'{region} Region')
              .set_properties(**{'text-align': 'center'})
              .set_table_styles([{'selector': 'caption',
                                  'props': [('font-size', '14px'), ('font-weight', 'bold'),
                                            ('color', color)]}]))
    display(styled)
    print()"""))

# ==============================================================
# CELL: Radar Charts
# ==============================================================
cells.append(code("viz-radar", r"""fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw=dict(polar=True))

radar_metrics = ['n_NetRtg', 'n_ORtg', 'n_DRtg', 'n_NET']
radar_labels = ['Net Rating', 'Off Rating', 'Def Rating', 'NET Rank']

if has_sor and 'n_SOR' in df.columns:
    radar_metrics += ['n_SOR', 'n_WAB']
    radar_labels += ['SOR', 'WAB']

if has_quad and 'n_Q1' in df.columns:
    radar_metrics.append('n_Q1')
    radar_labels.append('Q1 Win%')

n_met = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, n_met, endpoint=False).tolist()
angles += angles[:1]

for ax, region in zip(axes.flat, REGION_COLORS):
    region_top = df[df['Region'] == region].nlargest(3, 'Composite')

    for _, row in region_top.iterrows():
        values = [row[m] / 100 for m in radar_metrics]
        values += values[:1]
        is_ff = row['StdName'] in FINAL_FOUR
        label = f"({int(row['Seed'])}) {row['StdName']}"
        lw = 3 if is_ff else 1.5
        ls = '-' if is_ff else '--'
        ax.plot(angles, values, linewidth=lw, linestyle=ls, label=label)
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, size=8)
    ax.set_ylim(0, 1)
    ax.set_title(f'{region} Region', size=13, fontweight='bold',
                color=REGION_COLORS[region], pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

plt.suptitle('Regional Contender Comparison (Radar)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()"""))

# ==============================================================
# CELL: Seed Constraint + ORtg Scatter
# ==============================================================
cells.append(code("viz-seed-ortg", r"""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

# --- Left: Seed Constraint Configurations ---
top_cfgs = configs_df.head(5).reset_index(drop=True)
bar_colors = ['#2ecc71' if i == 0 else '#95a5a6' for i in range(len(top_cfgs))]

for i, (_, cfg) in enumerate(top_cfgs.iterrows()):
    teams_label = f"{cfg['East']}\n{cfg['South']}\n{cfg['West']}\n{cfg['Midwest']}"
    ax1.barh(i, cfg['Weighted_Score'], color=bar_colors[i], edgecolor='white', height=0.6)
    ax1.text(cfg['Weighted_Score'] + 0.3, i, f"Seeds: {cfg['Seeds']}", va='center', fontsize=7)
    ax1.text(1, i, teams_label, va='center', fontsize=6.5,
             fontweight='bold' if i == 0 else 'normal')

ax1.set_xlabel('Probability-Weighted Composite Score')
ax1.set_title('Top Configurations (Seed Sum >= 15)', fontweight='bold')
ax1.set_yticks([])
ax1.invert_yaxis()

# --- Right: ORtg vs Seed ---
for _, row in df.iterrows():
    color = REGION_COLORS[row['Region']]
    is_ff = row['StdName'] in FINAL_FOUR
    marker = '*' if is_ff else 'o'
    size = 200 if is_ff else 50
    ax2.scatter(row['Seed'], row['ORtg'], c=color, s=size, marker=marker,
               edgecolors='black' if is_ff else 'none',
               linewidths=1.5 if is_ff else 0, alpha=0.8, zorder=10 if is_ff else 3)

# Label FF picks + high-ORtg upsets
ortg_q75 = df['ORtg'].quantile(0.75)
for _, row in df.iterrows():
    if row['StdName'] in FINAL_FOUR or (row['Seed'] >= 5 and row['ORtg'] >= ortg_q75):
        ax2.annotate(f"({int(row['Seed'])}) {row['StdName']}", (row['Seed'], row['ORtg']),
                    textcoords='offset points', xytext=(5, 5), fontsize=7,
                    fontweight='bold' if row['StdName'] in FINAL_FOUR else 'normal')

ax2.set_xlabel('Seed', fontsize=12)
ax2.set_ylabel('Offensive Rating (ORtg)', fontsize=12)
ax2.set_title('ORtg by Seed -- Dark Horse Candidates', fontweight='bold')
handles = [mpatches.Patch(color=c, label=r) for r, c in REGION_COLORS.items()]
ax2.legend(handles=handles, fontsize=8)

plt.tight_layout()
plt.show()"""))

# ==============================================================
# CELL: Section 9 Header
# ==============================================================
cells.append(md("s9-hdr", r"""## 9. Essay Bullet Points & Summary

Per-pick statistical breakdown and two essay draft structures (per-team paragraphs & cohesive narrative)."""))

# ==============================================================
# CELL: Essay
# ==============================================================
cells.append(code("essay", r"""print('=' * 85)
print('FINAL FOUR PICKS -- DETAILED BREAKDOWN')
print('=' * 85)

for team_name in FINAL_FOUR:
    row = df[df['StdName'] == team_name].iloc[0]
    region = row['Region']
    seed = int(row['Seed'])
    pick_type = 'Chalk' if seed <= 2 else ('Dark Horse' if seed >= 5 else 'Moderate upset')

    print(f'\n--- ({seed}) {team_name} -- {region} Region [{pick_type}] ---')
    print(f'  KenPom Rank: #{int(row["Rk"])}  |  NET Rank: #{int(row["NET_Rank"])}')
    print(f'  ORtg: {row["ORtg"]:.1f} (#{row["ORtg_Rank"]} in field)  |  DRtg: {row["DRtg"]:.1f} (#{row["DRtg_Rank"]} in field)')
    print(f'  NetRtg: {row["NetRtg"]:.2f}  |  SOS: {row["SOS_NetRtg"]:.2f}')
    if has_sor and pd.notna(row.get('SOR')):
        print(f'  SOR: #{int(row["SOR"])}  |  WAB: {row["WAB"]:.0f}  |  KPI: #{int(row["KPI"])}  |  BPI: #{int(row["BPI"])}')
    if has_quad and 'Q1_W' in df.columns and pd.notna(row.get('Q1_W')):
        print(f'  Q1 Record: {int(row["Q1_W"])}-{int(row["Q1_L"])} ({row["Q1_WinPct"]:.1%})')
    print(f'  Final Four Probability: {row["FF_Prob"]:.1%}')
    print(f'  Composite Score: {row["Composite"]:.1f} (#{int(row["CompRank"])} in field)')

seed_sum = sum(FINAL_FOUR_SEEDS.values())
print(f'\n{"=" * 85}')
seeds_str = ' + '.join(str(FINAL_FOUR_SEEDS[t]) for t in FINAL_FOUR)
print(f'SEED SUM: {seeds_str} = {seed_sum}')
print(f'{"=" * 85}')

# --- Thesis ---
print('\n\nTHESIS STATEMENT:')
print('My Final Four is built on a composite model weighing offensive efficiency,')
print('defensive quality, schedule strength, and NET rankings -- optimized under')
print(f'the seed-sum constraint (>= 15) via {N_SIMS:,}-iteration Monte Carlo bracket simulation.')

# --- Draft 1: Per-team paragraphs ---
print('\n\n' + '=' * 85)
print('ESSAY DRAFT 1: Per-Team Paragraphs (~200 words)')
print('=' * 85 + '\n')

for team_name in FINAL_FOUR:
    row = df[df['StdName'] == team_name].iloc[0]
    seed = int(row['Seed'])
    region = row['Region']
    sor_str = f', SOR #{int(row["SOR"])}' if has_sor and pd.notna(row.get('SOR')) else ''
    wab_str = f', WAB {row["WAB"]:.0f}' if has_sor and pd.notna(row.get('WAB')) else ''
    para = (f'{team_name} ({seed}-seed, {region}): Ranked #{int(row["Rk"])} in KenPom '
            f'and #{int(row["NET_Rank"])} in NET '
            f'with an offensive rating of {row["ORtg"]:.1f} (#{row["ORtg_Rank"]} in the tournament field), '
            f'{team_name} {"is a dominant force" if seed <= 2 else "is a data-backed upset pick"}. '
            f'Their net efficiency of {row["NetRtg"]:.1f}{sor_str}{wab_str} '
            f'confirm they have been tested against elite competition. '
            f'My simulation gives them a {row["FF_Prob"]:.0%} chance of reaching the Final Four.')
    if team_name == 'Duke':
        para += (' One caveat: starting point guard Caleb Foster is out with a right foot fracture '
                 'suffered March 7. Cayden Boozer has stepped into a larger role and performed well '
                 'through the ACC Tournament, but these efficiency numbers reflect a full-strength roster '
                 '-- making Duke my highest-upside but also highest-variance pick.')
    print(para + '\n')

# --- Draft 2: Cohesive narrative ---
print('\n' + '=' * 85)
print('ESSAY DRAFT 2: Cohesive Narrative (~200 words)')
print('=' * 85 + '\n')

chalk = [t for t in FINAL_FOUR if FINAL_FOUR_SEEDS[t] <= 2]
moderate = [t for t in FINAL_FOUR if 3 <= FINAL_FOUR_SEEDS[t] <= 4]
dark_horses = [t for t in FINAL_FOUR if FINAL_FOUR_SEEDS[t] >= 5]

parts = []
parts.append('My Final Four selections are driven by a composite analytical model that '
             'weighs KenPom efficiency ratings, real NCAA NET rankings, Strength of Record (SOR), '
             'Wins Above Bubble (WAB), and quad record performance -- blending predictive efficiency '
             'metrics with the resume-based measures the selection committee actually uses.')

if chalk:
    chalk_str = ' and '.join(chalk)
    parts.append(f'{chalk_str} {"anchors" if len(chalk) == 1 else "anchor"} my bracket '
                 f'as {"a top-tier squad" if len(chalk) == 1 else "top-tier squads"} with '
                 f'dominant efficiency profiles and favorable paths to the Final Four.')
    if 'Duke' in chalk:
        parts.append('A key risk factor: Duke will be without starting point guard Caleb Foster '
                     '(right foot fracture) for likely the entire tournament. '
                     'While Cayden Boozer has filled in capably, my model reflects full-strength '
                     'season numbers, so Duke carries more variance than the raw composite suggests.')

if moderate:
    for t in moderate:
        r = df[df['StdName'] == t].iloc[0]
        parts.append(f'{t} ({int(r["Seed"])}-seed) is a calculated moderate upset: '
                     f'ranked #{int(r["Rk"])} in KenPom with a {r["ORtg"]:.1f} offensive rating '
                     f'(#{r["ORtg_Rank"]} in the field), their efficiency metrics suggest '
                     f'they are significantly underseeded and primed for a deep run.')

if dark_horses:
    for t in dark_horses:
        r = df[df['StdName'] == t].iloc[0]
        parts.append(f'{t} ({int(r["Seed"])}-seed) is a data-driven dark horse: '
                     f'ranked #{int(r["Rk"])} in KenPom with a net rating of {r["NetRtg"]:.1f} '
                     f'and strength of schedule of {r["SOS_NetRtg"]:.1f}, they outperform '
                     f'their seed in every advanced metric.')

parts.append(f'With a seed sum of {seed_sum}, this configuration satisfies the constraint '
             f'while maximizing composite quality. Each pick was validated through a '
             f'{N_SIMS:,}-iteration Monte Carlo simulation that accounts for bracket path '
             f'difficulty, matchup variance, and the inherent unpredictability that makes '
             f'March Madness compelling.')

narrative = ' '.join(parts)
words = narrative.split()
# Trim to ~200 words
if len(words) > 210:
    narrative = ' '.join(words[:200]) + '...'
    words = words[:200]
print(narrative)
print(f'\n(Word count: {len(words)})')"""))

# ==============================================================
# Assemble notebook
# ==============================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("final_four_analysis.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Created final_four_analysis.ipynb")
