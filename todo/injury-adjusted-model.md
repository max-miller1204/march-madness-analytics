# Future Enhancement: Injury-Adjusted Efficiency Model

## Problem

KenPom/composite ratings reflect full-season performance. When a key player is injured (e.g., Duke losing Caleb Foster for the 2026 tournament), the model overestimates team strength. Not all injuries are equal -- a star PG has far more impact than a bench player.

## Proposed Approach: Production-Share Discount

For each injured player, estimate their share of team production and discount accordingly:

```python
INJURIES = {
    'Duke': [{'player': 'Caleb Foster', 'ppg': 14.0, 'apg': 5.0}],
}

REPLACEMENT_FACTOR = 0.4  # replacement produces ~40% of starter's output

for team, injuries in INJURIES.items():
    team_ppg = df.loc[df['StdName'] == team, 'ORtg'].values[0]  # proxy
    for inj in injuries:
        player_share = (inj['ppg'] + 0.5 * inj['apg']) / team_ppg
        impact = player_share * (1 - REPLACEMENT_FACTOR)
        discount = df.loc[df['StdName'] == team, 'NetRtg'].values[0] * impact
        df.loc[df['StdName'] == team, 'NetRtg'] -= discount
        df.loc[df['StdName'] == team, 'ORtg'] -= discount * 0.7  # offense hit
```

## Alternative Approaches

### Manual impact factor
Simple but subjective -- assign a flat NetRtg discount per player (e.g., Foster = -4.0).

### With/without game splits
Compare team efficiency in games with vs. without the player. Most accurate but requires game-level data and often has small sample size issues.

## Where to integrate

- Apply adjustments **after** composite scoring (Section 5) but **before** Monte Carlo simulation (Section 6)
- Add a new notebook section between Sections 5 and 6
- Print a table showing original vs. adjusted ratings for injured teams
