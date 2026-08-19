"""Survey nflverse games.csv before wiring it into the feature store.

Three questions decide how the loader has to be written:
  1. Which seasons actually carry betting lines? (the file starts in 1999,
     the odds do not)
  2. What team abbreviations appear, including relocated franchises?
  3. Does nflverse agree with the Fact Book about 2025?
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CSV = Path("data/nflverse/games.csv")
rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
print(f"{len(rows):,} games, {len(rows[0])} columns\n")

seasons = sorted({int(r["season"]) for r in rows})
print(f"seasons: {seasons[0]}-{seasons[-1]}")
print("game_type values:", dict(Counter(r["game_type"] for r in rows)))

# --- 1. Coverage of the betting columns by season ------------------------
print("\n" + "=" * 78)
print("BETTING-DATA COVERAGE BY SEASON (% of games with a value)")
print("=" * 78)
fields = ["spread_line", "total_line", "home_moneyline", "away_moneyline",
          "home_spread_odds", "over_odds", "home_rest", "temp", "wind"]
by_season = defaultdict(list)
for r in rows:
    by_season[int(r["season"])].append(r)

print(f"{'SEASON':<8}{'GAMES':>6}" + "".join(f"{f[:12]:>14}" for f in fields))
for s in seasons:
    rs = by_season[s]
    pcts = [sum(1 for r in rs if r[f].strip()) / len(rs) for f in fields]
    if s < 2006 and s % 2:
        continue  # thin the pre-odds years for readability
    print(f"{s:<8}{len(rs):>6}" + "".join(f"{p:>13.0%} " for p in pcts))

# --- 2. Team abbreviations ----------------------------------------------
print("\n" + "=" * 78)
print("TEAM ABBREVIATIONS (with the seasons each appears in)")
print("=" * 78)
team_seasons = defaultdict(set)
for r in rows:
    for key in ("home_team", "away_team"):
        team_seasons[r[key]].add(int(r["season"]))
for team in sorted(team_seasons):
    yrs = sorted(team_seasons[team])
    span = f"{yrs[0]}-{yrs[-1]}" if len(yrs) > 1 else str(yrs[0])
    flag = "  <- relocated/renamed" if yrs[-1] < seasons[-1] else ""
    print(f"  {team:<5} {span}{flag}")

# --- 3. The seasons we care about ---------------------------------------
print("\n" + "=" * 78)
print("RECENT SEASONS IN DETAIL")
print("=" * 78)
for s in (2024, 2025, 2026):
    rs = by_season.get(s, [])
    if not rs:
        print(f"{s}: absent from the file")
        continue
    played = sum(1 for r in rs if r["home_score"].strip())
    lined = sum(1 for r in rs if r["spread_line"].strip())
    ml = sum(1 for r in rs if r["home_moneyline"].strip())
    types = dict(Counter(r["game_type"] for r in rs))
    print(f"{s}: {len(rs)} games | played {played} | spread {lined} | "
          f"moneyline {ml} | {types}")

# --- 4. Neutral-site and roof/surface vocabulary -------------------------
recent = [r for r in rows if int(r["season"]) >= 2020]
print("\nlocation:", dict(Counter(r["location"] for r in recent)))
print("roof    :", dict(Counter(r["roof"] for r in recent)))
print("surface :", dict(Counter(r["surface"] for r in recent)))

# --- 5. Sign-convention spot check --------------------------------------
# spread_line should correlate positively with (home_score - away_score).
print("\n" + "=" * 78)
print("SIGN CHECK on spread_line (nflverse: positive = home favoured)")
print("=" * 78)
checkable = [r for r in rows
             if r["spread_line"].strip() and r["result"].strip()
             and int(r["season"]) >= 2015]
hits = sum(1 for r in checkable
           if (float(r["spread_line"]) > 0) == (float(r["result"]) > 0)
           and float(r["result"]) != 0)
decided = sum(1 for r in checkable if float(r["result"]) != 0)
print(f"favourite (per positive spread_line) won {hits}/{decided} = "
      f"{hits / decided:.1%} of decided games since 2015")
print("A number near 66% confirms positive = home favoured; near 34% would")
print("mean the sign is inverted from what the docs say.")

sample = [r for r in checkable if int(r["season"]) == 2025][:5]
print("\nsample 2025 rows:")
for r in sample:
    print(f"  {r['away_team']:>3} @ {r['home_team']:<3} "
          f"spread_line={r['spread_line']:>6} total={r['total_line']:>5} "
          f"final {r['away_score']}-{r['home_score']} "
          f"ml {r['away_moneyline']}/{r['home_moneyline']}")
