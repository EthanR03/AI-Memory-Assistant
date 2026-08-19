"""Proof-of-concept: turn Fact Book text into model-ready numbers.

Parses the 2025 final standings (pdf p.245) and the per-team game log
(pdf p.243-244), then derives two baseline power ratings:

  Pythagorean win% - PF^2.37 / (PF^2.37 + PA^2.37), the standard NFL exponent.
  SoS-naive margin - average scoring margin per game.

The point is not that these are good predictors. It is that the book's
numbers survive extraction intact and land in a dataframe-shaped record,
which is the precondition for any predictor at all.
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

parts = re.split(r"=== PAGE (\d+) ===",
                 Path("data/extracted/factbook.pypdf.txt").read_text(encoding="utf-8"))
pages = {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}

# --- Final 2025 standings -------------------------------------------------
# Rows look like:  "* New England 14 3 0 .824 490 320"
ROW = re.compile(
    r"^[\s*#]*([A-Z][A-Za-z.\- ]+?)\s+(\d{1,2})\s+(\d{1,2})\s+(\d)\s+0?\.\d+\s+(\d{2,4})\s+(\d{2,4})\s*$",
    re.MULTILINE)

teams = {}
for name, w, l, t, pf, pa in ROW.findall(pages[245]):
    teams[name.strip()] = dict(w=int(w), l=int(l), t=int(t), pf=int(pf), pa=int(pa))

print(f"parsed {len(teams)} teams from final standings\n")

# --- Game logs: every 2025 game with its score ---------------------------
# Rows look like:  "W 41 - 40 Baltimore"  /  "L 20 - 23 at New England"
GAME = re.compile(r"^([WLT])\s+(\d{1,2})\s*-\s*(\d{1,2})\s+(at\s+|vs\.\s+)?(.+?)\+?\s*$",
                  re.MULTILINE)
HEADER = re.compile(r"^([A-Z][A-Z.\s'\-49]+?)\s*\((\d{1,2})-(\d{1,2})(?:-\d)?\)\s*$",
                    re.MULTILINE)

logs: dict[str, list[dict]] = {}
for pg in (243, 244):
    text = pages[pg]
    marks = [(m.start(), m.group(1).title()) for m in HEADER.finditer(text)]
    for i, (pos, club) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        block = text[pos:end]
        logs[club] = [
            dict(result=r, pf=int(a), pa=int(b), home=(loc or "").strip() == "", opp=opp.strip())
            for r, a, b, loc, opp in GAME.findall(block)
        ]

total_games = sum(len(v) for v in logs.values())
print(f"parsed {len(logs)} game logs, {total_games} team-games "
      f"({total_games // 2} unique games incl. postseason)\n")

# --- Home/road split, straight from the logs -----------------------------
home_w = sum(1 for g in (x for v in logs.values() for x in v) if g["home"] and g["result"] == "W")
home_n = sum(1 for g in (x for v in logs.values() for x in v) if g["home"])
print(f"home teams went {home_w}-{home_n - home_w} in 2025 "
      f"({home_w / home_n:.1%}) across {home_n} logged home games")

margins = [g["pf"] - g["pa"] for v in logs.values() for g in v if g["home"]]
print(f"mean home scoring margin: {sum(margins) / len(margins):+.2f} points\n")

# --- Baseline power ratings ----------------------------------------------
EXP = 2.37
rows = []
for name, s in teams.items():
    games = s["w"] + s["l"] + s["t"]
    pyth = s["pf"] ** EXP / (s["pf"] ** EXP + s["pa"] ** EXP)
    rows.append((name, s["w"], s["l"], s["pf"], s["pa"],
                 (s["pf"] - s["pa"]) / games, pyth, pyth * games - (s["w"] + 0.5 * s["t"])))

rows.sort(key=lambda r: -r[6])
print("=" * 74)
print(f"{'TEAM':<16}{'W':>3}{'L':>4}{'PF':>6}{'PA':>6}{'MARGIN/G':>10}{'PYTH%':>8}{'LUCK':>8}")
print("=" * 74)
for name, w, l, pf, pa, mg, pyth, luck in rows:
    print(f"{name:<16}{w:>3}{l:>4}{pf:>6}{pa:>6}{mg:>+10.1f}{pyth:>8.3f}{luck:>+8.1f}")
print("=" * 74)
print("LUCK = Pythagorean wins minus actual wins. Negative = won more than")
print("       its point differential justifies, i.e. a regression candidate.")
