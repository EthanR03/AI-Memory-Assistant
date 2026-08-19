"""Map the structure of the extracted Fact Book text.

Answers three questions before we commit to a schema:
  1. What sections exist and where do they start? (from the book's own index)
  2. Which pages carry the per-team / per-player data a predictor needs?
  3. How dense is each page - i.e. which ranges are tables vs. prose?
"""
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

TXT = Path("data/extracted/factbook.pypdf.txt")
raw = TXT.read_text(encoding="utf-8")
parts = re.split(r"=== PAGE (\d+) ===", raw)
pages = {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}

print(f"pages: {len(pages)}   chars: {len(raw):,}\n")

# --- 1. The index's own "section start" entries --------------------------
index_text = "".join(pages.get(n, "") for n in range(3, 5))
sections = re.findall(r"^(.+?),?\s*section start\s*\.+\s*(\d+)\s*$",
                      index_text, re.MULTILINE)
print("=" * 70)
print("SECTIONS (per the book's index, printed page numbers)")
print("=" * 70)
for name, page in sorted(sections, key=lambda s: int(s[1])):
    print(f"  p.{page:>4}  {name.strip()}")

# --- 2. Offset between printed page numbers and PDF page numbers ---------
# Find a printed page number in a page footer/header to calibrate.
print("\n" + "=" * 70)
print("PRINTED-PAGE -> PDF-PAGE CALIBRATION")
print("=" * 70)
for pdf_n in (50, 250, 500, 700, 850):
    head = pages.get(pdf_n, "")[:120].replace("\n", " | ")
    print(f"  pdf {pdf_n:>3}: {head[:110]}")

# --- 3. Keyword heat map: where does each topic live? --------------------
TOPICS = {
    "schedule/results": r"\b(at|vs\.?)\b.*\d{1,2}:\d{2}|Week \d+",
    "standings": r"\bW\s+L\s+T\s+Pct\b|Standings",
    "team stats": r"Team Statistics|TEAM STATISTICS|Yards Per Game",
    "passing stats": r"\bAtt\s+Comp\s+Yds\b|Passer Rating|Passing Leaders",
    "rushing stats": r"\bRushing\b.*\bAtt\s+Yds\b|Rushing Leaders",
    "defense/sacks": r"\bSacks?\b.*\bYds\b|Interceptions By",
    "coaches": r"Head Coach|Coaching Records",
    "rosters": r"Veteran Roster|2026 Draft Choices|\bPos\.\s+Ht\.\s+Wt\.",
    "injuries/rules": r"Injury Report|Playing Rules|Rule Change",
    "records": r"All-Time Record|Most .* Season|Record Holders",
    "playoffs": r"Playoff|Super Bowl|Championship Game",
    "hall of fame": r"Hall of Fame",
}
print("\n" + "=" * 70)
print("TOPIC HEAT MAP (pdf pages where each pattern appears)")
print("=" * 70)
for topic, pat in TOPICS.items():
    rx = re.compile(pat)
    hits = [n for n, t in pages.items() if rx.search(t)]
    if not hits:
        print(f"  {topic:<18} (none)")
        continue
    # collapse into contiguous ranges for readability
    ranges, start, prev = [], hits[0], hits[0]
    for n in hits[1:]:
        if n - prev > 3:
            ranges.append((start, prev))
            start = n
        prev = n
    ranges.append((start, prev))
    span = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges[:12])
    print(f"  {topic:<18} {len(hits):>4} pages: {span}")

# --- 4. All 32 team names: which pages are "their" pages? ----------------
TEAMS = [
    "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
    "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
    "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
    "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars",
    "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers",
    "Los Angeles Rams", "Miami Dolphins", "Minnesota Vikings",
    "New England Patriots", "New Orleans Saints", "New York Giants",
    "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers",
    "San Francisco 49ers", "Seattle Seahawks", "Tampa Bay Buccaneers",
    "Tennessee Titans", "Washington Commanders",
]
print("\n" + "=" * 70)
print("PER-TEAM PAGE BLOCKS (pages mentioning a team 5+ times)")
print("=" * 70)
for team in TEAMS:
    hits = [n for n, t in pages.items() if t.count(team) >= 5]
    print(f"  {team:<24} {len(hits):>3} pages  {hits[:14]}")

# --- 5. Densest pages (likeliest tables) --------------------------------
print("\n" + "=" * 70)
print("DENSEST PAGES (most digits - i.e. the statistical tables)")
print("=" * 70)
digits = Counter({n: sum(c.isdigit() for c in t) for n, t in pages.items()})
for n, d in digits.most_common(15):
    first = next((ln.strip() for ln in pages[n].splitlines() if ln.strip()), "")
    print(f"  pdf p.{n:<4} {d:>5} digits  | {first[:70]}")
