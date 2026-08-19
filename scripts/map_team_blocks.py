"""Locate each club's page block and label what each page inside it holds.

The Fact Book gives every club a fixed-length block. Finding the block
boundaries lets us ingest team data as *structured records* instead of
blind 800-character chunks - which is what a game predictor needs.
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

parts = re.split(r"=== PAGE (\d+) ===",
                 Path("data/extracted/factbook.pypdf.txt").read_text(encoding="utf-8"))
pages = {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}

# A club block opens with the address/telephone header on its first page.
opener = re.compile(r"Telephone:\s*\(\d{3}\)")
# The club name appears as a running head on later pages of the block.
CLUBS = ["BUFFALO BILLS", "MIAMI DOLPHINS", "NEW ENGLAND PATRIOTS", "NEW YORK JETS",
         "BALTIMORE RAVENS", "CINCINNATI BENGALS", "CLEVELAND BROWNS", "PITTSBURGH STEELERS",
         "HOUSTON TEXANS", "INDIANAPOLIS COLTS", "JACKSONVILLE JAGUARS", "TENNESSEE TITANS",
         "DENVER BRONCOS", "KANSAS CITY CHIEFS", "LAS VEGAS RAIDERS", "LOS ANGELES CHARGERS",
         "DALLAS COWBOYS", "NEW YORK GIANTS", "PHILADELPHIA EAGLES", "WASHINGTON COMMANDERS",
         "CHICAGO BEARS", "DETROIT LIONS", "GREEN BAY PACKERS", "MINNESOTA VIKINGS",
         "ATLANTA FALCONS", "CAROLINA PANTHERS", "NEW ORLEANS SAINTS", "TAMPA BAY BUCCANEERS",
         "ARIZONA CARDINALS", "LOS ANGELES RAMS", "SAN FRANCISCO 49ERS", "SEATTLE SEAHAWKS"]

starts = sorted(n for n, t in pages.items() if opener.search(t))
print(f"club-block opener pages ({len(starts)}): {starts}\n")

# Which club owns each block? Take the running head from the block's 3rd page.
print("=" * 78)
print(f"{'CLUB':<24} {'BLOCK':<12} PAGE ROLES")
print("=" * 78)
ROLE_TESTS = [
    ("2026-schedule", r"2026 SCHEDULE"),
    ("2025-results+stats", r"2025 TEAM RECORD|2025 TEAM STATISTICS"),
    ("coaches", r"Head Coach\b.*Pro Career|ASSISTANT COACHES"),
    ("roster", r"VETERAN ROSTER|Pos\.\s*Ht\.|DRAFT CHOICES"),
    ("history/records", r"ALL-TIME|Retired Numbers|First-Round"),
]
for i, s in enumerate(starts):
    end = starts[i + 1] - 1 if i + 1 < len(starts) else s + 5
    # The club's own name is the running head at the very top of the block's
    # inner pages. Cross-references to other clubs appear mid-page, so anchor
    # the match to the first few lines only.
    heads: list[str] = []
    for n in range(s, end + 1):
        top = "\n".join(pages.get(n, "").strip().splitlines()[:2]).upper()
        heads += [c for c in CLUBS if c in top]
    club = max(set(heads), key=heads.count) if heads else "?"
    roles = []
    for n in range(s, end + 1):
        hits = [name for name, pat in ROLE_TESTS
                if re.search(pat, pages.get(n, ""), re.IGNORECASE | re.DOTALL)]
        roles.append(f"{n}:{'/'.join(hits) or 'other'}")
    print(f"{club:<24} {s}-{end:<8} {'  '.join(roles)}")
