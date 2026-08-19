"""Turn Fact Book page text into structured records.

Each club gets a fixed six-page block (pdf 44-237) and the book repeats
every game in *both* clubs' blocks. That redundancy is the parser's best
friend: we read each club's schedule independently, then reconcile the
two views of every game. A disagreement means a parse error, not a data
error, so the reconciliation doubles as the test suite.
"""
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from .teams import TEAMS, resolve, UnknownTeam

# --- Page anchors --------------------------------------------------------
# The four statistics matrices, and the pages the season summary lives on.
STATS_PAGES = {
    (2025, "offense"): [257, 259],
    (2025, "defense"): [258, 260],
}
STANDINGS_PAGE = 245
GAMELOG_PAGES = [243, 244]
QB_RECORDS_PAGE = 313

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


@dataclass
class GameView:
    """One club's view of one game, before reconciliation."""
    season: int
    game_type: str          # 'REG' or 'POST'
    game_date: date | None  # None for a flex-scheduled TBD game
    team: str
    opponent: str
    is_home: bool | None    # None => neutral site
    team_score: int | None
    opp_score: int | None
    overtime: bool = False
    site: str | None = None
    note: str | None = None
    order: int = 0          # position in the club's printed list


# Known contradictions in the printed book, with the page that settles
# them. The club blocks are typeset per-club and occasionally disagree;
# where another section of the book arbitrates, the correction goes here
# so the fix is visible and auditable rather than hidden in a parser.
ERRATA: dict[tuple, dict] = {
    # New England's club page prints "W 30-6" for the wild card game, but
    # the official playoff results (pdf 520) and the box score (pdf 521)
    # both give 16-3, matching the Chargers' own page.
    (2025, "POST", "NE", "LAC"): {
        "home_score": 16, "away_score": 3,
        "why": "pdf 520/521 playoff results override NE club page typo",
    },
}


@dataclass
class ParseReport:
    """Anything the parsers could not make sense of, for the build log."""
    warnings: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _parse_date(token: str, season: int) -> date | None:
    """'Sept. 7' -> date(2025, 9, 7), rolling January into the next year."""
    m = re.match(r"([A-Za-z]+)\.?\s+(\d{1,2})", token.strip())
    if not m:
        return None
    month = _MONTHS.get(m.group(1).lower().rstrip("."))
    if not month:
        return None
    year = season + 1 if month <= 3 else season
    return date(year, month, int(m.group(2)))


def week_of(game_date: date | None, season: int, anchors: dict[int, str]) -> int | None:
    """Week number from a bare date.

    NFL weeks run Wednesday-to-Tuesday, so counting seven-day blocks from
    the season's anchor Wednesday recovers the week the book omits.
    """
    if game_date is None:
        return None
    anchor = date.fromisoformat(anchors[season])
    return ((game_date - anchor).days // 7) + 1


# --- Club blocks ---------------------------------------------------------

BLOCK_LENGTH = 6
# The club section runs pdf 44-237. The league-wide schedule grid at the
# front of the book (pdf 5-10) also contains the words "2026 SCHEDULE",
# so the search has to be bounded or it picks up six phantom blocks.
CLUB_SECTION = range(40, 240)


def find_club_blocks(pages: dict[int, str], report: ParseReport) -> dict[str, range]:
    """Locate each club's six-page block, keyed by canonical team id."""
    starts = sorted(n for n, t in pages.items()
                    if "2026 SCHEDULE" in t and n in CLUB_SECTION)
    blocks: dict[str, range] = {}

    full_names = {info["full_name"].upper(): tid for tid, info in TEAMS.items()}
    for start in starts:
        span = range(start, start + BLOCK_LENGTH)
        # The club's name is a running head at the top of its pages. Names
        # of *other* clubs appear mid-page in records and opponent lists,
        # so only the first two lines of each page are trustworthy.
        heads: list[str] = []
        for n in span:
            top = "\n".join(pages.get(n, "").strip().splitlines()[:2]).upper()
            heads += [tid for name, tid in full_names.items() if name in top]
        if not heads:
            report.warn(f"club block at page {start}: no club name found")
            continue
        team_id = max(set(heads), key=heads.count)
        if team_id in blocks:
            report.warn(f"club block at page {start}: duplicate club {team_id}")
            continue
        blocks[team_id] = span

    missing = set(TEAMS) - set(blocks)
    if missing:
        report.warn(f"no club block found for: {sorted(missing)}")
    return blocks


# --- Club facts (stadium, surface, coach) --------------------------------

_STADIUM = re.compile(r"^Stadium:\s*(.+?)\s*$", re.MULTILINE)
_SURFACE = re.compile(r"^Playing Surface:\s*(.+?)\s*$", re.MULTILINE)
_CAPACITY = re.compile(r"^Capacity:\s*([\d,]+)", re.MULTILINE)
_COACH = re.compile(r"^(.+?),\s*Head Coach\s*$", re.MULTILINE)


def parse_club_facts(pages: dict[int, str], blocks: dict[str, range]) -> list[dict]:
    """Venue and head coach for each club, from pages 1 and 3 of its block."""
    rows = []
    for team_id, span in blocks.items():
        blob = "\n".join(pages.get(n, "") for n in span)
        stadium = _STADIUM.search(blob)
        surface = _SURFACE.search(blob)
        capacity = _CAPACITY.search(blob)
        coach = _COACH.search(blob)
        rows.append({
            **{k: TEAMS[team_id][k] for k in
               ("team_id", "location", "nickname", "full_name",
                "conference", "division")},
            "stadium": stadium.group(1).strip() if stadium else None,
            "surface": surface.group(1).strip() if surface else None,
            "capacity": int(capacity.group(1).replace(",", "")) if capacity else None,
            "head_coach": coach.group(1).strip() if coach else None,
        })
    return rows


# --- 2025 results, from each club's "2025 TEAM RECORD" page --------------

# "Sept. 7 at Buffalo L 40-41"      / "Jan. 17 BUFFALO  W 33-30 (OT)"
# "Feb. 8 SEATTLE (SB LX) L 13-29"  - the round marker sits after the club
_RESULT_ROW = re.compile(
    r"^\s*([A-Za-z]+\.?\s+\d{1,2})\s+"      # date
    r"(at\s+|vs\.?\s+)?"                     # site marker
    r"([A-Za-z][A-Za-z.\s'\-&]*?)\s*"        # opponent
    r"(?:\(([^)]*)\)\s*)?"                   # optional round marker, e.g. (SB LX)
    r"\s([WLT])\s+"                          # result
    r"(\d{1,3})\s*-\s*(\d{1,3})"             # team score - opponent score
    r"(\s*\(OT\))?\s*$",
    re.MULTILINE)


def parse_2025_results(pages: dict[int, str], blocks: dict[str, range],
                       report: ParseReport) -> list[GameView]:
    """Every 2025 game, read once from each participating club's block."""
    views: list[GameView] = []
    for team_id, span in blocks.items():
        blob = "\n".join(pages.get(n, "") for n in span)
        section = re.search(r"2025 TEAM RECORD(.*?)(?:SCORE BY PERIODS|$)",
                            blob, re.S)
        if not section:
            report.warn(f"{team_id}: no 2025 TEAM RECORD section")
            continue
        text = section.group(1)

        # Split regular season from postseason so each game is typed.
        post_at = text.find("POSTSEASON")
        parts = [("REG", text[:post_at if post_at > 0 else len(text)])]
        if post_at > 0:
            parts.append(("POST", text[post_at:]))

        found = 0
        for game_type, chunk in parts:
            order = 0
            for dt, marker, opp, note, _res, us, them, ot in _RESULT_ROW.findall(chunk):
                try:
                    opponent = resolve(opp)
                except UnknownTeam:
                    continue  # header rows like "Date Opponent Result"
                marker = (marker or "").strip().lower()
                views.append(GameView(
                    season=2025, game_type=game_type,
                    game_date=_parse_date(dt, 2025),
                    team=team_id, opponent=opponent,
                    is_home=None if marker.startswith("vs") else (marker != "at"),
                    team_score=int(us), opp_score=int(them),
                    overtime=bool(ot.strip()),
                    note=note.strip() or None,
                    order=order,
                ))
                order += 1
                found += 1
        if found < 17:
            report.warn(f"{team_id}: only {found} games parsed from 2025 record")
    return views


# --- 2025 neutral-site flags, from the game-log pages --------------------

_LOG_ROW = re.compile(r"^([WLT])\s+(\d{1,3})\s*-\s*(\d{1,3})\s+"
                      r"(at\s+|vs\.?\s+)?(.+?)\+?\s*$", re.MULTILINE)
_LOG_HEADER = re.compile(r"^([A-Z][A-Z.\s'\-49]+?)\s*\(\d{1,2}-\d{1,2}(?:-\d)?\)\s*$",
                         re.MULTILINE)


def parse_neutral_2025(pages: dict[int, str], report: ParseReport) -> set[frozenset[str]]:
    """Matchups the game logs mark as neutral-site ("vs." plus a '+').

    The club blocks print neutral games in the same style as home games,
    so this is the only place the distinction survives.
    """
    neutral: set[frozenset[str]] = set()
    for page in GAMELOG_PAGES:
        text = pages.get(page, "")
        marks = [(m.start(), m.group(1)) for m in _LOG_HEADER.finditer(text)]
        for i, (pos, club) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
            try:
                team_id = resolve(club)
            except UnknownTeam:
                report.warn(f"game log p.{page}: unknown club header {club!r}")
                continue
            for _r, _a, _b, marker, opp in _LOG_ROW.findall(text[pos:end]):
                if not (marker or "").strip().lower().startswith("vs"):
                    continue
                opponent = resolve(opp, strict=False)
                if opponent:
                    neutral.add(frozenset({team_id, opponent}))
    return neutral


# --- 2026 schedule, from each club's block page 1 ------------------------

# "Sep. 13 at Houston .......1:00 PM"  /  "Sep. 27 vs Dallas (Rio de Janeiro) .4:25 PM"
_SCHED_ROW = re.compile(
    r"^\s*(?:([A-Za-z]+\.?\s+\d{1,2})|(TBD))\s+"
    r"(at\s+|vs\.?\s+)?"
    r"([A-Za-z][A-Za-z.\s'\-&]*?)"
    r"(?:\s*\(([^)]+)\))?"                   # neutral-site venue
    # Dot leader before the kickoff time. Usually a long run, but the book
    # prints a single dot after a venue bracket, so accept any number.
    r"\s*\.*\s*(?:\d{1,2}:\d{2}\s*[AP]M|TBD)\s*$",
    re.MULTILINE)


def parse_2026_schedule(pages: dict[int, str], blocks: dict[str, range],
                        report: ParseReport) -> list[GameView]:
    """The unplayed 2026 slate, read once from each club's block."""
    views: list[GameView] = []
    for team_id, span in blocks.items():
        text = pages.get(span.start, "")
        section = re.search(r"REGULAR SEASON(.*?)(?:\*\s*All times|Stadium:|$)",
                            text, re.S)
        if not section:
            report.warn(f"{team_id}: no 2026 REGULAR SEASON section")
            continue

        found = 0
        order = 0
        for dt, tbd, marker, opp, venue in _SCHED_ROW.findall(section.group(1)):
            if tbd and not opp.strip():
                continue
            try:
                opponent = resolve(opp)
            except UnknownTeam:
                continue  # the "BYE ......" filler row
            marker = (marker or "").strip().lower()
            venue = venue.strip()
            # An international game carries its venue in brackets. Only one
            # of the two clubs is reliably given the "vs" marker (Dallas
            # prints "Baltimore (Rio de Janeiro)" with no marker at all),
            # so the venue itself is what makes a game neutral.
            is_neutral = bool(venue) or marker.startswith("vs")
            views.append(GameView(
                season=2026, game_type="REG",
                game_date=_parse_date(dt, 2026) if dt else None,
                team=team_id, opponent=opponent,
                is_home=None if is_neutral else (marker != "at"),
                team_score=None, opp_score=None,
                site=venue or None,
                order=order,
            ))
            order += 1
            found += 1
        if found < 16:
            report.warn(f"{team_id}: only {found} games parsed from 2026 schedule")
    return views


# --- Reconciliation ------------------------------------------------------

def _dates_are_ordered(views: list[GameView]) -> bool:
    """Does this club's printed list run in chronological order?

    The book lists each club's games in order, so a club whose dates are
    non-decreasing is self-consistent. When two clubs disagree about a
    date, the self-consistent one is the one to believe.
    """
    dates = [v.game_date for v in sorted(views, key=lambda v: v.order) if v.game_date]
    return all(a <= b for a, b in zip(dates, dates[1:]))


def reconcile(views: list[GameView], report: ParseReport,
              neutral: set[frozenset[str]] | None = None) -> list[dict]:
    """Merge each club's view of a game into one row, validating as we go.

    Every game is printed in two clubs' blocks, so agreement between the
    two is the correctness check. Matching is done on *printed order*
    rather than on date: divisional rivals meet twice, and where the book
    mis-dates one of those meetings, order still lines the pair up.
    """
    neutral = neutral or set()

    # Which clubs list their own games in a sane order, per season/type.
    by_club: dict[tuple, list[GameView]] = {}
    for v in views:
        by_club.setdefault((v.season, v.game_type, v.team), []).append(v)
    ordered_clubs = {k for k, vs in by_club.items() if _dates_are_ordered(vs)}

    groups: dict[tuple, list[GameView]] = {}
    for v in views:
        groups.setdefault(
            (v.season, v.game_type, frozenset({v.team, v.opponent})), []).append(v)

    games: list[dict] = []
    for (season, game_type, pair), members in groups.items():
        a, b = sorted(pair)
        va = sorted([v for v in members if v.team == a], key=lambda v: v.order)
        vb = sorted([v for v in members if v.team == b], key=lambda v: v.order)

        if len(va) != len(vb):
            report.warn(f"{season} {game_type} {a}/{b}: {a} lists {len(va)} meeting(s), "
                        f"{b} lists {len(vb)} - cannot pair reliably")

        # Nth meeting on one club's page is the Nth on the other's.
        for i in range(max(len(va), len(vb))):
            games.append(_merge_pair(
                season, game_type, pair,
                va[i] if i < len(va) else None,
                vb[i] if i < len(vb) else None,
                neutral, ordered_clubs, report))

    games.sort(key=lambda g: (g["season"], g["game_date"] or date.max,
                              g["home_team"]))
    return games


def _merge_pair(season: int, game_type: str, pair: frozenset[str],
                va: GameView | None, vb: GameView | None,
                neutral: set[frozenset[str]], ordered_clubs: set[tuple],
                report: ParseReport) -> dict:
    """Fold two clubs' views of one game into a single row."""
    members = [v for v in (va, vb) if v]

    # --- Date: prefer a club that lists its own season in order --------
    dated = [v for v in members if v.game_date]
    game_date = None
    if dated:
        trusted = [v for v in dated
                   if (v.season, v.game_type, v.team) in ordered_clubs]
        game_date = (trusted or dated)[0].game_date
        disagreeing = {v.game_date for v in dated}
        if len(disagreeing) > 1:
            report.warn(
                f"{season} {game_type} {'/'.join(sorted(pair))}: clubs print "
                f"different dates {sorted(d.isoformat() for d in disagreeing)} "
                f"- using {game_date} from the club whose list is in order")

    # --- Home / away ---------------------------------------------------
    is_neutral = pair in neutral or all(v.is_home is None for v in members)
    home_claims = [v.team for v in members if v.is_home is True]

    if is_neutral or len(home_claims) != 1:
        if not is_neutral:
            report.warn(f"{season} {game_date} {'/'.join(sorted(pair))}: "
                        f"{len(home_claims)} clubs claim home - treating as neutral")
        is_neutral = True
        home, away = sorted(pair)
    else:
        home = home_claims[0]
        away = next(t for t in pair if t != home)

    home_view = next((v for v in members if v.team == home), None)
    away_view = next((v for v in members if v.team == away), None)

    home_score = away_score = None
    if home_view and home_view.team_score is not None:
        home_score, away_score = home_view.team_score, home_view.opp_score
    elif away_view and away_view.team_score is not None:
        home_score, away_score = away_view.opp_score, away_view.team_score

    # --- Cross-check: both clubs must report the same scoreline --------
    if home_view and away_view and home_view.team_score is not None \
            and away_view.team_score is not None:
        if (home_view.team_score, home_view.opp_score) != \
                (away_view.opp_score, away_view.team_score):
            fix = ERRATA.get((season, game_type, home, away))
            if fix:
                home_score, away_score = fix["home_score"], fix["away_score"]
                report.warn(f"{season} {game_date} {home} v {away}: clubs disagree "
                            f"({home_view.team_score}-{home_view.opp_score} vs "
                            f"{away_view.opp_score}-{away_view.team_score}); "
                            f"applied errata {home_score}-{away_score} "
                            f"[{fix['why']}]")
            else:
                report.warn(f"{season} {game_date} {home} v {away}: score mismatch "
                            f"{home_view.team_score}-{home_view.opp_score} vs "
                            f"{away_view.opp_score}-{away_view.team_score} "
                            f"- no errata entry, using home club's page")

    if len(members) < 2:
        report.warn(f"{season} {game_date} {'/'.join(sorted(pair))}: "
                    f"only one club lists this game")

    return {
        "season": season, "game_type": game_type, "game_date": game_date,
        "home_team": home, "away_team": away,
        "home_score": home_score, "away_score": away_score,
        "neutral": int(is_neutral),
        "overtime": int(any(v.overtime for v in members)),
        "site": next((v.site for v in members if v.site), None),
        "note": next((v.note for v in members if v.note), None),
        "played": int(home_score is not None),
    }


# --- 2025 team statistics matrices ---------------------------------------

_NUMERIC = re.compile(r"^[-+]?[\d][\d.,:/%-]*$")


def _to_float(token: str) -> float | None:
    """Numeric value of a matrix cell; '29:11' becomes minutes as a float."""
    token = token.replace(",", "").rstrip("%")
    if ":" in token:
        mm, _, ss = token.partition(":")
        try:
            return int(mm) + int(ss) / 60
        except ValueError:
            return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_team_stats(pages: dict[int, str], report: ParseReport) -> list[dict]:
    """The 16-column offence/defence matrices, as long-format rows.

    Sub-metrics are indented under a heading and reuse labels across
    sections ("Net Yds. Gained" appears under both Rushes and Passes), so
    each metric is qualified by the heading above it.
    """
    rows: list[dict] = []
    for (season, side), page_numbers in STATS_PAGES.items():
        for page in page_numbers:
            text = pages.get(page, "")
            columns: list[str] | None = None
            group = ""

            for line in text.splitlines():
                if not line.strip():
                    continue
                tokens = line.split()

                # The header line is 16 consecutive club abbreviations.
                if columns is None:
                    ids = [resolve(t, strict=False) for t in tokens]
                    if len(ids) == 16 and all(ids):
                        columns = ids
                    continue

                # Peel numeric cells off the end; what remains is the label.
                values: list[str] = []
                i = len(tokens)
                while i > 0 and len(values) < 16 and _NUMERIC.match(tokens[i - 1]):
                    values.append(tokens[i - 1])
                    i -= 1
                values.reverse()
                label = " ".join(tokens[:i]).strip()

                if len(values) != 16 or not re.search(r"[A-Za-z]", label):
                    continue

                indented = line[:1].isspace()
                if indented and group:
                    metric = f"{group}: {label}"
                else:
                    group, metric = label, label

                for team_id, raw in zip(columns, values):
                    rows.append({
                        "season": season, "team_id": team_id, "side": side,
                        "metric": metric, "value": _to_float(raw), "raw": raw,
                    })

            if columns is None:
                report.warn(f"stats p.{page}: could not find the 16-club header row")
    return rows


# --- Final standings -----------------------------------------------------

_STANDING_ROW = re.compile(
    r"^\s*([*#]?)\s*([A-Z][A-Za-z.\- ]+?)\s+(\d{1,2})\s+(\d{1,2})\s+(\d)\s+"
    r"0?\.\d+\s+(\d{2,4})\s+(\d{2,4})\s*$", re.MULTILINE)


def parse_standings(pages: dict[int, str], report: ParseReport) -> list[dict]:
    """2025 final standings: record, points for, points against, seeding."""
    rows = []
    for marker, name, w, l, t, pf, pa in _STANDING_ROW.findall(pages.get(STANDINGS_PAGE, "")):
        team_id = resolve(name, strict=False)
        if not team_id:
            continue
        rows.append({
            "season": 2025, "team_id": team_id,
            "wins": int(w), "losses": int(l), "ties": int(t),
            "points_for": int(pf), "points_against": int(pa),
            "division_champ": int(marker == "*"), "wild_card": int(marker == "#"),
        })
    if len(rows) != 32:
        report.warn(f"standings: parsed {len(rows)} clubs, expected 32")
    return rows


# --- Active quarterback starting records ---------------------------------

_QB_ROW = re.compile(r"^([A-Z][A-Za-z.'\-\s]+?)\s+(\d{1,3})\s+(\d{1,3})\s+(\d)\s+"
                     r"(\.\d{3})\s*$", re.MULTILINE)


def parse_qb_records(pages: dict[int, str], report: ParseReport) -> list[dict]:
    """Career starting records of active QBs (minimum 10 starts)."""
    rows = [
        {"player": name.strip(), "wins": int(w), "losses": int(l),
         "ties": int(t), "win_pct": float(pct)}
        for name, w, l, t, pct in _QB_ROW.findall(pages.get(QB_RECORDS_PAGE, ""))
    ]
    if not rows:
        report.warn("qb records: no rows parsed")
    return rows
