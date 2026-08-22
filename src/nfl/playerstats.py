"""Season-level player statistics, for the chat layer.

    python -m src.nfl.playerstats [--force-download] [--from 1999]

`players.py` gave the store identity - born, drafted, college, size.
This gives it production: yards, touchdowns, sacks, catches, EPA. Between
them, "who is X" stops being a win-loss record and "who led the league in
receiving in 2023" stops being a caveat.

SEASON grain, deliberately. nflverse publishes the same data by week,
which is four times the bytes and the wrong shape for chat - and, more
importantly, the season files are UNSAFE for modelling in a way the
weekly ones are not: a season total for year N contains the very game a
Stage-N model is trying to predict. Nothing here is walk-forward. If the
predictor ever wants player inputs it must pull the weekly files itself
and aggregate them forwards; do not join this table into a backtest.

Only `reg` and `post` are loaded. nflverse also ships `regpost`, which is
the two added together - storing all three invites a SUM that counts
every yard twice, the same trap qb_records already sets by being
regular-season only.
"""
import argparse
import csv
import re
from datetime import date, datetime, timezone
from pathlib import Path

from .. import config
from . import db, nflverse

BASE_URL = ("https://github.com/nflverse/nflverse-data/releases/"
            "download/stats_player/stats_player_{kind}_{season}.csv")

FIRST_SEASON = 1999          # matches the games table's span

# 'reg' and 'post' partition the season; 'regpost' would overlap both.
KINDS = ("reg", "post")

# 51 of the feed's 148 columns. Dropped: the explosive-play buckets
# (passing_10/16/20/40), the per-distance field goal breakdowns and their
# _list twins, 2pt conversions, the six fumble sub-types, punt detail,
# and the analyst ratios (pacr, racr, wopr, air_yards_share) - none of
# which anyone asks a chat assistant about.
STAT_COLUMNS = [
    "player_id", "season", "season_type", "recent_team", "position", "games",

    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "sacks_suffered", "passing_first_downs",
    "passing_air_yards", "passing_epa", "passing_cpoe",

    "carries", "rushing_yards", "rushing_tds", "rushing_first_downs",
    "rushing_fumbles_lost", "rushing_epa",

    "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_first_downs", "receiving_air_yards", "receiving_epa",
    "target_share",

    "def_tackles_solo", "def_tackle_assists", "def_tackles_for_loss",
    "def_sacks", "def_qb_hits", "def_interceptions", "def_pass_defended",
    "def_fumbles_forced", "def_tds",

    "fg_made", "fg_att", "fg_pct", "fg_long", "pat_made", "pat_att",

    "punt_return_yards", "kickoff_return_yards", "special_teams_tds",
    "fumbles_lost_total",

    "fantasy_points", "fantasy_points_ppr",
]

# Everything that is a whole count. The rest (EPA, CPOE, shares, pct)
# stays REAL.
_INT_COLUMNS = {
    "season", "games", "completions", "attempts", "passing_yards",
    "passing_tds", "passing_interceptions", "sacks_suffered",
    "passing_first_downs", "carries", "rushing_yards", "rushing_tds",
    "rushing_first_downs", "rushing_fumbles_lost", "receptions", "targets",
    "receiving_yards", "receiving_tds", "receiving_first_downs",
    "def_tackles_solo", "def_tackle_assists", "def_interceptions",
    "def_pass_defended", "def_tds", "fg_made", "fg_att", "fg_long",
    "pat_made", "pat_att", "punt_return_yards", "kickoff_return_yards",
    "special_teams_tds", "fumbles_lost_total",
}


# A real GSIS id is two digits, a dash, seven digits. The feed also
# carries two synthetic buckets that are not players at all: '0', which
# collects unattributed 1999-2000 plays and would otherwise lead a
# "most games" leaderboard with 515 games and no name, and 'XX-0000001'.
# These are the only two ids in the whole release that fail the pattern.
_GSIS_ID = re.compile(r"\d{2}-\d{7}\Z")


def _path(kind: str, season: int) -> Path:
    return config.NFLVERSE_STATS / f"stats_player_{kind}_{season}.csv"


def download(kind: str, season: int, force: bool = False) -> Path | None:
    """Fetch one season file, or return None if the release has no such year.

    A missing year is expected rather than exceptional: the current
    season has no postseason file until January, and asking for next
    season is a normal thing for a caller to do.
    """
    dest = _path(kind, season)
    if dest.exists() and not force:
        return dest
    import requests

    url = BASE_URL.format(kind=kind, season=season)
    response = requests.get(url, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    return dest


def _num(text: str, column: str):
    text = (text or "").strip()
    if not text or text.upper() in {"NA", "NULL"}:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return int(value) if column in _INT_COLUMNS else value


def load_rows(path: Path) -> list[dict]:
    """Read one season file into dicts shaped like the `player_stats` table."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            player_id = (r.get("player_id") or "").strip()
            if not _GSIS_ID.match(player_id):
                continue
            row = {}
            for column in STAT_COLUMNS:
                raw = r.get(column, "")
                if column in ("player_id", "season_type", "recent_team",
                              "position"):
                    value = (raw or "").strip() or None
                    if column == "recent_team" and value:
                        value = nflverse.team_id(value)
                else:
                    value = _num(raw, column)
                row[column] = value
            rows.append(row)
    return rows


def run(force_download: bool = False, first: int = FIRST_SEASON) -> None:
    last = date.today().year
    rows: list[dict] = []
    missing: list[str] = []

    print(f"Fetching stats_player {first}-{last} ({', '.join(KINDS)})...")
    for season in range(first, last + 1):
        for kind in KINDS:
            path = download(kind, season, force=force_download)
            if path is None:
                missing.append(f"{kind} {season}")
                continue
            rows.extend(load_rows(path))
    print(f"  {len(rows):,} player-seasons loaded")
    if missing:
        print(f"  not published yet: {', '.join(missing)}")

    conn = db.connect(config.NFL_DB)
    with conn:
        conn.execute("DELETE FROM player_stats")
        db.replace_all(conn, "player_stats", rows, STAT_COLUMNS)
        conn.execute(
            "INSERT INTO ingest_log (table_name, rows, source, as_of, "
            "built_at) VALUES (?, ?, ?, ?, ?)",
            ("player_stats", len(rows), "nflverse",
             datetime.now(timezone.utc).date().isoformat(),
             datetime.now(timezone.utc).isoformat(timespec="seconds")))

    print("\n" + "=" * 72)
    print("PLAYER_STATS TABLE")
    print("=" * 72)
    summary = conn.execute("""
        SELECT COUNT(*) n, COUNT(DISTINCT player_id) players,
               MIN(season) lo, MAX(season) hi
          FROM player_stats
    """).fetchone()
    print(f"  player-seasons : {summary['n']:,}")
    print(f"  distinct players: {summary['players']:,}")
    print(f"  seasons        : {summary['lo']}-{summary['hi']}")
    for r in conn.execute("SELECT season_type, COUNT(*) n FROM player_stats "
                          "GROUP BY season_type ORDER BY 2 DESC"):
        print(f"    {r['season_type']:8} {r['n']:>7,}")

    # The name comes from `players`, so an unresolved id is a player who
    # would silently vanish from any leaderboard. Report it rather than
    # let it pass.
    orphans = conn.execute("""
        SELECT COUNT(DISTINCT s.player_id) n FROM player_stats s
         WHERE NOT EXISTS (SELECT 1 FROM players p
                            WHERE p.gsis_id = s.player_id)
    """).fetchone()["n"]
    known = conn.execute("SELECT COUNT(*) n FROM players").fetchone()["n"]
    if not known:
        print("\n  ! `players` is empty - run `python -m src.nfl.players` "
              "first, or every one of these rows is anonymous.")
    else:
        print(f"\n  ids resolving to a bio in `players`: "
              f"{summary['players'] - orphans:,}/{summary['players']:,}")
        if orphans:
            print(f"  ! {orphans} id(s) have stats but no bio; they carry a "
                  f"NULL name in any join.")

    conn.close()
    print(f"\nStore updated: {config.NFL_DB}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-download", action="store_true",
                    help="re-fetch the season files even if already cached")
    ap.add_argument("--from", dest="first", type=int, default=FIRST_SEASON,
                    help=f"earliest season to load (default {FIRST_SEASON})")
    run(**vars(ap.parse_args()))
