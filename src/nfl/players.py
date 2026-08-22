"""Player bios: load the nflverse `players` release into the store.

    python -m src.nfl.players [--force-download]

Until now the store had no player-level data at all, so "who is Joe
Burrow" could only be answered as a win-loss record. This adds the other
half: born, drafted, college, position, size, first and last season.

The join key is `gsis_id`, which is exactly what `games.home_qb_id` and
`away_qb_id` already carry - all 348 distinct QB ids in `games` resolve
here, so the join is on an id rather than on a name. That matters: name
joins break on suffixes ("Odell Beckham" vs "Odell Beckham Jr.") and on
the several pairs of players who share one outright.

This is BIOS ONLY. There are still no passing, rushing or receiving
numbers anywhere in the store - those live in nflverse's `stats_player`
release, which is a separate job.

Unlike games.csv this file is not committed. It is ~7 MB, nothing in the
build or the backtest needs it, and it is one HTTP request to fetch.
"""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from . import db, nflverse

PLAYERS_URL = ("https://github.com/nflverse/nflverse-data/releases/"
               "download/players/players.csv")

PLAYER_COLUMNS = [
    "gsis_id", "display_name", "first_name", "last_name", "suffix",
    "birth_date", "position", "position_group", "height", "weight",
    "college_name", "college_conference", "jersey_number",
    "rookie_season", "last_season", "latest_team", "status",
    "years_of_experience", "draft_year", "draft_round", "draft_pick",
    "draft_team", "pfr_id", "headshot",
]

# The feed carries a dozen more id crosswalks (esb, nfl, pff, otc, espn,
# smart) and NGS/PFF duplicates of position and status. None of them
# answer a question this assistant gets asked, so they are dropped rather
# than stored and then ignored.


def download(dest: Path, force: bool = False) -> Path:
    return nflverse.download(dest, force=force, url=PLAYERS_URL)


def _int(text: str) -> int | None:
    text = (text or "").strip()
    if not text or text.upper() in {"NA", "NULL"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _text(text: str) -> str | None:
    text = (text or "").strip()
    return text or None


def load_rows(path: Path) -> list[dict]:
    """Read players.csv into dicts shaped like the `players` table."""
    players: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            gsis = _text(r["gsis_id"])
            # gsis_id is the primary key and the whole point of the file;
            # a row without one cannot be joined to anything.
            if not gsis or not _text(r["display_name"]):
                continue

            team = _text(r["latest_team"])
            drafted_by = _text(r["draft_team"])
            players.append({
                "gsis_id": gsis,
                "display_name": _text(r["display_name"]),
                "first_name": _text(r["first_name"]),
                "last_name": _text(r["last_name"]),
                "suffix": _text(r["suffix"]),
                "birth_date": _text(r["birth_date"]),
                "position": _text(r["position"]),
                "position_group": _text(r["position_group"]),
                "height": _int(r["height"]),
                "weight": _int(r["weight"]),
                "college_name": _text(r["college_name"]),
                "college_conference": _text(r["college_conference"]),
                "jersey_number": _int(r["jersey_number"]),
                "rookie_season": _int(r["rookie_season"]),
                "last_season": _int(r["last_season"]),
                # Same relocation mapping the games table uses, so
                # players.latest_team joins teams.team_id. In practice
                # this feed only needs LA -> LAR; it has already folded
                # STL, OAK and SD into the current franchises.
                "latest_team": nflverse.team_id(team) if team else None,
                "status": _text(r["status"]),
                "years_of_experience": _int(r["years_of_experience"]),
                "draft_year": _int(r["draft_year"]),
                "draft_round": _int(r["draft_round"]),
                "draft_pick": _int(r["draft_pick"]),
                "draft_team": (nflverse.team_id(drafted_by)
                               if drafted_by else None),
                "pfr_id": _text(r["pfr_id"]),
                "headshot": _text(r["headshot"]),
            })
    return players


def run(force_download: bool = False) -> None:
    path = download(config.NFLVERSE_PLAYERS, force=force_download)
    rows = load_rows(path)
    seasons = [r["rookie_season"] for r in rows if r["rookie_season"]]
    print(f"nflverse players: {len(rows):,} rows, "
          f"rookie seasons {min(seasons)}-{max(seasons)}\n")

    conn = db.connect(config.NFL_DB)
    with conn:
        conn.execute("DELETE FROM players")
        db.replace_all(conn, "players", rows, PLAYER_COLUMNS)
        conn.execute(
            "INSERT INTO ingest_log (table_name, rows, source, as_of, "
            "built_at) VALUES (?, ?, ?, ?, ?)",
            ("players", len(rows), "nflverse",
             datetime.now(timezone.utc).date().isoformat(),
             datetime.now(timezone.utc).isoformat(timespec="seconds")))

    print("=" * 72)
    print("PLAYERS TABLE")
    print("=" * 72)
    summary = conn.execute("""
        SELECT COUNT(*) n,
               SUM(birth_date IS NOT NULL) born,
               SUM(draft_year IS NOT NULL) drafted,
               SUM(college_name IS NOT NULL) college
          FROM players
    """).fetchone()
    print(f"  players      : {summary['n']:,}")
    print(f"  with birthday: {summary['born']:,}")
    print(f"  drafted      : {summary['drafted']:,}  "
          f"({summary['n'] - summary['drafted']:,} undrafted)")
    print(f"  with college : {summary['college']:,}")

    # The join is the reason this table exists, so prove it every run
    # rather than assuming it. A drop here means the feed moved.
    coverage = conn.execute("""
        WITH qbs AS (SELECT DISTINCT home_qb_id AS id FROM games
                      WHERE home_qb_id IS NOT NULL
                      UNION
                     SELECT DISTINCT away_qb_id FROM games
                      WHERE away_qb_id IS NOT NULL)
        SELECT COUNT(*) n, SUM(p.gsis_id IS NOT NULL) matched
          FROM qbs LEFT JOIN players p ON p.gsis_id = qbs.id
    """).fetchone()
    print(f"\n  starting QBs in `games`  : {coverage['n']}")
    print(f"  resolved in `players`    : {coverage['matched']}")
    if coverage["matched"] < coverage["n"]:
        print(f"  ! {coverage['n'] - coverage['matched']} QB id(s) did not "
              f"resolve - the feed's key may have changed.")

    conn.close()
    print(f"\nStore updated: {config.NFL_DB}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-download", action="store_true",
                    help="re-fetch players.csv even if it is already cached")
    run(**vars(ap.parse_args()))
