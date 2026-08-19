"""Build the feature store from the extracted Fact Book text.

    python -m src.nfl.build

Safe to re-run: every table is keyed, so a rebuild updates rows in place.
Anything the parsers could not interpret is printed at the end rather
than swallowed - a silent parse failure in a stats table is exactly the
kind of error that shifts a whole row of numbers without anyone noticing.
"""
from datetime import datetime, timezone

from .. import config
from . import db, parsers
from .pages import load_pages


def run() -> None:
    pages = load_pages()
    report = parsers.ParseReport()
    print(f"Loaded {len(pages)} pages from {config.FACTBOOK_TXT.name}\n")

    blocks = parsers.find_club_blocks(pages, report)
    print(f"Club blocks located: {len(blocks)}/32")

    # --- Parse -----------------------------------------------------------
    club_rows = parsers.parse_club_facts(pages, blocks)
    neutral = parsers.parse_neutral_2025(pages, report)

    views_2025 = parsers.parse_2025_results(pages, blocks, report)
    views_2026 = parsers.parse_2026_schedule(pages, blocks, report)
    print(f"Club-level game views: {len(views_2025)} (2025), {len(views_2026)} (2026)")

    games = parsers.reconcile(views_2025, report, neutral) + \
        parsers.reconcile(views_2026, report)

    for g in games:
        g["week"] = parsers.week_of(g["game_date"], g["season"], config.SEASON_ANCHORS)
        # Week 18 is flex-scheduled, so the book prints it as TBD.
        if g["season"] == 2026 and g["week"] is None:
            g["week"] = 18
        # Postseason rounds continue the week count past the regular season.
        g["game_id"] = db.game_id(g["season"], g["game_type"], g["week"],
                                  g["home_team"], g["away_team"])
        g["game_date"] = g["game_date"].isoformat() if g["game_date"] else None

    stats = parsers.parse_team_stats(pages, report)
    standings = parsers.parse_standings(pages, report)
    qbs = parsers.parse_qb_records(pages, report)

    # --- Load ------------------------------------------------------------
    conn = db.connect(config.NFL_DB)
    built_at = datetime.now(timezone.utc).isoformat()
    loaded: list[tuple[str, int, str]] = []

    with conn:
        db.reset(conn)
        loaded.append(("teams", db.replace_all(conn, "teams", club_rows, [
            "team_id", "location", "nickname", "full_name", "conference",
            "division", "stadium", "surface", "capacity", "head_coach",
        ]), "pdf 44-237"))

        loaded.append(("games", db.replace_all(conn, "games", games, [
            "game_id", "season", "week", "game_type", "game_date",
            "home_team", "away_team", "neutral", "site", "played",
            "home_score", "away_score", "overtime",
        ]), "pdf 44-237, 243-244"))

        loaded.append(("team_season_stats", db.replace_all(
            conn, "team_season_stats", stats,
            ["season", "team_id", "side", "metric", "value", "raw"],
        ), "pdf 257-260"))

        loaded.append(("standings", db.replace_all(conn, "standings", standings, [
            "season", "team_id", "wins", "losses", "ties",
            "points_for", "points_against", "division_champ", "wild_card",
        ]), "pdf 245"))

        loaded.append(("qb_records", db.replace_all(conn, "qb_records", qbs, [
            "player", "wins", "losses", "ties", "win_pct",
        ]), "pdf 313"))

        conn.execute("DELETE FROM ingest_log")
        conn.executemany(
            "INSERT INTO ingest_log (table_name, rows, source, as_of, built_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(t, n, s, config.FACTBOOK_AS_OF, built_at) for t, n, s in loaded])

    # --- Report ----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'TABLE':<22}{'ROWS':>8}   SOURCE")
    print("=" * 60)
    for table, rows, source in loaded:
        print(f"{table:<22}{rows:>8}   {source}")
    print("=" * 60)

    counts = dict(conn.execute(
        "SELECT season || ' ' || game_type, COUNT(*) FROM games GROUP BY 1").fetchall())
    print("games by season/type:", counts)
    print(f"\nfeature store: {config.NFL_DB}")

    if report.warnings:
        print(f"\n{len(report.warnings)} warning(s):")
        for w in report.warnings:
            print(f"  ! {w}")
    else:
        print("\nNo warnings - every game reconciled across both clubs' pages.")

    conn.close()


if __name__ == "__main__":
    run()
