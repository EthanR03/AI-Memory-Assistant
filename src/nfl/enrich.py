"""Stage 2 - fold nflverse into the feature store.

    python -m src.nfl.enrich [--force-download]

Run after `python -m src.nfl.build`. This does two things:

1. **Cross-validates.** The Fact Book and nflverse are independent
   sources for the 2025 season and the 2026 schedule. Every game is
   matched between them and any disagreement is printed. Two sources
   agreeing on 285 games is far stronger evidence than either alone.

2. **Replaces the games table** with the nflverse view of all seasons.
   nflverse is a superset - 1999 onwards, with weeks, kickoff times,
   closing lines, rest days and weather - so it becomes the authority for
   game rows. The Fact Book keeps everything nflverse does not carry:
   the statistics matrices, venues, coaching histories and QB records.
"""
import argparse
from collections import defaultdict

from .. import config
from . import db, nflverse

GAME_COLUMNS = [
    "game_id", "season", "week", "game_type", "round", "game_date", "gametime",
    "home_team", "away_team", "neutral", "site", "played",
    "home_score", "away_score", "overtime",
    "spread_open", "spread_close", "total_open", "total_close",
    "ml_home", "ml_away", "spread_odds_home", "spread_odds_away",
    "over_odds", "under_odds",
    "home_rest", "away_rest", "roof", "surface", "temp", "wind",
    "nflverse_game_id", "div_game", "stadium",
    "home_qb", "away_qb", "home_coach", "away_coach", "referee",
    "note", "source",
]


def _match(factbook: list[dict], incoming: list[dict]) -> tuple[dict, list, list]:
    """Pair Fact Book games with nflverse games for the shared seasons.

    Keyed on the club pair rather than on home/away, because the two
    sources can designate a different "home" club for a neutral-site game.
    """
    def key(g):
        return (g["season"], g["game_type"], frozenset({g["home_team"], g["away_team"]}))

    fb_by_key = defaultdict(list)
    for g in factbook:
        fb_by_key[key(g)].append(g)
    nv_by_key = defaultdict(list)
    for g in incoming:
        nv_by_key[key(g)].append(g)

    # Divisional rivals meet twice, so a key can hold two games. Undated
    # Fact Book rows (flex-scheduled week 18, printed as TBD) must sort
    # last, not first, or the two meetings pair up crossways.
    def order(g):
        return (g["game_date"] is None, g["game_date"] or "")

    pairs, only_fb, only_nv = {}, [], []
    for k in set(fb_by_key) | set(nv_by_key):
        fb = sorted(fb_by_key.get(k, []), key=order)
        nv = sorted(nv_by_key.get(k, []), key=order)

        # Pass 1: an exact home/away agreement is unambiguous, so take
        # those pairings first and let date order settle only the rest.
        remaining_nv = list(nv)
        unmatched_fb = []
        for f in fb:
            hit = next((n for n in remaining_nv
                        if n["home_team"] == f["home_team"]
                        and n["away_team"] == f["away_team"]), None)
            if hit:
                remaining_nv.remove(hit)
                pairs[hit["nflverse_game_id"]] = (f, hit)
            else:
                unmatched_fb.append(f)

        # Pass 2: whatever is left over, in date order.
        for i in range(max(len(unmatched_fb), len(remaining_nv))):
            f = unmatched_fb[i] if i < len(unmatched_fb) else None
            n = remaining_nv[i] if i < len(remaining_nv) else None
            if f and n:
                pairs[n["nflverse_game_id"]] = (f, n)
            elif f:
                only_fb.append(f)
            else:
                only_nv.append(n)
    return pairs, only_fb, only_nv


def cross_validate(pairs: dict) -> list[str]:
    """Compare the two sources game by game."""
    issues = []
    for _, (f, n) in sorted(pairs.items()):
        label = (f"{n['season']} wk{n['week']:>2} "
                 f"{n['away_team']} @ {n['home_team']}")

        if f["home_score"] is not None and n["home_score"] is not None:
            # Scores are stored per club, so compare them per club rather
            # than positionally - the sources may disagree on who is home.
            fb_scores = {f["home_team"]: f["home_score"],
                         f["away_team"]: f["away_score"]}
            nv_scores = {n["home_team"]: n["home_score"],
                         n["away_team"]: n["away_score"]}
            if fb_scores != nv_scores:
                issues.append(
                    f"{label}: score mismatch - factbook "
                    f"{sorted(fb_scores.items())} vs nflverse "
                    f"{sorted(nv_scores.items())}")

        if f["home_team"] != n["home_team"] and not (f["neutral"] or n["neutral"]):
            issues.append(f"{label}: home/away disagree and neither source "
                          f"calls it neutral")
    return issues


def run(force_download: bool = False) -> None:
    path = nflverse.download(config.NFLVERSE_GAMES, force=force_download)
    incoming = nflverse.load_rows(path)
    seasons = sorted({g["season"] for g in incoming})
    print(f"nflverse: {len(incoming):,} games, {seasons[0]}-{seasons[-1]}\n")

    conn = db.connect(config.NFL_DB)
    factbook = [dict(r) for r in conn.execute(
        "SELECT * FROM games WHERE source = 'factbook'")]
    if not factbook:
        print("No Fact Book games found - run `python -m src.nfl.build` first.")
        return

    shared = sorted({g["season"] for g in factbook})
    print(f"Fact Book: {len(factbook)} games, seasons {shared}")

    # --- 1. Cross-validate -----------------------------------------------
    overlap = [g for g in incoming if g["season"] in shared]
    pairs, only_fb, only_nv = _match(factbook, overlap)
    issues = cross_validate(pairs)

    print("\n" + "=" * 72)
    print("CROSS-VALIDATION: Fact Book vs nflverse")
    print("=" * 72)
    print(f"  matched games          : {len(pairs)}")
    print(f"  only in the Fact Book  : {len(only_fb)}")
    print(f"  only in nflverse       : {len(only_nv)}")
    graded = sum(1 for _, (f, n) in pairs.items()
                 if f["home_score"] is not None and n["home_score"] is not None)
    print(f"  both sources scored    : {graded}")
    print(f"  disagreements          : {len(issues)}")
    for issue in issues[:20]:
        print(f"    ! {issue}")
    for g in (only_fb + only_nv)[:10]:
        src = "factbook" if g in only_fb else "nflverse"
        print(f"    ! unmatched ({src}): {g['season']} "
              f"{g['away_team']} @ {g['home_team']}")

    # --- 2. Replace the games table --------------------------------------
    # Carry over the few fields only the Fact Book has.
    extras = {n["nflverse_game_id"]: f for _, (f, n) in pairs.items()}
    rows = []
    for g in incoming:
        f = extras.get(g["nflverse_game_id"])
        row = dict(g)
        row["site"] = (f or {}).get("site")
        row["note"] = (f or {}).get("note")
        row["source"] = "nflverse+factbook" if f else "nflverse"
        row["spread_open"] = None   # nflverse carries one line, not open+close
        row["total_open"] = None
        row["game_id"] = db.game_id(g["season"], g["game_type"], g["week"],
                                    g["home_team"], g["away_team"])
        rows.append(row)

    with conn:
        conn.execute("DELETE FROM games")
        db.replace_all(conn, "games", rows, GAME_COLUMNS)

    # --- 3. Report --------------------------------------------------------
    print("\n" + "=" * 72)
    print("GAMES TABLE AFTER ENRICHMENT")
    print("=" * 72)
    summary = conn.execute("""
        SELECT COUNT(*) n,
               SUM(played) played,
               SUM(spread_close IS NOT NULL) spread,
               SUM(ml_home IS NOT NULL) moneyline,
               MIN(season) lo, MAX(season) hi
          FROM games
    """).fetchone()
    print(f"  games        : {summary['n']:,}  ({summary['lo']}-{summary['hi']})")
    print(f"  played       : {summary['played']:,}")
    print(f"  with spread  : {summary['spread']:,}")
    print(f"  with money   : {summary['moneyline']:,}")

    print("\n  modelling window (played games that also have a closing spread):")
    window = conn.execute("""
        SELECT MIN(season) lo, MAX(season) hi, COUNT(*) n
          FROM games WHERE played = 1 AND spread_close IS NOT NULL
    """).fetchone()
    print(f"    {window['n']:,} games, {window['lo']}-{window['hi']}"
          f"  (Stage 1 had 272)")

    lookahead = conn.execute("""
        SELECT COUNT(*) n FROM games
         WHERE season = 2026 AND spread_close IS NOT NULL
    """).fetchone()["n"]
    print(f"\n  2026 games with a lookahead line already posted: {lookahead}/272")

    conn.close()
    print(f"\nStore updated: {config.NFL_DB}")
    if issues:
        print(f"\n{len(issues)} cross-validation issue(s) above - investigate "
              f"before trusting the affected rows.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-download", action="store_true",
                    help="re-fetch games.csv even if it is already cached")
    run(**vars(ap.parse_args()))
