"""Check the feature store against facts the book states independently.

    python -m src.nfl.validate

The games table is built from each club's game-by-game page; the
standings table is built from a different page that reports the season
totals. If summing one reproduces the other, both parses are right. This
is the closest thing to a ground-truth test the book affords, so it runs
as a gate before any model is fitted on the data.
"""
import sys

from .. import config
from . import db


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def run() -> int:
    conn = db.connect(config.NFL_DB)
    results: list[bool] = []
    print("Validating the feature store\n")

    # --- Shape ----------------------------------------------------------
    print("Row counts")
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("teams", "games", "team_season_stats", "standings")}
    results.append(_check("32 clubs", counts["teams"] == 32, str(counts["teams"])))
    for season, expected in ((2025, 272), (2026, 272)):
        n = conn.execute(
            "SELECT COUNT(*) FROM games WHERE season=? AND game_type='REG'",
            (season,)).fetchone()[0]
        results.append(_check(f"{season} regular season = {expected} games",
                              n == expected, str(n)))
    n_post = conn.execute(
        "SELECT COUNT(*) FROM games WHERE season=2025 AND game_type='POST'"
    ).fetchone()[0]
    results.append(_check("2025 postseason = 13 games", n_post == 13, str(n_post)))

    # --- Every club plays 17 regular-season games -----------------------
    print("\nPer-club schedule integrity")
    for season in (2025, 2026):
        bad = conn.execute("""
            SELECT team, COUNT(*) n FROM (
                SELECT home_team AS team FROM games
                 WHERE season=? AND game_type='REG'
                UNION ALL
                SELECT away_team FROM games
                 WHERE season=? AND game_type='REG')
            GROUP BY team HAVING n != 17
        """, (season, season)).fetchall()
        results.append(_check(f"{season}: all 32 clubs play 17 games", not bad,
                              "; ".join(f"{r['team']}={r['n']}" for r in bad)))

    # --- Standings reconciliation (the real test) -----------------------
    print("\n2025 games summed vs. the book's own standings page")
    rows = conn.execute("""
        WITH per_team AS (
            SELECT home_team AS team, home_score AS pf, away_score AS pa
              FROM games WHERE season=2025 AND game_type='REG'
            UNION ALL
            SELECT away_team, away_score, home_score
              FROM games WHERE season=2025 AND game_type='REG'
        ), agg AS (
            SELECT team,
                   SUM(pf > pa)  AS w,
                   SUM(pf < pa)  AS l,
                   SUM(pf = pa)  AS t,
                   SUM(pf)       AS pf,
                   SUM(pa)       AS pa
              FROM per_team GROUP BY team
        )
        SELECT s.team_id, s.wins, s.losses, s.ties, s.points_for, s.points_against,
               a.w, a.l, a.t, a.pf, a.pa
          FROM standings s JOIN agg a ON a.team = s.team_id
         WHERE s.season = 2025
         ORDER BY s.team_id
    """).fetchall()

    rec_bad = [r for r in rows
               if (r["wins"], r["losses"], r["ties"]) != (r["w"], r["l"], r["t"])]
    pts_bad = [r for r in rows
               if (r["points_for"], r["points_against"]) != (r["pf"], r["pa"])]

    results.append(_check(
        "W-L-T from games matches standings for all 32 clubs", not rec_bad,
        "; ".join(f"{r['team_id']} {r['w']}-{r['l']}-{r['t']} vs book "
                  f"{r['wins']}-{r['losses']}-{r['ties']}" for r in rec_bad)))
    results.append(_check(
        "points for/against match standings for all 32 clubs", not pts_bad,
        "; ".join(f"{r['team_id']} {r['pf']}/{r['pa']} vs book "
                  f"{r['points_for']}/{r['points_against']}" for r in pts_bad)))

    # --- Cross-check against the statistics matrices --------------------
    print("\nStatistics matrices vs. games table")
    stat_rows = conn.execute("""
        SELECT team_id, value FROM team_season_stats
         WHERE season=2025 AND side='offense' AND metric='Combined Net Yds.'
    """).fetchall()
    results.append(_check("offence matrix covers all 32 clubs",
                          len(stat_rows) == 32, str(len(stat_rows))))

    metrics = conn.execute("""
        SELECT side, COUNT(DISTINCT metric) m, COUNT(DISTINCT team_id) t
          FROM team_season_stats WHERE season=2025 GROUP BY side
    """).fetchall()
    for r in metrics:
        results.append(_check(f"{r['side']}: 32 clubs populated",
                              r["t"] == 32, f"{r['m']} metrics, {r['t']} clubs"))

    # --- Sanity: no game has a club playing itself, no null scores ------
    print("\nSanity")
    self_play = conn.execute(
        "SELECT COUNT(*) FROM games WHERE home_team = away_team").fetchone()[0]
    results.append(_check("no club plays itself", self_play == 0, str(self_play)))

    unplayed_2025 = conn.execute(
        "SELECT COUNT(*) FROM games WHERE season=2025 AND home_score IS NULL"
    ).fetchone()[0]
    results.append(_check("every 2025 game has a score", unplayed_2025 == 0,
                          str(unplayed_2025)))

    scored_2026 = conn.execute(
        "SELECT COUNT(*) FROM games WHERE season=2026 AND home_score IS NOT NULL"
    ).fetchone()[0]
    results.append(_check("no 2026 game has a score yet", scored_2026 == 0,
                          str(scored_2026)))

    neutral = conn.execute(
        "SELECT season, COUNT(*) n FROM games WHERE neutral=1 AND season>=2025 "
        "GROUP BY season").fetchall()
    print("  [info] neutral-site games: " +
          ", ".join(f"{r['season']}: {r['n']}" for r in neutral))

    # --- Stage 2: market data -------------------------------------------
    market = conn.execute(
        "SELECT COUNT(*) n FROM games WHERE spread_close IS NOT NULL").fetchone()["n"]
    if market:
        print("\nStage 2 - market data")

        # The single most dangerous bug in this store would be a flipped
        # spread sign, and it would be invisible in every row-count check.
        # A home favourite must actually win most of the time.
        r = conn.execute("""
            SELECT SUM(home_score > away_score) w, COUNT(*) n
              FROM games
             WHERE played=1 AND spread_close < 0 AND home_score != away_score
        """).fetchone()
        rate = r["w"] / r["n"]
        results.append(_check(
            "spread sign: home favourites win 60-75% of decided games",
            0.60 <= rate <= 0.75, f"{rate:.1%} of {r['n']:,}"))

        # And the market's number should track the actual margin closely.
        r = conn.execute("""
            SELECT AVG(ABS(-spread_close - (home_score - away_score))) mae,
                   AVG(-spread_close - (home_score - away_score)) bias,
                   COUNT(*) n
              FROM games WHERE played=1 AND spread_close IS NOT NULL
        """).fetchone()
        results.append(_check("closing spread MAE vs actual margin under 12 pts",
                              r["mae"] < 12, f"{r['mae']:.2f} pts over {r['n']:,} games"))
        results.append(_check("closing spread is roughly unbiased (|bias| < 0.5)",
                              abs(r["bias"]) < 0.5, f"{r['bias']:+.3f} pts"))

        # Favourites cover close to half the time - that is what an
        # efficient market looks like, and a big deviation means a bug.
        r = conn.execute("""
            SELECT SUM((home_score - away_score) > -spread_close) covers,
                   COUNT(*) n
              FROM games
             WHERE played=1 AND spread_close IS NOT NULL
               AND (home_score - away_score) != -spread_close
        """).fetchone()
        cover = r["covers"] / r["n"]
        results.append(_check("home teams cover 47-53% of the time",
                              0.47 <= cover <= 0.53, f"{cover:.1%} of {r['n']:,}"))

        totals = conn.execute("""
            SELECT AVG(ABS(total_close - (home_score + away_score))) mae, COUNT(*) n
              FROM games WHERE played=1 AND total_close IS NOT NULL
        """).fetchone()
        print(f"  [info] closing total MAE: {totals['mae']:.2f} pts "
              f"over {totals['n']:,} games")

        coverage = conn.execute("""
            SELECT COUNT(*) n, SUM(played) played,
                   SUM(spread_close IS NOT NULL) spread,
                   SUM(ml_home IS NOT NULL) ml,
                   MIN(season) lo, MAX(season) hi FROM games
        """).fetchone()
        print(f"  [info] {coverage['n']:,} games {coverage['lo']}-{coverage['hi']}; "
              f"{coverage['played']:,} played, {coverage['spread']:,} with a spread, "
              f"{coverage['ml']:,} with a moneyline")
    else:
        print("\nStage 2 - market data: none loaded "
              "(run `python -m src.nfl.enrich`)")

    conn.close()
    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run())
