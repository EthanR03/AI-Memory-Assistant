"""Tools the assistant can call: a read-only SQL window onto nfl.db.

The memory assistant answers from prose it embedded. That is the right
mechanism for notes and the wrong one for the predictor, whose data is
7,500 structured game records - the lesson the Fact Book already taught
when 883 pages of tables went into 800-character chunks and came back
unusable.

So the predictor gets queried instead of retrieved. "Which team has the
best record since 2016" is not a fact stored anywhere in nfl.db; it is
two lines of SQL over games. Computing it returns the exact answer,
where embedding-and-recalling would return a plausible-looking one.

Safety here is structural rather than advisory. The connection is opened
`mode=ro`, so SQLite itself refuses writes no matter what SQL arrives -
a prompt-injected DROP fails at the driver, not at a regex we hope
holds. The statement checks on top of that exist to return a clear error
instead of a confusing one.

    python -m src.tools "SELECT home_team, COUNT(*) FROM games GROUP BY 1"
"""
import re
import sqlite3
import sys

from . import config

# Enough rows to aggregate over, few enough to not swamp the context
# window. Anything bigger is a question that wants a GROUP BY.
MAX_ROWS = 200

# Only these may start a statement. WITH is here because common table
# expressions are how most interesting questions get written.
_ALLOWED_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)

# Rejected with a specific message rather than a generic failure, since
# the model can usually rewrite the query once it knows why.
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|"
    r"ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b", re.IGNORECASE)


class QueryError(Exception):
    """A query that was refused or failed, phrased for the model to read."""


def _check(sql: str) -> str:
    """Reject anything that is not a single read-only statement."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise QueryError("Empty query.")
    if not _ALLOWED_START.match(sql):
        raise QueryError("Only SELECT (or WITH ... SELECT) queries are "
                         "allowed. This is a read-only tool.")
    if _FORBIDDEN.search(sql):
        raise QueryError("This tool cannot modify the database or read its "
                         "settings. Use SELECT only.")
    if ";" in sql:
        raise QueryError("Send one statement at a time; ';' is not allowed.")
    return sql


def _connect() -> sqlite3.Connection:
    """Open nfl.db read-only.

    Note this deliberately does NOT go through db.connect(), which runs
    the schema script and would both write to the file and fail against
    a read-only handle. The store is built by src.nfl.build/enrich; this
    module only ever looks at it.
    """
    if not config.NFL_DB.exists():
        raise QueryError(
            f"No store at {config.NFL_DB}. Build it with "
            "`python -m src.nfl.enrich` (works without the Fact Book PDF).")
    conn = sqlite3.connect(f"file:{config.NFL_DB.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str, max_rows: int = MAX_ROWS) -> dict:
    """Run one read-only SELECT and return its rows.

    Returns {"columns": [...], "rows": [[...]], "row_count": n,
             "truncated": bool} - a shape that serialises cleanly into a
    tool result without the model having to parse prose.
    """
    sql = _check(sql)
    conn = _connect()
    try:
        cursor = conn.execute(sql)
        fetched = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description or []]
    except sqlite3.Error as exc:
        raise QueryError(f"SQL error: {exc}") from exc
    finally:
        conn.close()

    truncated = len(fetched) > max_rows
    rows = [list(r) for r in fetched[:max_rows]]
    return {"columns": columns, "rows": rows,
            "row_count": len(rows), "truncated": truncated}


def format_result(result: dict) -> str:
    """Render a result as a compact table for a tool response or the CLI."""
    if not result["columns"]:
        return "(no columns)"
    if not result["rows"]:
        return "(no rows matched)"

    widths = [max(len(str(c)), *(len(str(r[i])) for r in result["rows"]))
              for i, c in enumerate(result["columns"])]
    line = "  ".join(str(c).ljust(w) for c, w in zip(result["columns"], widths))
    out = [line, "  ".join("-" * w for w in widths)]
    out += ["  ".join(str(v).ljust(w) for v, w in zip(row, widths))
            for row in result["rows"]]
    if result["truncated"]:
        out.append(f"... truncated at {len(result['rows'])} rows; "
                   "add a GROUP BY or LIMIT.")
    return "\n".join(out)


# --- What the model needs to know to write correct SQL -------------------
#
# Hand-written rather than read from sqlite_master, because the traps are
# not in the column list: the spread's sign, the fact that only 2025 has
# season stats, and that standings must be derived for any other year.
SCHEMA_NOTES = """\
SQLite store of NFL data. Tables:

games (7,548 rows, seasons 1999-2026; 7,276 played)
  game_id, season, week, game_type ('REG'|'POST'), round ('WC','DIV','CON','SB'),
  game_date, home_team, away_team, neutral, played, home_score, away_score,
  overtime, spread_close, total_close, ml_home, ml_away,
  home_rest, away_rest, div_game, roof, surface, temp, wind, stadium,
  home_qb, away_qb, home_qb_id, away_qb_id, home_coach, away_coach, referee
predictions (what the model forecast; 'elo' covers 1999-2026,
             'elo+situational' covers 2010-2026)
  game_id, model, p_home, pred_margin, market_margin, edge, built_at
teams (32 rows, CURRENT clubs only, as of 2026-07-14)
  team_id, location, nickname, full_name, conference, division,
  stadium, surface, capacity, head_coach
team_season_stats (2025 ONLY - 3,904 rows, 85 metrics)
  season, team_id, side ('offense'|'defense'), metric, value, raw
standings (2025 ONLY - 32 rows)
  season, team_id, wins, losses, ties, points_for, points_against,
  division_champ, wild_card
qb_records (67 rows, CAREER totals, no season dimension)
  player, wins, losses, ties, win_pct
player_stats (61,533 rows - season totals, 1999-2025)
  player_id (= players.gsis_id), season, season_type ('REG'|'POST'),
  recent_team, position, games,
  completions, attempts, passing_yards, passing_tds,
  passing_interceptions, sacks_suffered, passing_first_downs,
  passing_air_yards, passing_epa, passing_cpoe,
  carries, rushing_yards, rushing_tds, rushing_first_downs,
  rushing_fumbles_lost, rushing_epa,
  receptions, targets, receiving_yards, receiving_tds,
  receiving_first_downs, receiving_air_yards, receiving_epa, target_share,
  def_tackles_solo, def_tackle_assists, def_tackles_for_loss, def_sacks,
  def_qb_hits, def_interceptions, def_pass_defended, def_fumbles_forced,
  def_tds, fg_made, fg_att, fg_pct, fg_long, pat_made, pat_att,
  punt_return_yards, kickoff_return_yards, special_teams_tds,
  fumbles_lost_total, fantasy_points, fantasy_points_ppr
players (25,050 rows - BIOS, every player since 1974)
  gsis_id, display_name, first_name, last_name, suffix, birth_date,
  position, position_group, height, weight, college_name,
  college_conference, jersey_number, rookie_season, last_season,
  latest_team, status, years_of_experience, draft_year, draft_round,
  draft_pick, draft_team, pfr_id, headshot

Things that will produce wrong answers if ignored:

- SPREAD SIGN is the sportsbook convention: spread_close is NEGATIVE when
  the HOME club is favoured. The market's implied home margin is
  -spread_close. Actual home margin is home_score - away_score.
- A team's games are split across home_team and away_team. For per-team
  records, UNION ALL the two perspectives (see the example below).
- team_season_stats and standings hold 2025 ONLY. For any other season,
  DERIVE the record from games - do not report "no data".
- 2026 games are SCHEDULED, not played: played = 0 and scores are NULL.
  Filter on played = 1 for anything historical.
- temp and wind are only recorded for outdoor games, and are the reading
  AT KICKOFF. roof is 'outdoors', 'dome', 'closed' or 'open'.
- No game data before 1999 exists, and player_stats starts at 1999 too,
  so a career total for anyone who played earlier is PARTIAL - say so
  rather than reporting it as the career figure. Still absent from the
  store entirely: per-GAME (as opposed to per-season) statistics,
  rosters, injuries and depth charts.
- player_stats is SEASON TOTALS. It cannot answer "how did he do in week
  9" or "in the Chiefs game" - only "in 2023". Do not fake it by joining
  to a single game.
- player_stats carries NO NAME. Join it: `JOIN players p ON
  p.gsis_id = s.player_id`. Four ids in player_stats have no bio row and
  come back NULL, so use an INNER join for leaderboards.
- season_type 'REG' and 'POST' PARTITION the season, so a bare SUM over
  player_stats silently mixes them. DEFAULT TO 'REG' for any career or
  season total, because that is what the NFL means by a career statistic
  and what qb_records already counts - Joe Burrow's career passing yards
  are 20,810 (REG), not 22,636 (REG+POST). State the basis, and give the
  postseason as a separate figure when it is interesting. Only combine
  the two if the question actually asks for everything.
- player_stats figures are DERIVED FROM PLAY-BY-PLAY and are not the
  official NFL box score. They agree with it nearly always, but not
  always: Drew Brees 2011 reads 5,535 passing yards here when the
  official record is 5,476, which makes this table's all-time
  single-season passing leaderboard WRONG AT THE TOP - it returns Brees
  2011 ahead of Peyton Manning's 5,477 in 2013, which is the real record.
  Attribute these numbers to nflverse, never call one an NFL record, and
  if asked for a record specifically, say the store cannot certify it.
- player_stats.recent_team is the LAST club of that season, so a traded
  player's whole season sits under one team. It is not a per-club split.
- Unused categories are 0, not NULL - a quarterback has
  receiving_yards = 0. Filter on position or on attempts/targets/carries
  being non-zero before ranking, or punters appear in rushing tables.
- player_stats is NOT walk-forward: a season total for year N includes
  the game itself. Never join it to `predictions` or use it to justify a
  forecast; that is a leak. Use it for questions about the past only.
- 6,081 rows in `players` carry a non-GSIS id (an ESB id) because the
  feed had no GSIS id for them. Those players can never join to games or
  player_stats - a bio with no stats is usually this, not a data error.
- players.status IS STALE and must never be used to say who is playing
  now: Tom Brady, Peyton Manning and Joe Montana all read 'ACT'. Use
  last_season instead - last_season < the current season means the player
  is no longer active. latest_team has the same problem (it is the last
  team, not a current one).
- players.height is in INCHES (76 -> 6'4"), weight in pounds. birth_date
  is ISO, so age must be COMPUTED from today's date, not read off.
- players.draft_year NULL means UNDRAFTED - say so rather than reporting
  it as missing data. draft_pick is the OVERALL pick, not the pick within
  its round. college_name can list two schools ('LSU; Ohio State').
- Join players to games on the GSIS id, never the name:
  `JOIN players p ON p.gsis_id = g.home_qb_id`. All 348 starting QB ids
  in `games` resolve. Names collide and vary by suffix.
- qb_records counts REGULAR SEASON games only, while home_qb/away_qb in
  games covers both. Joe Burrow is 43-33-1 in qb_records and 48-35 across
  all games, and both are right - the difference is a 5-2 postseason
  record. State which basis you are using, and do not quote the two
  numbers as if one contradicts the other.
- predictions are WALK-FORWARD: each row was forecast by a model that had
  seen only earlier games, so they can be scored honestly against results.
  Join with `JOIN games USING(game_id)`.
- ALWAYS filter predictions on one model (`WHERE model = 'elo'`), or every
  game comes back twice. Default to 'elo' unless asked for the other, and
  SAY which model the numbers came from - 'elo+situational' is the WORSE
  of the two and must never be served as though it were the main model.
- pred_margin and market_margin are HOME margins, and edge is
  pred_margin - market_margin. A NEGATIVE home margin means the AWAY club
  is favoured: pred_margin = -2.9 for DEN @ KC means the model likes
  DENVER by 2.9, NOT Kansas City by 2.9. Convert to "AWAY by N" before
  writing it in prose, or the sentence will contradict itself.
- IMPORTANT when reporting a pick: this model does NOT beat the market.
  Over 2010-2025 it hit 64.6% against the closing spread's 66.4%, and no
  edge filter clears the 52.4% break-even by more than noise. Report a
  forecast as the model's opinion, and say plainly that it has no
  demonstrated edge. Never present a pick as profitable advice.
- home_qb_id/away_qb_id are GSIS ids, the join key into `players`. Join on
  the id, not the name.

Per-team records are written like this:

  WITH res AS (
    SELECT home_team AS team, home_score AS pf, away_score AS pa,
           game_type, round FROM games WHERE played = 1
    UNION ALL
    SELECT away_team, away_score, home_score,
           game_type, round FROM games WHERE played = 1)
  SELECT team, SUM(pf > pa) AS wins, SUM(pf < pa) AS losses
    FROM res WHERE ... GROUP BY team ORDER BY wins DESC;
"""

# Provider-neutral description; map to the OpenAI/Anthropic tool format
# at the call site rather than baking either one in here.
TOOL_SPEC = {
    "name": "query_nfl_db",
    "description": (
        "Run a read-only SQL SELECT against the NFL store to answer "
        "questions about games, teams, results, betting lines, "
        "conditions, player bios and season statistics from 1999 to "
        "2026. Prefer computing "
        "an answer with "
        "SQL over recalling it. " + SCHEMA_NOTES),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A single SELECT or WITH...SELECT statement.",
            },
        },
        "required": ["sql"],
    },
}


def run() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        print(SCHEMA_NOTES)
        return
    try:
        print(format_result(query(" ".join(sys.argv[1:]))))
    except QueryError as exc:
        print(f"error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    run()
