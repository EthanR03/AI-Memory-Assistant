"""The feature store: SQLite schema and load helpers.

The `games` table deliberately carries columns no Fact Book page can
fill - closing spreads, totals, moneylines, rest days, weather. They sit
NULL through Stage 1. When an odds feed arrives in Stage 2 it updates
those columns in place, so the model and the backtester keep reading the
same table and nothing upstream has to change.
"""
import re
import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS teams (
    team_id     TEXT PRIMARY KEY,
    location    TEXT NOT NULL,
    nickname    TEXT NOT NULL,
    full_name   TEXT NOT NULL,
    conference  TEXT NOT NULL,
    division    TEXT NOT NULL,
    stadium     TEXT,
    surface     TEXT,
    capacity    INTEGER,
    head_coach  TEXT
);

CREATE TABLE IF NOT EXISTS games (
    game_id     TEXT PRIMARY KEY,
    season      INTEGER NOT NULL,
    week        INTEGER,
    game_type   TEXT NOT NULL CHECK (game_type IN ('REG', 'POST')),
    game_date   TEXT,
    home_team   TEXT NOT NULL REFERENCES teams(team_id),
    away_team   TEXT NOT NULL REFERENCES teams(team_id),
    neutral     INTEGER NOT NULL DEFAULT 0,
    site        TEXT,
    played      INTEGER NOT NULL DEFAULT 0,
    home_score  INTEGER,
    away_score  INTEGER,
    overtime    INTEGER NOT NULL DEFAULT 0,

    -- Stage 2: market data, filled from nflverse.
    --
    -- SIGN CONVENTION: spread is quoted the way a sportsbook shows it,
    -- from the home team's perspective, so NEGATIVE means the home club
    -- is favoured (home -3.5). nflverse's own `spread_line` uses the
    -- OPPOSITE sign; src/nfl/nflverse.py negates it on the way in and
    -- validate.py asserts the result. Do not "fix" one without the other.
    spread_open   REAL,
    spread_close  REAL,
    total_open    REAL,
    total_close   REAL,
    ml_home       INTEGER,
    ml_away       INTEGER,
    spread_odds_home INTEGER,
    spread_odds_away INTEGER,
    over_odds     INTEGER,
    under_odds    INTEGER,

    -- Stage 3: situational context.
    home_rest   INTEGER,
    away_rest   INTEGER,
    roof        TEXT,
    surface     TEXT,
    temp        REAL,
    wind        REAL,

    -- Provenance and extras carried over from nflverse.
    nflverse_game_id TEXT,
    round       TEXT,          -- WC / DIV / CON / SB for postseason games
    div_game    INTEGER,
    stadium     TEXT,
    gametime    TEXT,
    home_qb     TEXT,
    away_qb     TEXT,
    -- GSIS ids ("00-0036442"), carried so a player feed can be joined on
    -- an id rather than on a name. Name joins break on suffixes and on
    -- the several pairs of players who share one.
    home_qb_id  TEXT,
    away_qb_id  TEXT,
    home_coach  TEXT,
    away_coach  TEXT,
    referee     TEXT,
    note        TEXT,

    source      TEXT NOT NULL DEFAULT 'factbook'
);

CREATE INDEX IF NOT EXISTS games_season_week ON games (season, week);
CREATE INDEX IF NOT EXISTS games_home ON games (home_team);
CREATE INDEX IF NOT EXISTS games_away ON games (away_team);

-- Long format: the matrices carry ~45 metrics per club per side, and a
-- long table absorbs new metrics from other sources without migrations.
CREATE TABLE IF NOT EXISTS team_season_stats (
    season   INTEGER NOT NULL,
    team_id  TEXT NOT NULL REFERENCES teams(team_id),
    side     TEXT NOT NULL CHECK (side IN ('offense', 'defense')),
    metric   TEXT NOT NULL,
    value    REAL,
    raw      TEXT,
    PRIMARY KEY (season, team_id, side, metric)
);

CREATE TABLE IF NOT EXISTS standings (
    season          INTEGER NOT NULL,
    team_id         TEXT NOT NULL REFERENCES teams(team_id),
    wins            INTEGER NOT NULL,
    losses          INTEGER NOT NULL,
    ties            INTEGER NOT NULL,
    points_for      INTEGER NOT NULL,
    points_against  INTEGER NOT NULL,
    division_champ  INTEGER NOT NULL DEFAULT 0,
    wild_card       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (season, team_id)
);

-- Player bios, from the nflverse `players` release. The join key is
-- gsis_id, which is exactly what games.home_qb_id / away_qb_id carry -
-- all 348 distinct QB ids in `games` resolve here.
--
-- Bios only: no passing, rushing or receiving numbers live in this feed.
CREATE TABLE IF NOT EXISTS players (
    gsis_id       TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    first_name    TEXT,
    last_name     TEXT,
    suffix        TEXT,
    birth_date    TEXT,          -- ISO; age must be computed, not read
    position      TEXT,
    position_group TEXT,
    height        INTEGER,       -- INCHES (76 = 6ft 4in)
    weight        INTEGER,       -- pounds
    college_name  TEXT,          -- may list two schools: 'LSU; Ohio State'
    college_conference TEXT,
    jersey_number INTEGER,
    rookie_season INTEGER,
    last_season   INTEGER,       -- the retired/active test; `status` is not
    latest_team   TEXT,          -- mapped to a current franchise id
    status        TEXT,          -- STALE for retired players; see players.py
    years_of_experience INTEGER,
    draft_year    INTEGER,       -- NULL means undrafted
    draft_round   INTEGER,
    draft_pick    INTEGER,       -- overall, not within the round
    draft_team    TEXT,
    pfr_id        TEXT,          -- Pro-Football-Reference key, for linking
    headshot      TEXT
);

CREATE INDEX IF NOT EXISTS players_name ON players (display_name);
CREATE INDEX IF NOT EXISTS players_position ON players (position);

-- Season-level production, from nflverse `stats_player`. One row per
-- player per season per season_type, joined to `players` on gsis_id.
--
-- NOT WALK-FORWARD, and not safe to join into a backtest: a season total
-- for year N includes the game a Stage-N model would be predicting. This
-- table exists to answer questions, not to fit models. See playerstats.py.
--
-- 'REG' and 'POST' partition the season. nflverse also publishes the two
-- summed together; that file is deliberately NOT loaded, so SUM() over
-- this table cannot double-count.
CREATE TABLE IF NOT EXISTS player_stats (
    player_id     TEXT NOT NULL,   -- gsis_id; join to players.gsis_id
    season        INTEGER NOT NULL,
    season_type   TEXT NOT NULL,   -- 'REG' | 'POST'
    recent_team   TEXT,            -- last club that season, not every club
    position      TEXT,
    games         INTEGER,

    completions   INTEGER, attempts INTEGER,
    passing_yards INTEGER, passing_tds INTEGER,
    passing_interceptions INTEGER, sacks_suffered INTEGER,
    passing_first_downs INTEGER, passing_air_yards REAL,
    passing_epa   REAL, passing_cpoe REAL,

    carries       INTEGER, rushing_yards INTEGER, rushing_tds INTEGER,
    rushing_first_downs INTEGER, rushing_fumbles_lost INTEGER,
    rushing_epa   REAL,

    receptions    INTEGER, targets INTEGER, receiving_yards INTEGER,
    receiving_tds INTEGER, receiving_first_downs INTEGER,
    receiving_air_yards REAL, receiving_epa REAL, target_share REAL,

    def_tackles_solo INTEGER, def_tackle_assists INTEGER,
    def_tackles_for_loss REAL, def_sacks REAL, def_qb_hits REAL,
    def_interceptions INTEGER, def_pass_defended INTEGER,
    def_fumbles_forced REAL, def_tds INTEGER,

    fg_made INTEGER, fg_att INTEGER, fg_pct REAL, fg_long INTEGER,
    pat_made INTEGER, pat_att INTEGER,

    punt_return_yards INTEGER, kickoff_return_yards INTEGER,
    special_teams_tds INTEGER, fumbles_lost_total INTEGER,

    fantasy_points REAL, fantasy_points_ppr REAL,

    PRIMARY KEY (player_id, season, season_type)
);

CREATE INDEX IF NOT EXISTS player_stats_season ON player_stats (season);
CREATE INDEX IF NOT EXISTS player_stats_player ON player_stats (player_id);

CREATE TABLE IF NOT EXISTS qb_records (
    player   TEXT PRIMARY KEY,
    wins     INTEGER NOT NULL,
    losses   INTEGER NOT NULL,
    ties     INTEGER NOT NULL,
    win_pct  REAL NOT NULL
);

-- What the model said, so the chat layer can SELECT a forecast instead
-- of re-running a 7,500-game walk-forward per question.
--
-- These are the WALK-FORWARD predictions: each row was produced by a
-- model that had seen only games before it. That is what makes the table
-- safe to score against later - it is a record of what would have been
-- forecast at the time, not a fit to the whole history.
CREATE TABLE IF NOT EXISTS predictions (
    game_id       TEXT NOT NULL REFERENCES games(game_id),
    model         TEXT NOT NULL,   -- 'elo', 'elo+situational', ...
    p_home        REAL,            -- modelled probability the home club wins
    pred_margin   REAL,            -- modelled home margin, in points
    market_margin REAL,            -- the closing spread's implied home margin
    edge          REAL,            -- pred_margin - market_margin, NULL if unpriced
    built_at      TEXT NOT NULL,
    PRIMARY KEY (game_id, model)
);

CREATE INDEX IF NOT EXISTS predictions_model ON predictions (model);

-- Provenance: which pages produced which table, and when.
CREATE TABLE IF NOT EXISTS ingest_log (
    table_name  TEXT NOT NULL,
    rows        INTEGER NOT NULL,
    source      TEXT NOT NULL,
    as_of       TEXT,
    built_at    TEXT NOT NULL
);
"""


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns the schema has gained since a store was first created.

    `CREATE TABLE IF NOT EXISTS` silently leaves an existing table alone,
    so a store built before Stage 2 would be missing the market columns
    and every write to them would fail. Adding them here keeps an old
    store usable without forcing a rebuild.
    """
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(games)")}
    if not existing:
        return
    wanted = re.findall(r"^\s{4}(\w+)\s+(TEXT|INTEGER|REAL)\b",
                        SCHEMA.split("CREATE TABLE IF NOT EXISTS games (")[1]
                        .split(");")[0], re.MULTILINE)
    for column, sql_type in wanted:
        if column not in existing:
            conn.execute(f"ALTER TABLE games ADD COLUMN {column} {sql_type}")


def game_id(season: int, game_type: str, week: int | None,
            home: str, away: str) -> str:
    """Stable id, so re-running the build updates rows instead of duplicating.

    Week 18 is flex-scheduled and printed as TBD, so week can be absent;
    the club pair still makes the id unique within a season.
    """
    wk = f"W{week:02d}" if week is not None else "WXX"
    return f"{season}-{game_type}-{wk}-{away}@{home}"


# Child tables first: every one of these references teams(team_id), and
# predictions references games(game_id), so it has to go before games.
_TABLE_ORDER = ["ingest_log", "predictions", "games", "team_season_stats",
                "standings", "qb_records", "player_stats", "players", "teams"]


def reset(conn: sqlite3.Connection) -> None:
    """Empty every table so a rebuild replaces rather than accumulates.

    Row keys are derived from the parse (a game_id embeds the week and the
    home club), so a parser fix can change a key and orphan the old row.
    Clearing first keeps the store an exact mirror of the current parse.
    """
    for table in _TABLE_ORDER:
        conn.execute(f"DELETE FROM {table}")


def replace_all(conn: sqlite3.Connection, table: str,
                rows: list[dict], columns: list[str]) -> int:
    """Upsert `rows` into `table` by primary key."""
    if not rows:
        return 0
    placeholders = ", ".join("?" for _ in columns)
    sql = (f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) "
           f"VALUES ({placeholders})")
    conn.executemany(sql, [[r.get(c) for c in columns] for r in rows])
    return len(rows)
