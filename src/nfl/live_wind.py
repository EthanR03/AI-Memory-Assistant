"""Live paper-trading of the frozen wind rule (WIND_RULE.md, v1).

    python -m src.nfl.live_wind picks              # log games 18-36h out
    python -m src.nfl.live_wind picks --dry-run    # show them, log nothing
    python -m src.nfl.live_wind price GAME_ID 41.5 -110 --book DK
    python -m src.nfl.live_wind grade              # once the games are final

Stage 3b says the edge sits in wind the forecast showed a day out. 2026
is the test of that, and it only counts if every pick is written down
before kickoff. `picks` writes each eligible game to
data/live/wind_picks.csv with the time it was pulled. Commit the log
after every run: the git history is what proves the picks came first.

`price` records the total and price a book actually offered on a bet, and
refuses once the game has kicked off. `grade` grades each bet against the
nflverse close at -110 and against that logged price. It also fetches the
archived day-ahead forecast the backtest would have used, to show how far
the live pull sits from it.

Every number that decides a bet comes from WIND_RULE.md. If this file and
the rule ever disagree, the rule wins and this file has the bug.
"""
import argparse
import csv
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .. import config
from . import nflverse
from .forecast_wind import (API as ARCHIVE_API, HOURLY as ARCHIVE_HOURLY,
                            MODELS as ARCHIVE_MODELS, STADIUMS, _request,
                            window_means)
from .ratings import BREAK_EVEN

RULE = "v1"
SEASON = 2026
THRESHOLD_MPH = 12.0
PULL_HOURS = (18, 36)    # pull between 36h and 18h before kickoff
CLOSE_ODDS = -110        # the close is graded at an assumed -110
LOOKAHEAD = timedelta(days=7)    # how far ahead "coming up" lists games

FORECAST_API = "https://api.open-meteo.com/v1/forecast"
LIVE_HOURLY = "wind_speed_10m,temperature_2m"
LIVE_MODELS = "jma_seamless,best_match"
JMA = "wind_speed_10m_jma_seamless"
DEFAULT = "wind_speed_10m_best_match"
JMA_TEMP = "temperature_2m_jma_seamless"
ARCHIVE_JMA = "wind_speed_10m_previous_day1_jma_seamless"   # Stage 3b's series

ET = ZoneInfo("America/New_York")    # nflverse gametime is ET

COLUMNS = [
    # Written by `picks`, before kickoff.
    "game_id", "week", "kickoff_et", "away", "home", "stadium",
    "pulled_at", "lead_h", "jma_mph", "default_mph", "jma_temp_f",
    "bet", "rule",
    # Written by `price`, before kickoff.
    "book", "book_total", "book_odds", "priced_at",
    # Written by `grade`, after the game.
    "points", "total_close", "close_result", "close_units",
    "book_result", "book_units", "archive_jma_mph", "graded_at",
]


# --- Time ----------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _local(dt: datetime) -> str:
    """The machine's own time zone - when to actually run the script."""
    return dt.astimezone().strftime("%a %m-%d %H:%M")


def _et(text: str) -> datetime:
    """An ET wall time, returned in UTC. Two datetimes sharing a ZoneInfo
    subtract as wall clocks, an hour out across a DST change - and week
    8's Sunday is one - so all the arithmetic here is done in UTC."""
    return datetime.fromisoformat(text).replace(tzinfo=ET).astimezone(
        timezone.utc)


def kickoff(g: dict) -> datetime:
    return _et(f"{g['game_date']}T{g['gametime']}")


def _logged_kickoff(row: dict) -> datetime:
    return _et(row["kickoff_et"])


def _lead(g: dict, now: datetime) -> float:
    return (kickoff(g) - now).total_seconds() / 3600


# --- Schedule and forecasts ----------------------------------------------

def load_schedule() -> list[dict]:
    """This season's regular-season games, freshly downloaded.

    Kickoffs get flexed and venues moved during a season, so every run
    fetches a new copy. It goes to data/live/, never over the committed
    games.csv the backtests read.
    """
    path = nflverse.download(config.LIVE_GAMES, force=True)
    return [g for g in nflverse.load_rows(path)
            if g["season"] == SEASON and g["game_type"] == "REG"
            and g["game_date"] and g["gametime"]]


def eligible(g: dict) -> bool:
    """Outdoors per the schedule. Blank means a retractable roof - out."""
    return (g["roof"] or "").lower() == "outdoors"


def _fetch(games: list[dict], url: str, hourly: str,
           models: str) -> dict[str, dict]:
    """Each game's window means from one Open-Meteo API, keyed by game id.

    One request per game date covers every stadium playing that day. The
    window runs into the next day so a late kickoff still has four hours.
    """
    by_day: dict[str, list[dict]] = {}
    for g in games:
        by_day.setdefault(g["game_date"], []).append(g)
    out = {}
    for day, rows in sorted(by_day.items()):
        names = sorted({g["stadium"] for g in rows})
        end = date.fromisoformat(day) + timedelta(days=1)
        results = _request({
            "latitude": ",".join(str(STADIUMS[s][0]) for s in names),
            "longitude": ",".join(str(STADIUMS[s][1]) for s in names),
            "start_date": day, "end_date": end.isoformat(),
            "hourly": hourly, "models": models,
            "wind_speed_unit": "mph", "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
        }, url)
        series = dict(zip(names, (r["hourly"] for r in results)))
        for g in rows:
            out[g["nflverse_game_id"]] = window_means(
                series[g["stadium"]], g["game_date"], g["gametime"])
    return out


def is_bet(jma: float | None) -> bool:
    # Hourly values come at 0.1 mph, so a window mean is a multiple of
    # 0.025. Round off float noise first, so an exact 12.0 is a bet.
    return jma is not None and round(jma, 3) >= THRESHOLD_MPH


def _fmt(x: float | None, digits: int = 2) -> str:
    return "" if x is None else f"{x:.{digits}f}"


# --- The log ---------------------------------------------------------------

def read_log() -> list[dict]:
    if not config.WIND_PICKS.exists():
        return []
    with config.WIND_PICKS.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_log(rows: list[dict]) -> None:
    config.WIND_PICKS.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["kickoff_et"], r["game_id"]))
    tmp = config.WIND_PICKS.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(config.WIND_PICKS)


# --- picks -----------------------------------------------------------------

def picks(at: datetime | None = None, dry_run: bool = False) -> None:
    """Pull and log every outdoor game now 18-36h from kickoff."""
    now = at or _now()
    log = read_log()
    logged = {r["game_id"] for r in log}
    upcoming = [g for g in load_schedule()
                if eligible(g) and kickoff(g) > now
                and g["nflverse_game_id"] not in logged]
    unknown = sorted({g["stadium"] for g in upcoming} - STADIUMS.keys(),
                     key=str)
    if unknown:
        raise KeyError(f"no coordinates for {unknown} - add them to "
                       "forecast_wind.STADIUMS (that does not change the rule)")

    lo, hi = PULL_HOURS
    due = [g for g in upcoming if lo <= _lead(g, now) <= hi]
    missed = [g for g in upcoming if _lead(g, now) < lo]
    later = [g for g in upcoming
             if _lead(g, now) > hi and kickoff(g) - now <= LOOKAHEAD]

    print(f"Rule {RULE}: under when the JMA forecast is >= {THRESHOLD_MPH} "
          f"mph. Now {_local(now)} local.")
    rows = []
    if due:
        forecasts = _fetch(due, FORECAST_API, LIVE_HOURLY, LIVE_MODELS)
        for g in due:
            w = forecasts[g["nflverse_game_id"]]
            rows.append({
                "game_id": g["nflverse_game_id"], "week": g["week"],
                "kickoff_et": f"{g['game_date']} {g['gametime']}",
                "away": g["away_team"], "home": g["home_team"],
                "stadium": g["stadium"], "pulled_at": _stamp(now),
                "lead_h": _fmt(_lead(g, now), 1),
                "jma_mph": _fmt(w.get(JMA)),
                "default_mph": _fmt(w.get(DEFAULT)),
                "jma_temp_f": _fmt(w.get(JMA_TEMP), 1),
                "bet": "UNDER" if is_bet(w.get(JMA)) else "",
                "rule": RULE,
            })
        print(f"\n{'GAME':<18}{'KICKOFF ET':<18}{'LEAD':>6}{'JMA':>7}"
              f"{'DEFAULT':>9}  BET")
        for r in rows:
            print(f"{r['game_id']:<18}{r['kickoff_et']:<18}"
                  f"{r['lead_h'] + 'h':>6}{r['jma_mph'] or '-':>7}"
                  f"{r['default_mph'] or '-':>9}  {r['bet']}")
    else:
        print("\nNo outdoor game is 18-36h from kickoff right now.")

    if rows and not dry_run:
        write_log(log + rows)
        bets = sum(r["bet"] == "UNDER" for r in rows)
        print(f"\nLogged {len(rows)} games to {config.WIND_PICKS.name}, "
              f"bets: {bets}. Commit it now.")
    elif rows:
        print("\nDry run - nothing logged.")

    if missed:
        print("\nToo late to pull (under 18h out), so no bet under the rule:")
        for g in missed:
            print(f"  {g['nflverse_game_id']:<18}{g['game_date']} {g['gametime']}")
    if later:
        print("\nComing up - run `picks` once inside each window (local time):")
        for g in later:
            ko = kickoff(g)
            print(f"  {g['nflverse_game_id']:<18}"
                  f"{_local(ko - timedelta(hours=hi))}  to  "
                  f"{_local(ko - timedelta(hours=lo))}")


# --- price -----------------------------------------------------------------

def price(game_id: str, total: float, odds: int, book: str) -> None:
    """Record the total and price a book offered on a logged bet."""
    if abs(odds) < 100:
        raise SystemExit(f"{odds} is not American odds (e.g. -110, +105)")
    log = read_log()
    row = next((r for r in log if r["game_id"] == game_id), None)
    if row is None:
        raise SystemExit(f"{game_id} is not in the log - run `picks` first")
    if row["bet"] != "UNDER":
        raise SystemExit(f"{game_id} is not a bet under rule {row['rule']}")
    if _now() >= _logged_kickoff(row):
        raise SystemExit(f"{game_id} has kicked off - a price logged now "
                         "proves nothing")
    if row["book_total"]:
        print(f"Replacing {row['book_total']} at {row['book_odds']} "
              f"({row['book'] or 'no book'}, {row['priced_at']}).")
    row.update(book=book, book_total=f"{total:g}", book_odds=str(odds),
               priced_at=_stamp(_now()))
    write_log(log)
    print(f"{game_id}: under {total:g} at {odds} {book}. Commit the log.")


# --- grade -----------------------------------------------------------------

def _payout(odds: int) -> float:
    """Units won on a 1-unit stake at American odds."""
    return 100 / -odds if odds < 0 else odds / 100


def grade_under(points: int, line: float, odds: int) -> tuple[str, float]:
    if points == line:
        return "P", 0.0
    return ("W", _payout(odds)) if points < line else ("L", -1.0)


def grade() -> None:
    """Grade every logged game that is final and has a closing total."""
    log = read_log()
    if not log:
        print("Nothing logged yet - run `picks` first.")
        return
    games = {g["nflverse_game_id"]: g for g in load_schedule()}
    final = [(r, games[r["game_id"]]) for r in log
             if not r["graded_at"] and r["game_id"] in games
             and games[r["game_id"]]["home_score"] is not None
             and games[r["game_id"]]["total_close"] is not None]

    if final:
        archive = _fetch([g for _, g in final], ARCHIVE_API,
                         ARCHIVE_HOURLY, ARCHIVE_MODELS)
        now = _stamp(_now())
        for r, g in final:
            points = g["home_score"] + g["away_score"]
            r["points"] = str(points)
            r["total_close"] = f"{g['total_close']:g}"
            r["archive_jma_mph"] = _fmt(archive[r["game_id"]].get(ARCHIVE_JMA))
            if r["bet"] == "UNDER":
                result, units = grade_under(points, g["total_close"],
                                            CLOSE_ODDS)
                r["close_result"], r["close_units"] = result, f"{units:+.3f}"
                if r["book_total"]:
                    result, units = grade_under(
                        points, float(r["book_total"]), int(r["book_odds"]))
                    r["book_result"], r["book_units"] = result, f"{units:+.3f}"
            r["graded_at"] = now
        write_log(log)
        print(f"Graded {len(final)} games. Commit the log.")
    summarize(log)


def _record(rows: list[dict], result: str, units: str) -> str:
    graded = [r for r in rows if r[result]]
    if not graded:
        return f"{0:>6}{'-':>10}{'-':>8}{'-':>9}"
    w, l, p = (sum(r[result] == x for r in graded) for x in "WLP")
    hit = f"{w / (w + l):.1%}" if w + l else "-"
    total = sum(float(r[units]) for r in graded)
    return f"{len(graded):>6}{f'{w}-{l}-{p}':>10}{hit:>8}{total:>+9.2f}"


def summarize(log: list[dict]) -> None:
    bets = [r for r in log if r["bet"] == "UNDER"]
    waiting = sum(not r["graded_at"] for r in log)
    print(f"\nRULE {RULE}, {SEASON}: games logged {len(log)}, bets "
          f"{len(bets)}, waiting on a final score {waiting}")
    print(f"\n{'':<22}{'BETS':>6}{'W-L-P':>10}{'HIT':>8}{'UNITS':>9}")
    print(f"{'vs close at -110':<22}"
          f"{_record(bets, 'close_result', 'close_units')}")
    print(f"{'vs logged price':<22}"
          f"{_record(bets, 'book_result', 'book_units')}")
    print(f"\nBreak-even at -110 is {BREAK_EVEN:.1%}.")

    pairs = [(float(r["jma_mph"]), float(r["archive_jma_mph"])) for r in log
             if r["jma_mph"] and r["archive_jma_mph"]]
    if pairs:
        gap = sum(live - arch for live, arch in pairs) / len(pairs)
        size = sum(abs(live - arch) for live, arch in pairs) / len(pairs)
        flips = sum(is_bet(live) != is_bet(arch) for live, arch in pairs)
        print(f"\nAudit, live JMA pull vs the archived day-ahead series the "
              f"backtest used ({len(pairs)} games): mean gap {gap:+.2f} mph, "
              f"mean |gap| {size:.2f} mph, {flips} would flip the bet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("picks", help="log the outdoor games 18-36h out")
    p.add_argument("--dry-run", action="store_true",
                   help="show the picks, log nothing")
    p.add_argument("--at", type=_et,
                   help="pretend it is this ET time, e.g. 2026-10-28T20:00 "
                        "(implies --dry-run; the forecast is still today's)")

    p = sub.add_parser("price", help="record a book's total and price on a "
                                     "bet, before kickoff")
    p.add_argument("game_id")
    p.add_argument("total", type=float)
    p.add_argument("odds", type=int)
    p.add_argument("--book", default="")

    sub.add_parser("grade", help="grade the finished games, print the record")

    args = parser.parse_args()
    if args.command == "picks":
        picks(args.at, args.dry_run or args.at is not None)
    elif args.command == "price":
        price(args.game_id, args.total, args.odds, args.book)
    else:
        grade()
