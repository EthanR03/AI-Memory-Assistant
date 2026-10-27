"""Stage 3b: is the wind edge there on wind anyone could have KNOWN?

    python -m src.nfl.forecast_wind              # fetches once, then cached
    python -m src.nfl.forecast_wind --offline    # cache only, no network

Stage 3 found the closing total leaves about -0.24 points per mph of wind
on the table, and the slope survived an out-of-sample split. Its own
caveat is why this module exists: `wind` is the reading AT KICKOFF, and
the market prices a forecast. If the edge is "games that turned out
windier than forecast", it is real and unplayable.

So the test is re-run on archived forecasts from Open-Meteo's Previous
Runs API, which keeps what each model said N days before a given hour.
Four wind numbers per game, each a mean over the kickoff hour and the
three after it:

  kickoff   nflverse's reading - what Stage 3 used. Hindsight.
  short     Open-Meteo's default model stitched from the first hours of
            each run. This is the Historical Forecast API's series (the
            two were checked equal); knowable a few hours out, about when
            the closing total is struck.
  day1_gfs  The default model's forecast made 24h earlier. The archive
            starts 2024-01-19, so this is two seasons.
  day1_jma  JMA GSM's forecast made 24h earlier. Coarse (~0.5 degree
            grid) but archived from 2018 - eight seasons instead of two.

Grid wind at 10m is not on the same scale as a stadium reading, so a raw
slope per forecast mph is not comparable with Stage 3's -0.24. The test
that decides it rescales each forecast onto kickoff mph (regress the
reading on the forecast), splits the reading into that KNOWABLE part and
the SURPRISE left over, and regresses the total's residual on both. If
the knowable slope carries the -0.24, the edge existed before kickoff.
If the surprise slope carries it, the lead is dead.
"""
import argparse
import json
import time
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .. import config
from . import db
from .ratings import BREAK_EVEN
from .situational import (_indoor, _print_bet_header, _print_bet_row,
                          _under_record, ols, wind_mph)

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODELS = "best_match,jma_seamless"
HOURLY = ("wind_speed_10m,wind_speed_10m_previous_day1,"
          "temperature_2m,temperature_2m_previous_day1")

SEASON_FROM = 2018    # JMA's day-ahead archive starts here
SEASON_TO = 2025
GAME_HOURS = 4        # kickoff hour plus the three after it

# Every venue an outdoor (or open-roof) game was played at since 2018,
# keyed by nflverse's `stadium` name - not `stadium_id`, which follows the
# home club (JAX00 is listed at Wembley). Renamings share coordinates.
STADIUMS = {
    "Acrisure Stadium": (40.4468, -80.0158),
    "Heinz Field": (40.4468, -80.0158),
    "Allianz Arena": (48.2188, 11.6247),
    "Arena Corinthians": (-23.5453, -46.4742),
    "Arrowhead Stadium": (39.0489, -94.4839),
    "GEHA Field at Arrowhead Stadium": (39.0489, -94.4839),
    "AT&T Stadium": (32.7473, -97.0945),
    "Azteca Stadium": (19.3029, -99.1505),
    "Estadio Banorte": (19.3029, -99.1505),
    "Bank of America Stadium": (35.2258, -80.8528),
    "Bernabeu": (40.4531, -3.6883),
    "CenturyLink Field": (47.5952, -122.3316),
    "Lumen Field": (47.5952, -122.3316),
    "Deutsche Bank Park": (50.0686, 8.6455),
    "Empower Field at Mile High": (39.7439, -105.0201),
    "Sports Authority Field at Mile High": (39.7439, -105.0201),
    "FedExField": (38.9078, -76.8645),
    "Northwest Stadium": (38.9078, -76.8645),
    "FirstEnergy Stadium": (41.5061, -81.6995),
    "Huntington Bank Field": (41.5061, -81.6995),
    "Gillette Stadium": (42.0909, -71.2643),
    "Hard Rock Stadium": (25.9580, -80.2389),
    "Lambeau Field": (44.5013, -88.0622),
    "Levi's Stadium": (37.4030, -121.9700),
    "Lincoln Financial Field": (39.9008, -75.1675),
    "Los Angeles Memorial Coliseum": (34.0141, -118.2879),
    "Lucas Oil Stadium": (39.7601, -86.1639),
    "M&T Bank Stadium": (39.2780, -76.6227),
    "Maracana Stadium": (-22.9122, -43.2302),
    "Mercedes-Benz Stadium": (33.7554, -84.4008),
    "MetLife Stadium": (40.8135, -74.0745),
    "New Era Field": (42.7738, -78.7870),
    "Highmark Stadium": (42.7738, -78.7870),
    "Nissan Stadium": (36.1665, -86.7713),
    "NRG Stadium": (29.6847, -95.4107),
    "Reliant Stadium": (29.6847, -95.4107),
    "Oakland-Alameda County Coliseum": (37.7516, -122.2005),
    "Ring Central Coliseum": (37.7516, -122.2005),
    "Paul Brown Stadium": (39.0955, -84.5161),
    "Paycor Stadium": (39.0955, -84.5161),
    "Raymond James Stadium": (27.9759, -82.5033),
    "Soldier Field": (41.8623, -87.6167),
    "State Farm Stadium": (33.5276, -112.2626),
    "StubHub Center": (33.8644, -118.2611),
    "TIAA Bank Stadium": (30.3239, -81.6373),
    "EverBank Stadium": (30.3239, -81.6373),
    "Tottenham Stadium": (51.6043, -0.0664),
    "Tottenham Hotspur Stadium": (51.6043, -0.0664),
    "Wembley Stadium": (51.5560, -0.2796),
}


# --- Measures ------------------------------------------------------------

@dataclass(frozen=True)
class Measure:
    name: str
    label: str
    wind: str    # key into a game row
    temp: str


KICKOFF = Measure("kickoff", "kickoff reading (Stage 3)", "wind", "temp")
FORECASTS = [
    Measure("short", "short-lead forecast (0-6h)",
            "wind_speed_10m_best_match", "temperature_2m_best_match"),
    Measure("day1_gfs", "24h-ahead, default model",
            "wind_speed_10m_previous_day1_best_match",
            "temperature_2m_previous_day1_best_match"),
    Measure("day1_jma", "24h-ahead, JMA GSM",
            "wind_speed_10m_previous_day1_jma_seamless",
            "temperature_2m_previous_day1_jma_seamless"),
]


# --- Data ----------------------------------------------------------------

def load_sample(conn) -> list[dict]:
    """Played, priced, outdoor games - the population of Stage 3's test B."""
    rows = conn.execute("""
        SELECT game_id, season, game_date, gametime, stadium, roof,
               home_score, away_score, total_close, div_game, temp, wind
          FROM games
         WHERE season BETWEEN ? AND ? AND played = 1
           AND home_score IS NOT NULL AND total_close IS NOT NULL
         ORDER BY game_date, game_id
    """, (SEASON_FROM, SEASON_TO)).fetchall()
    games = [dict(r) for r in rows if not _indoor(dict(r))]
    unknown = sorted({g["stadium"] for g in games} - STADIUMS.keys(), key=str)
    if unknown:
        raise KeyError(f"no coordinates for {unknown} - add them to STADIUMS")
    return games


def _cache_path(season: int):
    return config.OPENMETEO_DIR / f"previous_runs_{season}.json"


def _request(params: dict) -> list[dict]:
    import requests

    for attempt in range(4):
        response = requests.get(API, params=params, timeout=60)
        if response.status_code == 429:          # minutely limit; wait it out
            time.sleep(60 * (attempt + 1))
            continue
        response.raise_for_status()
        body = response.json()
        return body if isinstance(body, list) else [body]
    raise RuntimeError("Open-Meteo kept rate-limiting; rerun later")


def fetch(games: list[dict], offline: bool = False) -> dict[str, dict]:
    """Hourly forecasts for every (game date, stadium), cached per season.

    One request per game date covers every stadium playing that day, so
    eight seasons is a few hundred calls - well inside the free tier. The
    window runs to the next day so a 22:20 kickoff still has four hours.
    """
    out: dict[str, dict] = {}
    for season in sorted({g["season"] for g in games}):
        path = _cache_path(season)
        cache = json.loads(path.read_text()) if path.exists() else {}
        todo: dict[str, set[str]] = {}
        for g in games:
            if g["season"] != season:
                continue
            key = f"{g['game_date']}|{g['stadium']}"
            if key not in cache:
                todo.setdefault(g["game_date"], set()).add(g["stadium"])

        if todo and offline:
            print(f"  {season}: {sum(map(len, todo.values()))} venue-days "
                  "not cached; skipped (--offline)")
        for i, (day, stadiums) in enumerate(sorted(todo.items())):
            names = sorted(stadiums)
            end = date.fromisoformat(day) + timedelta(days=1)
            if offline:
                break
            results = _request({
                "latitude": ",".join(str(STADIUMS[s][0]) for s in names),
                "longitude": ",".join(str(STADIUMS[s][1]) for s in names),
                "start_date": day, "end_date": end.isoformat(),
                "hourly": HOURLY, "models": MODELS,
                "wind_speed_unit": "mph", "temperature_unit": "fahrenheit",
                "timezone": "America/New_York",   # nflverse gametime is ET
            })
            for name, result in zip(names, results):
                cache[f"{day}|{name}"] = result["hourly"]
            if i % 20 == 19:
                _save(path, cache)
            time.sleep(0.25)
        if todo and not offline:
            _save(path, cache)
            print(f"  {season}: fetched {len(todo)} game dates")
        out.update(cache)
    return out


def _save(path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))


def attach(games: list[dict], hourly: dict[str, dict]) -> None:
    """Average each forecast series over the game window, onto the row.

    A series with any missing hour in the window is None for that game -
    which is how `day1_gfs` drops out before 2024-01-19.
    """
    for g in games:
        # Stage 3's sanity rules for the reading itself (71 mph etc.).
        g["wind"] = wind_mph(g)
        h = hourly.get(f"{g['game_date']}|{g['stadium']}")
        kickoff = f"{g['game_date']}T{int(g['gametime'][:2]):02d}:00"
        if h is None or kickoff not in h["time"]:
            continue
        start = h["time"].index(kickoff)
        for key, series in h.items():
            if key == "time":
                continue
            window = series[start:start + GAME_HOURS]
            if len(window) == GAME_HOURS and None not in window:
                g[key] = sum(window) / GAME_HOURS


def _residual(g: dict) -> float:
    return (g["home_score"] + g["away_score"]) - g["total_close"]


def _has(g: dict, *keys: str) -> bool:
    return all(g.get(k) is not None for k in keys)


# --- Tests ---------------------------------------------------------------

def _slope(rows: list[dict], wind: str, temp: str) -> tuple[float, float, float]:
    """Wind coefficient of residual ~ 1 + wind + temp: (beta, se, t)."""
    X = np.array([[1.0, g[wind], g[temp]] for g in rows])
    y = np.array([_residual(g) for g in rows])
    fit = ols(X, y, ["intercept", "wind", "temp"])
    return float(fit.beta[1]), float(fit.se[1]), float(fit.t[1])


def coverage(games: list[dict]) -> None:
    print(f"{'SEASON':<8}{'GAMES':>7}" + "".join(
        f"{m.name:>11}" for m in [KICKOFF] + FORECASTS))
    print("-" * 59)
    for season in sorted({g["season"] for g in games}):
        rows = [g for g in games if g["season"] == season]
        print(f"{season:<8}{len(rows):>7}" + "".join(
            f"{sum(_has(g, m.wind, m.temp) for g in rows):>11}"
            for m in [KICKOFF] + FORECASTS))
    print("-" * 59)


def test_slopes(games: list[dict]) -> None:
    """Each wind number on its own, against the closing total's error."""
    print(f"\n{'WIND NUMBER':<28}{'N':>6}{'PTS/MPH':>9}{'SE':>7}{'T':>7}"
          f"{'PER SD':>8}   KICKOFF, SAME GAMES")
    print("-" * 90)
    for m in [KICKOFF] + FORECASTS:
        rows = [g for g in games if _has(g, m.wind, m.temp)]
        beta, se, t = _slope(rows, m.wind, m.temp)
        sd = float(np.std([g[m.wind] for g in rows]))
        same = ""
        if m is not KICKOFF:
            both = [g for g in rows if _has(g, "wind", "temp")]
            kb, _, kt = _slope(both, "wind", "temp")
            same = f"{kb:+.3f} (T {kt:.2f}, n={len(both)})"
        print(f"{m.label:<28}{len(rows):>6}{beta:>+9.3f}{se:>7.3f}{t:>7.2f}"
              f"{beta * sd:>+8.2f}   {same}")
    print("-" * 90)
    print("Model: actual total minus closing total ~ wind + temperature,")
    print("each from the same source. PER SD = points for a one-SD windier")
    print("game, the scale-free comparison across sources.")


def test_decomposition(games: list[dict]) -> None:
    """Split the kickoff reading into the part a forecast saw and the rest."""
    print(f"\n{'FORECAST':<28}{'N':>6}{'CORR':>6}{'KNOWABLE':>10}{'T':>7}"
          f"{'SURPRISE':>10}{'T':>7}")
    print("-" * 74)
    for m in FORECASTS:
        rows = [g for g in games if _has(g, m.wind, "wind", "temp")]
        f = np.array([g[m.wind] for g in rows])
        k = np.array([g["wind"] for g in rows])
        # Calibrate the forecast onto kickoff mph. This uses the reading,
        # never the score, so it cannot leak the outcome into the test.
        b, a = np.polyfit(f, k, 1)
        knowable = a + b * f
        surprise = k - knowable
        X = np.column_stack([np.ones(len(rows)), knowable, surprise,
                             [g["temp"] for g in rows]])
        y = np.array([_residual(g) for g in rows])
        fit = ols(X, y, ["intercept", "knowable", "surprise", "temp"])
        corr = float(np.corrcoef(f, k)[0, 1])
        print(f"{m.label:<28}{len(rows):>6}{corr:>6.2f}"
              f"{fit.beta[1]:>+10.3f}{fit.t[1]:>7.2f}"
              f"{fit.beta[2]:>+10.3f}{fit.t[2]:>7.2f}")
    print("-" * 74)
    print("Both slopes are in points per KICKOFF mph, so each is directly")
    print("comparable with Stage 3's -0.24. KNOWABLE is the wind the forecast")
    print("predicted; SURPRISE is what the reading added on the day.")


def test_unders(games: list[dict]) -> None:
    print()
    _print_bet_header()
    for m in [KICKOFF] + FORECASTS:
        rows = [g for g in games if _has(g, m.wind)]
        for threshold in (10, 12, 15):
            wins, losses, pushes = _under_record(
                rows, lambda g, w=m.wind, t=threshold: g[w] >= t)
            if wins + losses >= 30:
                _print_bet_row(f"{m.name} >= {threshold} mph",
                               wins, losses, pushes)
    print("-" * 72)
    print("Twelve rules on overlapping games, chosen without a holdout: read")
    print("this table for direction only. Forecast mph runs on a different")
    print("scale from the reading, so equal thresholds are not equal games.")


def robustness(games: list[dict], m: Measure) -> None:
    """The obvious ways a wind slope could be something else.

    Windy venues are also cold, northern, and home to particular offences,
    so a slope across stadiums could be "Soldier Field goes under". Venue
    fixed effects keep only the wind variation WITHIN a stadium. HC1
    errors guard against windy games simply being noisier.
    """
    rows = [g for g in games if _has(g, m.wind, m.temp)]
    y = np.array([_residual(g) for g in rows])
    base = np.array([[1.0, g[m.wind], g[m.temp]] for g in rows])

    def slope(X: np.ndarray, keep=None) -> tuple[float, float, float]:
        Xk, yk = (X, y) if keep is None else (X[keep], y[keep])
        fit = ols(Xk, yk, ["intercept", "wind"] + ["x"] * (X.shape[1] - 2))
        resid = yk - Xk @ fit.beta
        n, k = Xk.shape
        bread = np.linalg.pinv(Xk.T @ Xk)
        hc1 = bread @ (Xk.T * resid ** 2) @ Xk @ bread * n / (n - k)
        return float(fit.beta[1]), float(fit.t[1]), \
            float(fit.beta[1] / np.sqrt(hc1[1, 1]))

    # Venues, not names: Heinz Field and Acrisure Stadium are one ground.
    venue = [STADIUMS[g["stadium"]] for g in rows]
    venues = sorted(set(venue))
    fixed = np.array([[float(v == u) for u in venues[1:]] for v in venue])

    print(f"\n{'SPECIFICATION':<40}{'PTS/MPH':>9}{'T':>7}{'HC1 T':>8}")
    print("-" * 64)
    b, t, robust = slope(base)
    print(f"{'wind + temperature':<40}{b:>+9.3f}{t:>7.2f}{robust:>8.2f}")
    b, t, robust = slope(np.column_stack([base, fixed]))
    print(f"{f'+ {len(venues)} venue fixed effects':<40}"
          f"{b:>+9.3f}{t:>7.2f}{robust:>8.2f}")
    drops = sorted((slope(base, np.array([v != u for v in venue]))[0], u)
                   for u in venues)
    print("-" * 64)
    print(f"Leaving out one venue at a time, the slope runs "
          f"{drops[0][0]:+.3f} to {drops[-1][0]:+.3f}.")


def by_season(games: list[dict], m: Measure, threshold: float = 12) -> None:
    """Does the slope hold up year to year, or is it one season?

    The bet column reuses Stage 3's frozen threshold rather than choosing
    a new one on these games.
    """
    print(f"\n{'SEASON':<8}{'N':>6}{'PTS/MPH':>9}{'T':>7}"
          f"{f'UNDER >= {threshold:g}':>16}{'HIT':>8}")
    print("-" * 54)
    winning = 0
    seasons = sorted({g["season"] for g in games})
    for season in seasons:
        rows = [g for g in games
                if g["season"] == season and _has(g, m.wind, m.temp)]
        beta, _, t = _slope(rows, m.wind, m.temp)
        wins, losses, pushes = _under_record(
            rows, lambda g: g[m.wind] >= threshold)
        hit = wins / (wins + losses)
        winning += hit > BREAK_EVEN
        print(f"{season:<8}{len(rows):>6}{beta:>+9.3f}{t:>7.2f}"
              f"{f'{wins}-{losses}-{pushes}':>16}{hit:>8.1%}")
    print("-" * 54)
    print(f"{winning} of {len(seasons)} seasons beat {BREAK_EVEN:.1%}.")


# --- CLI -----------------------------------------------------------------

def run(offline: bool = False) -> None:
    conn = db.connect(config.NFL_DB)
    games = load_sample(conn)
    conn.close()
    print(f"Fetching archived forecasts for {len(games):,} outdoor games "
          f"({SEASON_FROM}-{SEASON_TO})...")
    attach(games, fetch(games, offline))

    print("\n" + "=" * 72)
    print("COVERAGE: games with each wind number (and its temperature)")
    print("=" * 72)
    coverage(games)

    print("\n\n" + "=" * 72)
    print("1. THE WIND SLOPE, ONE SOURCE AT A TIME")
    print("=" * 72)
    test_slopes(games)

    print("\n\n" + "=" * 72)
    print("2. KNOWABLE OR SURPRISE? (the test that decides it)")
    print("=" * 72)
    test_decomposition(games)

    print("\n\n" + "=" * 72)
    print("3. BETTING THE UNDER AT -110 BY WIND THRESHOLD (in-sample)")
    print("=" * 72)
    test_unders(games)

    jma = next(m for m in FORECASTS if m.name == "day1_jma")
    print("\n\n" + "=" * 72)
    print(f"4. {jma.label.upper()}: COULD IT BE SOMETHING OTHER THAN WIND?")
    print("=" * 72)
    robustness(games, jma)

    print("\n\n" + "=" * 72)
    print(f"5. {jma.label.upper()}, SEASON BY SEASON")
    print("=" * 72)
    by_season(games, jma)

    print(f"\nBreak-even at -110 is {BREAK_EVEN:.1%}. "
          "|Z| and |T| under ~2 are noise.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true",
                        help="use cached forecasts only")
    run(parser.parse_args().offline)
