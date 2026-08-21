"""nflverse client: download and normalise games.csv.

Source: https://github.com/nflverse/nfldata (Lee Sharpe's game data, the
file nflreadr::load_schedules() reads). One row per game, 1999 to the
current season, carrying the closing spread, total, moneylines, rest
days, roof, surface and weather.

This is the file that closes most of the gaps the Fact Book left: 27
seasons of results instead of one, and a market price to measure against.
"""
import csv
from datetime import date
from pathlib import Path

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

# nflverse abbreviations that differ from this project's ids. The three
# relocations are mapped to the current franchise because a ratings model
# wants continuity - the 2015 Rams and the 2025 Rams are one club.
TEAM_MAP = {
    "LA": "LAR",    # nflverse writes the Rams as LA, not LAR
    "STL": "LAR",   # St. Louis Rams, through 2015
    "OAK": "LV",    # Oakland Raiders, through 2019
    "SD": "LAC",    # San Diego Chargers, through 2016
}

# nflverse game_type -> (our game_type, round label)
GAME_TYPES = {
    "REG": ("REG", None),
    "WC": ("POST", "WC"),
    "DIV": ("POST", "DIV"),
    "CON": ("POST", "CON"),
    "SB": ("POST", "SB"),
}


def download(dest: Path, force: bool = False) -> Path:
    """Fetch games.csv unless it is already on disk."""
    if dest.exists() and not force:
        return dest
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(GAMES_URL, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def team_id(abbr: str) -> str:
    return TEAM_MAP.get(abbr, abbr)


def _num(text: str, cast=float):
    text = (text or "").strip()
    if not text or text.upper() in {"NA", "NULL"}:
        return None
    try:
        return cast(text)
    except ValueError:
        return None


def _clean(text: str) -> str | None:
    # The surface column carries stray trailing spaces ("grass ").
    text = (text or "").strip()
    return text or None


def load_rows(path: Path) -> list[dict]:
    """Read games.csv into normalised dicts keyed the way our store is."""
    games: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            kind = GAME_TYPES.get(r["game_type"])
            if kind is None:
                continue
            game_type, round_label = kind

            spread_line = _num(r["spread_line"])
            gameday = _num(r["gameday"], str)

            games.append({
                "nflverse_game_id": r["game_id"],
                "season": int(r["season"]),
                "week": int(r["week"]),
                "game_type": game_type,
                "round": round_label,
                "game_date": (date.fromisoformat(gameday).isoformat()
                              if gameday else None),
                "gametime": _clean(r["gametime"]),
                "home_team": team_id(r["home_team"]),
                "away_team": team_id(r["away_team"]),
                "neutral": int(r["location"].strip().lower() == "neutral"),
                "home_score": _num(r["home_score"], int),
                "away_score": _num(r["away_score"], int),
                "overtime": _num(r["overtime"], int) or 0,
                "played": int(bool(r["home_score"].strip())),

                # Sign flip: nflverse quotes a positive spread_line when the
                # home club is favoured; this store uses the sportsbook
                # convention where the home favourite is negative.
                "spread_close": -spread_line if spread_line is not None else None,
                "total_close": _num(r["total_line"]),
                "ml_home": _num(r["home_moneyline"], int),
                "ml_away": _num(r["away_moneyline"], int),
                "spread_odds_home": _num(r["home_spread_odds"], int),
                "spread_odds_away": _num(r["away_spread_odds"], int),
                "over_odds": _num(r["over_odds"], int),
                "under_odds": _num(r["under_odds"], int),

                "home_rest": _num(r["home_rest"], int),
                "away_rest": _num(r["away_rest"], int),
                "roof": _clean(r["roof"]),
                "surface": _clean(r["surface"]),
                "temp": _num(r["temp"]),
                "wind": _num(r["wind"]),

                "div_game": _num(r["div_game"], int),
                "stadium": _clean(r["stadium"]),
                "home_qb": _clean(r["home_qb_name"]),
                "away_qb": _clean(r["away_qb_name"]),
                "home_qb_id": _clean(r["home_qb_id"]),
                "away_qb_id": _clean(r["away_qb_id"]),
                "home_coach": _clean(r["home_coach"]),
                "away_coach": _clean(r["away_coach"]),
                "referee": _clean(r["referee"]),
                "source": "nflverse",
            })
    return games
