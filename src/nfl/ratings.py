"""Power ratings, a multi-season walk-forward backtest, and market tests.

    python -m src.nfl.ratings

Stage 1 fitted this on one season because that was all the Fact Book
gave. With nflverse loaded there are ~7,300 played games back to 1999,
each with a closing spread, so the model can finally be asked the only
question that matters for betting: not "does it pick winners" but "does
it beat the number".

The backtest is strictly walk-forward across the whole history - week N
of season S is predicted from ratings that have seen only earlier games -
and ratings are regressed toward the mean between seasons.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .. import config
from . import db

# --- Model constants -----------------------------------------------------
BASE_RATING = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 48.0
POINTS_PER_ELO = 25.0
CARRYOVER = 2 / 3

# Standard deviation of NFL game margin around the spread, used to turn a
# point spread into a win probability. ~13.5 is the long-run figure.
MARGIN_SD = 13.5

# A -110 bet needs this hit rate to break even: 110 / (110 + 100).
BREAK_EVEN = 110 / 210


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class Prediction:
    game_id: str
    season: int
    week: int
    home: str
    away: str
    p_home: float
    pred_margin: float           # modelled home margin, in points
    home_score: int | None
    away_score: int | None
    spread_close: float | None   # sportsbook convention: negative = home favoured
    total_close: float | None

    @property
    def actual_margin(self) -> int | None:
        if self.home_score is None:
            return None
        return self.home_score - self.away_score

    @property
    def home_won(self) -> bool | None:
        m = self.actual_margin
        return None if m is None or m == 0 else m > 0

    @property
    def market_margin(self) -> float | None:
        """The market's implied home margin."""
        return None if self.spread_close is None else -self.spread_close


class Elo:
    """Standard Elo with a margin-of-victory multiplier."""

    def __init__(self, k: float = K_FACTOR, hfa: float = HOME_ADVANTAGE,
                 base: float = BASE_RATING):
        self.k, self.hfa, self.base = k, hfa, base
        self.ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.ratings.setdefault(team, self.base)

    def edge(self, home: str, away: str, neutral: bool = False) -> float:
        return self.rating(home) - self.rating(away) + (0 if neutral else self.hfa)

    def expect(self, home: str, away: str, neutral: bool = False) -> float:
        return 1.0 / (1.0 + 10 ** (-self.edge(home, away, neutral) / 400.0))

    def update(self, home: str, away: str, home_score: int, away_score: int,
               neutral: bool = False) -> None:
        expected = self.expect(home, away, neutral)
        margin = home_score - away_score
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5

        edge = self.edge(home, away, neutral)
        winner_edge = edge if margin > 0 else -edge
        mov = math.log(abs(margin) + 1) * (2.2 / (winner_edge * 0.001 + 2.2))

        delta = self.k * mov * (actual - expected)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta

    def regress_to_mean(self, carryover: float = CARRYOVER) -> None:
        for team, r in self.ratings.items():
            self.ratings[team] = self.base + (r - self.base) * carryover


# --- Backtest ------------------------------------------------------------

def load_games(conn, season_from: int = 1999, season_to: int = 2026) -> list[dict]:
    rows = conn.execute("""
        SELECT game_id, season, week, game_type, home_team, away_team, neutral,
               home_score, away_score, played, spread_close, total_close
          FROM games
         WHERE season BETWEEN ? AND ? AND week IS NOT NULL
         ORDER BY season, week, game_id
    """, (season_from, season_to)).fetchall()
    return [dict(r) for r in rows]


def walk_forward(games: list[dict], model: Elo) -> list[Prediction]:
    """Predict every game from ratings that have seen only earlier games."""
    predictions: list[Prediction] = []
    seasons = sorted({g["season"] for g in games})

    for si, season in enumerate(seasons):
        if si:
            model.regress_to_mean()
        season_games = [g for g in games if g["season"] == season]

        for week in sorted({g["week"] for g in season_games}):
            slate = [g for g in season_games if g["week"] == week]
            # Predict the full week before applying any of its results.
            for g in slate:
                neutral = bool(g["neutral"])
                predictions.append(Prediction(
                    game_id=g["game_id"], season=season, week=week,
                    home=g["home_team"], away=g["away_team"],
                    p_home=model.expect(g["home_team"], g["away_team"], neutral),
                    pred_margin=model.edge(g["home_team"], g["away_team"], neutral)
                    / POINTS_PER_ELO,
                    home_score=g["home_score"], away_score=g["away_score"],
                    spread_close=g["spread_close"], total_close=g["total_close"],
                ))
            for g in slate:
                if g["home_score"] is not None:
                    model.update(g["home_team"], g["away_team"],
                                 g["home_score"], g["away_score"],
                                 bool(g["neutral"]))
    return predictions


# --- Scoring -------------------------------------------------------------

def _score(name: str, graded: list[Prediction],
           margin_of, prob_of) -> dict:
    """Accuracy, calibration and margin error for one set of forecasts."""
    if not graded:
        return {}
    probs = [prob_of(p) for p in graded]
    margins = [margin_of(p) for p in graded]
    correct = sum((pr > 0.5) == p.home_won for pr, p in zip(probs, graded))
    return {
        "name": name,
        "games": len(graded),
        "accuracy": correct / len(graded),
        "brier": sum((pr - float(p.home_won)) ** 2
                     for pr, p in zip(probs, graded)) / len(graded),
        "log_loss": -sum(math.log(max(pr if p.home_won else 1 - pr, 1e-12))
                         for pr, p in zip(probs, graded)) / len(graded),
        "margin_mae": sum(abs(m - p.actual_margin)
                          for m, p in zip(margins, graded)) / len(graded),
    }


def compare(predictions: list[Prediction], season_from: int) -> list[dict]:
    """Model vs market vs a naive home pick, on identical games."""
    graded = [p for p in predictions
              if p.home_won is not None and p.season >= season_from
              and p.spread_close is not None]
    if not graded:
        return []

    home_rate = sum(p.home_won for p in graded) / len(graded)
    return [
        _score("Elo", graded, lambda p: p.pred_margin, lambda p: p.p_home),
        _score("Closing spread (market)", graded,
               lambda p: p.market_margin,
               lambda p: normal_cdf(p.market_margin / MARGIN_SD)),
        _score("Always pick home", graded,
               lambda p: 0.0, lambda p: home_rate),
    ]


def against_the_spread(predictions: list[Prediction], season_from: int,
                       min_edge: float = 0.0) -> dict:
    """Grade the model's picks against the closing spread.

    `min_edge` filters to games where the model disagrees with the market
    by at least that many points - the usual way a bettor would select
    plays rather than betting every game.
    """
    wins = losses = pushes = 0
    edges: list[float] = []

    for p in predictions:
        if (p.season < season_from or p.actual_margin is None
                or p.market_margin is None):
            continue
        edge = p.pred_margin - p.market_margin
        if abs(edge) < min_edge:
            continue
        edges.append(edge)

        # Bet the side the model likes relative to the number.
        cover = p.actual_margin - p.market_margin
        if cover == 0:
            pushes += 1
        elif (cover > 0) == (edge > 0):
            wins += 1
        else:
            losses += 1

    graded = wins + losses
    if not graded:
        return {"graded": 0}

    hit = wins / graded
    se = math.sqrt(BREAK_EVEN * (1 - BREAK_EVEN) / graded)
    # Profit in units, betting 1 unit per game at -110.
    roi = (wins * (100 / 110) - losses) / graded

    return {
        "graded": graded, "wins": wins, "losses": losses, "pushes": pushes,
        "hit_rate": hit, "roi": roi,
        "z": (hit - BREAK_EVEN) / se,
        "mean_abs_edge": sum(abs(e) for e in edges) / len(edges),
    }


# --- Persistence ---------------------------------------------------------

def save_predictions(conn, model: str, predictions: list[Prediction]) -> int:
    """Write one model's walk-forward forecasts to the predictions table.

    Rows for `model` are cleared first, so re-running replaces a model's
    forecasts rather than leaving a mix of two vintages behind. Other
    models are untouched, which is what lets 'elo' and 'elo+situational'
    live in the table side by side.

    This exists so the assistant can answer "who wins in week 1, and by
    how much" with a SELECT. Recomputing a 7,500-game walk-forward for
    every question would work and would be absurd.
    """
    built_at = datetime.now(timezone.utc).isoformat()
    rows = [{
        "game_id": p.game_id,
        "model": model,
        "p_home": p.p_home,
        "pred_margin": p.pred_margin,
        "market_margin": p.market_margin,
        "edge": (None if p.market_margin is None
                 else p.pred_margin - p.market_margin),
        "built_at": built_at,
    } for p in predictions]

    with conn:
        conn.execute("DELETE FROM predictions WHERE model = ?", (model,))
        return db.replace_all(conn, "predictions", rows, [
            "game_id", "model", "p_home", "pred_margin",
            "market_margin", "edge", "built_at",
        ])


# --- CLI -----------------------------------------------------------------

HOLDOUT_FROM = 2010   # ratings have long since warmed up by here


def _print_table(rows: list[dict]) -> None:
    print(f"{'FORECAST':<26}{'GAMES':>7}{'ACCURACY':>11}{'BRIER':>9}"
          f"{'LOGLOSS':>10}{'MAE':>9}")
    print("-" * 72)
    for m in rows:
        print(f"{m['name']:<26}{m['games']:>7}{m['accuracy']:>11.1%}"
              f"{m['brier']:>9.4f}{m['log_loss']:>10.4f}{m['margin_mae']:>9.2f}")
    print("-" * 72)


def run() -> None:
    conn = db.connect(config.NFL_DB)
    games = load_games(conn)
    model = Elo()
    predictions = walk_forward(games, model)

    saved = save_predictions(conn, "elo", predictions)

    played = sum(1 for p in predictions if p.actual_margin is not None)
    print(f"Saved {saved:,} walk-forward forecasts to `predictions` "
          f"(model 'elo').\n")
    print(f"Walk-forward backtest over {played:,} played games, 1999-2025")
    print(f"Scored on {HOLDOUT_FROM}-2025, where a closing spread exists.\n")
    _print_table(compare(predictions, HOLDOUT_FROM))
    print("Lower is better for Brier, log loss and MAE.")

    # --- The honest test --------------------------------------------------
    print("\n\n" + "=" * 72)
    print(f"AGAINST THE CLOSING SPREAD ({HOLDOUT_FROM}-2025)")
    print("=" * 72)
    print(f"{'FILTER':<28}{'BETS':>7}{'W-L-P':>14}{'HIT':>8}"
          f"{'ROI':>9}{'Z':>7}")
    print("-" * 72)
    for min_edge in (0.0, 1.0, 2.0, 3.0, 5.0, 7.0):
        r = against_the_spread(predictions, HOLDOUT_FROM, min_edge)
        if not r["graded"]:
            continue
        label = "every game" if not min_edge else f"model disagrees by {min_edge:g}+"
        wlp = f"{r['wins']}-{r['losses']}-{r['pushes']}"
        print(f"{label:<28}{r['graded']:>7}{wlp:>14}{r['hit_rate']:>8.1%}"
              f"{r['roi']:>+9.1%}{r['z']:>7.2f}")
    print("-" * 72)
    print(f"Break-even at -110 is {BREAK_EVEN:.1%}. Z is standard deviations")
    print("above that break-even, so |Z| under ~2 is indistinguishable from luck.")

    # --- 2026 -------------------------------------------------------------
    final = dict(sorted(model.ratings.items(), key=lambda kv: -kv[1])[:8])
    print("\n\nTop of the ratings entering 2026 (post-2025 season, regressed)")
    print("-" * 48)
    names = {r["team_id"]: r["full_name"]
             for r in conn.execute("SELECT team_id, full_name FROM teams")}
    for i, (team, rating) in enumerate(final.items(), 1):
        print(f"{i:>3}. {names.get(team, team):<28}{rating:>8.0f}")

    upcoming = [p for p in predictions
                if p.season == 2026 and p.spread_close is not None]
    print(f"\n\nBiggest disagreements with the posted 2026 lines "
          f"({len(upcoming)} games priced)")
    print("-" * 72)
    print(f"{'WEEK':>5}  {'MATCHUP':<20}{'MODEL':>9}{'MARKET':>9}{'EDGE':>8}")
    print("-" * 72)
    ranked = sorted(upcoming, key=lambda p: -abs(p.pred_margin - p.market_margin))
    for p in ranked[:12]:
        edge = p.pred_margin - p.market_margin
        print(f"{p.week:>5}  {p.away + ' @ ' + p.home:<20}"
              f"{p.pred_margin:>+9.1f}{p.market_margin:>+9.1f}{edge:>+8.1f}")
    print("-" * 72)
    print("Margins are from the home club's perspective. A large edge here")
    print("means the model disagrees with the market - which, on the evidence")
    print("above, usually means the model is wrong.")

    conn.close()


if __name__ == "__main__":
    run()
