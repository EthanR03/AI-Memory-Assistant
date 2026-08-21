"""Stage 3: do the situational columns carry anything the market missed?

    python -m src.nfl.situational

`home_rest`, `away_rest`, `div_game`, `roof`, `temp` and `wind` have been
sitting in `games` since Stage 2, loaded and unused. The obvious thing to
do with them is bolt them onto Elo and see if the model predicts better.
That is the least interesting question, because the closing spread has
already priced every one of them - a bookmaker knows which club is off a
bye and which game is being played in a gale.

So this module asks three questions, cheapest and sharpest first:

  A. Does the market MIS-price them? Regress the market's own residual
     (actual margin minus the spread's implied margin) on the situational
     features. An efficient market leaves nothing here: every coefficient
     should be indistinguishable from zero. Any that is not is a mispriced
     angle, and it is worth points per game rather than a rounding error.

  B. The same test on totals, where the folklore is loudest - wind is
     supposed to push games under, and the market is supposed to be slow
     to adjust. Same regression, plus a straight under-betting test at
     -110 by wind threshold.

  C. Only then, the predictive model: Elo margin plus situational terms,
     refit walk-forward each season on prior seasons only, scored on the
     same accuracy and against-the-spread tables `ratings.py` uses.

A and B are tests of the market. C is a test of the model. They can
disagree, and if they do, A and B are the ones to believe: C's edge is
measured against the same closing line that A and B examine directly.
"""
import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .. import config
from . import db, ratings
from .ratings import BREAK_EVEN, MARGIN_SD, Elo, Prediction, normal_cdf

HOLDOUT_FROM = ratings.HOLDOUT_FROM   # 2010; ratings have warmed up by here

# Thresholds. Rest is measured in days between kickoffs, so a Thursday
# game after the previous Sunday is 4 and a bye week is 13-14.
SHORT_REST = 4
BYE_REST = 13
COLD_F = 32.0
HIGH_WIND_MPH = 15.0

INDOOR_ROOFS = {"dome", "closed"}

# nflverse carries two readings that are not weather: 71 mph at Heinz
# Field in 2016 and 70 mph at Cincinnati in 2008, both for games that
# were played normally (the Pittsburgh game scored 38). Sustained wind
# that high stops a football game rather than affecting its total, so
# above this a reading is treated as missing rather than extreme. Real
# gusts of 35-44 mph do occur and are kept.
WIND_SANITY_MPH = 45.0

# The wind result in test B is the one finding here, so it is split:
# fitted on games before this season, then checked on everything after.
SPLIT_AT = 2018


# --- Features ------------------------------------------------------------

@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    of: Callable[[dict], float]   # game row -> value
    unit: str = "pts"             # what one unit of the coefficient means


def _indoor(g: dict) -> bool:
    return (g["roof"] or "").strip().lower() in INDOOR_ROOFS


def _outdoor_temp(g: dict) -> float | None:
    """Temperature, but only where it can matter: outdoors, and recorded."""
    return None if _indoor(g) else g["temp"]


def wind_mph(g: dict) -> float | None:
    """Wind, outdoors, where the reading is believable."""
    if _indoor(g) or g["wind"] is None or g["wind"] > WIND_SANITY_MPH:
        return None
    return float(g["wind"])


def _rest(g: dict, side: str) -> int:
    r = g[f"{side}_rest"]
    return 7 if r is None else int(r)


# Margin features are written from the HOME club's perspective, matching
# the sign of the residual they explain. Weather terms are indicators
# rather than raw degrees/mph so that a missing reading on an outdoor game
# (126 of 3,089 since 2010) reads as "not extreme" instead of dropping the
# row - and so the coefficient is in points, directly comparable to a
# spread.
MARGIN_FEATURES = [
    Feature("rest_diff", "rest advantage (days)",
            lambda g: _rest(g, "home") - _rest(g, "away"), "pts/day"),
    # Short rest is almost never one-sided: of 243 games since 2010 with a
    # club on four days or fewer, 242 have BOTH clubs there. It is the
    # Thursday game, not a rest edge, so separate home/away indicators are
    # the same column twice - fitting both put the VIF over 200 and the
    # standard errors at +/-13 points. One indicator, and `rest_diff`
    # carries what asymmetry there is.
    Feature("both_short", f"Thursday game (both <={SHORT_REST}d rest)",
            lambda g: float(_rest(g, "home") <= SHORT_REST
                            and _rest(g, "away") <= SHORT_REST)),
    Feature("home_bye", f"home off a bye (>={BYE_REST}d)",
            lambda g: float(_rest(g, "home") >= BYE_REST)),
    Feature("away_bye", f"away off a bye (>={BYE_REST}d)",
            lambda g: float(_rest(g, "away") >= BYE_REST)),
    Feature("div_game", "divisional game",
            lambda g: float(g["div_game"] or 0)),
    Feature("indoor", "indoors (dome or closed roof)",
            lambda g: float(_indoor(g))),
    Feature("cold", f"cold (<={COLD_F:.0f}F)",
            lambda g: float((_outdoor_temp(g) or 99) <= COLD_F)),
    Feature("windy", f"windy (>={HIGH_WIND_MPH:.0f}mph)",
            lambda g: float((wind_mph(g) or 0) >= HIGH_WIND_MPH)),
]

# Totals are symmetric - neither club's rest is "for" the over - so this
# set is about conditions, and it is fitted on outdoor games with a real
# reading, where raw degrees and mph are meaningful.
TOTAL_FEATURES = [
    Feature("wind_mph", "wind", lambda g: wind_mph(g), "pts/mph"),
    Feature("temp_f", "temperature", lambda g: float(g["temp"]), "pts/F"),
    Feature("windy", f"windy (>={HIGH_WIND_MPH:.0f}mph)",
            lambda g: float(wind_mph(g) >= HIGH_WIND_MPH)),
    Feature("cold", f"cold (<={COLD_F:.0f}F)",
            lambda g: float(g["temp"] <= COLD_F)),
    Feature("div_game", "divisional game",
            lambda g: float(g["div_game"] or 0)),
]


def design(rows: list[dict], features: list[Feature]) -> np.ndarray:
    """Build the design matrix, with an intercept in column 0."""
    X = np.ones((len(rows), len(features) + 1))
    for j, f in enumerate(features, start=1):
        X[:, j] = [f.of(g) for g in rows]
    return X


# --- Least squares -------------------------------------------------------

@dataclass
class Fit:
    names: list[str]
    beta: np.ndarray
    se: np.ndarray
    n: int
    r2: float

    @property
    def t(self) -> np.ndarray:
        return self.beta / np.where(self.se > 0, self.se, np.nan)


def ols(X: np.ndarray, y: np.ndarray, names: list[str]) -> Fit:
    """Plain OLS with textbook standard errors.

    n is in the thousands here, so the t distribution is normal for every
    practical purpose and |t| > 2 is the usual two-sigma line.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = float(resid @ resid) / (n - k)
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * sigma2)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot else 0.0
    return Fit(names, beta, se, n, r2)


def max_vif(X: np.ndarray) -> float:
    """Worst variance inflation among the non-intercept columns.

    Rest is encoded twice - a continuous differential and short/bye
    indicators - so collinearity is a live concern rather than a
    formality. Above ~5 the individual coefficients are unstable even
    though the fit as a whole is fine.
    """
    worst = 1.0
    for j in range(1, X.shape[1]):
        others = np.delete(X, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
        resid = X[:, j] - others @ beta
        ss_tot = float(((X[:, j] - X[:, j].mean()) ** 2).sum())
        if ss_tot <= 0:
            continue
        r2 = 1.0 - float(resid @ resid) / ss_tot
        worst = max(worst, 1.0 / max(1.0 - r2, 1e-9))
    return worst


# --- Data ----------------------------------------------------------------

def load_games(conn, season_from: int = 1999, season_to: int = 2026) -> list[dict]:
    """Every column the situational tests read, ordered as ratings.py wants."""
    rows = conn.execute("""
        SELECT game_id, season, week, game_type, home_team, away_team, neutral,
               home_score, away_score, played, spread_close, total_close,
               home_rest, away_rest, div_game, roof, surface, temp, wind
          FROM games
         WHERE season BETWEEN ? AND ? AND week IS NOT NULL
         ORDER BY season, week, game_id
    """, (season_from, season_to)).fetchall()
    return [dict(r) for r in rows]


def market_sample(games: list[dict], season_from: int = HOLDOUT_FROM) -> list[dict]:
    """Played games, priced, from the holdout era."""
    return [g for g in games
            if g["played"] and g["season"] >= season_from
            and g["home_score"] is not None
            and g["spread_close"] is not None]


# --- Reporting -----------------------------------------------------------

def _label_of(name: str, features: list[Feature]) -> str:
    if name == "intercept":
        return "(intercept)"
    if name == "elo_margin":
        return "Elo margin (points per Elo point)"
    return next(f.label for f in features if f.name == name)


def _print_fit(fit: Fit, features: list[Feature], *,
               vif: float | None = None) -> None:
    print(f"{'FEATURE':<34}{'COEF':>9}{'SE':>8}{'T':>8}")
    print("-" * 72)
    for i, name in enumerate(fit.names):
        flag = "  <-- signal" if name != "intercept" and abs(fit.t[i]) > 2 else ""
        print(f"{_label_of(name, features):<34}{fit.beta[i]:>+9.3f}"
              f"{fit.se[i]:>8.3f}{fit.t[i]:>8.2f}{flag}")
    print("-" * 72)
    tail = f"   max VIF = {vif:.1f}" if vif is not None else ""
    print(f"n = {fit.n:,}   R2 = {fit.r2:.4f}{tail}")
    if vif is not None and vif > 5:
        print("NOTE: VIF above 5 - the rest terms overlap, so read the block")
        print("      of rest coefficients together rather than one by one.")


def _print_bet_row(label: str, wins: int, losses: int, pushes: int) -> None:
    graded = wins + losses
    hit = wins / graded
    se = math.sqrt(BREAK_EVEN * (1 - BREAK_EVEN) / graded)
    roi = (wins * (100 / 110) - losses) / graded
    wlp = f"{wins}-{losses}-{pushes}"
    print(f"{label:<28}{graded:>7}{wlp:>14}{hit:>8.1%}{roi:>+9.1%}"
          f"{(hit - BREAK_EVEN) / se:>7.2f}")


def _print_bet_header() -> None:
    print(f"{'FILTER':<28}{'BETS':>7}{'W-L-P':>14}{'HIT':>8}{'ROI':>9}{'Z':>7}")
    print("-" * 72)


def test_a_margin_residuals(games: list[dict]) -> Fit:
    """Regress the closing spread's own error on the situational features."""
    rows = market_sample(games)
    y = np.array([(g["home_score"] - g["away_score"]) - (-g["spread_close"])
                  for g in rows], dtype=float)
    X = design(rows, MARGIN_FEATURES)
    names = ["intercept"] + [f.name for f in MARGIN_FEATURES]
    fit = ols(X, y, names)

    print("=" * 72)
    print(f"A. IS THE MARKET MISPRICING ANY OF THIS? ({HOLDOUT_FROM}-2025)")
    print("=" * 72)
    print("Outcome: actual home margin minus the closing spread's implied")
    print("margin. A coefficient is how many points the market is off by")
    print("when that condition holds. Zero everywhere = nothing to exploit.\n")
    _print_fit(fit, MARGIN_FEATURES, vif=max_vif(X))
    return fit


def total_sample(games: list[dict], season_from: int = HOLDOUT_FROM,
                 season_to: int = 2025) -> list[dict]:
    """Outdoor games with a priced total and a believable weather reading."""
    return [g for g in market_sample(games, season_from)
            if g["season"] <= season_to and g["temp"] is not None
            and wind_mph(g) is not None and g["total_close"] is not None]


def _under_record(rows: list[dict], keep=lambda g: True) -> tuple[int, int, int]:
    """W-L-P for betting the under on every row that passes `keep`."""
    wins = losses = pushes = 0
    for g in rows:
        if not keep(g):
            continue
        diff = (g["home_score"] + g["away_score"]) - g["total_close"]
        if diff == 0:
            pushes += 1
        elif diff < 0:
            wins += 1            # the under cashes
        else:
            losses += 1
    return wins, losses, pushes


def test_b_total_residuals(games: list[dict]) -> Fit:
    """The same test on totals, restricted to outdoor games with a reading."""
    rows = total_sample(games)
    y = np.array([(g["home_score"] + g["away_score"]) - g["total_close"]
                  for g in rows], dtype=float)
    X = design(rows, TOTAL_FEATURES)
    names = ["intercept"] + [f.name for f in TOTAL_FEATURES]
    fit = ols(X, y, names)

    print("\n\n" + "=" * 72)
    print(f"B. THE SAME TEST ON TOTALS ({HOLDOUT_FROM}-2025, outdoors only)")
    print("=" * 72)
    print("Outcome: actual points scored minus the closing total. Negative")
    print("means the game went under by that much.\n")
    _print_fit(fit, TOTAL_FEATURES, vif=max_vif(X))

    # The folklore, tested the way a bettor would actually play it.
    print("\nBetting the under at -110 by wind threshold:")
    _print_bet_header()
    for threshold in (0, 10, 12, 15, 18, 20):
        wins, losses, pushes = _under_record(
            rows, lambda g, t=threshold: wind_mph(g) >= t)
        if wins + losses < 50:
            continue
        label = ("every outdoor game" if not threshold
                 else f"wind >= {threshold} mph")
        _print_bet_row(label, wins, losses, pushes)
    _print_bet_row("divisional game", *_under_record(
        rows, lambda g: g["div_game"]))
    print("-" * 72)
    print("SIX thresholds were tried above. The best of six ~2-sigma results")
    print("is roughly what noise produces, so the split below is the test")
    print("that counts.")
    _wind_out_of_sample(games)
    return fit


def _wind_out_of_sample(games: list[dict]) -> None:
    """Pick the wind rule on early seasons, then bet it blind on later ones.

    Everything above is in-sample: the thresholds were chosen after
    looking at the same games that scored them. This fits the wind slope
    and picks the best threshold using only seasons before SPLIT_AT, then
    grades that one frozen rule on seasons from SPLIT_AT on - which is
    the only version of the test a bettor actually gets to play.
    """
    early = total_sample(games, HOLDOUT_FROM, SPLIT_AT - 1)
    late = total_sample(games, SPLIT_AT)

    def slope(rows: list[dict]) -> tuple[float, float]:
        y = np.array([(g["home_score"] + g["away_score"]) - g["total_close"]
                      for g in rows], dtype=float)
        X = design(rows, TOTAL_FEATURES)
        f = ols(X, y, ["intercept"] + [t.name for t in TOTAL_FEATURES])
        i = f.names.index("wind_mph")
        return float(f.beta[i]), float(f.t[i])

    print("\n\nOUT-OF-SAMPLE CHECK ON WIND")
    print("-" * 72)
    b_early, t_early = slope(early)
    b_late, t_late = slope(late)
    print(f"wind slope, {HOLDOUT_FROM}-{SPLIT_AT - 1} "
          f"(n={len(early):,}):  {b_early:+.3f} pts/mph   T = {t_early:.2f}")
    print(f"wind slope, {SPLIT_AT}-2025 "
          f"(n={len(late):,}):  {b_late:+.3f} pts/mph   T = {t_late:.2f}")

    # Choose the threshold on the early half only.
    best, best_hit = None, 0.0
    for threshold in (0, 8, 10, 12, 15, 18, 20):
        wins, losses, _ = _under_record(
            early, lambda g, t=threshold: wind_mph(g) >= t)
        if wins + losses < 200:      # enough to choose on
            continue
        if wins / (wins + losses) > best_hit:
            best, best_hit = threshold, wins / (wins + losses)

    print(f"\nBest threshold on {HOLDOUT_FROM}-{SPLIT_AT - 1} alone: "
          f"wind >= {best} mph, hitting {best_hit:.1%}.")
    print(f"Betting that frozen rule on {SPLIT_AT}-2025:")
    _print_bet_header()
    _print_bet_row(f"wind >= {best} mph (holdout)", *_under_record(
        late, lambda g: wind_mph(g) >= best))
    print("-" * 72)

    # A rule that only works in two seasons is not a rule. Every season
    # is shown, in-sample ones included, so a lucky year is visible
    # rather than averaged away.
    print(f"\nThe same rule season by season (wind >= {best} mph):")
    print(f"{'SEASON':<10}{'BETS':>7}{'W-L-P':>14}{'HIT':>8}")
    print("-" * 44)
    winning = 0
    seasons = sorted({g["season"] for g in early + late})
    for season in seasons:
        rows = [g for g in early + late if g["season"] == season]
        wins, losses, pushes = _under_record(
            rows, lambda g: wind_mph(g) >= best)
        if not wins + losses:
            continue
        hit = wins / (wins + losses)
        winning += hit > BREAK_EVEN
        tag = "  " if season < SPLIT_AT else " *"
        print(f"{season}{tag:<6}{wins + losses:>7}"
              f"{f'{wins}-{losses}-{pushes}':>14}{hit:>8.1%}")
    print("-" * 44)
    print(f"* = holdout season. {winning} of {len(seasons)} seasons beat "
          f"{BREAK_EVEN:.1%}.")

    print("\nCAVEAT, and it is the important one: `wind` is the reading AT")
    print("KICKOFF, which nobody has when the bet is placed. The market")
    print("prices a FORECAST. Part of this edge is therefore 'games windier")
    print("than forecast', which is not a bet anyone can make. Testing it")
    print("properly needs a wind forecast as of bet time and a timestamped")
    print("total - the same odds history the CLV limitation already calls")
    print("for. Until then this is a lead, not a strategy.")


# --- The predictive model ------------------------------------------------

def walk_forward_augmented(games: list[dict], predictions: list[Prediction],
                           season_from: int = HOLDOUT_FROM,
                           ) -> tuple[list[Prediction], dict[int, Fit]]:
    """Elo margin plus situational terms, refit on prior seasons each year.

    The fit is strictly backward-looking: season S is predicted by a
    regression that has seen only seasons before S. That matters more
    here than it does for Elo, because a regression fitted on the same
    games it is scored against will find "edges" that are memorised
    noise.
    """
    by_id = {g["game_id"]: g for g in games}
    elo_by_id = {p.game_id: p for p in predictions}
    names = ["intercept", "elo_margin"] + [f.name for f in MARGIN_FEATURES]

    def row(g: dict) -> list[float]:
        return ([1.0, elo_by_id[g["game_id"]].pred_margin]
                + [f.of(g) for f in MARGIN_FEATURES])

    trainable = [g for g in games
                 if g["played"] and g["home_score"] is not None
                 and g["game_id"] in elo_by_id]
    by_season: dict[int, list[Prediction]] = {}
    for p in predictions:
        by_season.setdefault(p.season, []).append(p)

    out: list[Prediction] = []
    fits: dict[int, Fit] = {}
    for season in sorted(s for s in by_season if s >= season_from):
        train = [g for g in trainable if g["season"] < season]
        if len(train) < 500:
            continue
        X = np.array([row(g) for g in train])
        y = np.array([g["home_score"] - g["away_score"] for g in train],
                     dtype=float)
        fits[season] = fit = ols(X, y, names)

        for p in by_season[season]:
            margin = float(np.array(row(by_id[p.game_id])) @ fit.beta)
            out.append(replace(p, pred_margin=margin))
    return out, fits


def _score_margin(name: str, preds: list[Prediction], margin_of) -> dict:
    """Accuracy and calibration, scoring every forecast the same way.

    ratings.compare turns Elo's rating gap into a probability with a
    logistic and the market's spread with a normal CDF. Here every
    forecast is a margin in points, so all of them go through the same
    normal CDF and the comparison is like for like.
    """
    return ratings._score(name, preds, margin_of,
                          lambda p: normal_cdf(margin_of(p) / MARGIN_SD))


def test_c_model(games: list[dict],
                 predictions: list[Prediction]) -> list[Prediction]:
    augmented, fits = walk_forward_augmented(games, predictions)
    aug_by_id = {p.game_id: p for p in augmented}

    graded = [p for p in predictions
              if p.home_won is not None and p.season >= HOLDOUT_FROM
              and p.spread_close is not None and p.game_id in aug_by_id]
    graded_aug = [aug_by_id[p.game_id] for p in graded]

    print("\n\n" + "=" * 72)
    print(f"C. ELO + SITUATIONAL AS A FORECAST ({HOLDOUT_FROM}-2025)")
    print("=" * 72)
    print("Refit each season on earlier seasons only. Every forecast below")
    print("is a margin in points, scored through the same normal CDF.\n")

    rows = [
        _score_margin("Elo", graded, lambda p: p.pred_margin),
        _score_margin("Elo + situational", graded_aug, lambda p: p.pred_margin),
        _score_margin("Closing spread (market)", graded,
                      lambda p: p.market_margin),
    ]
    print(f"{'FORECAST':<26}{'GAMES':>7}{'ACCURACY':>11}{'BRIER':>9}"
          f"{'LOGLOSS':>10}{'MAE':>9}")
    print("-" * 72)
    for m in rows:
        print(f"{m['name']:<26}{m['games']:>7}{m['accuracy']:>11.1%}"
              f"{m['brier']:>9.4f}{m['log_loss']:>10.4f}{m['margin_mae']:>9.2f}")
    print("-" * 72)

    print("\nAgainst the closing spread, 1 unit at -110:")
    _print_bet_header()
    for min_edge in (0.0, 1.0, 2.0, 3.0, 5.0):
        r = ratings.against_the_spread(augmented, HOLDOUT_FROM, min_edge)
        if not r["graded"]:
            continue
        label = ("every game" if not min_edge
                 else f"model disagrees by {min_edge:g}+")
        _print_bet_row(label, r["wins"], r["losses"], r["pushes"])
    print("-" * 72)

    # What the model settled on, fitted through the end of 2025.
    last = fits[max(fits)]
    print(f"\nCoefficients from the final fit (through {max(fits) - 1}):")
    print(f"{'FEATURE':<34}{'COEF':>9}{'SE':>8}{'T':>8}")
    print("-" * 72)
    for i, name in enumerate(last.names):
        print(f"{_label_of(name, MARGIN_FEATURES):<34}{last.beta[i]:>+9.3f}"
              f"{last.se[i]:>8.3f}{last.t[i]:>8.2f}")
    print("-" * 72)
    return augmented


# --- CLI -----------------------------------------------------------------

def run() -> None:
    conn = db.connect(config.NFL_DB)
    games = load_games(conn)
    predictions = ratings.walk_forward(games, Elo())

    n = len(market_sample(games))
    print(f"Situational tests over {n:,} played, priced games "
          f"({HOLDOUT_FROM}-2025).\n")

    test_a_margin_residuals(games)
    test_b_total_residuals(games)
    augmented = test_c_model(games, predictions)

    # Stored alongside 'elo' rather than replacing it, so the chat layer
    # can show both forecasts for a game and the table keeps a record of
    # the model that lost as well as the one that lost by less.
    saved = ratings.save_predictions(conn, "elo+situational", augmented)
    print(f"\nSaved {saved:,} forecasts to `predictions` "
          f"(model 'elo+situational').")

    print(f"\nBreak-even at -110 is {BREAK_EVEN:.1%}. "
          "|Z| and |T| under ~2 are noise.")
    conn.close()


if __name__ == "__main__":
    run()
