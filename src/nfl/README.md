# NFL predictor

Two data sources feeding one SQLite feature store:

- **Stage 1** — the *2026 NFL Record & Fact Book* (883-page PDF), parsed
  into teams, 2025 team statistics, standings and QB records.
- **Stage 2** — [nflverse](https://github.com/nflverse/nfldata)
  `games.csv`: ~7,500 games back to 1999 with closing spreads, totals,
  moneylines, rest days and weather.

## Build it

`data/nflverse/games.csv` is committed, so Stage 2 works out of the box.
Stage 1 needs the Fact Book PDF, which is not in git — without it,
`enrich` seeds the 32 clubs from `teams.py` and skips cross-validation,
and everything the model actually reads still gets built.

```bash
# Stage 1 (optional): PDF -> page-delimited text (~45s) -> nfl.db
python scripts/extract_pdf.py data/2026-Record-and-Fact-Book.pdf \
    --out data/extracted/factbook.pypdf.txt
python -m src.nfl.build

# Stage 2: load nflverse, cross-validate against Stage 1 if it ran
python -m src.nfl.enrich                 # --force-download to refresh

# Check everything
python -m src.nfl.validate

# Backtest + market tests + 2026 projections
python -m src.nfl.ratings

# Stage 3: are the situational columns mispriced?
python -m src.nfl.situational
```

## What's in the store

| Table | Rows | From |
|---|---|---|
| `teams` | 32 | Fact Book club blocks, pdf 44–237 |
| `games` | 7,548 (1999–2026) | nflverse, cross-checked against the Fact Book |
| `team_season_stats` | 3,904 | the 16-column matrices, pdf 257–260 |
| `standings` | 32 | pdf 245 |
| `qb_records` | 67 | pdf 313 |

7,276 played games; 7,388 with a closing spread; 5,407 with moneylines.
The 2026 slate has 112 of 272 games already priced with lookahead lines.

## How the data is verified

**Within the Fact Book.** The book prints every game twice, once in each
club's block. The parser reads both views and reconciles them, so a
disagreement surfaces as a warning rather than corrupting a row. Pairing
is on printed order, not date — divisional rivals meet twice and the book
occasionally mis-dates a meeting.

**Across sources.** `enrich.py` matches all 557 Fact Book games against
nflverse. Current result: **557/557 matched, 0 disagreements**, including
285 games where both sources carry a final score. Two independent
pipelines agreeing on every score and every home/away designation is the
strongest evidence available that both are right.

**Against arithmetic.** `validate.py` runs 18 checks, including summing
272 individually-parsed 2025 results to reproduce the book's own
standings page (W-L-T and points for/against) for all 32 clubs.

Two genuine contradictions in the printed book are handled explicitly:
Pittsburgh mis-dates its second Baltimore game, and New England prints
"W 30-6" for a wild card game the official results page (pdf 520) scores
16-3. The latter sits in `parsers.ERRATA` with its citation.

### Spread sign convention

This store uses the **sportsbook** convention: `spread_close` is negative
when the home club is favoured (home -3.5). nflverse's `spread_line` uses
the opposite sign; `nflverse.py` negates it on the way in, and
`validate.py` asserts the result by checking that home favourites
actually win 60–75% of decided games. Do not change one without the other.

## Results

Walk-forward across 7,276 played games (week *N* predicted from ratings
that have seen only earlier games), scored on 2010–2025:

| Forecast | Games | Accuracy | Brier | Log loss | MAE |
|---|---|---|---|---|---|
| Elo | 4,350 | 64.6% | 0.2207 | 0.6313 | 10.40 |
| **Closing spread (market)** | 4,350 | **66.4%** | **0.2117** | **0.6114** | **10.07** |
| Always pick home | 4,350 | 55.7% | 0.2467 | 0.6866 | 11.40 |

Against the closing spread, betting 1 unit at -110:

| Filter | Bets | W-L-P | Hit | ROI | Z |
|---|---|---|---|---|---|
| every game | 4,254 | 2154-2100-109 | 50.6% | -3.3% | -2.28 |
| model disagrees by 2+ | 2,118 | 1089-1029-62 | 51.4% | -1.8% | -0.89 |
| model disagrees by 5+ | 447 | 235-212-8 | 52.6% | +0.4% | +0.08 |

**The model has no edge.** It loses to the closing spread on every
accuracy metric, and no disagreement filter clears the 52.4% break-even
by more than noise (all |Z| < 2). This is the expected result — NFL sides
are among the most efficient markets there are — and it is the reason the
harness exists: to measure that, rather than assume either way.

## Limitation: this cannot measure true CLV

Closing line value needs two prices: the number you bet and the number
the game closed at. nflverse carries **one** line per game (the close), so
`spread_open`/`total_open` stay NULL and the tables above measure
performance *against* the close, not CLV. That is still the right test for
whether a model has edge — but proving you can capture edge in practice
needs timestamped line history from an odds API.

## Stage 3: the situational columns

`python -m src.nfl.situational` asks three questions of `home_rest`,
`away_rest`, `div_game`, `roof`, `temp` and `wind`. The obvious one —
bolt them onto Elo — is the least useful, because the closing spread has
already priced them. So the sharp test comes first: regress the
**market's own residual** on each feature. If the market is efficient
every coefficient is zero, and anything that isn't is a mispriced angle.

**A. On margins, nothing.** Regressing (actual margin − spread's implied
margin) on all nine features over 4,363 games gives R² = 0.0008 and not
one |t| > 2. Rest, byes, divisional games, domes and cold are priced.

**B. On totals, wind is real.** Same regression on (points − closing
total), outdoor games only:

| Feature | Coef | SE | t |
|---|---|---|---|
| **wind** | **−0.243 pts/mph** | 0.068 | **−3.58** |
| divisional game | −1.211 | 0.504 | −2.40 |
| temperature | −0.002 | 0.017 | −0.14 |
| cold (≤32°F) | +0.718 | 1.099 | +0.65 |

Every extra 5 mph is worth about 1.2 points of total that the closing
number does not take out. Six under-betting thresholds were tried, best
around 2σ — which is what noise looks like — so the module picks the
threshold on **2010–2017 only** and then bets it blind:

| | Wind slope | Frozen rule (wind ≥ 12) |
|---|---|---|
| 2010–2017 (fit) | −0.239 pts/mph, t = −2.56 | 53.6% |
| **2018–2025 (holdout)** | **−0.239 pts/mph, t = −2.43** | **58.8%, +12.3% ROI, Z = 2.34** |

The slope replicates to three decimal places on games it never saw. That
is the most durable result in this project.

**But the betting rule is much shakier than the slope.** Only 9 of 16
seasons beat break-even, and 2024 and 2025 both lost (47.4%, 47.2%) —
the holdout's 58.8% leans on 2021–23. Believe the coefficient; treat the
threshold rule as noisy.

**The caveat that matters most:** `wind` is the reading *at kickoff*, and
nobody has it when the bet is placed — the market prices a *forecast*.
Some unknown share of this edge is "games windier than forecast", which
is not a bet anyone can place. Separating the two needs a wind forecast
as of bet time and a timestamped total.

**C. As a forecast, it adds nothing.** Elo + situational, refit
walk-forward on prior seasons only, is a touch *worse* than plain Elo
(64.4% vs 64.6%, MAE 10.43 vs 10.40) and still finds no edge against the
spread. Exactly what A predicts.

## Next

Stage 3 turned up one lead, and it points at the same missing piece the
CLV limitation does:

1. **Timestamped odds + weather forecasts** — the only way to find out
   whether the wind result is tradeable or an artefact of hindsight
   weather. Now the highest-value direction, because there is a specific
   hypothesis to test rather than a general hope.
2. **Better features** — nflverse play-by-play carries EPA, success rate
   and CPOE, all far more predictive per-game than box-score margin.
   Still the best route to a *sides* model, which Stage 3 confirms Elo
   plus situational context will not produce.

Paper-trade any of it for a full season before risking money.
