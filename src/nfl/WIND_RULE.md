# The wind rule, frozen

Frozen 2026-10-26, before any 2026 result was looked at. Stage 3b found
the edge (see "Stage 3b" in `README.md`). This file fixes the one bet that
2026 is allowed to test, so the season cannot be used to tune it. The
commit that adds this file is the freeze: `git log -- src/nfl/WIND_RULE.md`
should show that commit and nothing after it until the season is graded.

## Rule v1

**Bet the under, 1 unit, on every 2026 regular-season game where the JMA
day-ahead forecast for the game window is 12.0 mph or more.**

Each term, pinned down:

| Term | Definition |
|---|---|
| Games | 2026 regular season, live from week 8. The roof is `outdoors` in nflverse's schedule at pull time. `dome`, `closed`, `open` and blank are out. Blank means a retractable roof, which nobody can know a day out. |
| Forecast | Open-Meteo forecast API, model `jma_seamless` (JMA GSM outside Japan), hourly `wind_speed_10m` in mph, the newest run available at pull time. |
| Location | The stadium's coordinates in `forecast_wind.STADIUMS`, the ones the backtest used. A venue missing from it gets added. That does not change the rule. |
| Window | Mean of the hourly values at the kickoff hour (ET, minutes dropped, so 16:25 counts from 16:00) and the three hours after it. All four hours must be present, or there is no bet. |
| Timing | One pull per game, between 36 and 18 hours before kickoff. The first pull in that window is the pick, and a logged game is never re-pulled. A game nobody pulled in its window is no bet. |
| Trigger | Mean ≥ 12.0 mph: under. Anything less: no bet, but the forecast is still logged. |
| Stake | 1 unit per bet. No sizing by wind. |

Logged with each game but never used to decide one: the default model's
(`best_match`) wind over the same window, and the JMA temperature.

## Grading

1. **Against the close.** nflverse `total_line` after the game, at an
   assumed −110: a win is +0.909 units, a loss −1, a push 0. This is the
   number the backtest graded against, so it is the like-for-like test.
2. **Against the price actually available.** If a book's total and price
   were logged before kickoff, the bet is also graded at that number and
   that price. Nobody has measured this part of the edge yet.
3. **Audit.** After the game, the Previous Runs archive's JMA
   `wind_speed_10m_previous_day1` is fetched for the same window. It is
   the exact series the backtest used, so it shows how far the live pull
   sits from what the backtest would have seen. It never re-grades a bet.

Break-even at −110 is 52.4%. The backtest went 136-92-1 (59.6%) in
sample, about 29 bets a season. With live picks starting at week 8, 2026
will be a couple of dozen bets, which settles little by itself.

## Where v1 differs from the backtest

- **Regular season only.** The backtest also counted playoff games.
- **No opened retractable roofs.** The backtest counted games nflverse
  later marked `open`. A day out, that is unknowable.
- **Lead time.** The backtest's archived series sits about 24 hours
  before each hour. A live pull takes the newest run 18–36 hours out.
  The audit column measures the gap.

## Changing it

Not during 2026. A change is a v2 in a new commit, and v1's record stays
as it was graded. Weeks 1–7 may be backfilled from the archive only
after this commit, and are reported apart from the live picks, never
pooled with them.
