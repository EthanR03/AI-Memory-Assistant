"""live_wind.py: grading, the 12 mph line, kickoffs across the DST change,
and the picks -> price -> grade round trip.

No network and no real log: the schedule, the forecasts and the clock are
faked, and the log goes to a temp directory.
"""
import contextlib
import io
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from src import config
from src.nfl import live_wind as lw
from src.nfl.forecast_wind import window_means


class Grading(unittest.TestCase):
    def test_payout(self):
        self.assertAlmostEqual(lw._payout(-110), 100 / 110)
        self.assertAlmostEqual(lw._payout(120), 1.2)

    def test_grade_under(self):
        self.assertEqual(lw.grade_under(38, 41.5, -110), ("W", 100 / 110))
        self.assertEqual(lw.grade_under(45, 41.5, -110), ("L", -1.0))
        self.assertEqual(lw.grade_under(41, 41.0, -110), ("P", 0.0))
        self.assertEqual(lw.grade_under(30, 41.0, 105), ("W", 1.05))


class Threshold(unittest.TestCase):
    def test_exact_twelve_is_a_bet(self):
        self.assertTrue(lw.is_bet(12.0))

    def test_float_noise_does_not_cost_a_bet(self):
        # These four hours average exactly 12.0 on paper, but window_means
        # returns 11.999999999999998. sum() is compensated since Python
        # 3.12, so this is rare - about 1 in 2,000 windows that average
        # 12.0, all of them very uneven - but the real code can reach it.
        h = {"time": [f"2026-11-01T{h}:00" for h in (13, 14, 15, 16)],
             lw.JMA: [0.3, 1.2, 8.2, 38.3]}
        mean = window_means(h, "2026-11-01", "13:00")[lw.JMA]
        self.assertLess(mean, 12.0)
        self.assertTrue(lw.is_bet(mean))

    def test_next_mean_down_is_not(self):
        self.assertFalse(lw.is_bet(11.975))

    def test_missing_forecast_is_not(self):
        self.assertFalse(lw.is_bet(None))


class Kickoffs(unittest.TestCase):
    def test_edt_before_the_change(self):
        ko = lw.kickoff({"game_date": "2026-10-29", "gametime": "20:15"})
        self.assertEqual(ko, datetime(2026, 10, 30, 0, 15, tzinfo=timezone.utc))

    def test_est_from_nov_1(self):
        ko = lw.kickoff({"game_date": "2026-11-01", "gametime": "13:00"})
        self.assertEqual(ko, datetime(2026, 11, 1, 18, 0, tzinfo=timezone.utc))

    def test_window_opens_36_real_hours_out(self):
        # Wall-clock arithmetic on a shared ZoneInfo would give 05:00.
        ko = lw.kickoff({"game_date": "2026-11-01", "gametime": "13:00"})
        self.assertEqual(ko - timedelta(hours=36),
                         datetime(2026, 10, 31, 6, 0, tzinfo=timezone.utc))

    def test_at_parses_et(self):
        self.assertEqual(lw._et("2026-10-28T20:00"),
                         datetime(2026, 10, 29, 0, 0, tzinfo=timezone.utc))


def _game(game_id: str, game_date: str, gametime: str, stadium: str,
          roof: str | None = "outdoors") -> dict:
    _, week, away, home = game_id.split("_")
    return dict(nflverse_game_id=game_id, week=int(week), game_date=game_date,
                gametime=gametime, away_team=away, home_team=home, roof=roof,
                stadium=stadium, home_score=None, away_score=None,
                total_close=None)


class RoundTrip(unittest.TestCase):
    """Saturday 2026-10-31, noon ET - the day before the DST change."""

    NOW = datetime(2026, 10, 31, 12, 0, tzinfo=lw.ET)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.schedule = [
            _game("2026_08_AAA_BUF", "2026-11-01", "13:00", "Highmark Stadium"),
            _game("2026_08_BBB_CHI", "2026-11-01", "13:00", "Soldier Field"),
            _game("2026_08_CCC_HOU", "2026-11-01", "13:00", "NRG Stadium", None),
            _game("2026_08_DDD_GB", "2026-10-31", "23:59", "Lambeau Field"),
            _game("2026_09_EEE_NE", "2026-11-03", "20:15", "Gillette Stadium"),
        ]
        self.fetched = []
        self.now = self.NOW
        for target, name, value in [
            (config, "WIND_PICKS", Path(tmp.name) / "wind_picks.csv"),
            (lw, "load_schedule", lambda: [dict(g) for g in self.schedule]),
            (lw, "_fetch", self._fetch),
            (lw, "_now", lambda: self.now),
        ]:
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _fetch(self, games, url, hourly, models):
        self.fetched.append(sorted(g["nflverse_game_id"] for g in games))
        if url == lw.FORECAST_API:
            return {
                "2026_08_AAA_BUF": {lw.JMA: 14.2, lw.DEFAULT: 15.0, lw.JMA_TEMP: 41.0},
                "2026_08_BBB_CHI": {lw.JMA: 6.5, lw.DEFAULT: 7.0, lw.JMA_TEMP: 50.0},
            }
        return {"2026_08_AAA_BUF": {lw.ARCHIVE_JMA: 11.8},
                "2026_08_BBB_CHI": {lw.ARCHIVE_JMA: 7.1}}

    def _quiet(self, fn, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kwargs)

    def _picked(self):
        self._quiet(lw.picks, max_lead=26)
        return lw.read_log()

    # --- picks

    def test_logs_due_outdoor_games_only(self):
        log = self._picked()
        # Not the blank roof, not the one 12h out, not the one 3 days out.
        self.assertEqual([r["game_id"] for r in log],
                         ["2026_08_AAA_BUF", "2026_08_BBB_CHI"])
        self.assertEqual([r["bet"] for r in log], ["UNDER", ""])
        self.assertEqual(self.fetched, [["2026_08_AAA_BUF", "2026_08_BBB_CHI"]])
        self.assertEqual(log[0]["rule"], lw.RULE)

    def test_lead_counts_real_hours(self):
        # Sat 12:00 EDT to Sun 13:00 EST is 26 hours, not 25.
        self.assertEqual(self._picked()[0]["lead_h"], "26.0")

    def test_logged_game_is_never_repulled(self):
        self._picked()
        self.assertEqual(len(self._picked()), 2)
        self.assertEqual(len(self.fetched), 1)

    def test_dry_run_logs_nothing(self):
        self._quiet(lw.picks, dry_run=True, max_lead=26)
        self.assertEqual(lw.read_log(), [])

    def test_max_lead_waits(self):
        missed = self._quiet(lw.picks, max_lead=24)
        self.assertEqual(self.fetched, [])
        self.assertEqual(lw.read_log(), [])
        self.assertEqual(missed, ["2026_08_DDD_GB"])

    def test_max_lead_cannot_leave_the_rule_window(self):
        for bad in (18, 40):
            with self.assertRaises(ValueError):
                lw.picks(max_lead=bad)

    # --- price

    def test_price_is_stored(self):
        self._picked()
        self._quiet(lw.price, "2026_08_AAA_BUF", 42.5, -108, "DK")
        row = lw.read_log()[0]
        self.assertEqual((row["book"], row["book_total"], row["book_odds"]),
                         ("DK", "42.5", "-108"))

    def test_price_refuses_no_bet_and_unknown_games(self):
        self._picked()
        for game_id in ("2026_08_BBB_CHI", "2026_08_ZZZ_XX"):
            with self.assertRaises(SystemExit):
                lw.price(game_id, 40, -110, "")

    def test_price_refuses_after_kickoff(self):
        self._picked()
        self._quiet(lw.price, "2026_08_AAA_BUF", 42.5, -108, "DK")
        self.now = datetime(2026, 11, 1, 13, 5, tzinfo=lw.ET)
        with self.assertRaises(SystemExit):
            lw.price("2026_08_AAA_BUF", 41.0, -110, "DK")
        self.assertEqual(lw.read_log()[0]["book_total"], "42.5")

    # --- grade

    def test_nothing_graded_before_final(self):
        self._picked()
        self._quiet(lw.grade)
        self.assertEqual([r["graded_at"] for r in lw.read_log()], ["", ""])

    def test_grade(self):
        self._picked()
        self._quiet(lw.price, "2026_08_AAA_BUF", 42.5, -108, "DK")
        self.schedule[0].update(home_score=20, away_score=17, total_close=43.0)
        self.schedule[1].update(home_score=31, away_score=24, total_close=44.5)
        self._quiet(lw.grade)
        bet, no_bet = lw.read_log()
        self.assertEqual((bet["close_result"], bet["close_units"]), ("W", "+0.909"))
        self.assertEqual((bet["book_result"], bet["book_units"]), ("W", "+0.926"))
        self.assertEqual((no_bet["points"], no_bet["total_close"],
                          no_bet["close_result"]), ("55", "44.5", ""))
        self.assertEqual((bet["archive_jma_mph"], no_bet["archive_jma_mph"]),
                         ("11.80", "7.10"))

    def test_graded_games_are_not_regraded(self):
        self._picked()
        self.schedule[0].update(home_score=20, away_score=17, total_close=43.0)
        self.schedule[1].update(home_score=31, away_score=24, total_close=44.5)
        self._quiet(lw.grade)
        calls = len(self.fetched)
        self._quiet(lw.grade)
        self.assertEqual(len(self.fetched), calls)


if __name__ == "__main__":
    unittest.main()
