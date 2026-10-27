"""run_live_picks.py: what the hourly task commits, pushes and logs.

Each git test gets a fresh repo and a bare "origin" in a temp directory -
never this repo, never GitHub.
"""
import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "run_live_picks", ROOT / "scripts" / "run_live_picks.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def _temp_dir(test: unittest.TestCase) -> Path:
    # Git marks its objects read-only, which Windows refuses to delete.
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    test.addCleanup(tmp.cleanup)
    return Path(tmp.name)


@unittest.skipUnless(shutil.which("git"), "git is not on PATH")
class Publish(unittest.TestCase):
    def setUp(self):
        base = _temp_dir(self)
        self.origin, self.work = base / "origin.git", base / "work"
        self.git("init", "-q", "--bare", "-b", "main", str(self.origin), cwd=base)
        self.git("init", "-q", "-b", "main", str(self.work), cwd=base)
        self.git("config", "user.name", "test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        (self.work / "README.md").write_text("x\n")
        self.git("add", "README.md")
        self.git("commit", "-q", "-m", "init")
        self.git("remote", "add", "origin", str(self.origin))
        self.git("push", "-q", "-u", "origin", "main")

        self.picks = self.work / runner.PICKS
        self.picks.parent.mkdir(parents=True)
        patcher = mock.patch.object(runner, "ROOT", self.work)
        patcher.start()
        self.addCleanup(patcher.stop)

    def git(self, *args, cwd=None) -> str:
        return subprocess.run(["git", *args], cwd=cwd or self.work, check=True,
                              capture_output=True, text=True).stdout.strip()

    def origin_head(self) -> str:
        return self.git("--git-dir", str(self.origin), "rev-parse", "main")

    def add_pick(self, line: str) -> None:
        with self.picks.open("a") as fh:
            fh.write(line + "\n")

    def test_nothing_to_do(self):
        self.assertEqual(runner.publish(), "")

    def test_new_log_is_committed_and_pushed(self):
        (self.work / "README.md").write_text("dirty\n")
        self.add_pick("game_id,bet")
        self.assertEqual(runner.publish(), "committed, pushed")
        self.assertEqual(self.origin_head(), self.git("rev-parse", "HEAD"))
        self.assertEqual(self.git("show", "--name-only", "--format=", "HEAD"),
                         runner.PICKS)
        self.assertIn("README.md", self.git("status", "--short"))   # untouched
        self.assertEqual(runner.publish(), "")

    def test_own_unpushed_commit_is_never_pushed(self):
        (self.work / "NOTES.txt").write_text("mine\n")
        self.git("add", "NOTES.txt")
        self.git("commit", "-q", "-m", "my own work")
        before = self.origin_head()
        self.add_pick("game_id,bet")
        self.assertTrue(runner.publish().startswith("committed, NOT pushed"))
        self.assertEqual(self.origin_head(), before)
        self.assertEqual(runner.publish(), "")   # and not on the next run

    def test_failed_push_is_retried(self):
        self.add_pick("game_id,bet")
        self.git("add", runner.PICKS)
        self.git("commit", "-q", "-m", "Log wind picks, push failed")
        self.assertEqual(runner.publish(), "pushed")
        self.assertEqual(self.origin_head(), self.git("rev-parse", "HEAD"))

    def test_off_main_nothing_is_committed(self):
        self.git("checkout", "-q", "-b", "feature")
        head = self.git("rev-parse", "HEAD")
        self.add_pick("game_id,bet")
        self.assertTrue(runner.publish().startswith("on branch feature"))
        self.assertEqual(self.git("rev-parse", "HEAD"), head)


class RunLog(unittest.TestCase):
    """main() writes one line when quiet, the full output otherwise."""

    def setUp(self):
        self.log = _temp_dir(self) / "run.log"
        patcher = mock.patch.object(runner, "RUN_LOG", self.log)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_main(self, picks, publish=lambda: ""):
        with mock.patch.object(runner.live_wind, "picks", picks), \
             mock.patch.object(runner, "publish", publish):
            runner.main()
        return self.log.read_text(encoding="utf-8")

    def test_quiet_run_is_one_line(self):
        text = self.run_main(lambda max_lead: [])
        self.assertRegex(text, r"^\d{4}-\d\d-\d\d \d\d:\d\d nothing due\n$")

    def test_task_uses_the_shared_lead(self):
        seen = []
        self.run_main(lambda max_lead: seen.append(max_lead) or [])
        self.assertEqual(seen, [runner.live_wind.TASK_MAX_LEAD])

    def test_missed_game_is_flagged(self):
        text = self.run_main(lambda max_lead: ["2026_08_DDD_GB"])
        self.assertIn("MISSED, no bet: 2026_08_DDD_GB", text)

    def test_commit_and_output_are_kept(self):
        def picks(max_lead):
            print("Logged 2 games")
            return []
        text = self.run_main(picks, lambda: "committed, pushed")
        self.assertIn("committed, pushed", text)
        self.assertIn("Logged 2 games", text)

    def test_failure_is_logged_not_raised(self):
        def picks(max_lead):
            raise ConnectionError("no network")
        text = self.run_main(picks)
        self.assertIn("FAILED", text)
        self.assertIn("ConnectionError: no network", text)


if __name__ == "__main__":
    unittest.main()
