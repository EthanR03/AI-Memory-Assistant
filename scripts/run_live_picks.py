"""Run `live_wind picks` unattended, then commit and push the pick log.

Usage (what the scheduled task runs; pythonw so no console window opens):
    venv\\Scripts\\pythonw.exe scripts\\run_live_picks.py

scripts/schedule_live_picks.ps1 registers this with Windows Task Scheduler
to run every hour through the 2026 regular season. Most runs find nothing
due and add one line to data/live/run.log. When a game is due, its pick is
logged, and wind_picks.csv is committed and pushed straight away, so
GitHub's own push time backs up every pick.

It pulls a game once it is at most 26h out, not the moment it hits 36h.
The rule takes the first pull anywhere in 18-36h; 26h sits close to the
backtest's ~24h lead and still leaves eight hourly retries before the 18h
cutoff, for when the machine is asleep or offline.

Git is only touched for data/live/wind_picks.csv, and only on main. A push
goes out only when every commit main has that GitHub lacks is a pick-log
commit, so your own unpushed work is never pushed for you. A push that
failed is retried on the next run.
"""
import contextlib
import io
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.nfl import live_wind  # noqa: E402

MAX_LEAD = 26
RUN_LOG = config.LIVE_DIR / "run.log"
PICKS = config.WIND_PICKS.relative_to(ROOT).as_posix()
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)   # Windows only


def git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, creationflags=NO_WINDOW)
    if done.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout.strip()


def _only_picks_unpushed() -> bool:
    """True if main is ahead of GitHub only by pick-log commits."""
    shas = git("rev-list", "origin/main..main").split()
    return bool(shas) and all(
        git("diff-tree", "--no-commit-id", "--name-only", "-r", sha) == PICKS
        for sha in shas)


def publish() -> str:
    """Commit the pick log if `picks` changed it, then push if safe.

    Returns what happened, or "" when there was nothing to do.
    """
    changed = bool(git("status", "--porcelain", "--", PICKS))
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        return (f"on branch {branch}: picks logged but NOT committed - "
                "commit them on main by hand" if changed else "")
    done = []
    if changed:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        git("add", "--", PICKS)
        git("commit", "-m", f"Log wind picks, {stamp}", "--", PICKS)
        done.append("committed")
    if _only_picks_unpushed():
        git("push", "origin", "main")
        done.append("pushed")
    elif changed:
        done.append("NOT pushed: main has other commits GitHub lacks - "
                    "push by hand")
    return ", ".join(done)


def main() -> None:
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            missed = live_wind.picks(max_lead=MAX_LEAD)
        status = publish()
        if missed:
            # Only happens if every run in a game's window failed or the
            # machine was off throughout. Repeats hourly until kickoff.
            status = ", ".join(filter(None, [
                status, f"MISSED, no bet: {' '.join(missed)}"]))
    except Exception:
        status = "FAILED\n" + traceback.format_exc()

    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        if status:
            fh.write(f"\n=== {now} {status}\n{out.getvalue()}\n")
        else:
            fh.write(f"{now} nothing due\n")


if __name__ == "__main__":
    main()
