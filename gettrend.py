#!/usr/bin/env python3
"""
github_trending_history.py

Fetches a proxy for "trending Python repos" for every day between a start
date and today, using GitHub's Search API (repos created that day, sorted
by stars). Results are written incrementally to per-month JSON files under
an output directory, organized as:

    <output_dir>/<YYYY>/<YYYY-MM>.json

Each month file is a JSON object keyed by ISO date, mapping to a list of
repo records:

    {
      "2020-01-05": [
        {"date": "2020-01-05", "name": "...", "url": "...",
         "size": 1234, "topics": ["cli", "automation"], "stars": 42},
        ...
      ]
    }

Features:
  - Resumable: tracks completed dates in a progress file + falls back to
    rescanning existing output files if the progress file is missing.
  - Concurrent fetching (ThreadPoolExecutor) with thread-safe, incremental
    ("simultaneous") writes to per-month files, guarded by per-file locks.
  - Careful GitHub rate-limit handling: proactive token-bucket throttling
    plus reactive handling of primary (X-RateLimit-*) and secondary/abuse
    rate limits (Retry-After / exponential backoff).
  - Uses pathlib exclusively for filesystem work.
  - Verbose logging by default, with -v for debug-level detail.

Requirements:
    pip install requests

Auth:
    Set GITHUB_TOKEN env var (or pass --token) for a much higher Search
    API quota (30 req/min vs 10 req/min unauthenticated).

Example:
    export GITHUB_TOKEN=ghp_xxx
    python github_trending_history.py --output-dir ./output --per-day 25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("This script requires the 'requests' package. Install it with: pip install requests")
try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("This script requires 'python-dotenv'. Install it with: pip install python-dotenv")

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
DEFAULT_START_DATE = date(2009, 1, 1)
PROGRESS_FILENAME = ".progress.json"

logger = logging.getLogger("github_trending_history")


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Proactive token-bucket limiter, plus a shared cooldown that reactive
    handlers (403/429 responses) can trigger to pause every thread."""

    def __init__(self, max_calls_per_window: int, window_seconds: float = 60.0):
        self.max_calls = max(1, max_calls_per_window)
        self.window = window_seconds
        self._lock = threading.Lock()
        self._calls: list[float] = []
        self._paused_until: Optional[float] = None

    def acquire(self) -> None:
        while True:
            wait = 0.0
            with self._lock:
                now = time.monotonic()
                if self._paused_until and now < self._paused_until:
                    wait = self._paused_until - now
                else:
                    self._calls = [t for t in self._calls if now - t < self.window]
                    if len(self._calls) < self.max_calls:
                        self._calls.append(now)
                        return
                    wait = self.window - (now - self._calls[0]) + 0.1
            logger.debug("Rate limiter: throttling, sleeping %.2fs", wait)
            time.sleep(wait)

    def pause_for(self, seconds: float) -> None:
        with self._lock:
            now = time.monotonic()
            self._paused_until = max(self._paused_until or 0.0, now + seconds)


# --------------------------------------------------------------------------- #
# GitHub client
# --------------------------------------------------------------------------- #
class GitHubClient:
    def __init__(self, token: Optional[str], rate_limiter: RateLimiter, max_retries: int = 8):
        self.session = requests.Session()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "github-trending-history-script",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.session.headers.update(headers)
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries

    def get(self, url: str, params: dict) -> dict:
        attempt = 0
        while True:
            attempt += 1
            self.rate_limiter.acquire()
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                if attempt > self.max_retries:
                    raise
                backoff = min(60, 2**attempt)
                logger.warning(
                    "Network error (%s). Retrying in %.1fs [attempt %d/%d]",
                    exc,
                    backoff,
                    attempt,
                    self.max_retries,
                )
                time.sleep(backoff)
                continue

            remaining = resp.headers.get("X-RateLimit-Remaining")
            reset = resp.headers.get("X-RateLimit-Reset")
            if remaining is not None:
                logger.debug("Rate limit remaining: %s (reset epoch: %s)", remaining, reset)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (403, 429):
                if attempt > self.max_retries:
                    raise RuntimeError(f"Exceeded retries for {url} (status {resp.status_code})")
                retry_after = resp.headers.get("Retry-After")
                try:
                    message = resp.json().get("message", "")
                except ValueError:
                    message = ""
                if retry_after:
                    wait = float(retry_after)
                    logger.warning("Rate limited (%s). Sleeping %.1fs per Retry-After header.", message, wait)
                elif remaining == "0" and reset:
                    wait = max(1.0, float(reset) - time.time() + 1)
                    logger.warning("Primary rate limit exhausted. Sleeping %.1fs until reset.", wait)
                else:
                    wait = min(120, 2**attempt)
                    logger.warning(
                        "Secondary rate limit / abuse detection hit (%s). Backing off %.1fs.",
                        message,
                        wait,
                    )
                self.rate_limiter.pause_for(wait)
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                if attempt > self.max_retries:
                    resp.raise_for_status()
                backoff = min(60, 2**attempt)
                logger.warning("Server error %d. Retrying in %.1fs.", resp.status_code, backoff)
                time.sleep(backoff)
                continue

            resp.raise_for_status()


# --------------------------------------------------------------------------- #
# Persistence / resume
# --------------------------------------------------------------------------- #
class StateStore:
    """Handles per-month output files and resumable progress tracking."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress_path = output_dir / PROGRESS_FILENAME
        self._progress_lock = threading.Lock()
        self._file_locks_guard = threading.Lock()
        self._file_locks: dict[Path, threading.Lock] = {}
        self.completed: set[str] = self._load_progress()

    def _load_progress(self) -> set[str]:
        if self.progress_path.exists():
            try:
                data = json.loads(self.progress_path.read_text(encoding="utf-8"))
                completed = set(data.get("completed_dates", []))
                logger.info("Resuming: loaded %d completed date(s) from progress file.", len(completed))
                return completed
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read progress file (%s). Rescanning output directory instead.", exc)

        completed = set()
        if self.output_dir.exists():
            for month_file in self.output_dir.glob("*/*.json"):
                try:
                    data = json.loads(month_file.read_text(encoding="utf-8"))
                    completed.update(data.keys())
                except (json.JSONDecodeError, OSError):
                    continue
        if completed:
            logger.info("Resuming: recovered %d completed date(s) by scanning output files.", len(completed))
        return completed

    def month_path(self, day: date) -> Path:
        year_dir = self.output_dir / f"{day.year:04d}"
        return year_dir / f"{day.year:04d}-{day.month:02d}.json"

    def _lock_for(self, path: Path) -> threading.Lock:
        with self._file_locks_guard:
            lock = self._file_locks.setdefault(path, threading.Lock())
        return lock

    def save_day(self, day: date, repos: list[dict]) -> None:
        date_str = day.isoformat()
        path = self.month_path(day)
        lock = self._lock_for(path)
        with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    data = {}
            else:
                data = {}
            data[date_str] = repos
            tmp_path = path.with_suffix(".json.tmp")
            tmp_path.write_text(
                json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(path)  # atomic on POSIX and Windows
        self._mark_done(date_str)

    def _mark_done(self, date_str: str) -> None:
        with self._progress_lock:
            self.completed.add(date_str)
            tmp = self.progress_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"completed_dates": sorted(self.completed)}, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.progress_path)


# --------------------------------------------------------------------------- #
# Fetch logic
# --------------------------------------------------------------------------- #
def fetch_day(client: GitHubClient, day: date, per_day: int) -> list[dict]:
    date_str = day.isoformat()
    query = f"language:python created:{date_str}"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": min(per_day, 100),
    }
    data = client.get(GITHUB_SEARCH_URL, params=params)
    items = data.get("items", [])[:per_day]
    return [
        {
            "date": date_str,
            "name": item.get("full_name"),
            "url": item.get("html_url"),
            "size": item.get("size"),
            "topics": item.get("topics", []),
            "stars": item.get("stargazers_count"),
        }
        for item in items
    ]


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# --------------------------------------------------------------------------- #
# CLI / orchestration
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch daily 'trending' (proxy: most-starred, created that day) "
        "Python repos from GitHub, from 2020-01-01 to today.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--token", default=None, help="GitHub token (defaults to GITHUB_TOKEN env var)")
    parser.add_argument("--per-day", type=int, default=25, help="Repos kept per day (max 100)")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel worker threads")
    parser.add_argument(
        "--calls-per-minute",
        type=int,
        default=25,
        help="Proactive throttle for the Search API (GitHub limit: 30/min authenticated, 10/min unauthenticated)",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase log verbosity (-v for debug)")
    return parser.parse_args()


####################


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    env_path = Path.home() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.info("Loaded environment variables from %s", env_path)
    else:
        logger.debug("No .env file found at %s", env_path)

    if args.start_date > args.end_date:
        logger.error("start-date (%s) is after end-date (%s).", args.start_date, args.end_date)
        sys.exit(1)

    if args.per_day > 100:
        logger.warning("per-day capped at 100 (GitHub API limit); requested %d.", args.per_day)
        args.per_day = 100

    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.warning(
            "No GitHub token found (--token or GITHUB_TOKEN). "
            "Search API is limited to 10 requests/min unauthenticated; this will be slow."
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rate_limiter = RateLimiter(max_calls_per_window=args.calls_per_minute, window_seconds=60.0)
    client = GitHubClient(token, rate_limiter)
    store = StateStore(output_dir)

    all_dates = list(daterange(args.start_date, args.end_date))
    pending = [d for d in all_dates if d.isoformat() not in store.completed]

    logger.info(
        "Date range: %s -> %s (%d days total, %d already done, %d pending).",
        args.start_date,
        args.end_date,
        len(all_dates),
        len(all_dates) - len(pending),
        len(pending),
    )

    if not pending:
        logger.info("Nothing to do. All dates already fetched.")
        return

    errors: list[tuple[date, str]] = []
    errors_lock = threading.Lock()

    def worker(day: date) -> None:
        try:
            logger.info("Fetching %s ...", day.isoformat())
            repos = fetch_day(client, day, args.per_day)
            store.save_day(day, repos)
            logger.info("Saved %d repo(s) for %s -> %s", len(repos), day.isoformat(), store.month_path(day))
        except Exception as exc:  # noqa: BLE001 - we want to log and continue
            logger.error("Failed to fetch %s: %s", day.isoformat(), exc)
            with errors_lock:
                errors.append((day, str(exc)))

    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(worker, d): d for d in pending}
            for _ in as_completed(futures):
                pass  # errors are logged/collected inside worker()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Progress already saved so far; rerun to resume.")
        sys.exit(130)

    if errors:
        logger.warning(
            "%d day(s) failed this run. Rerun the script to retry them (completed days are skipped).",
            len(errors),
        )
        for day, msg in errors:
            logger.warning("  %s: %s", day.isoformat(), msg)

    logger.info("Done.")


if __name__ == "__main__":
    main()
