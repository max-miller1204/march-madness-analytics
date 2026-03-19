#!/usr/bin/env python3
"""Fixed-interval polling scheduler for the tournament refresh pipeline."""

import re
import signal
import sys
import time
from datetime import datetime


def parse_interval(interval_str: str) -> int:
    """Parse a human-friendly interval string to seconds.

    Supported formats:
        "30m"  -> 1800
        "1h"   -> 3600
        "15m"  -> 900
        "90s"  -> 90
        "1h30m" -> 5400
        "1800" -> 1800  (plain seconds)

    Returns:
        int: interval in seconds

    Raises:
        ValueError: if the string cannot be parsed
    """
    # Plain integer — treat as seconds
    if interval_str.isdigit():
        return int(interval_str)

    total = 0
    pattern = re.compile(r'(\d+)\s*([hms])', re.IGNORECASE)
    matches = pattern.findall(interval_str)
    if not matches:
        raise ValueError(
            f"Cannot parse interval '{interval_str}'. "
            "Use formats like '30m', '1h', '15m', '90s', or '1h30m'."
        )

    multipliers = {'h': 3600, 'm': 60, 's': 1}
    for value, unit in matches:
        total += int(value) * multipliers[unit.lower()]
    return total


def run_watch(interval_seconds: int, pipeline_fn, **kwargs):
    """Run *pipeline_fn* in a fixed-interval polling loop.

    Parameters
    ----------
    interval_seconds : int
        Seconds to sleep between runs.
    pipeline_fn : callable
        The function to invoke each cycle.  Receives **kwargs.
    **kwargs
        Forwarded to *pipeline_fn* on every invocation.

    The loop handles ``KeyboardInterrupt`` (Ctrl+C) gracefully, printing a
    summary before exiting.
    """
    stop = False

    def _handle_signal(signum, frame):
        nonlocal stop
        stop = True

    # Catch SIGINT (Ctrl+C) and SIGTERM for graceful shutdown
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    run_count = 0
    print(f"[watch] Starting scheduled refresh every {interval_seconds}s "
          f"({interval_seconds // 60}m).  Press Ctrl+C to stop.")

    while not stop:
        run_count += 1
        start = datetime.now()
        print(f"\n{'=' * 60}")
        print(f"[watch] Run #{run_count} started at {start:%Y-%m-%d %H:%M:%S}")
        print(f"{'=' * 60}")

        try:
            pipeline_fn(**kwargs)
        except Exception as exc:
            print(f"[watch] Run #{run_count} FAILED: {exc}")
        else:
            elapsed = (datetime.now() - start).total_seconds()
            print(f"[watch] Run #{run_count} completed in {elapsed:.1f}s")

        if stop:
            break

        print(f"[watch] Sleeping {interval_seconds}s until next run...")
        # Sleep in small increments so Ctrl+C is responsive
        slept = 0
        while slept < interval_seconds and not stop:
            chunk = min(1, interval_seconds - slept)
            time.sleep(chunk)
            slept += chunk

    print(f"\n[watch] Stopped after {run_count} run(s).  Goodbye.")
    sys.exit(0)
