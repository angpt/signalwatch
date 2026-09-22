from __future__ import annotations

import argparse
from datetime import datetime

from signalwatch.config import load_config
from signalwatch.models import UTC
from signalwatch.pipeline import SignalWatchApp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily mention monitoring across Google Search, Google News, and selected sites via SerpAPI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Collect mentions and write a plain-text digest.")
    run_parser.add_argument("--config", default="config/example.toml", help="Path to a TOML config file.")
    run_parser.add_argument(
        "--now",
        help="Optional ISO timestamp for deterministic runs, e.g. 2026-07-07T15:00:00+00:00.",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args.config, args.now)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _run(config_path: str, now_value: str | None) -> int:
    config = load_config(config_path)
    app = SignalWatchApp(config)
    try:
        now = _parse_now(now_value) if now_value else None
        report = app.run(now=now)
    finally:
        app.close()

    print(f"Config loaded: {config_path}")
    print(f"Summarizer provider: {config.summarizer.provider.value}")
    print(f"Digest written to {report.latest_digest_path}")
    print(f"New-only digest written to {report.new_latest_digest_path}")
    print(f"History snapshot: {report.history_digest_path}")
    print(f"New-only history snapshot: {report.new_history_digest_path}")
    print(f"Raw mentions collected: {report.raw_mentions}")
    print(f"New mentions stored: {report.new_mentions}")
    print(f"Visible mentions in window: {report.visible_mentions}")
    if report.summarizer_counts:
        counts = ", ".join(f"{name}={count}" for name, count in report.summarizer_counts)
    else:
        counts = "none"
    print(f"Summaries this run: {counts}")
    if report.failures:
        print(f"Collector failures: {len(report.failures)}")
        for failure in report.failures:
            print(f"- {failure.source} / {failure.keyword}: {failure.error}")
    return 0


def _parse_now(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
