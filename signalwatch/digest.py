from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

from signalwatch.models import CollectorFailure, StoredMention


def render_digest(
    mentions: Sequence[StoredMention],
    *,
    generated_at: datetime,
    since_hours: int,
    failures: Sequence[CollectorFailure] = (),
    title: str = "Daily mention digest",
    window_label: str | None = None,
    empty_message: str = "No mentions matched the current config in this window.",
) -> str:
    lines = [
        title,
        f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Window: {window_label or f'last {since_hours} hours'}",
        f"Items: {len(mentions)}",
        "",
    ]
    if not mentions:
        lines.append(empty_message)
        lines.append("")
    else:
        for index, mention in enumerate(mentions, start=1):
            source_label = ", ".join(mention.sources) if mention.sources else mention.source
            lines.extend(
                [
                    f"{index}. Keywords: {', '.join(mention.keywords) or 'unclassified'}",
                    f"Source: {source_label}",
                    f"Sentiment: {mention.sentiment.value}",
                    f"Suggested action: {mention.suggested_action.value}",
                    f"Score: {mention.score:.1f}",
                    f"Published: {_format_datetime(mention.published_at or mention.discovered_at)}",
                    f"Summary: {mention.summary}",
                    f"Why it matters: {mention.why_it_matters}",
                    f"Link: {mention.url}",
                    "",
                ]
            )
    if failures:
        lines.append("Collector issues:")
        for failure in failures:
            lines.append(f"- {failure.source} / {failure.keyword}: {failure.error}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_digest(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _format_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC")
