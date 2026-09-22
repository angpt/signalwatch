from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from urllib.parse import urlsplit

from signalwatch.models import KeywordConfig, RawMention
from signalwatch.utils import collapse_whitespace, parse_datetime

_EXCERPT_LIMIT = 4000


def parse_search_results(
    items: list[dict],
    keyword: KeywordConfig,
    source_name: str,
    *,
    now: datetime,
) -> list[RawMention]:
    results: list[RawMention] = []
    for item in items:
        link = item.get("link") or item.get("url")
        if not link:
            continue
        url = str(link)
        title = collapse_whitespace(str(item.get("title") or ""))
        excerpt = _build_excerpt(item)
        date_value = _first_non_empty(
            item.get("published_at"),
            item.get("date"),
            item.get("published"),
            item.get("iso_date"),
            item.get("publishedDate"),
            item.get("published_date"),
        )
        results.append(
            RawMention(
                keyword=keyword.name,
                source=source_name,
                url=url,
                published_at=parse_datetime(str(date_value), now=now) if date_value else None,
                external_id=None,
                title=title,
                excerpt=excerpt,
                source_context=_build_source_context(item, source_name=source_name, url=url),
                meta={},
            )
        )
    return results


def collect_per_domain(
    domains: Sequence[str],
    fetch: Callable[[str], list[RawMention]],
) -> list[RawMention]:
    """Run one search per allowlisted domain, isolating per-domain failures.

    A single slow or erroring domain used to abort the whole sweep and discard
    results already gathered from domains that had answered. Each domain is now
    independent, so the collector only fails when every domain failed.
    """
    results: list[RawMention] = []
    errors: list[str] = []
    for domain in domains:
        try:
            results.extend(fetch(domain))
        except Exception as exc:
            errors.append(f"{domain}: {exc}")
    if errors and len(errors) == len(domains):
        raise RuntimeError("; ".join(errors))
    return results


def filter_since(results: list[RawMention], since: datetime) -> list[RawMention]:
    return [result for result in results if result.published_at is None or result.published_at >= since]


def _build_excerpt(item: dict) -> str:
    for candidate in (
        item.get("snippet"),
        item.get("content"),
    ):
        normalized = collapse_whitespace(str(candidate or ""))
        if normalized:
            return _truncate_text(normalized)

    highlighted = item.get("snippet_highlighted_words")
    if isinstance(highlighted, list):
        normalized = collapse_whitespace(" ".join(str(part) for part in highlighted))
        if normalized:
            return _truncate_text(normalized)
    normalized = collapse_whitespace(str(item.get("extracted_content") or ""))
    if normalized:
        return _truncate_text(normalized)
    return ""


def _build_source_context(item: dict, *, source_name: str, url: str) -> str:
    source_info = item.get("source")
    if isinstance(source_info, dict):
        source_context = collapse_whitespace(str(source_info.get("name") or ""))
    else:
        source_context = collapse_whitespace(str(source_info or ""))
    if source_context:
        return source_context

    engines = item.get("engines")
    if isinstance(engines, list):
        source_context = collapse_whitespace(", ".join(str(engine) for engine in engines if engine))
        if source_context:
            return source_context

    parsed_url = item.get("parsed_url")
    if isinstance(parsed_url, list) and len(parsed_url) > 1:
        source_context = collapse_whitespace(str(parsed_url[1] or ""))
        if source_context:
            return source_context

    return collapse_whitespace(urlsplit(url).netloc) or source_name


def _first_non_empty(*values: object) -> object | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not collapse_whitespace(value):
            continue
        return value
    return None


def _truncate_text(value: str, *, limit: int = _EXCERPT_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"
