from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from signalwatch.models import UTC

TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "s",
    "tl",
}

TRACKING_QUERY_PREFIXES = ("utm_",)


def utc_now() -> datetime:
    return datetime.now(UTC)


def collapse_whitespace(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def build_search_query(query: str, exclude_terms: tuple[str, ...]) -> str:
    parts = [collapse_whitespace(query)]
    for term in exclude_terms:
        term = collapse_whitespace(term)
        if not term:
            continue
        if " " in term:
            parts.append(f'-"{term}"')
        else:
            parts.append(f"-{term}")
    return " ".join(part for part in parts if part)


def normalize_domain(value: str) -> str:
    domain = collapse_whitespace(value).lower()
    if "//" in domain:
        domain = domain.split("//", 1)[1]
    domain = domain.split("/", 1)[0].strip(".")
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def host_matches_domain(host: str, domain: str) -> bool:
    host = collapse_whitespace(host).lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    domain = normalize_domain(domain)
    if not host or not domain:
        return False
    return host == domain or host.endswith(f".{domain}")


def contains_excluded_term(text: str, exclude_terms: tuple[str, ...]) -> bool:
    """Match an exclude term in free text, on word boundaries."""
    if not exclude_terms:
        return False
    haystack = collapse_whitespace(text).lower()
    if not haystack:
        return False
    for term in exclude_terms:
        term = collapse_whitespace(term).lower()
        if not term:
            continue
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", haystack):
            return True
    return False


def _squash(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def url_contains_excluded_term(url: str, exclude_terms: tuple[str, ...]) -> bool:
    """Match an exclude term in a URL, where words run together in slugs."""
    if not exclude_terms:
        return False
    haystack = _squash(url)
    if not haystack:
        return False
    return any(_squash(term) and _squash(term) in haystack for term in exclude_terms)


def is_excluded_mention(url: str, texts: tuple[str | None, ...], exclude_terms: tuple[str, ...]) -> bool:
    if not exclude_terms:
        return False
    if url_contains_excluded_term(url, exclude_terms):
        return True
    return any(contains_excluded_term(text or "", exclude_terms) for text in texts)


def is_rejected_domain(url: str, rejected_domains: tuple[str, ...]) -> bool:
    if not rejected_domains:
        return False
    host = urlsplit(url.strip()).hostname or ""
    return any(host_matches_domain(host, domain) for domain in rejected_domains)


def canonicalize_url(url: str) -> str:
    split = urlsplit(url.strip())
    scheme = (split.scheme or "https").lower()
    netloc = split.netloc.lower()
    path = split.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    filtered_query = []
    for key, value in parse_qsl(split.query, keep_blank_values=False):
        key_lower = key.lower()
        if key_lower in TRACKING_QUERY_KEYS:
            continue
        if any(key_lower.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES):
            continue
        filtered_query.append((key, value))
    query = urlencode(filtered_query, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_datetime(value: str | None, now: datetime | None = None) -> datetime | None:
    if not value:
        return None
    value = collapse_whitespace(value)
    now = now or utc_now()
    lowered = value.lower()

    relative_match = re.fullmatch(r"(\d+)\s+(minute|hour|day|week|month)s?\s+ago", lowered)
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)
        delta_map = {
            "minute": timedelta(minutes=amount),
            "hour": timedelta(hours=amount),
            "day": timedelta(days=amount),
            "week": timedelta(weeks=amount),
            "month": timedelta(days=30 * amount),
        }
        return now - delta_map[unit]

    if lowered == "yesterday":
        return now - timedelta(days=1)

    iso_candidate = value.replace("Z", "+00:00")
    try:
        return ensure_utc(datetime.fromisoformat(iso_candidate))
    except ValueError:
        pass

    formats = (
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y, %I:%M %p, %z UTC",
        "%m/%d/%Y, %I:%M %p",
    )
    for fmt in formats:
        try:
            return ensure_utc(datetime.strptime(value, fmt))
        except ValueError:
            continue

    try:
        return ensure_utc(parsedate_to_datetime(value))
    except (TypeError, ValueError, IndexError):
        return None
