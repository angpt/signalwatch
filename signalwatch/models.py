from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

UTC = timezone.utc


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class SuggestedAction(StrEnum):
    REPLY = "reply"
    INVESTIGATE = "investigate"
    SHARE = "share"
    WATCH = "watch"
    IGNORE = "ignore"


class SummarizerProvider(StrEnum):
    HEURISTIC = "heuristic"
    LOCAL_LLM = "local_llm"


@dataclass(slots=True, frozen=True)
class KeywordConfig:
    name: str
    query: str
    exclude_terms: tuple[str, ...] = ()
    selected_domains: tuple[str, ...] = ()
    rejected_domains: tuple[str, ...] = ()
    max_results_per_source: int | None = None


@dataclass(slots=True, frozen=True)
class FilterConfig:
    rejected_domains: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class SerpApiProviderConfig:
    enabled: bool = True
    api_key_env: str = "SERPAPI_API_KEY"
    gl: str = "us"
    hl: str = "en"


@dataclass(slots=True, frozen=True)
class SourceConfig:
    google_news: bool = True
    google_search: bool = True
    selected_web: bool = True


@dataclass(slots=True, frozen=True)
class StorageConfig:
    database_path: Path
    latest_digest_path: Path
    history_dir: Path
    retention_days: int = 30
    new_latest_digest_path: Path | None = None
    new_history_dir: Path | None = None


@dataclass(slots=True, frozen=True)
class RunConfig:
    since_hours: int = 24
    timeout_seconds: int = 20
    max_results_per_query: int = 10
    user_agent: str = "signalwatch/0.1"


@dataclass(slots=True, frozen=True)
class LocalLLMConfig:
    """Any OpenAI-compatible chat-completions server (Ollama, llama.cpp, LM Studio, llamafile, vLLM)."""

    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = ""
    api_key_env: str = ""
    request_timeout_seconds: int = 240
    batch_size: int = 6
    max_input_chars: int = 4000
    max_batch_chars: int = 12000
    temperature: float = 0.0


@dataclass(slots=True, frozen=True)
class SummarizerConfig:
    provider: SummarizerProvider = SummarizerProvider.HEURISTIC
    fallback_to_heuristic: bool = True
    llm: LocalLLMConfig = field(default_factory=LocalLLMConfig)


@dataclass(slots=True, frozen=True)
class ProviderConfig:
    serpapi: SerpApiProviderConfig = field(default_factory=SerpApiProviderConfig)
    sources: SourceConfig = field(default_factory=SourceConfig)


@dataclass(slots=True, frozen=True)
class AppConfig:
    root_dir: Path
    storage: StorageConfig
    run: RunConfig
    providers: ProviderConfig
    keywords: tuple[KeywordConfig, ...]
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)


@dataclass(slots=True, frozen=True)
class RawMention:
    keyword: str
    source: str
    url: str
    published_at: datetime | None
    external_id: str | None = None
    title: str | None = None
    excerpt: str | None = None
    source_context: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EnrichedMention:
    keyword: str
    source: str
    url: str
    canonical_url: str
    published_at: datetime | None
    discovered_at: datetime
    summary: str
    sentiment: Sentiment
    suggested_action: SuggestedAction
    why_it_matters: str
    score: float
    external_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class StoredMention:
    id: int
    source: str
    url: str
    published_at: datetime | None
    discovered_at: datetime
    summary: str
    sentiment: Sentiment
    suggested_action: SuggestedAction
    why_it_matters: str
    score: float
    keywords: tuple[str, ...]
    sources: tuple[str, ...] = ()
    raw_text: str = ""


@dataclass(slots=True, frozen=True)
class CollectorFailure:
    source: str
    keyword: str
    error: str


@dataclass(slots=True, frozen=True)
class RunReport:
    generated_at: datetime
    raw_mentions: int
    new_mentions: int
    visible_mentions: int
    latest_digest_path: Path
    history_digest_path: Path
    new_latest_digest_path: Path
    new_history_digest_path: Path
    summarizer_counts: tuple[tuple[str, int], ...] = ()
    failures: tuple[CollectorFailure, ...] = ()
