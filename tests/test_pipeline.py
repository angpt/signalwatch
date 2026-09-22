from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from signalwatch.models import (
    AppConfig,
    EnrichedMention,
    FilterConfig,
    KeywordConfig,
    ProviderConfig,
    RawMention,
    RunConfig,
    Sentiment,
    SerpApiProviderConfig,
    SourceConfig,
    StorageConfig,
    SuggestedAction,
    UTC,
)
from signalwatch.pipeline import SignalWatchApp
from signalwatch.utils import canonicalize_url


class PipelineTests(unittest.TestCase):
    def test_pipeline_skips_enrichment_for_duplicate_canonical_urls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = AppConfig(
                root_dir=root,
                storage=StorageConfig(
                    database_path=root / "mentions.db",
                    latest_digest_path=root / "latest_digest.txt",
                    history_dir=root / "digests",
                ),
                run=RunConfig(),
                providers=ProviderConfig(
                    serpapi=SerpApiProviderConfig(enabled=False),
                ),
                keywords=(KeywordConfig(name="OpenAI", query="OpenAI"),),
            )
            app = SignalWatchApp(config)
            try:
                app.collectors = (
                    _StaticCollector(
                        "google_search",
                        [
                            _raw("https://example.com/post?utm_source=a", "google_search"),
                            _raw("https://example.com/post", "selected_web"),
                        ],
                    ),
                )
                summarizer = _CountingSummarizer()
                app.summarizer = summarizer

                app.run(now=datetime(2026, 7, 7, 15, 0, tzinfo=UTC))

                mentions = app.store.list_since(datetime(2026, 7, 6, 15, 0, tzinfo=UTC))
                self.assertEqual(summarizer.calls, 1)
                self.assertEqual(len(mentions), 1)
                self.assertEqual(set(mentions[0].sources), {"google_search", "selected_web"})
            finally:
                app.close()

    def test_pipeline_hides_mentions_for_removed_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = AppConfig(
                root_dir=root,
                storage=StorageConfig(
                    database_path=root / "mentions.db",
                    latest_digest_path=root / "latest_digest.txt",
                    history_dir=root / "digests",
                ),
                run=RunConfig(),
                providers=ProviderConfig(
                    serpapi=SerpApiProviderConfig(enabled=False),
                ),
                keywords=(KeywordConfig(name="OpenAI", query="OpenAI"),),
            )
            app = SignalWatchApp(config)
            try:
                app.collectors = ()
                app.store.add_or_update(
                    _enriched(
                        keyword="Legacy keyword",
                        url="https://example.com/legacy",
                    )
                )

                report = app.run(now=datetime(2026, 7, 7, 15, 0, tzinfo=UTC))
                digest = config.storage.latest_digest_path.read_text(encoding="utf-8")

                self.assertEqual(report.visible_mentions, 0)
                self.assertNotIn("Legacy keyword", digest)
                self.assertIn("No mentions matched the current config in this window.", digest)
            finally:
                app.close()

    def test_pipeline_trims_inactive_keywords_from_visible_mentions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = AppConfig(
                root_dir=root,
                storage=StorageConfig(
                    database_path=root / "mentions.db",
                    latest_digest_path=root / "latest_digest.txt",
                    history_dir=root / "digests",
                ),
                run=RunConfig(),
                providers=ProviderConfig(
                    serpapi=SerpApiProviderConfig(enabled=False),
                ),
                keywords=(KeywordConfig(name="OpenAI", query="OpenAI"),),
            )
            app = SignalWatchApp(config)
            try:
                app.collectors = ()
                app.store.add_or_update(_enriched(keyword="OpenAI"))
                app.store.add_or_update(
                    _enriched(
                        keyword="Legacy keyword",
                        url="https://example.com/post?utm_source=legacy",
                        source="selected_web",
                    )
                )

                report = app.run(now=datetime(2026, 7, 7, 15, 0, tzinfo=UTC))
                digest = config.storage.latest_digest_path.read_text(encoding="utf-8")

                self.assertEqual(report.visible_mentions, 1)
                self.assertIn("Keywords: OpenAI", digest)
                self.assertNotIn("Legacy keyword", digest)
            finally:
                app.close()

    def test_pipeline_drops_results_from_rejected_domains(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = AppConfig(
                root_dir=root,
                storage=StorageConfig(
                    database_path=root / "mentions.db",
                    latest_digest_path=root / "latest_digest.txt",
                    history_dir=root / "digests",
                ),
                run=RunConfig(),
                providers=ProviderConfig(
                    serpapi=SerpApiProviderConfig(enabled=False),
                ),
                keywords=(
                    KeywordConfig(name="OpenAI", query="OpenAI", rejected_domains=("medium.com",)),
                ),
                filters=FilterConfig(rejected_domains=("facebook.com",)),
            )
            app = SignalWatchApp(config)
            try:
                app.collectors = (
                    _StaticCollector(
                        "google_search",
                        [
                            _raw("https://m.facebook.com/groups/post", "google_search"),
                            _raw("https://medium.com/@someone/post", "google_search"),
                            _raw("https://example.com/keeper", "google_search"),
                        ],
                    ),
                )
                summarizer = _CountingSummarizer()
                app.summarizer = summarizer

                report = app.run(now=datetime(2026, 7, 7, 15, 0, tzinfo=UTC))
                digest = config.storage.latest_digest_path.read_text(encoding="utf-8")

                self.assertEqual(report.raw_mentions, 1)
                self.assertEqual(summarizer.calls, 1)
                self.assertEqual(report.visible_mentions, 1)
                self.assertIn("https://example.com/keeper", digest)
                self.assertNotIn("facebook.com", digest)
                self.assertNotIn("medium.com", digest)
            finally:
                app.close()

    def test_pipeline_hides_stored_mentions_from_newly_rejected_domains(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = AppConfig(
                root_dir=root,
                storage=StorageConfig(
                    database_path=root / "mentions.db",
                    latest_digest_path=root / "latest_digest.txt",
                    history_dir=root / "digests",
                ),
                run=RunConfig(),
                providers=ProviderConfig(
                    serpapi=SerpApiProviderConfig(enabled=False),
                ),
                keywords=(KeywordConfig(name="OpenAI", query="OpenAI"),),
                filters=FilterConfig(rejected_domains=("facebook.com",)),
            )
            app = SignalWatchApp(config)
            try:
                app.collectors = ()
                app.store.add_or_update(_enriched(keyword="OpenAI", url="https://www.facebook.com/old"))
                app.store.add_or_update(_enriched(keyword="OpenAI", url="https://example.com/old"))

                report = app.run(now=datetime(2026, 7, 7, 15, 0, tzinfo=UTC))
                digest = config.storage.latest_digest_path.read_text(encoding="utf-8")

                self.assertEqual(report.visible_mentions, 1)
                self.assertIn("https://example.com/old", digest)
                self.assertNotIn("facebook.com", digest)
            finally:
                app.close()

    def test_pipeline_hides_stored_mentions_matching_raw_title_even_if_summary_paraphrased_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = AppConfig(
                root_dir=root,
                storage=StorageConfig(
                    database_path=root / "mentions.db",
                    latest_digest_path=root / "latest_digest.txt",
                    history_dir=root / "digests",
                ),
                run=RunConfig(),
                providers=ProviderConfig(
                    serpapi=SerpApiProviderConfig(enabled=False),
                ),
                keywords=(
                    KeywordConfig(name="acme", query="acme", exclude_terms=("roadrunner",)),
                ),
            )
            app = SignalWatchApp(config)
            try:
                app.collectors = ()
                app.store.add_or_update(
                    _enriched(
                        keyword="acme",
                        url="https://example.com/cartoon-recap",
                        meta={
                            "title": "Roadrunner cartoon recap",
                            "excerpt": "acme anvils everywhere",
                            "source_context": "google_search",
                        },
                    )
                )

                report = app.run(now=datetime(2026, 7, 7, 15, 0, tzinfo=UTC))
                digest = config.storage.latest_digest_path.read_text(encoding="utf-8")

                self.assertEqual(report.visible_mentions, 0)
                self.assertNotIn("cartoon-recap", digest)
            finally:
                app.close()

    def test_pipeline_builds_collectors_from_source_toggles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = AppConfig(
                root_dir=root,
                storage=StorageConfig(
                    database_path=root / "mentions.db",
                    latest_digest_path=root / "latest_digest.txt",
                    history_dir=root / "digests",
                ),
                run=RunConfig(),
                providers=ProviderConfig(
                    serpapi=SerpApiProviderConfig(enabled=True),
                    sources=SourceConfig(
                        google_news=False,
                        google_search=False,
                        selected_web=True,
                    ),
                ),
                keywords=(KeywordConfig(name="OpenAI", query="OpenAI", selected_domains=("github.com",)),),
            )
            app = SignalWatchApp(config)
            try:
                self.assertEqual(
                    tuple(collector.source_name for collector in app.collectors),
                    ("selected_web",),
                )
            finally:
                app.close()

    def test_pipeline_writes_new_only_digest_for_new_mentions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = AppConfig(
                root_dir=root,
                storage=StorageConfig(
                    database_path=root / "mentions.db",
                    latest_digest_path=root / "latest_digest.txt",
                    history_dir=root / "digests",
                ),
                run=RunConfig(),
                providers=ProviderConfig(
                    serpapi=SerpApiProviderConfig(enabled=False),
                ),
                keywords=(KeywordConfig(name="OpenAI", query="OpenAI"),),
            )
            app = SignalWatchApp(config)
            try:
                app.store.add_or_update(
                    _enriched(
                        keyword="OpenAI",
                        url="https://example.com/existing",
                    )
                )
                app.collectors = (
                    _StaticCollector(
                        "google_search",
                        [
                            _raw("https://example.com/existing", "google_search"),
                            _raw("https://example.com/new", "google_search"),
                        ],
                    ),
                )
                app.summarizer = _CountingSummarizer()

                report = app.run(now=datetime(2026, 7, 8, 15, 0, tzinfo=UTC))
                new_digest = report.new_latest_digest_path.read_text(encoding="utf-8")

                self.assertEqual(report.new_mentions, 1)
                self.assertIn("New mention digest", new_digest)
                self.assertIn("https://example.com/new", new_digest)
                self.assertNotIn("https://example.com/existing", new_digest)
            finally:
                app.close()

    def test_pipeline_uses_batch_enrichment_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = AppConfig(
                root_dir=root,
                storage=StorageConfig(
                    database_path=root / "mentions.db",
                    latest_digest_path=root / "latest_digest.txt",
                    history_dir=root / "digests",
                ),
                run=RunConfig(),
                providers=ProviderConfig(
                    serpapi=SerpApiProviderConfig(enabled=False),
                ),
                keywords=(KeywordConfig(name="OpenAI", query="OpenAI"),),
            )
            app = SignalWatchApp(config)
            try:
                app.collectors = (
                    _StaticCollector(
                        "google_search",
                        [
                            _raw("https://example.com/first", "google_search"),
                            _raw("https://example.com/second", "selected_web"),
                        ],
                    ),
                )
                summarizer = _BatchSummarizer()
                app.summarizer = summarizer

                report = app.run(now=datetime(2026, 7, 8, 15, 0, tzinfo=UTC))

                self.assertEqual(report.new_mentions, 2)
                self.assertEqual(summarizer.start_calls, 1)
                self.assertEqual(summarizer.batch_calls, 1)
                self.assertEqual(summarizer.finish_calls, 1)
                self.assertEqual(report.summarizer_counts, (("local_llm", 2),))
            finally:
                app.close()


class _StaticCollector:
    def __init__(self, source_name: str, mentions: list[RawMention]) -> None:
        self.source_name = source_name
        self._mentions = mentions

    def collect(self, keyword, config, since, now) -> list[RawMention]:
        return list(self._mentions)


class _CountingSummarizer:
    def __init__(self) -> None:
        self.calls = 0

    def enrich(self, raw: RawMention, discovered_at: datetime) -> EnrichedMention:
        self.calls += 1
        return EnrichedMention(
            keyword=raw.keyword,
            source=raw.source,
            url=raw.url,
            canonical_url=canonicalize_url(raw.url),
            published_at=raw.published_at,
            discovered_at=discovered_at,
            summary="Summarized once.",
            sentiment=Sentiment.NEUTRAL,
            suggested_action=SuggestedAction.WATCH,
            why_it_matters="It is recent enough to keep on the radar.",
            score=55.0,
        )


class _BatchSummarizer:
    def __init__(self) -> None:
        self.start_calls = 0
        self.batch_calls = 0
        self.finish_calls = 0

    def start_run(self) -> None:
        self.start_calls += 1

    def finish_run(self) -> None:
        self.finish_calls += 1

    def enrich_many(self, raws: list[RawMention], discovered_at: datetime) -> list[EnrichedMention]:
        self.batch_calls += 1
        return [
            EnrichedMention(
                keyword=raw.keyword,
                source=raw.source,
                url=raw.url,
                canonical_url=canonicalize_url(raw.url),
                published_at=raw.published_at,
                discovered_at=discovered_at,
                summary=f"Batch summary for {raw.keyword}.",
                sentiment=Sentiment.NEUTRAL,
                suggested_action=SuggestedAction.WATCH,
                why_it_matters="It is recent enough to keep on the radar.",
                score=55.0,
                meta={"summarizer": "local_llm"},
            )
            for raw in raws
        ]


def _raw(url: str, source: str) -> RawMention:
    return RawMention(
        keyword="OpenAI",
        source=source,
        url=url,
        published_at=datetime(2026, 7, 7, 14, 0, tzinfo=UTC),
        title="OpenAI mention",
        excerpt="A matching result.",
        source_context=source,
    )


def _enriched(
    *,
    keyword: str,
    url: str = "https://example.com/post",
    source: str = "google_search",
    meta: dict | None = None,
) -> EnrichedMention:
    discovered_at = datetime(2026, 7, 7, 14, 0, tzinfo=UTC)
    return EnrichedMention(
        keyword=keyword,
        source=source,
        url=url,
        canonical_url=canonicalize_url(url),
        published_at=discovered_at,
        discovered_at=discovered_at,
        summary="Summarized once.",
        sentiment=Sentiment.NEUTRAL,
        suggested_action=SuggestedAction.WATCH,
        why_it_matters="It is recent enough to keep on the radar.",
        score=55.0,
        meta=meta or {},
    )


if __name__ == "__main__":
    unittest.main()
