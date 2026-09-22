from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from signalwatch.models import (
    AppConfig,
    KeywordConfig,
    ProviderConfig,
    RawMention,
    RunConfig,
    SerpApiProviderConfig,
    StorageConfig,
    UTC,
)
from signalwatch.providers.serpapi import (
    GoogleNewsCollector,
    GoogleSearchCollector,
    SelectedWebCollector,
)


class FakeHttpClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def get_json(self, url: str, *, params: dict | None = None, headers: dict | None = None) -> dict:
        self.calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        return self.payload


class PerDomainHttpClient:
    """Returns a different payload, or raises, depending on the site: domain."""

    def __init__(self, by_domain: dict[str, object]) -> None:
        self.by_domain = by_domain
        self.calls: list[dict] = []

    def get_json(self, url: str, *, params: dict | None = None, headers: dict | None = None) -> dict:
        params = params or {}
        self.calls.append({"url": url, "params": params, "headers": headers or {}})
        query = str(params.get("q", ""))
        for domain, outcome in self.by_domain.items():
            if f"site:{domain} " in query:
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        raise AssertionError(f"unexpected query: {query}")


class SerpApiCollectorTests(unittest.TestCase):
    def test_google_search_uses_daily_tbs(self) -> None:
        collector = GoogleSearchCollector(FakeHttpClient({"organic_results": []}))
        config = _build_config()
        keyword = KeywordConfig(name="OpenAI", query="OpenAI")

        with patch.dict(os.environ, {"SERPAPI_API_KEY": "secret"}, clear=False):
            collector.collect(keyword, config, _dt(2026, 7, 6, 15), _dt(2026, 7, 7, 15))

        self.assertEqual(collector.http_client.calls[0]["params"]["tbs"], "qdr:d")

    def test_in_band_errors_raise_collector_failures(self) -> None:
        collector = GoogleSearchCollector(FakeHttpClient({"error": "quota exceeded"}))
        config = _build_config()
        keyword = KeywordConfig(name="OpenAI", query="OpenAI")

        with patch.dict(os.environ, {"SERPAPI_API_KEY": "secret"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "SerpAPI error: quota exceeded"):
                collector.collect(keyword, config, _dt(2026, 7, 6, 15), _dt(2026, 7, 7, 15))

    def test_google_news_filters_old_results_and_uses_run_time_for_relative_dates(self) -> None:
        payload = {
            "news_results": [
                {"title": "Fresh result", "link": "https://example.com/fresh", "date": "2 hours ago"},
                {"title": "Old result", "link": "https://example.com/old", "date": "2 days ago"},
            ]
        }
        collector = GoogleNewsCollector(FakeHttpClient(payload))
        config = _build_config()
        keyword = KeywordConfig(name="OpenAI", query="OpenAI", max_results_per_source=5)

        with patch.dict(os.environ, {"SERPAPI_API_KEY": "secret"}, clear=False):
            results = collector.collect(keyword, config, _dt(2026, 7, 6, 15), _dt(2026, 7, 7, 15))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.com/fresh")
        self.assertEqual(results[0].published_at, _dt(2026, 7, 7, 13))

    def test_google_news_caps_results_to_requested_limit(self) -> None:
        payload = {
            "news_results": [
                {"title": "A", "link": "https://example.com/a", "date": "1 hour ago"},
                {"title": "B", "link": "https://example.com/b", "date": "2 hours ago"},
            ]
        }
        collector = GoogleNewsCollector(FakeHttpClient(payload))
        config = _build_config()
        keyword = KeywordConfig(name="OpenAI", query="OpenAI", max_results_per_source=1)

        with patch.dict(os.environ, {"SERPAPI_API_KEY": "secret"}, clear=False):
            results = collector.collect(keyword, config, _dt(2026, 7, 6, 15), _dt(2026, 7, 7, 15))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.com/a")

    def test_selected_web_keeps_results_when_one_domain_fails(self) -> None:
        client = PerDomainHttpClient(
            {
                "github.com": TimeoutError("The read operation timed out"),
                "news.ycombinator.com": {
                    "organic_results": [{"title": "B", "link": "https://news.ycombinator.com/b"}]
                },
            }
        )
        collector = SelectedWebCollector(client)
        config = _build_config()
        keyword = KeywordConfig(
            name="OpenAI",
            query="OpenAI",
            selected_domains=("github.com", "news.ycombinator.com"),
        )

        with patch.dict(os.environ, {"SERPAPI_API_KEY": "secret"}, clear=False):
            results = collector.collect(keyword, config, _dt(2026, 7, 6, 15), _dt(2026, 7, 7, 15))

        self.assertEqual([item.url for item in results], ["https://news.ycombinator.com/b"])

    def test_selected_web_queries_every_domain_after_an_early_failure(self) -> None:
        client = PerDomainHttpClient(
            {
                "github.com": TimeoutError("The read operation timed out"),
                "lobste.rs": {"organic_results": []},
                "news.ycombinator.com": {"organic_results": []},
            }
        )
        collector = SelectedWebCollector(client)
        config = _build_config()
        keyword = KeywordConfig(
            name="OpenAI",
            query="OpenAI",
            selected_domains=("github.com", "lobste.rs", "news.ycombinator.com"),
        )

        with patch.dict(os.environ, {"SERPAPI_API_KEY": "secret"}, clear=False):
            collector.collect(keyword, config, _dt(2026, 7, 6, 15), _dt(2026, 7, 7, 15))

        self.assertEqual(len(client.calls), 3)

    def test_selected_web_raises_only_when_every_domain_fails(self) -> None:
        client = PerDomainHttpClient(
            {
                "github.com": TimeoutError("The read operation timed out"),
                "lobste.rs": TimeoutError("The read operation timed out"),
            }
        )
        collector = SelectedWebCollector(client)
        config = _build_config()
        keyword = KeywordConfig(
            name="OpenAI",
            query="OpenAI",
            selected_domains=("github.com", "lobste.rs"),
        )

        with patch.dict(os.environ, {"SERPAPI_API_KEY": "secret"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "github.com: .*lobste.rs: "):
                collector.collect(keyword, config, _dt(2026, 7, 6, 15), _dt(2026, 7, 7, 15))


def _build_config() -> AppConfig:
    temp_dir = Path(tempfile.gettempdir()) / "signalwatch-tests"
    return AppConfig(
        root_dir=temp_dir,
        storage=StorageConfig(
            database_path=temp_dir / "mentions.db",
            latest_digest_path=temp_dir / "latest_digest.txt",
            history_dir=temp_dir / "digests",
            retention_days=30,
        ),
        run=RunConfig(
            since_hours=24,
            timeout_seconds=5,
            max_results_per_query=3,
            user_agent="signalwatch-test",
        ),
        providers=ProviderConfig(
            serpapi=SerpApiProviderConfig(enabled=True, api_key_env="SERPAPI_API_KEY"),
        ),
        keywords=(KeywordConfig(name="OpenAI", query="OpenAI"),),
    )


def _dt(year: int, month: int, day: int, hour: int) -> datetime:
    return datetime(year, month, day, hour, 0, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
