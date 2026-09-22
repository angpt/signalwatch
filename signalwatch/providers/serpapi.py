from __future__ import annotations

from datetime import datetime
import os

from signalwatch.http import HttpClient
from signalwatch.models import AppConfig, KeywordConfig, RawMention
from signalwatch.providers.search_common import collect_per_domain, filter_since, parse_search_results
from signalwatch.utils import build_search_query, collapse_whitespace


class _SerpApiCollectorBase:
    engine = ""
    source_name = ""

    def __init__(self, http_client: HttpClient) -> None:
        self.http_client = http_client

    def _api_key(self, config: AppConfig) -> str:
        settings = config.providers.serpapi
        if not settings.enabled:
            raise RuntimeError("SerpAPI is disabled.")
        api_key = os.getenv(settings.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing SerpAPI API key in ${settings.api_key_env}.")
        return api_key

    def _search(self, config: AppConfig, *, query: str, num: int, tbs: str | None = None) -> dict:
        settings = config.providers.serpapi
        api_key = self._api_key(config)
        params = {
            "api_key": api_key,
            "engine": self.engine,
            "q": query,
            "num": num,
            "gl": settings.gl,
            "hl": settings.hl,
        }
        if tbs:
            params["tbs"] = tbs
        payload = self.http_client.get_json(
            "https://serpapi.com/search.json",
            params=params,
        )
        if payload.get("error"):
            raise RuntimeError(f"SerpAPI error: {collapse_whitespace(str(payload['error']))}")
        return payload


class GoogleSearchCollector(_SerpApiCollectorBase):
    engine = "google"
    source_name = "google_search"

    def collect(
        self,
        keyword: KeywordConfig,
        config: AppConfig,
        since: datetime,
        now: datetime,
    ) -> list[RawMention]:
        if not config.providers.serpapi.enabled:
            return []
        num = keyword.max_results_per_source or config.run.max_results_per_query
        query = build_search_query(keyword.query, keyword.exclude_terms)
        payload = self._search(config, query=query, num=num, tbs="qdr:d")
        results = parse_search_results(payload.get("organic_results", []), keyword, self.source_name, now=now)
        return filter_since(results, since)


class GoogleNewsCollector(_SerpApiCollectorBase):
    engine = "google_news"
    source_name = "google_news"

    def collect(
        self,
        keyword: KeywordConfig,
        config: AppConfig,
        since: datetime,
        now: datetime,
    ) -> list[RawMention]:
        if not config.providers.serpapi.enabled:
            return []
        num = keyword.max_results_per_source or config.run.max_results_per_query
        query = build_search_query(keyword.query, keyword.exclude_terms)
        payload = self._search(config, query=query, num=num)
        items = payload.get("news_results") or []
        flattened: list[dict] = []
        for item in items:
            stories = item.get("stories")
            if stories:
                flattened.extend(stories)
            else:
                flattened.append(item)
        results = parse_search_results(flattened[:num], keyword, self.source_name, now=now)
        return filter_since(results, since)


class SelectedWebCollector(_SerpApiCollectorBase):
    engine = "google"
    source_name = "selected_web"

    def collect(
        self,
        keyword: KeywordConfig,
        config: AppConfig,
        since: datetime,
        now: datetime,
    ) -> list[RawMention]:
        if not config.providers.serpapi.enabled or not keyword.selected_domains:
            return []
        num = keyword.max_results_per_source or config.run.max_results_per_query
        query_base = build_search_query(keyword.query, keyword.exclude_terms)

        def fetch(domain: str) -> list[RawMention]:
            payload = self._search(config, query=f"site:{domain} {query_base}", num=num, tbs="qdr:d")
            parsed = parse_search_results(payload.get("organic_results", []), keyword, self.source_name, now=now)
            return filter_since(parsed, since)

        return collect_per_domain(keyword.selected_domains, fetch)
