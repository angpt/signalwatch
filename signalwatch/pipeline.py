from __future__ import annotations

from datetime import datetime, timedelta

from signalwatch.digest import render_digest, write_digest
from signalwatch.http import HttpClient
from signalwatch.models import AppConfig, CollectorFailure, RunReport, StoredMention, UTC
from signalwatch.providers import GoogleNewsCollector, GoogleSearchCollector, SelectedWebCollector
from signalwatch.storage import MentionStore
from signalwatch.summarize import build_summarizer
from signalwatch.utils import canonicalize_url, is_excluded_mention, is_rejected_domain


class SignalWatchApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._keywords_by_name = {keyword.name: keyword for keyword in config.keywords}
        self.http_client = HttpClient(
            timeout_seconds=config.run.timeout_seconds,
            user_agent=config.run.user_agent,
        )
        self.store = MentionStore(config.storage.database_path)
        self.collectors = self._build_collectors()
        self.summarizer = build_summarizer(
            config.summarizer,
            user_agent=config.run.user_agent,
        )

    def close(self) -> None:
        self.store.close()

    def _build_collectors(self) -> tuple:
        sources = self.config.providers.sources
        collectors = []
        if sources.google_news:
            collectors.append(GoogleNewsCollector(self.http_client))
        if sources.google_search:
            collectors.append(GoogleSearchCollector(self.http_client))
        if sources.selected_web:
            collectors.append(SelectedWebCollector(self.http_client))
        return tuple(collectors)

    def run(self, now: datetime | None = None) -> RunReport:
        generated_at = (now or datetime.now(UTC)).astimezone(UTC)
        since = generated_at - timedelta(hours=self.config.run.since_hours)
        self.store.purge(generated_at, retention_days=self.config.storage.retention_days)

        raw_mentions = 0
        new_mentions = 0
        failures: list[CollectorFailure] = []
        pending_by_canonical: dict[str, list] = {}

        try:
            self._start_summarizer_run()

            for keyword in self.config.keywords:
                for collector in self.collectors:
                    try:
                        collected = collector.collect(keyword, self.config, since, generated_at)
                    except Exception as exc:
                        failures.append(
                            CollectorFailure(
                                source=collector.source_name,
                                keyword=keyword.name,
                                error=str(exc),
                            )
                        )
                        continue
                    collected = [
                        raw
                        for raw in collected
                        if not is_rejected_domain(raw.url, self._rejected_domains_for(keyword.name))
                        and not is_excluded_mention(
                            raw.url,
                            (raw.title, raw.excerpt, raw.source_context),
                            keyword.exclude_terms,
                        )
                    ]
                    raw_mentions += len(collected)
                    for raw in collected:
                        canonical_url = canonicalize_url(raw.url)
                        if self.store.record_existing_match(
                            canonical_url,
                            keyword=raw.keyword,
                            source=raw.source,
                            published_at=raw.published_at,
                        ):
                            continue
                        pending_by_canonical.setdefault(canonical_url, []).append(raw)

            pending_raws = [group[0] for group in pending_by_canonical.values()]
            enriched_mentions = self._enrich_pending_mentions(pending_raws, generated_at)

            for enriched in enriched_mentions:
                created = self.store.add_or_update(enriched)
                if created:
                    new_mentions += 1
                for duplicate in pending_by_canonical.get(enriched.canonical_url, [])[1:]:
                    self.store.record_existing_match(
                        enriched.canonical_url,
                        keyword=duplicate.keyword,
                        source=duplicate.source,
                        published_at=duplicate.published_at,
                    )
        finally:
            self._finish_summarizer_run()

        visible_mentions = self._filter_visible_mentions(self.store.list_since(since))
        new_mentions_visible = self._filter_visible_mentions(self.store.list_discovered_at(generated_at))
        summarizer_counts = self.store.summarizer_counts_for_discovered_at(generated_at)
        digest = render_digest(
            visible_mentions,
            generated_at=generated_at,
            since_hours=self.config.run.since_hours,
            failures=failures,
        )
        new_digest = render_digest(
            new_mentions_visible,
            generated_at=generated_at,
            since_hours=self.config.run.since_hours,
            failures=failures,
            title="New mention digest",
            window_label="this run",
            empty_message="No new mentions matched the current config in this run.",
        )
        latest_path = self.config.storage.latest_digest_path
        history_path = self.config.storage.history_dir / f"digest-{generated_at.strftime('%Y%m%dT%H%M%SZ')}.txt"
        new_latest_path = self.config.storage.new_latest_digest_path or latest_path.with_name(
            f"{latest_path.stem}_new{latest_path.suffix}"
        )
        new_history_dir = self.config.storage.new_history_dir or self.config.storage.history_dir.with_name(
            f"{self.config.storage.history_dir.name}_new"
        )
        new_history_path = new_history_dir / f"digest-{generated_at.strftime('%Y%m%dT%H%M%SZ')}.txt"
        write_digest(latest_path, digest)
        write_digest(history_path, digest)
        write_digest(new_latest_path, new_digest)
        write_digest(new_history_path, new_digest)
        return RunReport(
            generated_at=generated_at,
            raw_mentions=raw_mentions,
            new_mentions=new_mentions,
            visible_mentions=len(visible_mentions),
            latest_digest_path=latest_path,
            history_digest_path=history_path,
            new_latest_digest_path=new_latest_path,
            new_history_digest_path=new_history_path,
            summarizer_counts=summarizer_counts,
            failures=tuple(failures),
        )

    def _exclude_terms_for(self, keyword_name: str) -> tuple[str, ...]:
        keyword = self._keywords_by_name.get(keyword_name)
        return keyword.exclude_terms if keyword else ()

    def _rejected_domains_for(self, keyword_name: str) -> tuple[str, ...]:
        keyword = self._keywords_by_name.get(keyword_name)
        keyword_domains = keyword.rejected_domains if keyword else ()
        return self.config.filters.rejected_domains + keyword_domains

    def _filter_visible_mentions(self, mentions: list[StoredMention]) -> list[StoredMention]:
        """Hide mentions that the current config would no longer collect.

        Keywords and rejected domains can change between runs, so already stored
        rows are re-checked here instead of only at collection time.
        """
        filtered: list[StoredMention] = []
        for mention in mentions:
            current_keywords = tuple(
                keyword
                for keyword in mention.keywords
                if keyword in self._keywords_by_name
                and not is_rejected_domain(mention.url, self._rejected_domains_for(keyword))
                and not is_excluded_mention(
                    mention.url,
                    (mention.raw_text,),
                    self._exclude_terms_for(keyword),
                )
            )
            if not current_keywords:
                continue
            if current_keywords == mention.keywords:
                filtered.append(mention)
                continue
            filtered.append(
                StoredMention(
                    id=mention.id,
                    source=mention.source,
                    url=mention.url,
                    published_at=mention.published_at,
                    discovered_at=mention.discovered_at,
                    summary=mention.summary,
                    sentiment=mention.sentiment,
                    suggested_action=mention.suggested_action,
                    why_it_matters=mention.why_it_matters,
                    score=mention.score,
                    keywords=current_keywords,
                    sources=mention.sources,
                    raw_text=mention.raw_text,
                )
            )
        return filtered

    def _start_summarizer_run(self) -> None:
        start_run = getattr(self.summarizer, "start_run", None)
        if callable(start_run):
            start_run()

    def _finish_summarizer_run(self) -> None:
        finish_run = getattr(self.summarizer, "finish_run", None)
        if callable(finish_run):
            finish_run()

    def _enrich_pending_mentions(self, raws, generated_at: datetime):
        if not raws:
            return []
        enrich_many = getattr(self.summarizer, "enrich_many", None)
        if callable(enrich_many):
            return list(enrich_many(raws, generated_at))
        return [self.summarizer.enrich(raw, generated_at) for raw in raws]
