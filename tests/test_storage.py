from __future__ import annotations

from datetime import datetime
import tempfile
import unittest
from pathlib import Path

from signalwatch.models import EnrichedMention, Sentiment, SuggestedAction, UTC
from signalwatch.storage import MentionStore


class StorageTests(unittest.TestCase):
    def test_store_dedupes_by_canonical_url_and_keeps_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MentionStore(Path(temp_dir) / "mentions.db")
            now = datetime(2026, 7, 7, 13, 0, tzinfo=UTC)

            first = _mention(keyword="OpenAI", now=now, source="google_search")
            second = _mention(
                keyword="ChatGPT",
                now=now,
                source="selected_web",
                url="https://example.com/post?utm_source=x",
            )

            self.assertTrue(store.add_or_update(first))
            self.assertFalse(store.add_or_update(second))

            mentions = store.list_since(datetime(2026, 7, 6, 13, 0, tzinfo=UTC))
            self.assertEqual(len(mentions), 1)
            self.assertEqual(set(mentions[0].keywords), {"ChatGPT", "OpenAI"})
            self.assertEqual(set(mentions[0].sources), {"google_search", "selected_web"})
            store.close()

    def test_duplicate_updates_keep_original_discovered_at(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MentionStore(Path(temp_dir) / "mentions.db")
            first_seen = datetime(2026, 7, 7, 13, 0, tzinfo=UTC)
            second_seen = datetime(2026, 7, 8, 13, 0, tzinfo=UTC)

            self.assertTrue(store.add_or_update(_mention(keyword="OpenAI", now=first_seen)))
            self.assertFalse(store.add_or_update(_mention(keyword="OpenAI", now=second_seen)))

            mentions = store.list_since(datetime(2026, 7, 7, 0, 0, tzinfo=UTC))
            self.assertEqual(len(mentions), 1)
            self.assertEqual(mentions[0].discovered_at, first_seen)

            later_window_mentions = store.list_since(datetime(2026, 7, 8, 0, 0, tzinfo=UTC))
            self.assertEqual(later_window_mentions, [])
            store.close()


def _mention(
    *,
    keyword: str,
    now: datetime,
    source: str = "google_search",
    url: str = "https://example.com/post",
) -> EnrichedMention:
    return EnrichedMention(
        keyword=keyword,
        source=source,
        url=url,
        canonical_url="https://example.com/post",
        published_at=now,
        discovered_at=now,
        summary="A result mentions the keyword.",
        sentiment=Sentiment.NEUTRAL,
        suggested_action=SuggestedAction.WATCH,
        why_it_matters="It is recent enough to keep on the radar.",
        score=55.0,
        external_id=None,
        meta={},
    )


if __name__ == "__main__":
    unittest.main()
