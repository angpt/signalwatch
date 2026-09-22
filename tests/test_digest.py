from __future__ import annotations

from datetime import datetime
import unittest

from signalwatch.digest import render_digest
from signalwatch.models import StoredMention, SuggestedAction, Sentiment, UTC


class DigestTests(unittest.TestCase):
    def test_render_digest_includes_expected_fields(self) -> None:
        mention = StoredMention(
            id=1,
            source="google_search",
            url="https://forum.example.com/threads/example/",
            published_at=datetime(2026, 7, 7, 12, 0, tzinfo=UTC),
            discovered_at=datetime(2026, 7, 7, 12, 5, tzinfo=UTC),
            summary="A forum thread mentions OpenAI and compares it with alternatives.",
            sentiment=Sentiment.NEGATIVE,
            suggested_action=SuggestedAction.REPLY,
            why_it_matters="The wording suggests active evaluation or a request for help.",
            score=82.0,
            keywords=("OpenAI",),
        )

        output = render_digest([mention], generated_at=datetime(2026, 7, 7, 13, 0, tzinfo=UTC), since_hours=24)

        self.assertIn("Daily mention digest", output)
        self.assertIn("Suggested action: reply", output)
        self.assertIn("Link: https://forum.example.com/threads/example/", output)

    def test_render_digest_supports_custom_title_and_empty_message(self) -> None:
        output = render_digest(
            [],
            generated_at=datetime(2026, 7, 7, 13, 0, tzinfo=UTC),
            since_hours=24,
            title="New mention digest",
            window_label="this run",
            empty_message="No new mentions matched the current config in this run.",
        )

        self.assertIn("New mention digest", output)
        self.assertIn("Window: this run", output)
        self.assertIn("No new mentions matched the current config in this run.", output)


if __name__ == "__main__":
    unittest.main()
