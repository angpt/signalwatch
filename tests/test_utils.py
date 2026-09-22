from __future__ import annotations

from datetime import datetime
import unittest

from signalwatch.models import UTC
from signalwatch.summarize import HeuristicSummarizer
from signalwatch.utils import (
    canonicalize_url,
    is_excluded_mention,
    is_rejected_domain,
    normalize_domain,
    parse_datetime,
)
from signalwatch.models import RawMention, Sentiment, SuggestedAction


class UtilsTests(unittest.TestCase):
    def test_canonicalize_url_strips_tracking_params(self) -> None:
        self.assertEqual(
            canonicalize_url("https://example.com/path/?utm_source=x&ref=home&id=1"),
            "https://example.com/path?id=1",
        )

    def test_canonicalize_url_strips_locale_param(self) -> None:
        base = "https://forum.example.com/threads/some-thread/"
        canonical = canonicalize_url(base)
        for locale in ("it", "pt-br", "es-419"):
            self.assertEqual(canonicalize_url(f"{base}?tl={locale}"), canonical)

    def test_excluded_term_matches_text(self) -> None:
        terms = ("banana", "fight night", "MX-5050")
        self.assertTrue(
            is_excluded_mention("https://example.com/a", ("Scaling Banana Exports",), terms)
        )
        self.assertTrue(
            is_excluded_mention("https://example.com/a", ("Acme Widgets headline fight night in Vegas",), terms)
        )
        self.assertTrue(
            is_excluded_mention("https://example.com/a", ("The Acme MX-5050 tape deck",), terms)
        )

    def test_excluded_term_matches_url_slug(self) -> None:
        self.assertTrue(
            is_excluded_mention(
                "https://social.example.com/posts/jane-acme-bananaexport_building-an-export-framework",
                (None,),
                ("banana",),
            )
        )

    def test_excluded_term_respects_word_boundaries(self) -> None:
        self.assertFalse(
            is_excluded_mention("https://example.com/a", ("Acme gateway routing",), ("judo", "tape deck"))
        )

    def test_no_exclude_terms_keeps_everything(self) -> None:
        self.assertFalse(is_excluded_mention("https://example.com/banana", ("banana",), ()))

    def test_parse_relative_datetime(self) -> None:
        now = datetime(2026, 7, 7, 12, 0, tzinfo=UTC)
        parsed = parse_datetime("2 hours ago", now=now)
        self.assertEqual(parsed, datetime(2026, 7, 7, 10, 0, tzinfo=UTC))

    def test_summarizer_assigns_negative_reply(self) -> None:
        summarizer = HeuristicSummarizer()
        raw = RawMention(
            keyword="OpenAI",
            source="google_search",
            url="https://forum.example.com/threads/example/",
            published_at=datetime(2026, 7, 7, 11, 0, tzinfo=UTC),
            title="Is OpenAI worth it or are there better alternatives?",
            excerpt="The onboarding is confusing and I hit a bug.",
            source_context="forum.example.com",
        )

        enriched = summarizer.enrich(raw, datetime(2026, 7, 7, 12, 0, tzinfo=UTC))

        self.assertEqual(enriched.sentiment, Sentiment.NEGATIVE)
        self.assertEqual(enriched.suggested_action, SuggestedAction.REPLY)
        self.assertEqual(enriched.meta["summarizer"], "heuristic")
        self.assertEqual(enriched.meta["summarizer_runner"], "heuristic")

    def test_normalize_domain_strips_scheme_path_and_www(self) -> None:
        self.assertEqual(normalize_domain("https://www.Facebook.com/pages"), "facebook.com")
        self.assertEqual(normalize_domain(" instagram.com. "), "instagram.com")
        self.assertEqual(normalize_domain("  "), "")

    def test_is_rejected_domain_matches_subdomains_only(self) -> None:
        rejected = ("facebook.com", "instagram.com")

        self.assertTrue(is_rejected_domain("https://www.facebook.com/some/post", rejected))
        self.assertTrue(is_rejected_domain("https://m.facebook.com/post?id=1", rejected))
        self.assertTrue(is_rejected_domain("http://instagram.com:8080/p/abc", rejected))
        self.assertFalse(is_rejected_domain("https://notfacebook.com/post", rejected))
        self.assertFalse(is_rejected_domain("https://facebook.com.evil.example/post", rejected))
        self.assertFalse(is_rejected_domain("https://example.com/facebook.com", rejected))
        self.assertFalse(is_rejected_domain("https://www.facebook.com/post", ()))


if __name__ == "__main__":
    unittest.main()
