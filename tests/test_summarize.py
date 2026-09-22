from __future__ import annotations

from datetime import datetime
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from signalwatch.models import (
    LocalLLMConfig,
    RawMention,
    Sentiment,
    SuggestedAction,
    SummarizerConfig,
    SummarizerProvider,
    UTC,
)
from signalwatch.summarize import (
    HeuristicSummarizer,
    LocalLLMSummarizer,
    SummarizerError,
    build_summarizer,
)

DISCOVERED_AT = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


class _ChatHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible /chat/completions server that echoes one item per mention."""

    requests: list[dict] = []
    auth_headers: list[str | None] = []
    reply_override: str | None = None

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        type(self).requests.append({"path": self.path, "body": body})
        type(self).auth_headers.append(self.headers.get("Authorization"))

        if type(self).reply_override is not None:
            content = type(self).reply_override
        else:
            mentions = json.loads(body["messages"][1]["content"])["mentions"]
            content = json.dumps(
                {
                    "items": [
                        {
                            "mention_index": item["mention_index"],
                            "summary": f"LLM summary for {item['keyword']}.",
                            "sentiment": "positive",
                            "suggested_action": "share",
                            "why_it_matters": "The local model handled this mention.",
                            "score": 77,
                        }
                        for item in mentions
                    ]
                }
            )
        payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        return None


class LocalLLMSummarizerTests(unittest.TestCase):
    def setUp(self) -> None:
        _ChatHandler.requests = []
        _ChatHandler.auth_headers = []
        _ChatHandler.reply_override = None
        self.server = HTTPServer(("127.0.0.1", 0), _ChatHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _config(self, **overrides) -> SummarizerConfig:
        llm = LocalLLMConfig(
            base_url=self.base_url,
            model="test-model",
            request_timeout_seconds=10,
            **overrides,
        )
        return SummarizerConfig(provider=SummarizerProvider.LOCAL_LLM, llm=llm)

    def test_build_summarizer_returns_heuristic_by_default(self) -> None:
        self.assertIsInstance(build_summarizer(SummarizerConfig()), HeuristicSummarizer)

    def test_enrich_many_batches_mentions_and_merges_model_output(self) -> None:
        summarizer = build_summarizer(self._config(batch_size=2))
        self.assertIsInstance(summarizer, LocalLLMSummarizer)

        enriched = summarizer.enrich_many(
            [_raw("alpha", "https://example.com/1"), _raw("beta", "https://example.com/2"), _raw("gamma", "https://example.com/3")],
            DISCOVERED_AT,
        )

        self.assertEqual(len(_ChatHandler.requests), 2)  # batch_size=2 -> 2 requests for 3 mentions
        self.assertEqual(_ChatHandler.requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(_ChatHandler.requests[0]["body"]["model"], "test-model")
        self.assertEqual([item.summary for item in enriched], [f"LLM summary for {k}." for k in ("alpha", "beta", "gamma")])
        self.assertEqual(enriched[0].sentiment, Sentiment.POSITIVE)
        self.assertEqual(enriched[0].suggested_action, SuggestedAction.SHARE)
        self.assertEqual(enriched[0].score, 77.0)
        self.assertEqual(enriched[0].meta["summarizer"], "local_llm")
        self.assertEqual(enriched[0].meta["model"], "test-model")

    def test_extracts_json_from_fenced_model_output(self) -> None:
        inner = json.dumps(
            {"items": [{"mention_index": 0, "summary": "Fenced.", "sentiment": "neutral", "suggested_action": "watch", "why_it_matters": "x", "score": 10}]}
        )
        _ChatHandler.reply_override = f"Sure! Here you go:\n```json\n{inner}\n```"

        enriched = build_summarizer(self._config()).enrich_many([_raw()], DISCOVERED_AT)

        self.assertEqual(enriched[0].summary, "Fenced.")
        self.assertEqual(enriched[0].meta["summarizer"], "local_llm")

    def test_falls_back_to_heuristic_on_invalid_output(self) -> None:
        _ChatHandler.reply_override = "not json at all"

        enriched = build_summarizer(self._config()).enrich_many([_raw()], DISCOVERED_AT)

        self.assertEqual(enriched[0].meta["summarizer"], "heuristic_fallback")
        self.assertEqual(enriched[0].meta["summarizer_runner"], "local_llm")
        self.assertIn("summarizer_error", enriched[0].meta)

    def test_falls_back_when_server_is_unreachable(self) -> None:
        config = SummarizerConfig(
            provider=SummarizerProvider.LOCAL_LLM,
            llm=LocalLLMConfig(base_url="http://127.0.0.1:1/v1", model="m", request_timeout_seconds=2),
        )

        enriched = build_summarizer(config).enrich_many([_raw()], DISCOVERED_AT)

        self.assertEqual(enriched[0].meta["summarizer"], "heuristic_fallback")

    def test_raises_without_fallback(self) -> None:
        _ChatHandler.reply_override = "not json at all"
        config = SummarizerConfig(
            provider=SummarizerProvider.LOCAL_LLM,
            fallback_to_heuristic=False,
            llm=LocalLLMConfig(base_url=self.base_url, model="m", request_timeout_seconds=10),
        )

        with self.assertRaises(SummarizerError):
            build_summarizer(config).enrich_many([_raw()], DISCOVERED_AT)

    def test_incomplete_batch_response_falls_back(self) -> None:
        _ChatHandler.reply_override = json.dumps({"items": []})

        enriched = build_summarizer(self._config()).enrich_many([_raw()], DISCOVERED_AT)

        self.assertEqual(enriched[0].meta["summarizer"], "heuristic_fallback")

    def test_sends_bearer_token_when_api_key_env_is_set(self) -> None:
        summarizer = build_summarizer(self._config(api_key_env="SIGNALWATCH_TEST_KEY"))

        with patch.dict("os.environ", {"SIGNALWATCH_TEST_KEY": "secret"}):
            summarizer.enrich_many([_raw()], DISCOVERED_AT)

        self.assertEqual(_ChatHandler.auth_headers, ["Bearer secret"])

    def test_no_auth_header_by_default(self) -> None:
        build_summarizer(self._config()).enrich_many([_raw()], DISCOVERED_AT)

        self.assertEqual(_ChatHandler.auth_headers, [None])


class HeuristicSummarizerTests(unittest.TestCase):
    def test_heuristic_marks_question_as_reply(self) -> None:
        raw = RawMention(
            keyword="widgets",
            source="google_search",
            url="https://example.com/q",
            published_at=DISCOVERED_AT,
            title="Which one is better, widgets or gadgets?",
            excerpt="",
        )

        enriched = HeuristicSummarizer().enrich(raw, DISCOVERED_AT)

        self.assertEqual(enriched.suggested_action, SuggestedAction.REPLY)
        self.assertEqual(enriched.meta["summarizer"], "heuristic")


def _raw(keyword: str = "widgets", url: str = "https://example.com/post") -> RawMention:
    return RawMention(
        keyword=keyword,
        source="selected_web",
        url=url,
        published_at=datetime(2026, 7, 14, 11, 30, tzinfo=UTC),
        title=f"{keyword} users compare local inference stacks",
        excerpt="Several people are weighing cost, setup friction, and model quality.",
        source_context="selected_web",
    )


if __name__ == "__main__":
    unittest.main()
