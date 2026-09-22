from __future__ import annotations

from contextlib import redirect_stderr
from io import BytesIO
from io import StringIO
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from signalwatch.http import HttpClient, HttpError


class HttpClientTests(unittest.TestCase):
    def test_http_errors_redact_query_parameters(self) -> None:
        client = HttpClient(timeout_seconds=1, user_agent="signalwatch-test")
        url = "https://serpapi.com/search.json?api_key=supersecret&q=openai"
        error = HTTPError(
            url=url,
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=BytesIO(b'{"error":"quota exceeded"}'),
        )

        with patch("signalwatch.http.urlopen", side_effect=error):
            with self.assertRaises(HttpError) as context:
                client.get_json(
                    "https://serpapi.com/search.json",
                    params={"api_key": "supersecret", "q": "openai"},
                )

        message = str(context.exception)
        self.assertIn("https://serpapi.com/search.json", message)
        self.assertNotIn("supersecret", message)
        self.assertNotIn("api_key=", message)

    def test_get_retries_once_after_url_error(self) -> None:
        client = HttpClient(timeout_seconds=1, user_agent="signalwatch-test")

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def read(self) -> bytes:
                return b'{"ok": true}'

        with patch("signalwatch.http.time.sleep") as sleep_mock:
            with patch(
                "signalwatch.http.urlopen",
                side_effect=[URLError("temporary failure"), FakeResponse()],
            ) as urlopen_mock:
                payload = client.get_json("https://example.com/status")

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(urlopen_mock.call_count, 2)
        sleep_mock.assert_called_once()

    def test_http_debug_logs_redact_auth_headers_and_query_params(self) -> None:
        class FakeResponse:
            status = 200

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def read(self) -> bytes:
                return b'{"ok": true}'

            def getcode(self) -> int:
                return self.status

        debug_output = StringIO()
        with patch.dict(os.environ, {"SIGNALWATCH_HTTP_DEBUG": "1"}, clear=False):
            client = HttpClient(timeout_seconds=1, user_agent="signalwatch-test")
            with redirect_stderr(debug_output):
                with patch("signalwatch.http.urlopen", return_value=FakeResponse()):
                    payload = client.post_json(
                        "http://127.0.0.1:8080/v1/chat/completions",
                        payload={"model": "test-model"},
                        headers={"Authorization": "Bearer sk-123"},
                    )

        self.assertEqual(payload, {"ok": True})
        logs = debug_output.getvalue()
        self.assertIn("POST http://127.0.0.1:8080/v1/chat/completions", logs)
        self.assertIn('"Authorization": "Bearer REDACTED"', logs)
        self.assertNotIn("sk-123", logs)

        debug_output = StringIO()
        with patch.dict(os.environ, {"SIGNALWATCH_HTTP_DEBUG": "1"}, clear=False):
            client = HttpClient(timeout_seconds=1, user_agent="signalwatch-test")
            with redirect_stderr(debug_output):
                with patch("signalwatch.http.urlopen", return_value=FakeResponse()):
                    client.get_json(
                        "https://serpapi.com/search.json",
                        params={"api_key": "supersecret", "q": "openai"},
                    )

        logs = debug_output.getvalue()
        self.assertIn("q=openai", logs)
        self.assertIn("api_key=REDACTED", logs)
        self.assertNotIn("supersecret", logs)


if __name__ == "__main__":
    unittest.main()
