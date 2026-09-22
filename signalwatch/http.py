from __future__ import annotations

import json
import os
from urllib.parse import parse_qsl
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


class HttpError(RuntimeError):
    pass


class HttpClient:
    def __init__(self, *, timeout_seconds: int, user_agent: str) -> None:
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent
        self.debug_enabled = _env_flag("SIGNALWATCH_HTTP_DEBUG")

    def get_json(self, url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict:
        body = self._request("GET", url, params=params, headers=headers)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise HttpError(f"Invalid JSON response from {_safe_url(url)}: {exc}") from exc

    def post_form_json(
        self,
        url: str,
        *,
        form: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict:
        merged_headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if headers:
            merged_headers.update(headers)
        body = urlencode(form).encode("utf-8")
        response = self._request("POST", url, headers=merged_headers, body=body)
        try:
            return json.loads(response)
        except json.JSONDecodeError as exc:
            raise HttpError(f"Invalid JSON response from {_safe_url(url)}: {exc}") from exc

    def post_json(
        self,
        url: str,
        *,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict:
        merged_headers = {"Content-Type": "application/json"}
        if headers:
            merged_headers.update(headers)
        body = json.dumps(payload).encode("utf-8")
        response = self._request("POST", url, headers=merged_headers, body=body)
        try:
            return json.loads(response)
        except json.JSONDecodeError as exc:
            raise HttpError(f"Invalid JSON response from {_safe_url(url)}: {exc}") from exc

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> str:
        if params:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{urlencode(params, doseq=True)}"

        merged_headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        if headers:
            merged_headers.update(headers)
        max_attempts = 2 if method.upper() == "GET" else 1
        self._log_request(method, url, headers=merged_headers, body=body)

        for attempt in range(max_attempts):
            request = Request(url, data=body, headers=merged_headers, method=method)
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = response.read().decode("utf-8", errors="replace")
                    self._log_response(method, url, status=_response_status(response), body=payload)
                    return payload
            except HTTPError as exc:
                if _should_retry_http_error(exc) and attempt + 1 < max_attempts:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                payload = exc.read().decode("utf-8", errors="replace")
                self._log_response(method, url, status=exc.code, body=payload)
                raise HttpError(
                    f"{method} {_safe_url(url)} failed with HTTP {exc.code}: {_summarize_payload(payload)}"
                ) from exc
            except URLError as exc:
                if attempt + 1 < max_attempts:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                self._debug(f"[http] <- {method} {_safe_url(url)} error={exc.reason}")
                raise HttpError(f"{method} {_safe_url(url)} failed: {exc.reason}") from exc

        raise HttpError(f"{method} {_safe_url(url)} failed after {max_attempts} attempts.")

    def _log_request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes | None,
    ) -> None:
        if not self.debug_enabled:
            return
        redacted_headers = _redact_headers(headers)
        request_bits = [f"[http] -> {method} {_debug_url(url)}", f"headers={json.dumps(redacted_headers, sort_keys=True)}"]
        body_summary = _summarize_request_body(body, headers=headers)
        if body_summary:
            request_bits.append(f"body={body_summary}")
        self._debug(" ".join(request_bits))

    def _log_response(self, method: str, url: str, *, status: int, body: str) -> None:
        if not self.debug_enabled:
            return
        self._debug(f"[http] <- {method} {_safe_url(url)} status={status} body={_summarize_payload(body, limit=800)}")

    def _debug(self, message: str) -> None:
        if not self.debug_enabled:
            return
        print(message, file=sys.stderr)


def _safe_url(url: str) -> str:
    split = urlsplit(url)
    return urlunsplit((split.scheme, split.netloc, split.path, "", ""))


def _debug_url(url: str) -> str:
    split = urlsplit(url)
    if not split.query:
        return url
    filtered_query = []
    for key, value in parse_qsl(split.query, keep_blank_values=True):
        filtered_query.append((key, _redact_secret_value(value) if _is_sensitive_name(key) else value))
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(filtered_query, doseq=True), ""))


def _summarize_payload(payload: str, *, limit: int = 500) -> str:
    compact = " ".join(payload.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _should_retry_http_error(exc: HTTPError) -> bool:
    return exc.code in {429, 500, 502, 503, 504}


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    status = getattr(response, "code", None)
    if isinstance(status, int):
        return status
    getcode = getattr(response, "getcode", None)
    if callable(getcode):
        try:
            result = getcode()
        except Exception:
            result = None
        if isinstance(result, int):
            return result
    return 200


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        redacted[key] = _redact_secret_value(value, preserve_bearer=_is_sensitive_name(key)) if _is_sensitive_name(key) else value
    return redacted


def _summarize_request_body(body: bytes | None, *, headers: dict[str, str]) -> str:
    if body is None:
        return ""
    content_type = str(headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
    decoded = body.decode("utf-8", errors="replace")

    if content_type == "application/json":
        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError:
            return _summarize_payload(decoded, limit=1200)
        return _summarize_payload(json.dumps(_redact_structure(parsed), separators=(",", ":")), limit=1200)

    if content_type == "application/x-www-form-urlencoded":
        parsed_form = dict(parse_qsl(decoded, keep_blank_values=True))
        return _summarize_payload(json.dumps(_redact_structure(parsed_form), separators=(",", ":")), limit=1200)

    return f"<{len(body)} bytes>"


def _redact_structure(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _is_sensitive_name(key):
                redacted[key] = _redact_secret_value(str(item), preserve_bearer=True)
            else:
                redacted[key] = _redact_structure(item)
        return redacted
    if isinstance(value, list):
        return [_redact_structure(item) for item in value]
    return value


def _is_sensitive_name(name: str) -> bool:
    normalized = name.strip().lower()
    if normalized in {
        "api_key",
        "apikey",
        "authorization",
        "client_secret",
        "password",
        "refresh_token",
        "secret",
        "token",
    }:
        return True
    return normalized.endswith(("_key", "_password", "_secret", "_token"))


def _redact_secret_value(value: str, *, preserve_bearer: bool = False) -> str:
    compact = value.strip()
    if preserve_bearer and compact.lower().startswith("bearer "):
        return "Bearer REDACTED"
    return "REDACTED"
