from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import os

from signalwatch.http import HttpClient
from signalwatch.models import (
    EnrichedMention,
    LocalLLMConfig,
    RawMention,
    Sentiment,
    SuggestedAction,
    SummarizerConfig,
    SummarizerProvider,
)
from signalwatch.utils import canonicalize_url, collapse_whitespace

POSITIVE_TERMS = {
    "love",
    "great",
    "helpful",
    "impressed",
    "win",
    "launched",
    "launch",
    "featured",
    "recommend",
    "excited",
    "faster",
    "growth",
}

NEGATIVE_TERMS = {
    "bad",
    "broken",
    "confusing",
    "hate",
    "issue",
    "outage",
    "problem",
    "slow",
    "terrible",
    "worse",
    "worst",
    "bug",
    "complaint",
    "frustrating",
}

REPLY_TERMS = {
    "alternative",
    "alternatives",
    "anyone",
    "compare",
    "comparison",
    "recommendation",
    "recommendations",
    "vs",
    "worth it",
    "which one",
}

SHARE_TERMS = {
    "announced",
    "launch",
    "launched",
    "release",
    "released",
    "funding",
    "partnership",
    "feature",
    "featured",
}

BATCH_SYSTEM_PROMPT = """You summarize daily monitoring mentions for a founder-oriented digest.
Return exactly one JSON object with this shape:
{"items":[{"mention_index":0,"summary":"...","sentiment":"neutral","suggested_action":"watch","why_it_matters":"...","score":63}]}

Rules:
- Use one item for each input mention.
- Copy mention_index exactly from the input.
- summary should be plain text, concise, and mention the company or keyword if helpful.
- sentiment must be one of: positive, neutral, negative, mixed.
- suggested_action must be one of: reply, investigate, share, watch, ignore.
- why_it_matters should be one short operational sentence.
- score must be a number from 0 to 100.
- Return JSON only, with no markdown fences or extra narration."""


class SummarizerError(RuntimeError):
    pass


@dataclass(slots=True)
class HeuristicSummarizer:
    max_summary_length: int = 220

    def start_run(self) -> None:
        return None

    def finish_run(self) -> None:
        return None

    def enrich(self, raw: RawMention, discovered_at: datetime) -> EnrichedMention:
        combined = self._combined_text(raw)
        sentiment = self._infer_sentiment(combined)
        action = self._suggest_action(raw, combined, sentiment)
        summary = self._build_summary(raw)
        why_it_matters = self._why_it_matters(raw, combined, sentiment, action)
        score = self._score(raw, combined, sentiment, action)
        return EnrichedMention(
            keyword=raw.keyword,
            source=raw.source,
            url=raw.url,
            canonical_url=canonicalize_url(raw.url),
            published_at=raw.published_at,
            discovered_at=discovered_at,
            summary=summary,
            sentiment=sentiment,
            suggested_action=action,
            why_it_matters=why_it_matters,
            score=score,
            external_id=raw.external_id,
            meta={
                "title": raw.title or "",
                "excerpt": raw.excerpt or "",
                "source_context": raw.source_context or "",
                "summarizer": "heuristic",
                "summarizer_runner": "heuristic",
            },
        )

    def enrich_many(self, raws: list[RawMention], discovered_at: datetime) -> list[EnrichedMention]:
        return [self.enrich(raw, discovered_at) for raw in raws]

    def _combined_text(self, raw: RawMention) -> str:
        parts = [raw.title or "", raw.excerpt or "", raw.source_context or ""]
        return collapse_whitespace(" ".join(part for part in parts if part))

    def _infer_sentiment(self, text: str) -> Sentiment:
        lowered = text.lower()
        positive = sum(term in lowered for term in POSITIVE_TERMS)
        negative = sum(term in lowered for term in NEGATIVE_TERMS)
        if positive and negative:
            return Sentiment.MIXED
        if negative > positive:
            return Sentiment.NEGATIVE
        if positive > negative:
            return Sentiment.POSITIVE
        return Sentiment.NEUTRAL

    def _suggest_action(self, raw: RawMention, text: str, sentiment: Sentiment) -> SuggestedAction:
        lowered = text.lower()
        if "?" in text or any(term in lowered for term in REPLY_TERMS):
            return SuggestedAction.REPLY
        if sentiment == Sentiment.NEGATIVE:
            return SuggestedAction.INVESTIGATE
        if raw.source == "google_news" and any(term in lowered for term in SHARE_TERMS):
            return SuggestedAction.SHARE
        if sentiment == Sentiment.POSITIVE and any(term in lowered for term in SHARE_TERMS):
            return SuggestedAction.SHARE
        return SuggestedAction.WATCH

    def _build_summary(self, raw: RawMention) -> str:
        source_label = {
            "google_news": "A Google News result",
            "google_search": "A Google Search result",
            "selected_web": "A selected public web result",
        }.get(raw.source, "A web result")

        gist = collapse_whitespace(" ".join(part for part in [raw.title or "", raw.excerpt or ""] if part))
        if not gist:
            gist = f"mentions {raw.keyword}."
        else:
            gist = self._trim_sentence(gist)
            if not gist.endswith("."):
                gist = f"{gist}."
        if raw.keyword.lower() not in gist.lower():
            return f"{source_label} mentions {raw.keyword}: {gist}"
        return f"{source_label}: {gist}"

    def _why_it_matters(
        self,
        raw: RawMention,
        text: str,
        sentiment: Sentiment,
        action: SuggestedAction,
    ) -> str:
        reasons: list[str] = []
        if action == SuggestedAction.REPLY:
            reasons.append("The wording suggests active evaluation or a request for help.")
        elif action == SuggestedAction.INVESTIGATE:
            reasons.append("The mention reads negative enough to be worth checking quickly.")
        elif action == SuggestedAction.SHARE:
            reasons.append("This looks like positive or notable coverage that may be worth amplifying.")
        else:
            reasons.append("It is recent enough to keep on the radar.")

        if raw.source == "selected_web":
            reasons.append("It came from one of your explicitly allowlisted sites.")
        elif raw.source == "google_news":
            reasons.append("News coverage can shape perception beyond a single community.")

        if sentiment == Sentiment.MIXED:
            reasons.append("The tone is mixed, which usually means nuance rather than a simple win or loss.")

        return " ".join(reasons[:2])

    def _score(self, raw: RawMention, text: str, sentiment: Sentiment, action: SuggestedAction) -> float:
        score = 40.0
        source_bonus = {
            "google_news": 12.0,
            "google_search": 7.0,
            "selected_web": 10.0,
        }.get(raw.source, 0.0)
        score += source_bonus

        if sentiment == Sentiment.NEGATIVE:
            score += 18.0
        elif sentiment == Sentiment.POSITIVE:
            score += 8.0
        elif sentiment == Sentiment.MIXED:
            score += 12.0

        action_bonus = {
            SuggestedAction.REPLY: 18.0,
            SuggestedAction.INVESTIGATE: 16.0,
            SuggestedAction.SHARE: 10.0,
            SuggestedAction.WATCH: 4.0,
            SuggestedAction.IGNORE: 0.0,
        }[action]
        score += action_bonus

        text_length = len(text)
        if text_length >= 120:
            score += 4.0
        elif text_length <= 30:
            score -= 6.0

        return round(max(0.0, min(100.0, score)), 1)

    def _trim_sentence(self, value: str) -> str:
        value = collapse_whitespace(value)
        if len(value) <= self.max_summary_length:
            return value
        return value[: self.max_summary_length - 1].rstrip() + "…"


@dataclass(slots=True)
class LocalLLMSummarizer:
    """Batches mentions into chat-completions requests against a local OpenAI-compatible server."""

    llm: LocalLLMConfig
    fallback: HeuristicSummarizer | None = None
    user_agent: str = "signalwatch/0.1"
    _api_client: HttpClient = field(init=False)

    def __post_init__(self) -> None:
        self._api_client = HttpClient(
            timeout_seconds=self.llm.request_timeout_seconds,
            user_agent=self.user_agent,
        )

    def start_run(self) -> None:
        return None

    def finish_run(self) -> None:
        return None

    def enrich(self, raw: RawMention, discovered_at: datetime) -> EnrichedMention:
        return self.enrich_many([raw], discovered_at)[0]

    def enrich_many(self, raws: list[RawMention], discovered_at: datetime) -> list[EnrichedMention]:
        if not raws:
            return []

        heuristic = self.fallback or HeuristicSummarizer()
        baselines = [heuristic.enrich(raw, discovered_at) for raw in raws]
        responses_by_index: dict[int, dict] = {}

        try:
            for batch in self._split_batches(raws):
                responses_by_index.update(self._invoke_batch(batch))
        except Exception as exc:
            if self.fallback is None:
                raise
            return [_fallback_enriched(baseline, exc, runner="local_llm") for baseline in baselines]

        enriched_mentions: list[EnrichedMention] = []
        for index, baseline in enumerate(baselines):
            response = responses_by_index.get(index)
            if response is None:
                error = SummarizerError(f"Model response did not include mention_index {index}.")
                if self.fallback is None:
                    raise error
                enriched_mentions.append(_fallback_enriched(baseline, error, runner="local_llm"))
                continue
            enriched_mentions.append(
                _merge_summary_response(
                    baseline,
                    response,
                    {
                        "summarizer": "local_llm",
                        "summarizer_runner": "local_llm",
                        "summarizer_server": self.llm.base_url,
                        "model": self.llm.model,
                    },
                )
            )
        return enriched_mentions

    def _headers(self) -> dict[str, str]:
        if not self.llm.api_key_env:
            return {}
        api_key = os.getenv(self.llm.api_key_env, "")
        if not api_key:
            raise SummarizerError(f"Missing API key in ${self.llm.api_key_env}.")
        return {"Authorization": f"Bearer {api_key}"}

    def _split_batches(self, raws: list[RawMention]) -> list[list[tuple[int, RawMention]]]:
        batches: list[list[tuple[int, RawMention]]] = []
        current: list[tuple[int, RawMention]] = []
        current_chars = 0

        for index, raw in enumerate(raws):
            item_size = len(json.dumps(self._batch_item(index, raw), separators=(",", ":")))
            if current and (
                len(current) >= self.llm.batch_size
                or current_chars + item_size > self.llm.max_batch_chars
            ):
                batches.append(current)
                current = []
                current_chars = 0
            current.append((index, raw))
            current_chars += item_size

        if current:
            batches.append(current)
        return batches

    def _batch_item(self, index: int, raw: RawMention) -> dict[str, object]:
        payload = _build_raw_payload(raw, self.llm.max_input_chars)
        payload.pop("allowed_sentiments", None)
        payload.pop("allowed_actions", None)
        payload["mention_index"] = index
        return payload

    def _invoke_batch(self, batch: list[tuple[int, RawMention]]) -> dict[int, dict]:
        items = [self._batch_item(index, raw) for index, raw in batch]
        request = {
            "model": self.llm.model,
            "messages": [
                {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "allowed_sentiments": [item.value for item in Sentiment],
                            "allowed_actions": [item.value for item in SuggestedAction],
                            "mentions": items,
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
            "temperature": self.llm.temperature,
        }
        response = self._api_client.post_json(
            f"{self.llm.base_url}/chat/completions",
            payload=request,
            headers=self._headers(),
        )
        parsed = _extract_json_object(_extract_chat_message_content(response))
        parsed_items = parsed.get("items")
        if not isinstance(parsed_items, list):
            raise SummarizerError("Batch response JSON must contain an 'items' list.")

        responses: dict[int, dict] = {}
        expected_indexes = {index for index, _ in batch}
        for item in parsed_items:
            if not isinstance(item, dict):
                raise SummarizerError("Each batch response item must be a JSON object.")
            try:
                mention_index = int(item.get("mention_index"))
            except (TypeError, ValueError) as exc:
                raise SummarizerError("Each batch response item must include an integer mention_index.") from exc
            if mention_index not in expected_indexes:
                raise SummarizerError(f"Batch response referenced unexpected mention_index {mention_index}.")
            responses[mention_index] = item

        missing = expected_indexes - responses.keys()
        if missing:
            raise SummarizerError(
                f"Batch response omitted mention indexes: {', '.join(str(item) for item in sorted(missing))}."
            )
        return responses


def build_summarizer(
    config: SummarizerConfig,
    *,
    user_agent: str = "signalwatch/0.1",
) -> HeuristicSummarizer | LocalLLMSummarizer:
    heuristic = HeuristicSummarizer()
    if config.provider == SummarizerProvider.HEURISTIC:
        return heuristic
    return LocalLLMSummarizer(
        llm=config.llm,
        fallback=heuristic if config.fallback_to_heuristic else None,
        user_agent=user_agent,
    )


def _build_raw_payload(raw: RawMention, max_input_chars: int) -> dict[str, object]:
    title, excerpt, source_context = _truncate_fields(raw, max_input_chars)
    combined = collapse_whitespace(" ".join(part for part in (title, excerpt, source_context) if part))
    return {
        "keyword": raw.keyword,
        "source": raw.source,
        "url": raw.url,
        "published_at": raw.published_at.isoformat() if raw.published_at else None,
        "title": title,
        "excerpt": excerpt,
        "source_context": source_context,
        "combined_text": combined,
        "allowed_sentiments": [item.value for item in Sentiment],
        "allowed_actions": [item.value for item in SuggestedAction],
    }


def _truncate_fields(raw: RawMention, max_input_chars: int) -> tuple[str, str, str]:
    remaining = max(0, max_input_chars)
    fields: list[str] = []
    for value in (raw.title or "", raw.excerpt or "", raw.source_context or ""):
        normalized = collapse_whitespace(value)
        if not normalized or remaining <= 0:
            fields.append("")
            continue
        if len(normalized) <= remaining:
            trimmed = normalized
        else:
            trimmed = normalized[: max(0, remaining - 1)].rstrip()
            if trimmed:
                trimmed += "…"
        fields.append(trimmed)
        remaining -= len(trimmed)
    while len(fields) < 3:
        fields.append("")
    return fields[0], fields[1], fields[2]


def _fallback_enriched(
    baseline: EnrichedMention,
    exc: Exception,
    *,
    runner: str,
) -> EnrichedMention:
    meta = dict(baseline.meta)
    meta["summarizer"] = "heuristic_fallback"
    meta["summarizer_error"] = str(exc)
    meta["summarizer_runner"] = runner
    return replace(baseline, meta=meta)


def _merge_summary_response(
    baseline: EnrichedMention,
    response: dict,
    meta_updates: dict[str, object],
) -> EnrichedMention:
    meta = dict(baseline.meta)
    meta.update(meta_updates)
    extra_meta = response.get("meta")
    if isinstance(extra_meta, dict):
        for key, value in extra_meta.items():
            if isinstance(key, str):
                meta[key] = value

    return replace(
        baseline,
        summary=_coerce_text(response.get("summary")) or baseline.summary,
        sentiment=_coerce_sentiment(response.get("sentiment"), baseline.sentiment),
        suggested_action=_coerce_action(response.get("suggested_action"), baseline.suggested_action),
        why_it_matters=_coerce_text(response.get("why_it_matters")) or baseline.why_it_matters,
        score=_coerce_score(response.get("score"), baseline.score),
        meta=meta,
    )


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    return collapse_whitespace(str(value))


def _coerce_sentiment(value: object, default: Sentiment) -> Sentiment:
    normalized = _coerce_text(value).lower()
    if not normalized:
        return default
    try:
        return Sentiment(normalized)
    except ValueError:
        return default


def _coerce_action(value: object, default: SuggestedAction) -> SuggestedAction:
    normalized = _coerce_text(value).lower()
    if not normalized:
        return default
    try:
        return SuggestedAction(normalized)
    except ValueError:
        return default


def _coerce_score(value: object, default: float) -> float:
    if value is None:
        return default
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return round(max(0.0, min(100.0, score)), 1)


def _extract_chat_message_content(response: dict) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SummarizerError("Server response was missing choices.")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise SummarizerError("Server response choice was not a JSON object.")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise SummarizerError("Server response choice was missing a message object.")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "\n".join(parts)
    raise SummarizerError("Server response message did not include text content.")


def _extract_json_object(value: str) -> dict:
    text = value.strip()
    if not text:
        raise SummarizerError("Model output was empty.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    start = text.find("{")
    if start == -1:
        raise SummarizerError("Model output did not contain JSON.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise SummarizerError("Model output did not contain a valid JSON object.") from exc
                if isinstance(parsed, dict):
                    return parsed
                break

    raise SummarizerError("Model output did not contain a valid JSON object.")
