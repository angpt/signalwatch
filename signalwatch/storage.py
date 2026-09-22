from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from signalwatch.models import EnrichedMention, StoredMention, SuggestedAction, Sentiment
from signalwatch.utils import ensure_utc


class MentionStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def purge(self, now: datetime, *, retention_days: int) -> None:
        cutoff = ensure_utc(now) - timedelta(days=retention_days)
        self.connection.execute("DELETE FROM mentions WHERE discovered_at < ?", (cutoff.isoformat(),))
        self.connection.commit()

    def add_or_update(self, mention: EnrichedMention) -> bool:
        existing = self.connection.execute(
            "SELECT id FROM mentions WHERE canonical_url = ?",
            (mention.canonical_url,),
        ).fetchone()

        payload = (
            mention.source,
            mention.external_id,
            mention.url,
            mention.canonical_url,
            mention.published_at.isoformat() if mention.published_at else None,
            mention.discovered_at.isoformat(),
            mention.summary,
            mention.sentiment.value,
            mention.suggested_action.value,
            mention.why_it_matters,
            mention.score,
            json.dumps(mention.meta, sort_keys=True),
        )

        if existing is None:
            cursor = self.connection.execute(
                """
                INSERT INTO mentions (
                    source,
                    external_id,
                    url,
                    canonical_url,
                    published_at,
                    discovered_at,
                    summary,
                    sentiment,
                    suggested_action,
                    why_it_matters,
                    score,
                    meta_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            mention_id = int(cursor.lastrowid)
            created = True
        else:
            mention_id = int(existing["id"])
            self.connection.execute(
                """
                UPDATE mentions
                SET
                    external_id = COALESCE(external_id, ?),
                    published_at = COALESCE(published_at, ?),
                    summary = CASE WHEN ? > score THEN ? ELSE summary END,
                    sentiment = CASE WHEN ? > score THEN ? ELSE sentiment END,
                    suggested_action = CASE WHEN ? > score THEN ? ELSE suggested_action END,
                    why_it_matters = CASE WHEN ? > score THEN ? ELSE why_it_matters END,
                    score = MAX(score, ?),
                    meta_json = CASE WHEN ? > score THEN ? ELSE meta_json END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    mention.external_id,
                    mention.published_at.isoformat() if mention.published_at else None,
                    mention.score,
                    mention.summary,
                    mention.score,
                    mention.sentiment.value,
                    mention.score,
                    mention.suggested_action.value,
                    mention.score,
                    mention.why_it_matters,
                    mention.score,
                    mention.score,
                    json.dumps(mention.meta, sort_keys=True),
                    mention_id,
                ),
            )
            created = False

        self._attach_keyword_and_source(mention_id, keyword=mention.keyword, source=mention.source)
        self.connection.commit()
        return created

    def record_existing_match(
        self,
        canonical_url: str,
        *,
        keyword: str,
        source: str,
        published_at: datetime | None = None,
    ) -> bool:
        existing = self.connection.execute(
            "SELECT id FROM mentions WHERE canonical_url = ?",
            (canonical_url,),
        ).fetchone()
        if existing is None:
            return False

        mention_id = int(existing["id"])
        self._attach_keyword_and_source(mention_id, keyword=keyword, source=source)
        self.connection.execute(
            """
            UPDATE mentions
            SET
                published_at = COALESCE(published_at, ?),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (published_at.isoformat() if published_at else None, mention_id),
        )
        self.connection.commit()
        return True

    def list_since(self, since: datetime) -> list[StoredMention]:
        return self._list_where("m.discovered_at >= ?", (ensure_utc(since).isoformat(),))

    def list_discovered_at(self, discovered_at: datetime) -> list[StoredMention]:
        return self._list_where("m.discovered_at = ?", (ensure_utc(discovered_at).isoformat(),))

    def summarizer_counts_for_discovered_at(self, discovered_at: datetime) -> tuple[tuple[str, int], ...]:
        rows = self.connection.execute(
            """
            SELECT
                COALESCE(NULLIF(json_extract(meta_json, '$.summarizer'), ''), 'untagged') AS summarizer,
                COUNT(*) AS count
            FROM mentions
            WHERE discovered_at = ?
            GROUP BY summarizer
            ORDER BY summarizer
            """,
            (ensure_utc(discovered_at).isoformat(),),
        ).fetchall()
        return tuple((str(row["summarizer"]), int(row["count"])) for row in rows)

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS mentions (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                external_id TEXT,
                url TEXT NOT NULL,
                canonical_url TEXT NOT NULL,
                published_at TEXT,
                discovered_at TEXT NOT NULL,
                summary TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                suggested_action TEXT NOT NULL,
                why_it_matters TEXT NOT NULL,
                score REAL NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, canonical_url)
            );

            CREATE TABLE IF NOT EXISTS mention_keywords (
                mention_id INTEGER NOT NULL REFERENCES mentions(id) ON DELETE CASCADE,
                keyword TEXT NOT NULL,
                UNIQUE(mention_id, keyword)
            );

            CREATE TABLE IF NOT EXISTS mention_sources (
                mention_id INTEGER NOT NULL REFERENCES mentions(id) ON DELETE CASCADE,
                source TEXT NOT NULL,
                UNIQUE(mention_id, source)
            );

            CREATE INDEX IF NOT EXISTS idx_mentions_discovered_at ON mentions(discovered_at);
            CREATE INDEX IF NOT EXISTS idx_mentions_score ON mentions(score);
            """
        )
        self._backfill_sources()
        self._merge_duplicate_canonical_urls()
        self.connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_mentions_canonical_url_unique ON mentions(canonical_url)"
        )
        self.connection.commit()

    def _attach_keyword_and_source(self, mention_id: int, *, keyword: str, source: str) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO mention_keywords (mention_id, keyword) VALUES (?, ?)",
            (mention_id, keyword),
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO mention_sources (mention_id, source) VALUES (?, ?)",
            (mention_id, source),
        )

    def _backfill_sources(self) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO mention_sources (mention_id, source)
            SELECT id, source FROM mentions
            """
        )

    def _list_where(self, where_clause: str, params: tuple[str, ...]) -> list[StoredMention]:
        rows = self.connection.execute(
            f"""
            SELECT
                m.id,
                m.source,
                m.url,
                m.published_at,
                m.discovered_at,
                m.summary,
                m.sentiment,
                m.suggested_action,
                m.why_it_matters,
                m.score,
                m.meta_json,
                (
                    SELECT GROUP_CONCAT(keyword, '||')
                    FROM mention_keywords
                    WHERE mention_id = m.id
                ) AS keywords,
                (
                    SELECT GROUP_CONCAT(source, '||')
                    FROM mention_sources
                    WHERE mention_id = m.id
                ) AS sources
            FROM mentions m
            WHERE {where_clause}
            ORDER BY m.score DESC, COALESCE(m.published_at, m.discovered_at) DESC, m.id DESC
            """,
            params,
        ).fetchall()
        mentions: list[StoredMention] = []
        for row in rows:
            keywords = tuple(sorted(filter(None, (row["keywords"] or "").split("||"))))
            sources = tuple(sorted(filter(None, (row["sources"] or "").split("||"))))
            mentions.append(
                StoredMention(
                    id=int(row["id"]),
                    source=str(row["source"]),
                    url=str(row["url"]),
                    published_at=_parse_row_datetime(row["published_at"]),
                    discovered_at=_parse_row_datetime(row["discovered_at"]),
                    summary=str(row["summary"]),
                    sentiment=Sentiment(str(row["sentiment"])),
                    suggested_action=SuggestedAction(str(row["suggested_action"])),
                    why_it_matters=str(row["why_it_matters"]),
                    score=float(row["score"]),
                    keywords=keywords,
                    sources=sources,
                    raw_text=_raw_text_from_meta(row["meta_json"]),
                )
            )
        return mentions

    def _merge_duplicate_canonical_urls(self) -> None:
        duplicate_rows = self.connection.execute(
            """
            SELECT canonical_url
            FROM mentions
            GROUP BY canonical_url
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        for row in duplicate_rows:
            self._merge_duplicate_group(str(row["canonical_url"]))

    def _merge_duplicate_group(self, canonical_url: str) -> None:
        rows = self.connection.execute(
            """
            SELECT
                id,
                source,
                url,
                published_at,
                discovered_at,
                summary,
                sentiment,
                suggested_action,
                why_it_matters,
                score,
                meta_json,
                external_id
            FROM mentions
            WHERE canonical_url = ?
            ORDER BY score DESC, discovered_at ASC, id ASC
            """,
            (canonical_url,),
        ).fetchall()
        if len(rows) < 2:
            return

        primary = rows[0]
        primary_id = int(primary["id"])
        duplicate_ids = [int(item["id"]) for item in rows[1:]]
        earliest_discovered = min(
            _parse_row_datetime(str(item["discovered_at"])) for item in rows if item["discovered_at"]
        )
        published_candidates = [
            _parse_row_datetime(str(item["published_at"])) for item in rows if item["published_at"]
        ]
        published_at = min(published_candidates) if published_candidates else None

        placeholders = ", ".join("?" for _ in duplicate_ids)
        params = tuple([primary_id, *duplicate_ids])
        self.connection.execute(
            f"""
            INSERT OR IGNORE INTO mention_keywords (mention_id, keyword)
            SELECT ?, keyword
            FROM mention_keywords
            WHERE mention_id IN ({placeholders})
            """,
            params,
        )
        self.connection.execute(
            f"""
            INSERT OR IGNORE INTO mention_sources (mention_id, source)
            SELECT ?, source
            FROM mention_sources
            WHERE mention_id IN ({placeholders})
            """,
            params,
        )
        self.connection.execute(
            """
            UPDATE mentions
            SET
                published_at = ?,
                discovered_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                published_at.isoformat() if published_at else None,
                earliest_discovered.isoformat(),
                primary_id,
            ),
        )
        delete_placeholders = ", ".join("?" for _ in duplicate_ids)
        self.connection.execute(
            f"DELETE FROM mentions WHERE id IN ({delete_placeholders})",
            tuple(duplicate_ids),
        )


def _raw_text_from_meta(meta_json: str | None) -> str:
    """Recover the original title/excerpt/source for retroactive exclude_terms checks.

    The stored summary/why_it_matters are LLM-paraphrased and may drop a term that
    exclude_terms is meant to catch, so re-filtering uses this raw text instead.
    """
    if not meta_json:
        return ""
    try:
        meta = json.loads(meta_json)
    except (TypeError, ValueError):
        return ""
    if not isinstance(meta, dict):
        return ""
    parts = (meta.get("title"), meta.get("excerpt"), meta.get("source_context"))
    return " ".join(str(part) for part in parts if part)


def _parse_row_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
