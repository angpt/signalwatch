from __future__ import annotations

from datetime import datetime
from typing import Protocol

from signalwatch.models import AppConfig, KeywordConfig, RawMention


class Collector(Protocol):
    source_name: str

    def collect(
        self,
        keyword: KeywordConfig,
        config: AppConfig,
        since: datetime,
        now: datetime,
    ) -> list[RawMention]:
        ...
