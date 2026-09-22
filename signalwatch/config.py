from __future__ import annotations

from pathlib import Path
import tomllib

from signalwatch.models import (
    AppConfig,
    FilterConfig,
    KeywordConfig,
    LocalLLMConfig,
    ProviderConfig,
    RunConfig,
    SerpApiProviderConfig,
    SourceConfig,
    StorageConfig,
    SummarizerConfig,
    SummarizerProvider,
)
from signalwatch.utils import normalize_domain


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    root_dir = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    storage_data = data.get("storage", {})
    run_data = data.get("run", {})
    providers_data = data.get("providers", {})
    summarizer_data = data.get("summarizer", {})
    filters_data = data.get("filters", {})
    latest_digest_path = _resolve_path(root_dir, storage_data.get("latest_digest_path", "outputs/latest_digest.txt"))
    history_dir = _resolve_path(root_dir, storage_data.get("history_dir", "outputs/digests"))
    new_latest_digest_value = storage_data.get("new_latest_digest_path")
    new_history_dir_value = storage_data.get("new_history_dir")

    storage = StorageConfig(
        database_path=_resolve_path(root_dir, storage_data.get("database_path", "work/state/mentions.db")),
        latest_digest_path=latest_digest_path,
        history_dir=history_dir,
        retention_days=int(storage_data.get("retention_days", 30)),
        new_latest_digest_path=(
            _resolve_path(root_dir, str(new_latest_digest_value))
            if new_latest_digest_value
            else _default_new_latest_digest_path(latest_digest_path)
        ),
        new_history_dir=(
            _resolve_path(root_dir, str(new_history_dir_value))
            if new_history_dir_value
            else _default_new_history_dir(history_dir)
        ),
    )
    run = RunConfig(
        since_hours=int(run_data.get("since_hours", 24)),
        timeout_seconds=int(run_data.get("timeout_seconds", 20)),
        max_results_per_query=int(run_data.get("max_results_per_query", 10)),
        user_agent=str(run_data.get("user_agent", "signalwatch/0.1")),
    )
    serpapi = providers_data.get("serpapi", {})
    sources = providers_data.get("sources", {})
    providers = ProviderConfig(
        serpapi=SerpApiProviderConfig(
            enabled=bool(serpapi.get("enabled", True)),
            api_key_env=str(serpapi.get("api_key_env", "SERPAPI_API_KEY")),
            gl=str(serpapi.get("gl", "us")),
            hl=str(serpapi.get("hl", "en")),
        ),
        sources=SourceConfig(
            google_news=bool(sources.get("google_news", True)),
            google_search=bool(sources.get("google_search", True)),
            selected_web=bool(sources.get("selected_web", True)),
        ),
    )
    llm_data = summarizer_data.get("llm", {})
    provider = _parse_summarizer_provider(summarizer_data.get("provider", SummarizerProvider.HEURISTIC.value))
    summarizer = SummarizerConfig(
        provider=provider,
        fallback_to_heuristic=bool(summarizer_data.get("fallback_to_heuristic", True)),
        llm=LocalLLMConfig(
            base_url=str(llm_data.get("base_url", "http://127.0.0.1:11434/v1")).rstrip("/"),
            model=str(llm_data.get("model", "")),
            api_key_env=str(llm_data.get("api_key_env", "")),
            request_timeout_seconds=int(llm_data.get("request_timeout_seconds", 240)),
            batch_size=int(llm_data.get("batch_size", 6)),
            max_input_chars=int(llm_data.get("max_input_chars", 4000)),
            max_batch_chars=int(llm_data.get("max_batch_chars", 12000)),
            temperature=float(llm_data.get("temperature", 0.0)),
        ),
    )
    if summarizer.provider == SummarizerProvider.LOCAL_LLM:
        if not summarizer.llm.base_url:
            raise ValueError("Local LLM summarizer requires [summarizer.llm].base_url to be set.")
        if not summarizer.llm.model.strip():
            raise ValueError("Local LLM summarizer requires [summarizer.llm].model to be set.")
    filters = FilterConfig(
        rejected_domains=_parse_domains(filters_data.get("rejected_domains", [])),
    )
    keywords = tuple(_parse_keyword(item) for item in data.get("keywords", []))
    if not keywords:
        raise ValueError("Config must include at least one [[keywords]] entry.")

    return AppConfig(
        root_dir=root_dir,
        storage=storage,
        run=run,
        providers=providers,
        summarizer=summarizer,
        keywords=keywords,
        filters=filters,
    )


def _parse_keyword(data: dict) -> KeywordConfig:
    name = str(data["name"])
    query = str(data.get("query", name))
    return KeywordConfig(
        name=name,
        query=query,
        exclude_terms=tuple(str(item) for item in data.get("exclude_terms", [])),
        selected_domains=tuple(str(item) for item in data.get("selected_domains", [])),
        rejected_domains=_parse_domains(data.get("rejected_domains", [])),
        max_results_per_source=(
            int(data["max_results_per_source"]) if data.get("max_results_per_source") is not None else None
        ),
    )


def _parse_domains(values: object) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError("rejected_domains must be a list of domain strings.")
    normalized = tuple(normalize_domain(str(item)) for item in values)
    return tuple(domain for domain in normalized if domain)


def _resolve_path(root_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def _default_new_latest_digest_path(latest_digest_path: Path) -> Path:
    return latest_digest_path.with_name(f"{latest_digest_path.stem}_new{latest_digest_path.suffix}")


def _default_new_history_dir(history_dir: Path) -> Path:
    if history_dir.name == "digests":
        return history_dir.with_name("new_digests")
    if history_dir.name.endswith("_digests"):
        return history_dir.with_name(history_dir.name.replace("_digests", "_new_digests"))
    return history_dir.with_name(f"{history_dir.name}_new")


def _parse_summarizer_provider(value: object) -> SummarizerProvider:
    normalized = str(value).strip().lower()
    try:
        return SummarizerProvider(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported summarizer provider {value!r}. Expected one of: "
            f"{', '.join(item.value for item in SummarizerProvider)}."
        ) from exc
