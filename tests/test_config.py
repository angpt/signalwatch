from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from signalwatch.config import load_config
from signalwatch.models import SummarizerProvider


class ConfigTests(unittest.TestCase):
    def test_load_config_resolves_paths_from_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            config_path = root / "config" / "test.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [storage]
                    database_path = "work/state/mentions.db"

                    [providers.serpapi]
                    enabled = false

                    [[keywords]]
                    name = "Example"
                    query = '"example"'
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.root_dir, root.resolve())
            self.assertEqual(config.storage.database_path, (root / "work" / "state" / "mentions.db").resolve())
            self.assertEqual(
                config.storage.new_latest_digest_path,
                (root / "outputs" / "latest_digest_new.txt").resolve(),
            )
            self.assertEqual(
                config.storage.new_history_dir,
                (root / "outputs" / "new_digests").resolve(),
            )
            self.assertEqual(config.keywords[0].name, "Example")

    def test_load_config_supports_source_toggles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            config_path = root / "config" / "test.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [providers.serpapi]
                    enabled = true

                    [providers.sources]
                    google_news = false
                    google_search = false
                    selected_web = true

                    [[keywords]]
                    name = "Example"
                    query = '"example"'
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertFalse(config.providers.sources.google_news)
            self.assertFalse(config.providers.sources.google_search)
            self.assertTrue(config.providers.sources.selected_web)

    def test_load_config_normalizes_rejected_domains(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            config_path = root / "config" / "test.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [filters]
                    rejected_domains = ["https://www.Facebook.com/", "instagram.com", " "]

                    [providers.serpapi]
                    enabled = false

                    [[keywords]]
                    name = "Example"
                    query = '"example"'
                    rejected_domains = ["Medium.com"]
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.filters.rejected_domains, ("facebook.com", "instagram.com"))
            self.assertEqual(config.keywords[0].rejected_domains, ("medium.com",))

    def test_load_config_defaults_rejected_domains_to_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            config_path = root / "config" / "test.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [providers.serpapi]
                    enabled = false

                    [[keywords]]
                    name = "Example"
                    query = '"example"'
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.filters.rejected_domains, ())
            self.assertEqual(config.keywords[0].rejected_domains, ())

    def test_load_config_supports_local_llm_summarizer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            config_path = root / "config" / "test.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [summarizer]
                    provider = "local_llm"
                    fallback_to_heuristic = false

                    [summarizer.llm]
                    base_url = "http://localhost:8080/v1/"
                    model = "gemma3:4b"
                    api_key_env = "LOCAL_LLM_KEY"
                    request_timeout_seconds = 90
                    batch_size = 3
                    max_input_chars = 2000
                    max_batch_chars = 9000
                    temperature = 0.2

                    [[keywords]]
                    name = "Example"
                    query = '"example"'
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.summarizer.provider, SummarizerProvider.LOCAL_LLM)
            self.assertFalse(config.summarizer.fallback_to_heuristic)
            self.assertEqual(config.summarizer.llm.base_url, "http://localhost:8080/v1")
            self.assertEqual(config.summarizer.llm.model, "gemma3:4b")
            self.assertEqual(config.summarizer.llm.api_key_env, "LOCAL_LLM_KEY")
            self.assertEqual(config.summarizer.llm.request_timeout_seconds, 90)
            self.assertEqual(config.summarizer.llm.batch_size, 3)
            self.assertEqual(config.summarizer.llm.max_input_chars, 2000)
            self.assertEqual(config.summarizer.llm.max_batch_chars, 9000)
            self.assertEqual(config.summarizer.llm.temperature, 0.2)

    def test_load_config_requires_model_for_local_llm_summarizer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            config_path = root / "config" / "test.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [summarizer]
                    provider = "local_llm"

                    [[keywords]]
                    name = "Example"
                    query = '"example"'
                    """
                ).strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "requires \\[summarizer.llm\\].model"):
                load_config(config_path)

    def test_load_config_rejects_unknown_summarizer_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            config_path = root / "config" / "test.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [summarizer]
                    provider = "command"

                    [[keywords]]
                    name = "Example"
                    query = '"example"'
                    """
                ).strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unsupported summarizer provider"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
