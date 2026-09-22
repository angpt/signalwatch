# signalwatch

Local-first keyword monitoring. Once a day (or whenever you run it), `signalwatch`
searches Google Search, Google News, and a short allowlist of sites through
[SerpAPI](https://serpapi.com), summarizes each new result with a **local LLM**, and
writes a plain-text digest. No servers, no dashboards, and no third-party
dependencies: it is standard-library Python.

Each mention in the digest has a summary, sentiment, suggested action
(`reply`, `investigate`, `share`, `watch`, `ignore`), a "why it matters" line, a score,
the URL, and the keywords that matched.

## Privacy

Only derived output is stored: summaries, sentiment, action, score, URL, timestamps,
and matched keywords. Raw titles, snippets, and page text are not saved as such; they
are used to build the summary and then discarded. (The built-in heuristic summarizer
quotes a trimmed title and snippet in its summary, so expect that text in the database
when you use it.) The only outbound calls are to SerpAPI (your keywords) and to the
LLM endpoint you configure (search snippets).

## Quick start

Requires Python 3.11+ and a [SerpAPI key](https://serpapi.com/manage-api-key).

```bash
git clone <this repo> && cd signalwatch
export SERPAPI_API_KEY=...

# Option A: no LLM, built-in keyword heuristic
python3 -m signalwatch run --config config/heuristic.toml

# Option B: local LLM (Ollama shown; see below for others)
ollama pull llama3.2
python3 -m signalwatch run --config config/example.toml
```

Copy `config/example.toml`, edit the `[[keywords]]`, and you are set. Or install the
`signalwatch` command with `pip install -e .`.

Output:

- `outputs/latest_digest.txt`: everything in the `since_hours` window
- `outputs/latest_digest_new.txt`: only mentions first seen in this run
- timestamped copies in `outputs/digests/` and `outputs/new_digests/`
- SQLite state in `work/state/mentions.db` (URL dedupe and retention)

Paths in a config resolve relative to the repo root (the parent of `config/`).

## Local LLM

`[summarizer] provider = "local_llm"` talks to any server that implements the
OpenAI `POST /v1/chat/completions` API. Mentions are batched (`batch_size`) into a few
requests per run, and the model is asked to return one JSON object.

| Server | `base_url` | Start it with |
| --- | --- | --- |
| Ollama | `http://127.0.0.1:11434/v1` | `ollama serve` |
| llama.cpp | `http://127.0.0.1:8080/v1` | `llama-server -m model.gguf` |
| llamafile | `http://127.0.0.1:8080/v1` | `./model.llamafile --server` |
| LM Studio | `http://127.0.0.1:1234/v1` | start its local server |

```toml
[summarizer]
provider = "local_llm"
fallback_to_heuristic = true

[summarizer.llm]
base_url = "http://127.0.0.1:8080/v1"
model = "llama3.2"
api_key_env = ""            # set to an env var name if your server needs a bearer token
request_timeout_seconds = 240
batch_size = 6
```

signalwatch does not start or stop the model server; run it yourself. If the server is
unreachable, times out, or returns something unparseable, `fallback_to_heuristic = true`
(the default) summarizes those mentions with the keyword heuristic instead. The console
summary at the end of a run shows how many summaries came from each summarizer. Small models can struggle
with the JSON format, so lower `batch_size` if you see a lot of fallbacks.

## Configuration

See [config/example.toml](config/example.toml) for every option. The main pieces:

**Keywords**

```toml
[[keywords]]
name = "acme"                       # label used in the digest
query = "\"acme\" OR \"acme corp\""  # Google query syntax
exclude_terms = ["roadrunner"]      # dropped locally and sent as -term
selected_domains = ["news.ycombinator.com", "lobste.rs"]
rejected_domains = ["medium.com"]   # on top of the global list
max_results_per_source = 5
```

**Sources** (`[providers.sources]`)

- `google_search`: Google web results from the last 24 hours
- `google_news`: Google News results
- `selected_web`: `site:domain query` for each entry in `selected_domains`, one SerpAPI
  search per domain. This is also how you monitor Reddit, Hacker News, GitHub, and so on
  without their APIs.

Each search costs one SerpAPI credit, so a keyword with all three sources and four selected
domains uses 6 searches per run. Toggle sources off to control spend.

**Filters**

- `[filters] rejected_domains` drops whole sites on every keyword. Subdomains match, and
  `facebook.com` does not match `notfacebook.com`.
- `exclude_terms` matches title, snippet, and source on word boundaries, and matches the URL
  as a squashed substring.
- Both apply to already-stored rows too, so tightening a filter hides old results from
  the next digest.

**Debugging**: `SIGNALWATCH_HTTP_DEBUG=1` logs outbound requests to stderr, with API keys
and auth headers redacted.

A failing keyword/source pair does not abort the run. Failures are listed in the digest and
printed on the console.

## Scheduling

```cron
0 8 * * *  cd /path/to/signalwatch && SERPAPI_API_KEY=... python3 -m signalwatch run --config config/example.toml
```

## Development

```bash
python3 -m unittest discover -s tests
```

## License

MIT, see [LICENSE](LICENSE).
