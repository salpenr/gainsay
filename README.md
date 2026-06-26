# Gainsay

**A local engine for intellectual due process.**

Most answer engines are built to tell you what is true. Gainsay is built to show you *how* a
conclusion was reached — and to leave you free to disagree with it. It runs entirely on your own
machine, cites every claim back to the source it came from, and surfaces where those sources
contradict one another. The promise is not a better answer; it is an answer you can **inspect,
reproduce, and contest** — without having to trust anyone, including us.

> *Trust the process enough that you never have to trust the authority.*

## What Gainsay promises

These are guarantees, not features. They do not depend on any particular implementation, and the
project binds itself to uphold them — see the [**Constitution**](./CONSTITUTION.md):

1. **The evidence is inspectable** — every claim carries a tag back to the exact source it came from.
2. **Disagreement is surfaced** — where sources conflict, the conflict is shown, not smoothed away.
3. **The weighing is visible** — you can see *why* a conclusion currently prevails, not merely that it does.
4. **Every conclusion is contestable** — no verdict is final; enough is shown for you to challenge it.
5. **The process is reproducible** — the same question over the same evidence yields a path you can re-walk, locally.
6. **Change is traceable** — when the evidence shifts, the answer has a history, not only a present.

## What Gainsay does *not* promise

It does **not** claim to tell you what is true, to be a final authority, or to be correct because it
is confident. It can be wrong. The point was never that it does not err — it is that when it does, the
path to seeing and correcting the error stays open. *Every conclusion stays challengeable.*

## How to verify these guarantees yourself

You should not have to take the promises on faith — that would defeat the purpose. Each is checkable:

- **Inspectable / surfaced / weighed** — every answer prints its sources `[W#] [B#] [S#]` and a
  per-claim **support-vs-contradiction** panel. Read them; nothing is asserted that isn't traceable.
- **Contestable** — run `--verify` and read where the engine flags its *own* answer as unsupported or
  contradicted by its sources.
- **Reproducible** — ask the same question twice, or run `--no-web` for a fully-offline, local-only path
  you can re-walk.
- **Private** — every web fetch is logged locally to `~/.gainsay/web-audit/`; inspect exactly what
  touched the network. Nothing else leaves your machine.
- **Governed** — read the [Constitution](./CONSTITUTION.md) and hold the project to it. Every change is
  reviewed against one question: *does this strengthen or weaken the guarantees above?*

---

## How it works

```
question
   |
   v
decompose (optional, "deep" mode) ----> sub-questions
   |
   v
retrieve from each enabled tier:
   web search  [W#]      (privacy-routed; query-only egress)
   your library [B#]     (local vector store over YOUR documents)
   scholarly   [S#]      (OpenAlex / Semantic Scholar / arXiv)
   |
   v
rerank (LLM-as-judge, retrieve-many-then-rerank, diversity-capped)
   |
   v
synthesize a cited answer (local model, tools off, sources fenced as data)
   |
   v
verify / disagreement engine: per-claim support vs. contradiction across sources
   |
   v
cited, reranked, contradiction-checked answer
```

### The tiers

| Tag    | Tier               | Trust    | Notes                                                        |
|--------|--------------------|----------|-------------------------------------------------------------|
| `[W#]` | Live web search    | Untrusted| Only the search query leaves your machine.                  |
| `[B#]` | Your own library   | Trusted  | A local vector store over documents **you** index.          |
| `[S#]` | Scholarly sources  | Untrusted| OpenAlex / Semantic Scholar / arXiv connectors.             |

Gainsay ships with an **empty library index**. The `[B#]` tier is a *capability*, not a bundled corpus
— you decide what goes in it (see "Index your own documents" below).

## How the guarantees are kept

The promises above are enforced *structurally*, not by good intentions:

- **Cited by construction** (Guarantee 1). Every claim carries a tag back to the exact source it came
  from. A structural check rejects answers that cite sources that were never retrieved — so a citation
  cannot be fabricated.
- **The disagreement engine** (Guarantees 2–4). After the answer is written, Gainsay extracts its
  load-bearing claims and, for each, shows which retrieved sources *support* it, which *contradict* it,
  and where they conflict — the thing a single hosted model structurally cannot do, because it has no
  persistent multi-source corpus to cross-check against. The analysis is computed live and deliberately
  **not** persisted as a confidence score (a model's confidence guess, written to disk, just calcifies a
  guess into a "fact").
- **Tools-off synthesis** (protects every guarantee). Retrieved web text is third-party content and a
  prime indirect-prompt-injection vector (OWASP LLM01). The structural guarantee — which holds even
  though this is open source — is that all retrieved text is fenced inside an explicit data boundary and
  synthesized with **tools disabled**: the worst a poisoned page can do is skew an answer, never trigger
  an action. Layered on top are best-effort heuristics (homoglyph folding, defanging, an injection
  tripwire) — treat those as a tripwire to **extend for your own threat model**, not a guarantee.
- **Local and private** (Guarantee 5 + privacy). Synthesis, embeddings, and reranking all run against a
  local Ollama server. The only thing that leaves the machine is the keyword search query — and you can
  disable even that with `--no-web`.

---

## Requirements

- **Python 3.10+**
- **A local [Ollama](https://ollama.com) install**, with:
  - a chat model for synthesis (default `gpt-oss:20b`)
  - an embedding model for the library tier (e.g. `nomic-embed-text`)
- `pip install gainsay` pulls in the helpful extras automatically (`numpy` for fast search,
  `beautifulsoup4` for HTML cleaning, `charset-normalizer` for encoding). The core loop runs on the
  standard library alone; offline translation is the one opt-in extra (`pip install "gainsay[translate]"`).

```bash
# 1. install Gainsay
pip install gainsay

# 2. install Ollama from https://ollama.com, then pull the models:
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

Prefer running from a clone without installing? Use `python -m gainsay "your question"` in place of the
`gainsay` command shown below.

---

## Usage

```bash
# ask a question (web + your library, reranked, cited)
gainsay "what is retrieval-augmented generation?"

# control how much evidence to pull
gainsay --web 6 --books 4 "explain RRF rank fusion"

# library only (fully offline; nothing leaves your machine)
gainsay --no-web "what does my style guide say about headings?"

# web only (skip the library tier)
gainsay --no-books "latest stable release of sqlite"

# deep mode: decompose the question into sub-questions first
gainsay --deep "compare two approaches to vector search"

# turn on the disagreement engine explicitly
gainsay --verify "is X true?"

# add the scholarly tier
gainsay --scholar "evidence for diffusion model guidance scaling"

# machine-readable output (for scripting / integration)
gainsay --json "your question"
```

Common flags:

| Flag           | Effect                                                         |
|----------------|----------------------------------------------------------------|
| `--web N`      | Number of web results to search (default 5).                   |
| `--books N`    | Number of library passages to pull (default 4).                |
| `--fetch-top N`| How many top web results to fully fetch (default 3).           |
| `--no-web`     | Library only (fully offline).                                  |
| `--no-books`   | Web only.                                                      |
| `--deep`       | Agentic query decomposition.                                   |
| `--verify`     | Run the disagreement / contradiction engine.                   |
| `--scholar`    | Enable the scholarly tier.                                     |
| `--no-rerank`  | Disable LLM-as-judge reranking.                                |
| `--model NAME` | Override the synthesis model.                                  |
| `--json`       | Emit machine-readable JSON.                                    |

There is also a streaming web UI:

```bash
gainsay-web
# then open the printed local URL in your browser
```

---

## Index your own documents

Gainsay ships with an **empty** index. The library (`[B#]`) tier becomes useful once you point it at
documents you own. Indexing reads your files, splits them into chunks, embeds each chunk with your local
Ollama embedding model, and stores the vectors in a local sqlite database — nothing is uploaded.

```python
from gainsay import rag

# index a folder (or a single file) of YOUR documents
rag.index_path(r"/path/to/your/documents")

# sanity-check what retrieval returns
for chunk in rag.search("a question about your documents", k=5):
    print(chunk["path"], "->", chunk["text"][:120])
```

Re-run `index_path` whenever your documents change; indexing is incremental. Supported inputs include
plain text, Markdown, and HTML (HTML is stripped to text before embedding).

> **Bring your own corpus.** Gainsay does not bundle any copyrighted material. Point it at public-domain
> texts, your own notes, or documents you are licensed to use.

---

## Configuration

Gainsay reads a few optional environment variables:

| Variable               | Purpose                                          | Default                  |
|------------------------|--------------------------------------------------|--------------------------|
| `GAINSAY_MODEL`        | Ollama model used for synthesis.                          | `gpt-oss:20b`     |
| `GAINSAY_RERANK_MODEL` | Ollama model for reranking (falls back to `GAINSAY_MODEL`). | `GAINSAY_MODEL` |
| `GAINSAY_RERANK`       | Set `0` to disable LLM reranking.                         | on                |
| `GAINSAY_HYBRID`       | Set `0` to disable hybrid (BM25 + embedding) retrieval.   | on                |
| `TRANSLATE_MODEL`       | Ollama model for translating foreign-language sources.    | `qwen3:14b`       |

The synthesis model defaults to `gpt-oss:20b`. On a smaller machine you can set `GAINSAY_MODEL` to a
lighter model like `llama3.1:8b`. The library index path defaults to a per-user application-data
directory and can be left as-is for a single-user install.

---

## Privacy model

- The **only** outbound network traffic from the core loop is the keyword search query sent to a web
  search backend. You can disable even that with `--no-web` for fully-offline, library-only answers.
- Your full question, the retrieved passages, the synthesized answer, and the disagreement analysis are
  all produced by your **local** model and never uploaded.
- Web and scholarly sources are treated as untrusted data. Their text is defanged and fenced, and an
  injection tripwire flags suspicious passages so you can judge them with extra suspicion.

---

## Acknowledgments

Gainsay was built with the help of others, and shaped by one teacher's example.

- **Tina Huang** — data scientist and educator, whose teaching on *verifying* AI output rather than taking
  it at face value was the encouragement behind this project. Gainsay turns that lesson into structure: it
  cites every claim and cross-checks the answer against its own sources.
  *Homage only — she has not reviewed or endorsed Gainsay.* ([youtube.com/@TinaHuang1](https://www.youtube.com/@TinaHuang1))
- **Claude** (Anthropic) — assisted with the engineering, the design discussions, and the drafting of this
  project's Constitution and documentation.
- **ChatGPT** (OpenAI) — a second perspective for reviewing code and pressure-testing decisions along the way.

These AI tools were collaborators in the work, not its authority. Every decision about what the project
promises — and what it refuses to promise — was the author's.

---

## License

MIT — see [LICENSE](./LICENSE).
