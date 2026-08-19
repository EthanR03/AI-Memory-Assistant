# ai-mem-assistant

Two things live in this repo, sharing one config module and one `data/`
folder:

1. **A RAG memory assistant** (`src/`) — drop notes into `data/`, ingest
   them into a local vector store, and ask questions that are answered
   only from what you actually wrote.
2. **An NFL game predictor** (`src/nfl/`, `scripts/`) — the 2026 *Record
   & Fact Book* PDF plus 27 seasons of nflverse game data, parsed into a
   SQLite feature store, with a walk-forward backtest that measures a
   ratings model against the closing spread.

They started as one project — the predictor grew out of trying to ingest
an 883-page PDF into the assistant, and it turned out that book wants
structured records, not 800-character chunks.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate elsewhere
pip install -r requirements.txt
echo OPENAI_API_KEY=sk-... > .env
```

## The memory assistant

A textbook RAG pipeline, one stage per module, each readable on its own:

| Module | Stage |
|---|---|
| `loader.py` | read `.txt` / `.md` files out of `data/` |
| `chunker.py` | split into overlapping ~800-char chunks |
| `embedder.py` | embed with OpenAI `text-embedding-3-small` |
| `store.py` | upsert into a persistent ChromaDB collection |
| `retriever.py` | embed the question, pull the nearest chunks |
| `generator.py` | answer from those chunks *only*, citing the file |

```bash
python -m src.ingest                      # build the store (re-runnable)
python -m src.search "goal of the project"   # see the raw retrieved chunks
python -m src.ask    "goal of the project"   # get a grounded answer
```

`python -m src.ask` with no arguments opens an interactive prompt. The
generator is instructed to say it has no memory of something rather than
guess, so a wrong answer is a retrieval bug you can see with
`src.search`.

Settings — chunk size, overlap, model names, paths — all live in
`src/config.py`.

## The NFL predictor

Full write-up, schema, verification method and results:
**[`src/nfl/README.md`](src/nfl/README.md)**.

The short version:

```bash
python scripts/extract_pdf.py data/2026-Record-and-Fact-Book.pdf \
    --out data/extracted/factbook.pypdf.txt
python -m src.nfl.build      # PDF text -> nfl.db
python -m src.nfl.enrich     # add nflverse, cross-validate the overlap
python -m src.nfl.validate   # 18 consistency checks
python -m src.nfl.ratings    # backtest + market tests + 2026 projections
```

That yields 7,548 games (1999–2026) with closing lines, 32 team blocks,
3,904 team-season stat rows, and a walk-forward Elo backtest. The
headline result is negative and stated as such: the model hits 64.6%
against the market's 66.4% and finds no spread filter that clears
break-even by more than noise. Measuring that was the point.

`scripts/` holds the one-off exploration that shaped the schema —
`analyze_factbook.py`, `map_team_blocks.py`, `explore_nflverse.py`,
`baseline_ratings.py`. They print findings; they don't write to the
store.

## What is not in git

`.gitignore` keeps out the 22 MB source PDF, the extracted text,
`data/nflverse/`, `nfl.db`, `chroma_db/` and `.env`. Everything derived
is rebuilt by the commands above; the two inputs you supply yourself are
the Fact Book PDF (into `data/`) and your OpenAI key (into `.env`).
