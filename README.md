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
| `tools.py` | a read-only SQL window onto `nfl.db` |
| `generator.py` | pick a tool, run it, answer from what came back |

```bash
python -m src.ingest                      # build the store (re-runnable)
python -m src.search "goal of the project"   # see the raw retrieved chunks
python -m src.ask    "goal of the project"   # get a grounded answer
```

`python -m src.ask` with no arguments opens an interactive prompt.

### One assistant, two backends

`generator.py` is not a fixed pipeline any more. It hands the model two
tools and lets it choose:

- **`search_notes`** — semantic search over your notes, for what *you*
  wrote and decided.
- **`query_nfl_db`** — read-only SQL over the predictor's store, for
  anything factual about football: games, lines, weather, model
  forecasts, bios for every player since 1974 and season statistics
  from 1999 on.

That split is the lesson the Fact Book taught, applied deliberately:
prose gets retrieved, records get queried. "Which club has the best
record since 2016" is not a fact stored anywhere — it is two lines of
SQL, and computing it returns the exact answer where embed-and-recall
returns a plausible one.

```
$ python -m src.ask "What NFL team has been the most successful in the last 10 years?"
The Kansas City Chiefs ... with 135 wins and 53 losses from 2016 to 2025.
(retrieved from: nfl.db)

$ python -m src.ask "Who is Joe Burrow?"
Joe Burrow ... born 10 Dec 1996, 6'4", LSU, first overall in 2020 ...
20,810 career passing yards, 43-33-1 as a regular-season starter.
(retrieved from: nfl.db)

$ python -m src.ask "Who led the NFL in receiving yards in 2023?"
Tyreek Hill led the NFL in receiving yards in 2023 with 1,799 yards.
(retrieved from: nfl.db)

$ python -m src.ask "What is the goal of the project?"
... build a retrieval-based memory assistant ... (source: sample.md)
(retrieved from: sample.md)
```

The assistant is told to say it has no memory of something rather than
guess, so a wrong notes answer is a retrieval bug you can see with
`src.search`. It is also told that the predictor **has no demonstrated
edge**, so a forecast is reported as the model's opinion and never as
betting advice.

The loop runs at `temperature=0` on purpose: at the default, the same
question wrote different SQL each run and returned 137 wins once and 135
the next. One of those was wrong, and for query generation there is no
upside to sampling.

The SQL tool is also usable on its own, which is the easiest way to check
a query before trusting an answer built on it:

```bash
python -m src.tools "SELECT COUNT(*) FROM games WHERE played = 1"
python -m src.tools          # no args: print the schema notes
```

It opens the store `mode=ro`, so SQLite refuses writes whatever SQL
arrives — a prompt-injected `DROP` fails at the driver, not at a regex.

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
python -m src.nfl.situational  # are the situational columns mispriced?
python -m src.nfl.players    # optional: bios, for chat
python -m src.nfl.playerstats  # optional: season stats, for chat
```

That yields 7,548 games (1999–2026) with closing lines, 32 team blocks,
3,904 team-season stat rows, and a walk-forward Elo backtest. The
headline result is negative and stated as such: the model hits 64.6%
against the market's 66.4% and finds no spread filter that clears
break-even by more than noise. Measuring that was the point.

Stage 3 then asks whether the market *mis*-prices rest, byes, divisional
games, roof and weather. On margins it does not — nine features against
the spread's own residual give R² = 0.0008 and no |t| > 2. On **totals**
there is one live result: every mph of wind is worth **−0.24 points** the
closing number does not take out, and the slope replicates almost exactly
on a held-out half (−0.239 fitted on 2010–17, −0.239 on 2018–25). It is a
lead rather than a strategy — `wind` is the reading at kickoff, and the
market prices a forecast.

`scripts/` holds the one-off exploration that shaped the schema —
`analyze_factbook.py`, `map_team_blocks.py`, `explore_nflverse.py`,
`baseline_ratings.py`. They print findings; they don't write to the
store.

## What is not in git

`.gitignore` keeps out the 22 MB source PDF, the extracted text,
`nfl.db`, `chroma_db/` and `.env`. Everything derived is rebuilt by the
commands above.

`data/nflverse/games.csv` (~2 MB) *is* committed, so a fresh clone can
build a working store with no network access and no PDF:

```bash
python -m src.nfl.enrich     # seeds the 32 clubs, loads 7,548 games
python -m src.nfl.ratings    # full backtest, identical numbers
```

Skipping the Fact Book costs you the cross-validation step (there is no
second source to check against), the statistics matrices, venues,
coaching histories and QB records — so `src.nfl.validate` reports 15/16
there, failing only the check that wants `team_season_stats` populated.
(Two of the 18 checks don't run at all without the book, which is why the
denominator drops to 16.) Everything the model itself reads comes from
nflverse.

One honest caveat on that 15/16: the two checks that sum the 2025 games
against the book's standings page **pass vacuously** on a PDF-less store,
because `standings` is empty and an empty comparison finds no mismatch.
They are real checks when the book is loaded and green-but-meaningless
when it isn't.

The player feeds — `data/nflverse/players.csv` (~7 MB) and
`data/nflverse/stats/` (54 files, ~25 MB) — are *not* committed. Nothing
in the build or the backtest reads them; they only serve the chat layer,
so `src.nfl.players` and `src.nfl.playerstats` fetch them on demand.

The CSV is a snapshot; refresh it with
`python -m src.nfl.enrich --force-download` rather than deleting it, and
expect in-season results and lookahead lines to drift until you do.

The two inputs you supply yourself are the Fact Book PDF (into `data/`)
and your OpenAI key (into `.env`).
