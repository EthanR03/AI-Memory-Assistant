"""Central configuration for the ingestion pipeline.

Keeping every setting in one place means you tune a value here once
instead of hunting through every file. Import this module anywhere you
need a setting.
"""
from pathlib import Path

from dotenv import load_dotenv

# Load variables from your .env file (e.g. OPENAI_API_KEY) into the
# process environment. The OpenAI client reads OPENAI_API_KEY from there
# automatically, so you never have to pass the key around by hand.
load_dotenv()

# --- Paths ---------------------------------------------------------------
# PROJECT_ROOT is the folder that contains this `src/` package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"          # you drop source files here
CHROMA_DIR = PROJECT_ROOT / "chroma_db"   # the vector store lives here on disk

# --- Embedding model -----------------------------------------------------
# text-embedding-3-small: cheap, fast, 1536 dimensions, very solid quality.
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE = 800       # roughly how many characters per chunk
CHUNK_OVERLAP = 150    # characters each chunk shares with its neighbour

# --- Vector store --------------------------------------------------------
COLLECTION_NAME = "memories"

# --- NFL predictor -------------------------------------------------------
# Text extracted from the Record & Fact Book PDF (see scripts/extract_pdf.py)
# and the SQLite feature store the parsers build from it.
EXTRACTED_DIR = DATA_DIR / "extracted"
FACTBOOK_TXT = EXTRACTED_DIR / "factbook.pypdf.txt"
NFL_DB = PROJECT_ROOT / "nfl.db"

# Stage 2: nflverse game data (results + closing lines, 1999-present).
NFLVERSE_DIR = DATA_DIR / "nflverse"
NFLVERSE_GAMES = NFLVERSE_DIR / "games.csv"

# Player bios (nflverse `players` release). Unlike games.csv this is NOT
# committed - it is 7 MB, it is not needed to build or backtest, and
# `python -m src.nfl.players` fetches it on demand.
NFLVERSE_PLAYERS = NFLVERSE_DIR / "players.csv"

# The book's data is frozen at this date; anything later (cuts, trades,
# injuries) is NOT in it. Worth surfacing in any answer built on it.
FACTBOOK_AS_OF = "2026-07-14"

# Season anchors: the Wednesday on/before each season's first game. Used to
# derive week numbers from the bare dates the book prints.
SEASON_ANCHORS = {2025: "2025-09-03", 2026: "2026-09-09"}
