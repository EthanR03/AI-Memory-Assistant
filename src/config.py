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
