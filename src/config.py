"""
Configuration settings for RAGfolio
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
VERSIONS_DIR = DATA_DIR / "versions"

# API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-sonnet")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # for all-MiniLM-L6-v2

# Processing Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", "50"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))

# Database Configuration
DATABASE_PATH = PROJECT_ROOT / os.getenv("DATABASE_PATH", "data/metadata.db")
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "index.faiss"

# Version Control
USE_GIT_VERSIONING = os.getenv("USE_GIT_VERSIONING", "true").lower() == "true"

# Supported file types
SUPPORTED_EXTENSIONS = {
    ".md", ".txt", ".docx", ".pdf"
}

# Priority weights
RECENCY_WEIGHT = 0.3
PRIORITY_WEIGHT = 0.2
SIMILARITY_WEIGHT = 0.5

# Create directories if they don't exist
for directory in [DOCUMENTS_DIR, DATA_DIR, EMBEDDINGS_DIR, VERSIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
