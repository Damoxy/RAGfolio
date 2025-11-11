"""
Document indexing and processing system
"""
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import click

try:
    import faiss
except ImportError:
    faiss = None

from sentence_transformers import SentenceTransformer

try:
    from .config import (
        DOCUMENTS_DIR, DATABASE_PATH, FAISS_INDEX_PATH, 
        EMBEDDING_MODEL, CHUNK_SIZE, OVERLAP_SIZE, SUPPORTED_EXTENSIONS
    )
    from .document_parser import DocumentParser, chunk_text
    from .version_manager import VersionManager
except ImportError:
    # For direct execution
    from config import (
        DOCUMENTS_DIR, DATABASE_PATH, FAISS_INDEX_PATH, 
        EMBEDDING_MODEL, CHUNK_SIZE, OVERLAP_SIZE, SUPPORTED_EXTENSIONS
    )
    from document_parser import DocumentParser, chunk_text
    from version_manager import VersionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Index documents with embeddings and metadata"""
    
    def __init__(self):
        self.parser = DocumentParser()
        self.version_manager = VersionManager()
        self.embedding_model = None
        self.faiss_index = None
        self.dimension = None
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metadata"""
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_name TEXT NOT NULL,
                    title TEXT,
                    content_hash TEXT,
                    file_size INTEGER,
                    created_time TEXT,
                    modified_time TEXT,
                    processed_time TEXT,
                    version_id TEXT,
                    priority TEXT DEFAULT 'medium',
                    topics TEXT,  -- JSON array of topics
                    metadata TEXT  -- JSON metadata
                )
            """)
            
            # Chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding_index INTEGER,  -- Index in FAISS
                    char_start INTEGER,
                    char_end INTEGER,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # Index table for tracking FAISS index state
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS index_info (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            conn.commit()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model"""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
    
    def _load_faiss_index(self):
        """Load or create FAISS index"""
        if faiss is None:
            raise ImportError("FAISS is required for vector indexing")
        
        self._load_embedding_model()
        
        if FAISS_INDEX_PATH.exists():
            logger.info("Loading existing FAISS index")
            self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        else:
            logger.info(f"Creating new FAISS index with dimension {self.dimension}")
            self.faiss_index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
            self._save_faiss_index()
    
    def _save_faiss_index(self):
        """Save FAISS index to disk"""
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(FAISS_INDEX_PATH))
    
    def initialize(self):
        """Initialize the indexer (create tables, load models)"""
        logger.info("Initializing RAGfolio indexer...")
        self._init_database()
        self._load_embedding_model()
        self._load_faiss_index()
        
        # Store model info in database
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO index_info (key, value) VALUES (?, ?)",
                ("embedding_model", EMBEDDING_MODEL)
            )
            cursor.execute(
                "INSERT OR REPLACE INTO index_info (key, value) VALUES (?, ?)",
                ("dimension", str(self.dimension))
            )
            conn.commit()
        
        logger.info("Indexer initialized successfully")
    
    def add_document(self, file_path: Path, force_reprocess: bool = False) -> bool:
        """
        Add a single document to the index
        
        Args:
            file_path: Path to the document
            force_reprocess: Force reprocessing even if up to date
            
        Returns:
            True if document was processed, False if skipped
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file type: {file_path}")
            return False
        
        # Check if processing is needed
        if not force_reprocess and not self.version_manager.needs_reprocessing(file_path, EMBEDDING_MODEL):
            logger.info(f"Skipping {file_path} (up to date)")
            return False
        
        logger.info(f"Processing {file_path}")
        
        try:
            # Parse document
            content, metadata = self.parser.parse_document(file_path)
            
            if not content.strip():
                logger.warning(f"No content found in {file_path}")
                return False
            
            # Chunk content
            chunks = chunk_text(content, CHUNK_SIZE, OVERLAP_SIZE)
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            
            # Get version info
            version_info = self.version_manager.get_document_version(file_path)
            
            # Load models if needed
            self._load_embedding_model()
            self._load_faiss_index()
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
            
            # Store in database and FAISS
            self._store_document_and_chunks(file_path, content, chunks, embeddings, metadata, version_info)
            
            # Track processing
            self.version_manager.track_document_processing(file_path, len(chunks), EMBEDDING_MODEL)
            
            logger.info(f"Successfully processed {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False
    
    def _store_document_and_chunks(self, file_path: Path, content: str, chunks: List[str], 
                                 embeddings: np.ndarray, metadata: Dict[str, Any], 
                                 version_info: Dict[str, Any]):
        """Store document and chunks in database and FAISS index"""
        
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Remove existing document and chunks
            cursor.execute("DELETE FROM documents WHERE file_path = ?", (str(file_path),))
            
            # Insert document
            cursor.execute("""
                INSERT INTO documents (
                    file_path, file_name, title, content_hash, file_size,
                    created_time, modified_time, processed_time, version_id,
                    priority, topics, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(file_path),
                file_path.name,
                metadata.get('title', file_path.stem),
                version_info.get('full_hash', ''),
                metadata.get('file_size', 0),
                metadata.get('created_time', ''),
                metadata.get('modified_time', ''),
                datetime.now().isoformat(),
                version_info.get('version_id', ''),
                metadata.get('priority', 'medium'),
                json.dumps(metadata.get('topics', [])),
                json.dumps(metadata)
            ))
            
            document_id = cursor.lastrowid
            
            # Add embeddings to FAISS index
            faiss_start_idx = self.faiss_index.ntotal
            
            # Normalize embeddings for cosine similarity
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.faiss_index.add(normalized_embeddings.astype(np.float32))
            
            # Insert chunks
            for i, chunk in enumerate(chunks):
                cursor.execute("""
                    INSERT INTO chunks (
                        document_id, chunk_index, content, embedding_index,
                        char_start, char_end
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    document_id,
                    i,
                    chunk,
                    faiss_start_idx + i,
                    0,  # TODO: Calculate actual character positions
                    len(chunk)
                ))
            
            conn.commit()
        
        # Save updated FAISS index
        self._save_faiss_index()
    
    def index_all_documents(self, force_reprocess: bool = False):
        """Index all documents in the documents directory"""
        logger.info(f"Indexing all documents in {DOCUMENTS_DIR}")
        
        # Find all supported documents
        document_files = []
        for extension in SUPPORTED_EXTENSIONS:
            document_files.extend(DOCUMENTS_DIR.rglob(f"*{extension}"))
        
        logger.info(f"Found {len(document_files)} documents to process")
        
        processed = 0
        skipped = 0
        
        for file_path in document_files:
            if self.add_document(file_path, force_reprocess):
                processed += 1
            else:
                skipped += 1
        
        logger.info(f"Processing complete: {processed} processed, {skipped} skipped")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Document count
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            # Chunk count
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            
            # FAISS index size
            faiss_size = self.faiss_index.ntotal if self.faiss_index else 0
            
            # Get model info
            cursor.execute("SELECT value FROM index_info WHERE key = 'embedding_model'")
            model_result = cursor.fetchone()
            model_name = model_result[0] if model_result else "Unknown"
            
            return {
                "documents": doc_count,
                "chunks": chunk_count,
                "faiss_vectors": faiss_size,
                "embedding_model": model_name,
                "database_path": str(DATABASE_PATH),
                "faiss_index_path": str(FAISS_INDEX_PATH)
            }
    
    def rebuild_index(self):
        """Rebuild the entire index from scratch"""
        logger.info("Rebuilding index from scratch...")
        
        # Clear database
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM documents")
            conn.commit()
        
        # Clear FAISS index
        if self.faiss_index:
            self.faiss_index.reset()
        else:
            self._load_faiss_index()
        
        # Reindex all documents
        self.index_all_documents(force_reprocess=True)
        
        logger.info("Index rebuild complete")


# CLI Interface
@click.command()
@click.option('--initialize', is_flag=True, help='Initialize the indexer')
@click.option('--add', type=click.Path(exists=True, path_type=Path), help='Add a single document')
@click.option('--index-all', is_flag=True, help='Index all documents')
@click.option('--force', is_flag=True, help='Force reprocessing of all documents')
@click.option('--rebuild', is_flag=True, help='Rebuild the entire index')
@click.option('--stats', is_flag=True, help='Show index statistics')
def main(initialize, add, index_all, force, rebuild, stats):
    """RAGfolio Document Indexer"""
    
    indexer = DocumentIndexer()
    
    if initialize:
        indexer.initialize()
    
    if add:
        indexer.add_document(add, force_reprocess=force)
    
    if index_all:
        indexer.index_all_documents(force_reprocess=force)
    
    if rebuild:
        indexer.rebuild_index()
    
    if stats:
        stats_data = indexer.get_index_stats()
        click.echo("Index Statistics:")
        for key, value in stats_data.items():
            click.echo(f"  {key}: {value}")


if __name__ == '__main__':
    main()
