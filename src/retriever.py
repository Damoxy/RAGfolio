"""
Semantic retrieval system with recency and priority weighting
"""
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import re
import click

try:
    import faiss
except ImportError:
    faiss = None

from sentence_transformers import SentenceTransformer

try:
    from .config import (
        DATABASE_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL, MAX_RESULTS,
        RECENCY_WEIGHT, PRIORITY_WEIGHT, SIMILARITY_WEIGHT
    )
except ImportError:
    # For direct execution
    from config import (
        DATABASE_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL, MAX_RESULTS,
        RECENCY_WEIGHT, PRIORITY_WEIGHT, SIMILARITY_WEIGHT
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Retrieve documents using semantic search with recency and priority weighting"""
    
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.dimension = None
        
        # Priority weights
        self.priority_weights = {
            'high': 1.0,
            'medium': 0.7,
            'low': 0.4
        }
    
    def _load_models(self):
        """Load embedding model and FAISS index"""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        if self.faiss_index is None:
            if not FAISS_INDEX_PATH.exists():
                raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
            
            logger.info("Loading FAISS index")
            self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
    
    def search(self, query: str, max_results: int = None, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search with filtering and ranking
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            filters: Optional filters (topics, priority, date_range, etc.)
            
        Returns:
            List of search results with metadata and scores
        """
        if max_results is None:
            max_results = MAX_RESULTS
        
        # Load models
        self._load_models()
        
        # Parse query for special filters
        query, parsed_filters = self._parse_query(query)
        
        # Merge filters
        if filters:
            parsed_filters.update(filters)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
        
        # Search FAISS index (get more results for filtering and reranking)
        search_k = min(max_results * 3, self.faiss_index.ntotal)
        similarities, indices = self.faiss_index.search(query_embedding.astype(np.float32), search_k)
        
        # Get chunk and document information
        results = self._get_chunk_details(indices[0], similarities[0])
        
        # Apply filters
        if parsed_filters:
            results = self._apply_filters(results, parsed_filters)
        
        # Apply recency and priority weighting
        results = self._apply_weighting(results)
        
        # Sort by final score and limit results
        results.sort(key=lambda x: x['final_score'], reverse=True)
        results = results[:max_results]
        
        # Add ranking information
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return results
    
    def _parse_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Parse query for special filter syntax"""
        filters = {}
        
        # Extract date filters: after:2024-01-01, before:2024-12-31
        date_pattern = r'(after|before):(\d{4}-\d{2}-\d{2})'
        matches = re.findall(date_pattern, query)
        for operator, date_str in matches:
            if 'date_range' not in filters:
                filters['date_range'] = {}
            filters['date_range'][operator] = date_str
        
        # Remove date filters from query
        query = re.sub(date_pattern, '', query)
        
        # Extract topic filters: topic:meditation
        topic_pattern = r'topic:(\w+)'
        topic_matches = re.findall(topic_pattern, query)
        if topic_matches:
            filters['topics'] = topic_matches
        
        # Remove topic filters from query
        query = re.sub(topic_pattern, '', query)
        
        # Extract priority filters: priority:high
        priority_pattern = r'priority:(high|medium|low)'
        priority_matches = re.findall(priority_pattern, query)
        if priority_matches:
            filters['priority'] = priority_matches
        
        # Remove priority filters from query
        query = re.sub(priority_pattern, '', query)
        
        # Clean up query
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query, filters
    
    def _get_chunk_details(self, indices: np.ndarray, similarities: np.ndarray) -> List[Dict[str, Any]]:
        """Get detailed information for retrieved chunks"""
        results = []
        
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            for idx, similarity in zip(indices, similarities):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                # Get chunk information
                cursor.execute("""
                    SELECT c.id, c.content, c.chunk_index, c.char_start, c.char_end,
                           d.id as doc_id, d.file_path, d.file_name, d.title,
                           d.created_time, d.modified_time, d.processed_time,
                           d.version_id, d.priority, d.topics, d.metadata
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.embedding_index = ?
                """, (int(idx),))
                
                row = cursor.fetchone()
                if row:
                    topics = json.loads(row[14]) if row[14] else []
                    metadata = json.loads(row[15]) if row[15] else {}
                    
                    result = {
                        'chunk_id': row[0],
                        'content': row[1],
                        'chunk_index': row[2],
                        'char_start': row[3],
                        'char_end': row[4],
                        'document_id': row[5],
                        'file_path': row[6],
                        'file_name': row[7],
                        'title': row[8],
                        'created_time': row[9],
                        'modified_time': row[10],
                        'processed_time': row[11],
                        'version_id': row[12],
                        'priority': row[13],
                        'topics': topics,
                        'metadata': metadata,
                        'similarity_score': float(similarity),
                    }
                    results.append(result)
        
        return results
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to search results"""
        filtered = results
        
        # Topic filter
        if 'topics' in filters:
            filter_topics = [t.lower() for t in filters['topics']]
            filtered = [
                r for r in filtered
                if any(topic.lower() in filter_topics for topic in r['topics'])
            ]
        
        # Priority filter
        if 'priority' in filters:
            filter_priorities = filters['priority']
            filtered = [r for r in filtered if r['priority'] in filter_priorities]
        
        # Date range filter
        if 'date_range' in filters:
            date_range = filters['date_range']
            
            if 'after' in date_range:
                after_date = datetime.fromisoformat(date_range['after'])
                filtered = [
                    r for r in filtered
                    if datetime.fromisoformat(r['modified_time']) >= after_date
                ]
            
            if 'before' in date_range:
                before_date = datetime.fromisoformat(date_range['before'])
                filtered = [
                    r for r in filtered
                    if datetime.fromisoformat(r['modified_time']) <= before_date
                ]
        
        return filtered
    
    def _apply_weighting(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply recency and priority weighting to results"""
        now = datetime.now()
        
        for result in results:
            # Base similarity score
            similarity = result['similarity_score']
            
            # Recency score (newer documents get higher scores)
            modified_time = datetime.fromisoformat(result['modified_time'])
            days_old = (now - modified_time).days
            
            # Exponential decay for recency (half-life of 180 days)
            recency_score = np.exp(-days_old / 180)
            
            # Priority score
            priority_score = self.priority_weights.get(result['priority'], 0.5)
            
            # Combined final score
            final_score = (
                SIMILARITY_WEIGHT * similarity +
                RECENCY_WEIGHT * recency_score +
                PRIORITY_WEIGHT * priority_score
            )
            
            result.update({
                'recency_score': recency_score,
                'priority_score': priority_score,
                'final_score': final_score,
                'days_old': days_old
            })
        
        return results
    
    def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT c.id, c.content, c.chunk_index, c.char_start, c.char_end,
                       d.file_path, d.file_name, d.title, d.priority, d.topics, d.metadata
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.id = ?
                ORDER BY c.chunk_index
            """, (document_id,))
            
            chunks = []
            for row in cursor.fetchall():
                topics = json.loads(row[9]) if row[9] else []
                metadata = json.loads(row[10]) if row[10] else {}
                
                chunks.append({
                    'chunk_id': row[0],
                    'content': row[1],
                    'chunk_index': row[2],
                    'char_start': row[3],
                    'char_end': row[4],
                    'file_path': row[5],
                    'file_name': row[6],
                    'title': row[7],
                    'priority': row[8],
                    'topics': topics,
                    'metadata': metadata
                })
            
            return chunks
    
    def get_similar_documents(self, document_id: int, max_results: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document"""
        chunks = self.get_document_chunks(document_id)
        if not chunks:
            return []
        
        # Use the first chunk as the query (or could combine multiple chunks)
        query_content = chunks[0]['content']
        
        # Search for similar content
        results = self.search(query_content, max_results=max_results * 2)
        
        # Filter out chunks from the same document and group by document
        seen_documents = set()
        similar_docs = []
        
        for result in results:
            if result['document_id'] != document_id and result['document_id'] not in seen_documents:
                seen_documents.add(result['document_id'])
                similar_docs.append(result)
                
                if len(similar_docs) >= max_results:
                    break
        
        return similar_docs
    
    def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on indexed content"""
        # This is a simple implementation - could be enhanced with more sophisticated methods
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Get topics that match the partial query
            cursor.execute("""
                SELECT DISTINCT topics FROM documents
                WHERE topics LIKE ? AND topics != ''
                LIMIT ?
            """, (f'%{partial_query}%', limit))
            
            suggestions = []
            for row in cursor.fetchall():
                topics = json.loads(row[0]) if row[0] else []
                for topic in topics:
                    if partial_query.lower() in topic.lower() and topic not in suggestions:
                        suggestions.append(topic)
                        if len(suggestions) >= limit:
                            break
            
            return suggestions


# CLI Interface
@click.command()
@click.argument('query')
@click.option('--max-results', default=10, help='Maximum number of results')
@click.option('--topic', multiple=True, help='Filter by topics')
@click.option('--priority', multiple=True, type=click.Choice(['high', 'medium', 'low']), help='Filter by priority')
@click.option('--after', help='Show documents after date (YYYY-MM-DD)')
@click.option('--before', help='Show documents before date (YYYY-MM-DD)')
@click.option('--verbose', is_flag=True, help='Show detailed scoring information')
def main(query, max_results, topic, priority, after, before, verbose):
    """Search the document corpus"""
    
    retriever = DocumentRetriever()
    
    # Build filters
    filters = {}
    if topic:
        filters['topics'] = list(topic)
    if priority:
        filters['priority'] = list(priority)
    if after or before:
        filters['date_range'] = {}
        if after:
            filters['date_range']['after'] = after
        if before:
            filters['date_range']['before'] = before
    
    # Perform search
    results = retriever.search(query, max_results, filters)
    
    # Display results
    if not results:
        click.echo("No results found.")
        return
    
    for result in results:
        click.echo(f"\n--- Result {result['rank']} ---")
        click.echo(f"Title: {result['title']}")
        click.echo(f"File: {result['file_name']}")
        click.echo(f"Priority: {result['priority']}")
        click.echo(f"Topics: {', '.join(result['topics'])}")
        click.echo(f"Score: {result['final_score']:.3f}")
        
        if verbose:
            click.echo(f"  Similarity: {result['similarity_score']:.3f}")
            click.echo(f"  Recency: {result['recency_score']:.3f}")
            click.echo(f"  Priority Weight: {result['priority_score']:.3f}")
            click.echo(f"  Days Old: {result['days_old']}")
        
        click.echo(f"\nContent Preview:")
        content = result['content'][:300]
        if len(result['content']) > 300:
            content += "..."
        click.echo(content)
        click.echo("-" * 50)


if __name__ == '__main__':
    main()
