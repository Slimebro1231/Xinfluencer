"""Vector search and retrieval utilities."""

from typing import List, Dict, Optional
from vector.db import VectorDB
from vector.embed import TextEmbedder
import logging

logger = logging.getLogger(__name__)

class VectorSearcher:
    """High-level vector search interface."""
    
    def __init__(self, vector_db: Optional[VectorDB] = None, embedder: Optional[TextEmbedder] = None):
        """Initialize searcher with vector DB and embedder."""
        self.vector_db = vector_db or VectorDB()
        self.embedder = embedder or TextEmbedder()
    
    def search_text(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for similar content using text query."""
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search vector database
        results = self.vector_db.search(query_embedding, limit=limit)
        
        logger.info(f"Found {len(results)} results for query: {query[:50]}...")
        return results
    
    def hybrid_search(self, query: str, filters: Optional[Dict] = None, limit: int = 10) -> List[Dict]:
        """Perform hybrid search with optional metadata filters."""
        # For now, just do vector search
        # TODO: Add metadata filtering and keyword matching
        return self.search_text(query, limit)
    
    def get_context_for_generation(self, query: str, max_chunks: int = 5) -> str:
        """Retrieve relevant context for text generation."""
        results = self.search_text(query, limit=max_chunks)
        
        # Combine retrieved chunks into context
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(f"[Context {i+1}] {result['text']}")
        
        context = "\n\n".join(context_parts)
        logger.info(f"Retrieved {len(results)} chunks for context generation")
        return context 