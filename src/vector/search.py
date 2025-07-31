"""Lightweight vector search and retrieval utilities."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
from pathlib import Path

from .db import VectorDB
from .embed import TextEmbedder

logger = logging.getLogger(__name__)

class VectorSearcher:
    """Lightweight vector search with CPU-based similarity computation."""
    
    def __init__(self, vector_db: VectorDB, embedder: TextEmbedder, use_gpu: bool = False):
        """
        Initialize lightweight vector searcher.
        
        Args:
            vector_db: Vector database instance
            embedder: Text embedder instance
            use_gpu: Not used in lightweight version, kept for compatibility
        """
        self.vector_db = vector_db
        self.embedder = embedder
        self.use_gpu = False  # Lightweight version doesn't use GPU
        
        logger.info("Lightweight VectorSearcher initialized")
    
    def search_similar(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using lightweight CPU computation.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            threshold: Similarity threshold
            
        Returns:
            List of similar documents with scores
        """
        start_time = time.time()
        
        try:
            # Embed the query
            query_embedding = self.embedder.embed_text(query)
            
            # Convert to numpy array if it's a list
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            
            # Use lightweight CPU search
            results = self._search_cpu(query_embedding, top_k)
            
            # Filter by threshold
            filtered_results = [r for r in results if r['score'] >= threshold]
            
            search_time = time.time() - start_time
            logger.info(f"Lightweight vector search completed in {search_time:.3f}s, found {len(filtered_results)} results")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Lightweight vector search failed: {e}")
            return []
    
    def _search_cpu(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Lightweight CPU search using cosine similarity."""
        try:
            # Get all vectors from database
            all_data = self.vector_db.get_all_vectors()
            
            if not all_data:
                return []
            
            # Compute similarities
            similarities = []
            for item in all_data:
                try:
                    vector = np.array(item['vector'])
                    # Compute cosine similarity
                    dot_product = np.dot(vector, query_embedding)
                    norm_vector = np.linalg.norm(vector)
                    norm_query = np.linalg.norm(query_embedding)
                    
                    if norm_vector > 0 and norm_query > 0:
                        similarity = dot_product / (norm_vector * norm_query)
                    else:
                        similarity = 0.0
                    
                    similarities.append((item['id'], similarity))
                except Exception as e:
                    logger.warning(f"Failed to compute similarity for item {item.get('id', 'unknown')}: {e}")
                    continue
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k results with proper format
            results = []
            for i, (doc_id, score) in enumerate(similarities[:top_k]):
                # Find the original data for this ID
                original_data = next((item for item in all_data if item['id'] == doc_id), None)
                if original_data:
                    results.append({
                        'text': original_data['payload']['text'],
                        'score': float(score),
                        'tweet_id': original_data['payload']['tweet_id'],
                        'metadata': original_data['payload'].get('metadata', {})
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Lightweight CPU search failed: {e}")
            return []
    
    def batch_search(self, queries: List[str], top_k: int = 5, threshold: float = 0.5) -> List[List[Dict[str, Any]]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of top results per query
            threshold: Similarity threshold
            
        Returns:
            List of results for each query
        """
        start_time = time.time()
        
        try:
            # Embed all queries
            query_embeddings = self.embedder.embed_texts(queries)
            
            # Use lightweight CPU batch search
            results = self._batch_search_cpu(query_embeddings, top_k)
            
            # Filter by threshold
            filtered_results = []
            for query_results in results:
                filtered = [r for r in query_results if r['score'] >= threshold]
                filtered_results.append(filtered)
            
            batch_time = time.time() - start_time
            logger.info(f"Lightweight batch search completed in {batch_time:.3f}s for {len(queries)} queries")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Lightweight batch search failed: {e}")
            return [[] for _ in queries]
    
    def _batch_search_cpu(self, query_embeddings: np.ndarray, top_k: int) -> List[List[Dict[str, Any]]]:
        """Lightweight CPU batch search."""
        try:
            all_data = self.vector_db.get_all_vectors()
            
            if not all_data:
                return [[] for _ in range(len(query_embeddings))]
            
            results = []
            for query_embedding in query_embeddings:
                similarities = []
                for item in all_data:
                    try:
                        vector = np.array(item['vector'])
                        similarity = np.dot(vector, query_embedding) / (np.linalg.norm(vector) * np.linalg.norm(query_embedding))
                        similarities.append((item['id'], similarity))
                    except Exception as e:
                        logger.warning(f"Failed to compute batch similarity: {e}")
                        continue
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                query_results = []
                for i, (doc_id, score) in enumerate(similarities[:top_k]):
                    query_results.append({
                        'id': doc_id,
                        'score': float(score),
                        'rank': i + 1
                    })
                
                results.append(query_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Lightweight CPU batch search failed: {e}")
            return [[] for _ in range(len(query_embeddings))]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'gpu_enabled': False,
            'faiss_available': False,
            'faiss_gpu_available': False,
            'cupy_available': False,
            'vectors_loaded': len(self.vector_db.get_all_vectors()) > 0,
            'search_type': 'lightweight_cpu'
        }

    def get_context_for_generation(self, query: str, top_k: int = 5) -> str:
        """
        Get formatted context for text generation based on query.
        
        Args:
            query: The query to search for
            top_k: Number of top results to include
            
        Returns:
            Formatted context string for generation
        """
        results = self.search_similar(query, top_k=top_k)
        
        if not results:
            return "No relevant context found."
        
        # Format context for generation
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result.get('text', '')
            score = result.get('score', 0.0)
            context_parts.append(f"[{i}] (Score: {score:.3f}) {text}")
        
        return "\n".join(context_parts) 