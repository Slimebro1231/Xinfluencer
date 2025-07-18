"""Vector search and retrieval utilities."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
from pathlib import Path

# Try to import GPU-accelerated libraries
try:
    import faiss
    FAISS_AVAILABLE = True
    # Check if GPU version is available
    try:
        import faiss.contrib.torch_utils
        FAISS_GPU_AVAILABLE = True
    except ImportError:
        FAISS_GPU_AVAILABLE = False
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from .db import VectorDB
from .embed import TextEmbedder

logger = logging.getLogger(__name__)

class VectorSearcher:
    """Enhanced vector search with GPU acceleration support."""
    
    def __init__(self, vector_db: VectorDB, embedder: TextEmbedder, use_gpu: bool = True):
        """
        Initialize vector searcher with GPU acceleration support.
        
        Args:
            vector_db: Vector database instance
            embedder: Text embedder instance
            use_gpu: Whether to use GPU acceleration if available
        """
        self.vector_db = vector_db
        self.embedder = embedder
        self.use_gpu = use_gpu and self._check_gpu_availability()
        
        # Initialize FAISS index if available
        self.faiss_index = None
        self.vectors_loaded = False
        
        if FAISS_AVAILABLE and self.use_gpu:
            self._initialize_faiss_gpu()
        
        logger.info(f"VectorSearcher initialized with GPU: {self.use_gpu}")
        if self.use_gpu:
            logger.info(f"FAISS-GPU: {FAISS_GPU_AVAILABLE}, CuPy: {CUPY_AVAILABLE}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _initialize_faiss_gpu(self):
        """Initialize FAISS GPU index."""
        try:
            if FAISS_GPU_AVAILABLE:
                # Create GPU index
                dimension = self.embedder.model.get_sentence_embedding_dimension()
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                
                # Move to GPU if available
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
                
                logger.info("FAISS GPU index initialized successfully")
            else:
                logger.warning("FAISS GPU not available, falling back to CPU")
                dimension = self.embedder.model.get_sentence_embedding_dimension()
                self.faiss_index = faiss.IndexFlatIP(dimension)
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS GPU index: {e}")
            self.faiss_index = None
    
    def _load_vectors_to_faiss(self):
        """Load vectors from database to FAISS index."""
        if not self.faiss_index or self.vectors_loaded:
            return
        
        try:
            # Get all vectors from database
            vectors = []
            ids = []
            
            # This would need to be implemented based on your VectorDB structure
            # For now, we'll assume a method to get all vectors
            all_data = self.vector_db.get_all_vectors()
            
            if all_data:
                for item in all_data:
                    vectors.append(item['vector'])
                    ids.append(item['id'])
                
                vectors_array = np.array(vectors, dtype=np.float32)
                self.faiss_index.add(vectors_array)
                self.vector_ids = ids
                self.vectors_loaded = True
                
                logger.info(f"Loaded {len(vectors)} vectors to FAISS index")
            
        except Exception as e:
            logger.error(f"Failed to load vectors to FAISS: {e}")
    
    def search_similar(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using GPU acceleration when available.
        
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
            
            if self.use_gpu and self.faiss_index:
                # Use FAISS GPU search
                results = self._search_faiss_gpu(query_embedding, top_k)
            elif CUPY_AVAILABLE and self.use_gpu:
                # Use CuPy for GPU computation
                results = self._search_cupy_gpu(query_embedding, top_k)
            else:
                # Fallback to CPU search
                results = self._search_cpu(query_embedding, top_k)
            
            # Filter by threshold
            filtered_results = [r for r in results if r['score'] >= threshold]
            
            search_time = time.time() - start_time
            logger.info(f"Vector search completed in {search_time:.3f}s, found {len(filtered_results)} results")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _search_faiss_gpu(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search using FAISS GPU."""
        try:
            # Ensure vectors are loaded
            self._load_vectors_to_faiss()
            
            if not self.vectors_loaded:
                return []
            
            # Search
            query_array = query_embedding.reshape(1, -1).astype(np.float32)
            scores, indices = self.faiss_index.search(query_array, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # FAISS returns -1 for invalid indices
                    results.append({
                        'id': self.vector_ids[idx],
                        'score': float(score),
                        'rank': i + 1
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS GPU search failed: {e}")
            return []
    
    def _search_cupy_gpu(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search using CuPy GPU computation."""
        try:
            # Get all vectors from database
            all_data = self.vector_db.get_all_vectors()
            
            if not all_data:
                return []
            
            # Convert to CuPy arrays
            query_gpu = cp.array(query_embedding)
            vectors_gpu = cp.array([item['vector'] for item in all_data])
            
            # Compute cosine similarities
            # Normalize vectors for cosine similarity
            query_norm = cp.linalg.norm(query_gpu)
            vectors_norm = cp.linalg.norm(vectors_gpu, axis=1, keepdims=True)
            
            # Avoid division by zero
            query_norm = cp.where(query_norm == 0, 1, query_norm)
            vectors_norm = cp.where(vectors_norm == 0, 1, vectors_norm)
            
            query_normalized = query_gpu / query_norm
            vectors_normalized = vectors_gpu / vectors_norm
            
            # Compute similarities
            similarities = cp.dot(vectors_normalized, query_normalized)
            
            # Get top-k indices
            top_indices = cp.argsort(similarities)[::-1][:top_k]
            top_scores = similarities[top_indices]
            
            # Convert back to CPU
            top_indices_cpu = cp.asnumpy(top_indices)
            top_scores_cpu = cp.asnumpy(top_scores)
            
            results = []
            for i, (idx, score) in enumerate(zip(top_indices_cpu, top_scores_cpu)):
                results.append({
                    'id': all_data[idx]['id'],
                    'score': float(score),
                    'rank': i + 1
                })
            
            return results
            
        except Exception as e:
            logger.error(f"CuPy GPU search failed: {e}")
            return []
    
    def _search_cpu(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Fallback CPU search."""
        try:
            # Get all vectors from database
            all_data = self.vector_db.get_all_vectors()
            
            if not all_data:
                return []
            
            # Compute similarities
            similarities = []
            for item in all_data:
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
            logger.error(f"CPU search failed: {e}")
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
            
            if self.use_gpu and self.faiss_index:
                # Use FAISS GPU batch search
                results = self._batch_search_faiss_gpu(query_embeddings, top_k)
            elif CUPY_AVAILABLE and self.use_gpu:
                # Use CuPy for batch GPU computation
                results = self._batch_search_cupy_gpu(query_embeddings, top_k)
            else:
                # Fallback to CPU batch search
                results = self._batch_search_cpu(query_embeddings, top_k)
            
            # Filter by threshold
            filtered_results = []
            for query_results in results:
                filtered = [r for r in query_results if r['score'] >= threshold]
                filtered_results.append(filtered)
            
            batch_time = time.time() - start_time
            logger.info(f"Batch search completed in {batch_time:.3f}s for {len(queries)} queries")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            return [[] for _ in queries]
    
    def _batch_search_faiss_gpu(self, query_embeddings: np.ndarray, top_k: int) -> List[List[Dict[str, Any]]]:
        """Batch search using FAISS GPU."""
        try:
            self._load_vectors_to_faiss()
            
            if not self.vectors_loaded:
                return [[] for _ in range(len(query_embeddings))]
            
            # Search
            query_array = query_embeddings.astype(np.float32)
            scores, indices = self.faiss_index.search(query_array, top_k)
            
            results = []
            for query_idx, (query_scores, query_indices) in enumerate(zip(scores, indices)):
                query_results = []
                for i, (score, idx) in enumerate(zip(query_scores, query_indices)):
                    if idx != -1:
                        query_results.append({
                            'id': self.vector_ids[idx],
                            'score': float(score),
                            'rank': i + 1
                        })
                results.append(query_results)
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS GPU batch search failed: {e}")
            return [[] for _ in range(len(query_embeddings))]
    
    def _batch_search_cupy_gpu(self, query_embeddings: np.ndarray, top_k: int) -> List[List[Dict[str, Any]]]:
        """Batch search using CuPy GPU."""
        try:
            all_data = self.vector_db.get_all_vectors()
            
            if not all_data:
                return [[] for _ in range(len(query_embeddings))]
            
            # Convert to CuPy arrays
            queries_gpu = cp.array(query_embeddings)
            vectors_gpu = cp.array([item['vector'] for item in all_data])
            
            # Normalize
            queries_norm = cp.linalg.norm(queries_gpu, axis=1, keepdims=True)
            vectors_norm = cp.linalg.norm(vectors_gpu, axis=1, keepdims=True)
            
            queries_norm = cp.where(queries_norm == 0, 1, queries_norm)
            vectors_norm = cp.where(vectors_norm == 0, 1, vectors_norm)
            
            queries_normalized = queries_gpu / queries_norm
            vectors_normalized = vectors_gpu / vectors_norm
            
            # Compute similarities
            similarities = cp.dot(queries_normalized, vectors_normalized.T)
            
            # Get top-k for each query
            top_indices = cp.argsort(similarities, axis=1)[:, ::-1][:, :top_k]
            top_scores = cp.take_along_axis(similarities, top_indices, axis=1)
            
            # Convert back to CPU
            top_indices_cpu = cp.asnumpy(top_indices)
            top_scores_cpu = cp.asnumpy(top_scores)
            
            results = []
            for query_idx, (indices, scores) in enumerate(zip(top_indices_cpu, top_scores_cpu)):
                query_results = []
                for i, (idx, score) in enumerate(zip(indices, scores)):
                    query_results.append({
                        'id': all_data[idx]['id'],
                        'score': float(score),
                        'rank': i + 1
                    })
                results.append(query_results)
            
            return results
            
        except Exception as e:
            logger.error(f"CuPy GPU batch search failed: {e}")
            return [[] for _ in range(len(query_embeddings))]
    
    def _batch_search_cpu(self, query_embeddings: np.ndarray, top_k: int) -> List[List[Dict[str, Any]]]:
        """Fallback CPU batch search."""
        try:
            all_data = self.vector_db.get_all_vectors()
            
            if not all_data:
                return [[] for _ in range(len(query_embeddings))]
            
            results = []
            for query_embedding in query_embeddings:
                similarities = []
                for item in all_data:
                    vector = np.array(item['vector'])
                    similarity = np.dot(vector, query_embedding) / (np.linalg.norm(vector) * np.linalg.norm(query_embedding))
                    similarities.append((item['id'], similarity))
                
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
            logger.error(f"CPU batch search failed: {e}")
            return [[] for _ in range(len(query_embeddings))]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'gpu_enabled': self.use_gpu,
            'faiss_available': FAISS_AVAILABLE,
            'faiss_gpu_available': FAISS_GPU_AVAILABLE,
            'cupy_available': CUPY_AVAILABLE,
            'vectors_loaded': self.vectors_loaded
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