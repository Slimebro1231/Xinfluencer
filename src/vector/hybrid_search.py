"""Lightweight hybrid search combining dense and sparse retrieval."""

from typing import List, Dict, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
import logging
import re

logger = logging.getLogger(__name__)

class HybridSearch:
    """Lightweight hybrid search combining BM25 sparse retrieval with dense embeddings."""
    
    def __init__(self, dense_search, documents: List[str], alpha: float = 0.5):
        """Initialize hybrid search with dense search and documents."""
        self.dense_search = dense_search
        self.documents = documents
        self.alpha = alpha  # Weight for dense vs sparse (0.5 = equal weight)
        
        # Initialize BM25
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        logger.info(f"Lightweight hybrid search initialized with {len(documents)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Remove special characters and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def search(self, query: str, top_k: int = 10, rerank: bool = True) -> List[Dict]:
        """Perform hybrid search with lightweight reranking."""
        try:
            # Dense search
            dense_results = self.dense_search.search(query, top_k=top_k * 2)
            dense_scores = {result['id']: result['score'] for result in dense_results}
            
            # Sparse search (BM25)
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Normalize scores
            dense_scores_norm = self._normalize_scores(list(dense_scores.values()))
            bm25_scores_norm = self._normalize_scores(bm25_scores)
            
            # Combine scores
            combined_scores = {}
            for i, doc_id in enumerate(dense_scores.keys()):
                if i < len(bm25_scores_norm):
                    combined_score = (self.alpha * dense_scores_norm[i] + 
                                    (1 - self.alpha) * bm25_scores_norm[i])
                    combined_scores[doc_id] = combined_score
            
            # Sort by combined scores
            sorted_results = sorted(combined_scores.items(), 
                                  key=lambda x: x[1], reverse=True)[:top_k]
            
            # Format results
            results = []
            for doc_id, score in sorted_results:
                doc_index = int(doc_id) if doc_id.isdigit() else 0
                if doc_index < len(self.documents):
                    results.append({
                        'id': doc_id,
                        'text': self.documents[doc_index],
                        'score': score,
                        'dense_score': dense_scores.get(doc_id, 0),
                        'sparse_score': bm25_scores[doc_index] if doc_index < len(bm25_scores) else 0
                    })
            
            # Lightweight reranking based on keyword overlap
            if rerank and len(results) > 1:
                results = self._lightweight_rerank(query, results)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to dense search only
            return self.dense_search.search(query, top_k=top_k)
    
    def _normalize_scores(self, scores) -> List[float]:
        """Normalize scores to [0, 1] range."""
        # Handle numpy arrays safely
        if scores is None:
            return []
        
        # Convert numpy array to list if needed
        if isinstance(scores, np.ndarray):
            if scores.size == 0:
                return []
            scores = scores.tolist()
        elif len(scores) == 0:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _lightweight_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Lightweight reranking based on keyword overlap and semantic similarity."""
        try:
            query_tokens = set(self._tokenize(query))
            
            for result in results:
                doc_tokens = set(self._tokenize(result['text']))
                
                # Calculate keyword overlap
                overlap = len(query_tokens.intersection(doc_tokens))
                overlap_score = overlap / max(len(query_tokens), 1)
                
                # Calculate semantic similarity boost based on crypto keywords
                crypto_keywords = {'bitcoin', 'btc', 'crypto', 'gold', 'etf', 'market', 'price', 'bull', 'bear'}
                crypto_overlap = len(crypto_keywords.intersection(doc_tokens))
                crypto_boost = min(crypto_overlap * 0.1, 0.3)  # Max 30% boost
                
                # Combine scores
                final_score = result['score'] * (1 + overlap_score + crypto_boost)
                result['final_score'] = final_score
            
            # Sort by final score
            results.sort(key=lambda x: x['final_score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.warning(f"Lightweight reranking failed: {e}")
            return results 