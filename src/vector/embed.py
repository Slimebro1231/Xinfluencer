"""Text embedding utilities for semantic search."""

from typing import List, Dict
import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import os

logger = logging.getLogger(__name__)

class TextEmbedder:
    """Proper text embedding using sentence-transformers for semantic search."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load sentence transformer model
            self.model = SentenceTransformer(model_name, device=self.device)
            
            logger.info(f"Initialized embedding model: {model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        try:
            # Generate embeddings using sentence-transformers
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # Convert to list format
            embeddings_list = embeddings.tolist()
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Add embeddings to text chunks."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks 