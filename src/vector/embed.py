"""Text embedding utilities."""

from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class TextEmbedder:
    """Text embedding using Sentence Transformers."""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """Initialize the embedding model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Loaded embedding model: {model_name} on {self.device}")
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        # Generate embeddings using SentenceTransformer
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Add embeddings to text chunks."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks 