"""Text embedding utilities."""

from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)

class TextEmbedder:
    """Text embedding using Sentence Transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Loaded embedding model: {model_name} on {self.device}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
            
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use mean pooling
            embeddings = self._mean_pooling(outputs, encoded['attention_mask'])
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().tolist()
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Add embeddings to text chunks."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks 