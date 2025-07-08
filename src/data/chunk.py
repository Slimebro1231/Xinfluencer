"""Text chunking for embeddings."""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 256, overlap: int = 50) -> List[Dict]:
    """
    Split text into overlapping chunks for embedding.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of chunk dictionaries with 'text' key
    """
    if len(text) <= chunk_size:
        return [{"text": text}]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to break at word boundaries
        if end < len(text):
            # Find the last space before the chunk end
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"text": chunk_text})
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    logger.debug(f"Split text into {len(chunks)} chunks")
    return chunks 