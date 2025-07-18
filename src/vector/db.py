"""Vector database interface for Qdrant."""

from typing import List, Dict, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class VectorDB:
    """Vector database interface using in-memory storage for demo."""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize mock vector database."""
        self.host = host
        self.port = port
        self.collection_name = "tweet_chunks"
        self.data = []  # In-memory storage for demo
        self.next_id = 0
        logger.info(f"Initialized mock vector database (host={host}, port={port})")
        
    def create_collection(self, vector_size: int = 768):
        """Create collection for tweet chunks."""
        logger.info(f"Mock collection '{self.collection_name}' created (vector_size={vector_size})")
        self.data = []  # Reset data
        self.next_id = 0
    
    def upsert_chunks(self, chunks: List[Dict]):
        """Store text chunks with embeddings."""
        for chunk in chunks:
            point = {
                "id": self.next_id,
                "vector": chunk["embedding"],
                "payload": {
                    "text": chunk["text"],
                    "tweet_id": chunk["tweet_id"],
                    "timestamp": chunk["timestamp"],
                    "metadata": chunk.get("metadata", {})
                }
            }
            self.data.append(point)
            self.next_id += 1
        
        logger.info(f"Upserted {len(chunks)} chunks to mock vector DB (total: {len(self.data)})")
    
    def search(self, query_vector: List[float], limit: int = 10) -> List[Dict]:
        """Search for similar chunks using cosine similarity."""
        if not self.data:
            return []
        
        # Convert query vector to numpy array
        query_vec = np.array(query_vector)
        
        # Calculate cosine similarity with all stored vectors
        similarities = []
        for point in self.data:
            stored_vec = np.array(point["vector"])
            
            # Cosine similarity
            dot_product = np.dot(query_vec, stored_vec)
            norm_query = np.linalg.norm(query_vec)
            norm_stored = np.linalg.norm(stored_vec)
            
            if norm_query > 0 and norm_stored > 0:
                similarity = dot_product / (norm_query * norm_stored)
            else:
                similarity = 0.0
            
            similarities.append({
                "point": point,
                "score": similarity
            })
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top results
        results = []
        for item in similarities[:limit]:
            results.append({
                "text": item["point"]["payload"]["text"],
                "score": item["score"],
                "tweet_id": item["point"]["payload"]["tweet_id"],
                "metadata": item["point"]["payload"].get("metadata", {})
            })
        
        logger.info(f"Found {len(results)} similar chunks (query vector size: {len(query_vector)})")
        return results
    
    def get_all_vectors(self) -> List[Dict]:
        """Get all vectors from the database for compatibility with search."""
        return [
            {
                "id": point["id"],
                "vector": point["vector"],
                "payload": point["payload"]
            }
            for point in self.data
        ]


# Real Qdrant implementation (commented out for demo)
"""
class VectorDB:
    def __init__(self, host: str = "localhost", port: int = 6333):
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance, PointStruct
        
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "tweet_chunks"
        
    def create_collection(self, vector_size: int = 768):
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Collection might already exist: {e}")
    
    def upsert_chunks(self, chunks: List[Dict]):
        from qdrant_client.models import PointStruct
        
        points = []
        for i, chunk in enumerate(chunks):
            point = PointStruct(
                id=i,
                vector=chunk["embedding"],
                payload={
                    "text": chunk["text"],
                    "tweet_id": chunk["tweet_id"],
                    "timestamp": chunk["timestamp"],
                    "metadata": chunk.get("metadata", {})
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Upserted {len(points)} chunks to vector DB")
    
    def search(self, query_vector: List[float], limit: int = 10) -> List[Dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "tweet_id": hit.payload["tweet_id"],
                "metadata": hit.payload.get("metadata", {})
            }
            for hit in results
        ]
""" 