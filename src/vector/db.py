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
    
    def rebuild_from_json_data(self, json_files: List[str] = None, embedder=None) -> int:
        """
        Rebuild vector database from JSON data files.
        
        Args:
            json_files: List of JSON file paths to process. If None, uses default files.
            embedder: TextEmbedder instance for creating embeddings
            
        Returns:
            Number of vectors added to the database
        """
        import json
        import os
        from pathlib import Path
        
        if json_files is None:
            json_files = [
                "data/collected/kol_collection_20250728_095921.json",
                "data/collected/high_engagement_20250728_091647.json", 
                "data/collected/unified_collection_20250729_040626.json",
                "data/collected/kol_collection_20250728_101224.json",
                "data/collected/unified_collection_20250729_024917.json",
                "data/collected/unified_collection_20250729_023833.json",
                "data/collected/comprehensive_collection_20250728_091647.json",
                "data/collected/unified_collection_20250729_053635.json",
                "data/collected/unified_collection_20250729_081609.json",
                "data/collected/comprehensive_collection_20250728_095957.json"
            ]
        
        logger.info(f"Rebuilding vector database from {len(json_files)} JSON files")
        
        # Clear existing data
        self.data = []
        self.next_id = 0
        logger.info("Cleared existing vector database")
        
        total_vectors = 0
        
        for json_file in json_files:
            if not os.path.exists(json_file):
                logger.warning(f"File not found: {json_file}")
                continue
                
            logger.info(f"Processing {json_file}...")
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract tweets from author-keyed structure
                all_tweets = []
                if isinstance(data, dict):
                    for author, author_tweets in data.items():
                        if isinstance(author_tweets, list):
                            all_tweets.extend(author_tweets)
                elif isinstance(data, list):
                    all_tweets = data
                
                logger.info(f"Found {len(all_tweets)} tweets in {json_file}")
                
                # Add tweets to vector database
                for i, tweet in enumerate(all_tweets):
                    try:
                        # Extract tweet text
                        if isinstance(tweet, dict):
                            text = tweet.get("text", "")
                            tweet_id = str(tweet.get("id", f"unknown_{i}"))
                        else:
                            text = tweet.text
                            tweet_id = str(tweet.id)
                        
                        # Skip if no text or too short
                        if not text or len(text.strip()) < 20:
                            continue
                        
                        # Create embedding if embedder is available
                        if embedder:
                            embedding = embedder.embed_text(text)
                        else:
                            # Fallback: create a simple hash-based vector for demo
                            import hashlib
                            hash_obj = hashlib.md5(text.encode())
                            embedding = [float(int(hash_obj.hexdigest()[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)] * 24  # 768-dim vector
                        
                        # Add to vector database
                        point = {
                            "id": self.next_id,
                            "vector": embedding,
                            "payload": {
                                "text": text,
                                "tweet_id": tweet_id,
                                "source": json_file,
                                "metadata": {
                                    "processed_at": str(Path(__file__).parent.parent.parent / "data" / "cache" / "vector_rebuild"),
                                    "embedding_type": "semantic" if embedder else "hash_fallback"
                                }
                            }
                        }
                        
                        self.data.append(point)
                        self.next_id += 1
                        total_vectors += 1
                        
                        if total_vectors % 100 == 0:
                            logger.info(f"Added {total_vectors} vectors...")
                            
                    except Exception as e:
                        logger.error(f"Error processing tweet {i}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                continue
        
        logger.info(f"Vector database rebuilt with {total_vectors} vectors")
        return total_vectors
    
    def search_similar(self, query_text: str, top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        """
        Search for similar texts using semantic similarity.
        
        Args:
            query_text: Text to search for
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar texts with scores
        """
        if not self.data:
            return []
        
        # Create query embedding (fallback to hash-based for demo)
        import hashlib
        hash_obj = hashlib.md5(query_text.encode())
        query_embedding = [float(int(hash_obj.hexdigest()[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)] * 24
        
        # Search using existing search method
        results = self.search(query_embedding, limit=top_k)
        
        # Filter by threshold
        filtered_results = [r for r in results if r["score"] >= threshold]
        
        return filtered_results


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