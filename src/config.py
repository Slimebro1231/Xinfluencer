"""Configuration settings for Xinfluencer AI."""

import os
from pathlib import Path
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    twitter_bearer_token: str = ""
    openai_api_key: str = ""
    huggingface_token: str = ""
    
    # Model settings
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    lora_rank: int = 16
    lora_alpha: int = 32
    
    # Vector database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    vector_dim: int = 1024
    collection_name: str = "crypto_tweets"
    
    # Data processing
    chunk_size: int = 256
    chunk_overlap: int = 50
    max_tweets_per_user: int = 100
    
    # Quality filters
    toxicity_threshold: float = 0.8
    bot_score_threshold: float = 0.9
    min_perplexity: float = 10.0
    max_perplexity: float = 90.0
    
    # Generation settings
    max_new_tokens: int = 280
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 8
    epochs: int = 1
    ppo_steps: int = 128
    
    # Monitoring
    prometheus_port: int = 8000
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings() 