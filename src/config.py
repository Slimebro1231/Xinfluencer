"""Configuration management for Xinfluencer AI."""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class TwitterConfig(BaseSettings):
    """Twitter API configuration."""
    api_key: str = Field(..., env="TWITTER_API_KEY")
    api_secret: str = Field(..., env="TWITTER_API_SECRET")
    bearer_token: str = Field(..., env="TWITTER_BEARER_TOKEN")
    access_token: str = Field(..., env="TWITTER_ACCESS_TOKEN")
    access_token_secret: str = Field(..., env="TWITTER_ACCESS_TOKEN_SECRET")
    
    class Config:
        env_file = ".env"

class VectorDBConfig(BaseSettings):
    """Vector database configuration."""
    host: str = Field(default="localhost", env="QDRANT_HOST")
    port: int = Field(default=6333, env="QDRANT_PORT")
    collection_name: str = Field(default="tweet_chunks")
    
    class Config:
        env_file = ".env"

class ModelConfig(BaseSettings):
    """Model configuration."""
    generation_model: str = Field(default="microsoft/DialoGPT-medium", env="GENERATION_MODEL")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    class Config:
        env_file = ".env"

class PipelineConfig(BaseSettings):
    """Pipeline configuration."""
    max_tweets_per_kol: int = Field(default=50, env="MAX_TWEETS_PER_KOL")
    chunk_size: int = Field(default=256, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    
    class Config:
        env_file = ".env"

class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/xinfluencer.log", env="LOG_FILE")
    
    class Config:
        env_file = ".env"

class Config(BaseSettings):
    """Main configuration class."""
    twitter: TwitterConfig = TwitterConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    model: ModelConfig = ModelConfig()
    pipeline: PipelineConfig = PipelineConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # KOL list - you can customize this
    crypto_kols: List[str] = [
        "elonmusk", "VitalikButerin", "novogratz", "CryptoCobain",
        "CryptoBullish", "TheCryptoDog", "CryptoKaleo", "KoroushAK",
        "cz_binance", "SBF_FTX", "michael_saylor", "peter_schiff",
        "realDonaldTrump", "JoeBiden", "SECGov", "CFTCgov"
    ]
    
    class Config:
        env_file = ".env"

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config 