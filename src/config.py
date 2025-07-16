"""Configuration management for Xinfluencer AI."""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class H200Config(BaseSettings):
    """H200 server configuration."""
    pem_file: str = Field(..., alias="H200_PEM_FILE")
    user: str = Field("ubuntu", alias="H200_USER")
    host: str = Field("157.10.162.127", alias="H200_HOST")
    remote_dir: str = Field("/home/ubuntu/xinfluencer", alias="H200_REMOTE_DIR")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

class TwitterConfig(BaseSettings):
    """Twitter API configuration."""
    username: Optional[str] = Field(default=None, alias="TWITTER_USERNAME")
    email: Optional[str] = Field(default=None, alias="TWITTER_EMAIL")
    password: Optional[str] = Field(default=None, alias="TWITTER_PASSWORD")
    bearer_token: Optional[str] = Field(default=None, alias="TWITTER_BEARER_TOKEN")
    
    class Config:
        env_file = ".env"
        populate_by_name = True
        extra = "ignore"

class VectorDBConfig(BaseSettings):
    """Vector database configuration."""
    host: str = Field("localhost", alias="VECTOR_DB_HOST")
    port: int = Field(6333, alias="VECTOR_DB_PORT")
    collection_name: str = Field(default="tweet_chunks", alias="QDRANT_COLLECTION_NAME")
    
    class Config:
        env_file = ".env"
        populate_by_name = True
        extra = "ignore"

class ModelConfig(BaseSettings):
    """Model configuration."""
    generation_model: str = Field(default="mistralai/Mistral-7B-v0.1", alias="GENERATION_MODEL")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    
    class Config:
        env_file = ".env"
        populate_by_name = True
        extra = "ignore"

class PipelineConfig(BaseSettings):
    """Pipeline configuration."""
    max_tweets_per_kol: int = Field(default=50, alias="MAX_TWEETS_PER_KOL")
    chunk_size: int = Field(default=256, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    
    class Config:
        env_file = ".env"
        populate_by_name = True
        extra = "ignore"

class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="logs/xinfluencer.log", alias="LOG_FILE")
    
    class Config:
        env_file = ".env"
        populate_by_name = True
        extra = "ignore"

class Config(BaseSettings):
    """Main configuration class."""
    twitter: TwitterConfig = TwitterConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    h200: H200Config = H200Config()
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
        env_nested_delimiter = '__'
        extra = "ignore"  # Allow extra fields

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config 