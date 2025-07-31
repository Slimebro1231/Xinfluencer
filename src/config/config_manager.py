"""
Configuration Manager for centralized access to all system configuration.
Provides getter and setter methods for easy navigation and modification.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Centralized configuration manager with getter/setter methods."""
    
    def __init__(self, config_dir: str = "src/config"):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all configuration files
        self.collection_config = self._load_config("collection_config.json")
        self.evaluation_config = self._load_config("../evaluation/config.json")
        
        logger.info("Configuration manager initialized")
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            logger.warning(f"Configuration file {filename} not found, creating default")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {filename}")
            return config
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return {}
    
    def _save_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Save configuration to JSON file."""
        config_path = self.config_dir / filename
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved configuration to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
            return False
    
    # Collection Configuration Getters
    
    def get_crypto_keywords(self, category: str = "primary") -> List[str]:
        """Get crypto keywords by category."""
        return self.collection_config.get("crypto_keywords", {}).get(category, [])
    
    def get_all_crypto_keywords(self) -> List[str]:
        """Get all crypto keywords from all categories."""
        keywords = []
        crypto_keywords = self.collection_config.get("crypto_keywords", {})
        for category in ["primary", "secondary", "rwa_specific"]:
            keywords.extend(crypto_keywords.get(category, []))
        return keywords
    
    def get_crypto_kols(self, category: str = "primary") -> List[str]:
        """Get crypto KOLs by category."""
        return self.collection_config.get("crypto_kols", {}).get(category, [])
    
    def get_all_crypto_kols(self) -> List[str]:
        """Get all crypto KOLs from all categories."""
        kols = []
        crypto_kols = self.collection_config.get("crypto_kols", {})
        for category in ["primary", "gold", "btc", "other"]:
            kols.extend(crypto_kols.get(category, []))
        return kols
    
    def get_collection_limits(self) -> Dict[str, Any]:
        """Get collection limits configuration."""
        return self.collection_config.get("collection_limits", {})
    
    def get_quality_filters(self) -> Dict[str, Any]:
        """Get quality filters configuration."""
        return self.collection_config.get("quality_filters", {})
    
    def get_search_queries(self, query_type: str = "trending_crypto") -> List[str]:
        """Get search queries by type."""
        return self.collection_config.get("search_queries", {}).get(query_type, [])
    
    def get_evaluation_weights(self) -> Dict[str, Any]:
        """Get evaluation weights configuration."""
        return self.collection_config.get("evaluation_weights", {})
    
    def get_training_selection(self) -> Dict[str, Any]:
        """Get training selection configuration."""
        return self.collection_config.get("training_selection", {})
    
    def get_advanced_components(self) -> Dict[str, Any]:
        """Get advanced components configuration."""
        return self.collection_config.get("advanced_components", {})
    
    # Collection Configuration Setters
    
    def set_crypto_keywords(self, category: str, keywords: List[str]) -> bool:
        """Set crypto keywords for a category."""
        if "crypto_keywords" not in self.collection_config:
            self.collection_config["crypto_keywords"] = {}
        self.collection_config["crypto_keywords"][category] = keywords
        return self._save_config(self.collection_config, "collection_config.json")
    
    def set_crypto_kols(self, category: str, kols: List[str]) -> bool:
        """Set crypto KOLs for a category."""
        if "crypto_kols" not in self.collection_config:
            self.collection_config["crypto_kols"] = {}
        self.collection_config["crypto_kols"][category] = kols
        return self._save_config(self.collection_config, "collection_config.json")
    
    def set_collection_limits(self, limits: Dict[str, Any]) -> bool:
        """Set collection limits configuration."""
        self.collection_config["collection_limits"] = limits
        return self._save_config(self.collection_config, "collection_config.json")
    
    def set_quality_filters(self, filters: Dict[str, Any]) -> bool:
        """Set quality filters configuration."""
        self.collection_config["quality_filters"] = filters
        return self._save_config(self.collection_config, "collection_config.json")
    
    def set_search_queries(self, query_type: str, queries: List[str]) -> bool:
        """Set search queries for a type."""
        if "search_queries" not in self.collection_config:
            self.collection_config["search_queries"] = {}
        self.collection_config["search_queries"][query_type] = queries
        return self._save_config(self.collection_config, "collection_config.json")
    
    def set_evaluation_weights(self, weights: Dict[str, Any]) -> bool:
        """Set evaluation weights configuration."""
        self.collection_config["evaluation_weights"] = weights
        return self._save_config(self.collection_config, "collection_config.json")
    
    def set_training_selection(self, selection: Dict[str, Any]) -> bool:
        """Set training selection configuration."""
        self.collection_config["training_selection"] = selection
        return self._save_config(self.collection_config, "collection_config.json")
    
    def set_advanced_components(self, components: Dict[str, Any]) -> bool:
        """Set advanced components configuration."""
        self.collection_config["advanced_components"] = components
        return self._save_config(self.collection_config, "collection_config.json")
    
    # Evaluation Configuration Getters
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.evaluation_config
    
    def get_tweet_quality_config(self) -> Dict[str, Any]:
        """Get tweet quality configuration."""
        return self.evaluation_config.get("tweet_quality", {})
    
    def get_ab_testing_config(self) -> Dict[str, Any]:
        """Get A/B testing configuration."""
        return self.evaluation_config.get("evaluation", {}).get("ab_testing", {})
    
    def get_engagement_evaluation_config(self) -> Dict[str, Any]:
        """Get engagement evaluation configuration."""
        return self.evaluation_config.get("engagement_evaluation", {})
    
    # Utility Methods
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of all configuration."""
        return {
            "collection": {
                "crypto_keywords_count": len(self.get_all_crypto_keywords()),
                "crypto_kols_count": len(self.get_all_crypto_kols()),
                "collection_limits": self.get_collection_limits(),
                "quality_filters": self.get_quality_filters()
            },
            "evaluation": {
                "tweet_quality": self.get_tweet_quality_config(),
                "ab_testing": self.get_ab_testing_config(),
                "engagement_evaluation": self.get_engagement_evaluation_config()
            },
            "advanced_components": self.get_advanced_components()
        }
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate configuration and return any issues."""
        issues = {
            "warnings": [],
            "errors": []
        }
        
        # Check required sections
        required_sections = ["crypto_keywords", "crypto_kols", "collection_limits"]
        for section in required_sections:
            if section not in self.collection_config:
                issues["errors"].append(f"Missing required section: {section}")
        
        # Check for empty lists
        if not self.get_all_crypto_keywords():
            issues["warnings"].append("No crypto keywords configured")
        
        if not self.get_all_crypto_kols():
            issues["warnings"].append("No crypto KOLs configured")
        
        # Check limits
        limits = self.get_collection_limits()
        if limits.get("posts_per_hour", 0) > 1000:
            issues["warnings"].append("posts_per_hour exceeds recommended limit")
        
        return issues
    
    def reload_config(self) -> bool:
        """Reload all configuration from files."""
        try:
            self.collection_config = self._load_config("collection_config.json")
            self.evaluation_config = self._load_config("../evaluation/config.json")
            logger.info("Configuration reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False

# Global instance for easy access
config_manager = ConfigManager() 