"""
Tweet Quality Evaluation Module
Centralized tweet quality scoring for evaluation and training.
"""

import json
import os
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
# Optional imports for semantic search and embedding
try:
    from src.vector.embed import TextEmbedder
    from src.vector.hybrid_search import HybridSearch
    from src.vector.search import VectorSearcher
    from src.vector.db import VectorDB
    SEMANTIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Vector components not available: {e}")
    SEMANTIC_AVAILABLE = False
    TextEmbedder = None
    HybridSearch = None
    VectorSearcher = None
    VectorDB = None

class TweetQualityEvaluator:
    """
    Centralized tweet quality evaluator for the evaluation system.
    Uses config from src/evaluation/config.json.
    Integrates with training and A/B testing pipelines.
    """
    
    def __init__(self, config_path: Optional[str] = None, embedder: Optional[TextEmbedder] = None):
        """
        Initialize the tweet quality evaluator.
        
        Args:
            config_path: Path to evaluation config file
            embedder: TextEmbedder instance for semantic similarity
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.embedder = embedder or (TextEmbedder() if TextEmbedder and SEMANTIC_AVAILABLE else None)
        self.quality_config = self.config['tweet_quality']
        self.scoring_config = self.config['scoring']
        
        # Initialize semantic search components
        self._init_semantic_search()
        
        # Initialize reference texts and embeddings only if embedder is available
        if self.embedder:
            self._init_reference_embeddings()
        else:
            print("Warning: TextEmbedder not available, semantic features disabled")
            self.gold_ref_embeds = []
            self.btc_ref_embeds = []
            self.defi_ref_embeds = []
    
    def _init_semantic_search(self):
        """Initialize semantic search components for enhanced relevance scoring."""
        self.semantic_config = self.scoring_config.get('semantic_search', {})
        self.semantic_enabled = self.semantic_config.get('enabled', False) and SEMANTIC_AVAILABLE
        
        if self.semantic_enabled:
            try:
                # Initialize vector database and search components
                self.vector_db = VectorDB()
                self.vector_searcher = VectorSearcher(self.vector_db, self.embedder)
                
                # Initialize hybrid search if enabled
                if self.semantic_config.get('hybrid_search_enabled', False):
                    # Get documents from vector database for hybrid search
                    all_vectors = self.vector_db.get_all_vectors()
                    documents = [item['payload']['text'] for item in all_vectors if 'text' in item['payload']]
                    if documents:  # Only create hybrid search if we have documents
                        self.hybrid_search = HybridSearch(self.vector_searcher, documents)
                    else:
                        self.hybrid_search = None
                        print("Warning: No documents in vector database for hybrid search")
                else:
                    self.hybrid_search = None
                    
                print(f"Semantic search initialized successfully. Documents: {len(documents) if 'documents' in locals() else 0}")
            except Exception as e:
                print(f"Warning: Semantic search initialization failed: {e}")
                self.semantic_enabled = False
                self.vector_searcher = None
                self.hybrid_search = None
        else:
            self.vector_searcher = None
            self.hybrid_search = None
    
    def _init_reference_embeddings(self):
        """Initialize reference embeddings for relevance scoring."""
        # Combine reference texts with narrative phrases
        self.gold_refs = (
            self.quality_config['reference_texts']['gold'] + 
            self.quality_config['narrative_phrases']['gold']
        )
        self.btc_refs = (
            self.quality_config['reference_texts']['btc'] + 
            self.quality_config['narrative_phrases']['btc']
        )
        self.defi_refs = self.quality_config['narrative_phrases']['defi_rwa']
        
        # Precompute embeddings
        self.gold_ref_embeds = self.embedder.embed_texts(self.gold_refs)
        self.btc_ref_embeds = self.embedder.embed_texts(self.btc_refs)
        self.defi_ref_embeds = self.embedder.embed_texts(self.defi_refs)
    
    def _tweet_to_dict(self, tweet) -> Dict[str, Any]:
        """Convert tweet to dictionary format, handling both dict and TweetData objects."""
        if isinstance(tweet, dict):
            return tweet
        elif hasattr(tweet, '__dict__'):
            # Convert TweetData object to dict
            tweet_dict = {
                'id': getattr(tweet, 'id', ''),
                'text': getattr(tweet, 'text', ''),
                'author_id': getattr(tweet, 'author_id', ''),
                'author_username': getattr(tweet, 'author_username', ''),
                'created_at': getattr(tweet, 'created_at', ''),
                'public_metrics': getattr(tweet, 'public_metrics', {}),
                'context_annotations': getattr(tweet, 'context_annotations', []),
                'referenced_tweets': getattr(tweet, 'referenced_tweets', []),
                'non_public_metrics': getattr(tweet, 'non_public_metrics', {})
            }
            return tweet_dict
        else:
            return {}
    
    def engagement_score(self, tweet) -> float:
        """
        Calculate engagement score using config weights and logarithmic scaling.
        
        Args:
            tweet: Tweet dictionary or TweetData object with public_metrics
            
        Returns:
            Weighted engagement score with logarithmic normalization
        """
        tweet_dict = self._tweet_to_dict(tweet)
        metrics = tweet_dict.get('public_metrics', {})
        raw_score = 0.0
        
        for metric, weight in self.quality_config['engagement_weights'].items():
            # Skip comment fields
            if metric.startswith('_'):
                continue
                
            value = metrics.get(metric, 0)
            # Ensure value is numeric
            try:
                value = float(value) if value is not None else 0.0
            except (ValueError, TypeError):
                value = 0.0
                
            # Ensure weight is numeric
            try:
                weight = float(weight) if weight is not None else 0.0
            except (ValueError, TypeError):
                weight = 0.0
                
            raw_score += value * weight
        
        # Apply logarithmic normalization for better differentiation
        if self.scoring_config['engagement_normalization'] == 'logarithmic':
            log_base = self.scoring_config.get('log_base', 10.0)
            scale_factor = self.scoring_config['engagement_scale_factor']
            
            if raw_score > 0:
                # Apply logarithmic scaling: log(1 + score/scale_factor) / log(1 + max_expected/scale_factor)
                normalized_score = math.log(1 + raw_score / scale_factor, log_base)
                # Normalize to 0-1 range (assuming max expected engagement is around 50000)
                max_expected = 50000
                max_normalized = math.log(1 + max_expected / scale_factor, log_base)
                normalized_score = min(normalized_score / max_normalized, 1.0)
            else:
                normalized_score = 0.0
        else:
            # Fallback to tanh normalization
            scale_factor = self.scoring_config['engagement_scale_factor']
            normalized_score = np.tanh(raw_score / scale_factor)
        
        return float(normalized_score)
    
    def _max_cosine_similarity(self, query_embed: List[float], ref_embeds: List[List[float]]) -> float:
        """Calculate maximum cosine similarity between query and reference embeddings."""
        if not ref_embeds:
            return 0.0
        
        max_similarity = 0.0
        query_norm = np.linalg.norm(query_embed)
        
        if query_norm == 0:
            return 0.0
        
        for ref_embed in ref_embeds:
            ref_norm = np.linalg.norm(ref_embed)
            if ref_norm == 0:
                continue
            
            # Cosine similarity
            similarity = np.dot(query_embed, ref_embed) / (query_norm * ref_norm)
            max_similarity = max(max_similarity, similarity)
        
        return float(max_similarity)
    
    def _semantic_relevance_score(self, text: str) -> float:
        """
        Calculate semantic relevance using vector search.
        
        Args:
            text: Tweet text
            
        Returns:
            Semantic relevance score between 0 and 1
        """
        if not self.semantic_enabled or not self.vector_searcher:
            return 0.0
        
        try:
            # Use hybrid search if available, otherwise fallback to vector search
            if self.hybrid_search:
                results = self.hybrid_search.search(text, top_k=self.semantic_config['top_k'], rerank=self.semantic_config.get('cross_encoder_enabled', False))
            else:
                results = self.vector_searcher.search_similar(text, top_k=self.semantic_config['top_k'], threshold=self.semantic_config['similarity_threshold'])
            
            if not results:
                return 0.0
            
            # Calculate weighted average of top results
            total_weight = 0.0
            weighted_sum = 0.0
            
            for i, result in enumerate(results[:self.semantic_config['top_k']]):
                # Weight by rank (higher rank = higher weight)
                weight = 1.0 / (i + 1)
                score = result.get('score', 0.0)
                
                weighted_sum += score * weight
                total_weight += weight
            
            if total_weight > 0:
                semantic_score = weighted_sum / total_weight
            else:
                semantic_score = 0.0
            
            return float(semantic_score)
            
        except Exception as e:
            print(f"Warning: Semantic relevance calculation failed: {e}")
            return 0.0
    
    def relevance_score(self, tweet) -> float:
        """
        Compute relevance score using semantic search against reference texts and narratives.
        This method uses only proper semantic search methods without any fallbacks.
        
        Args:
            tweet: Tweet dictionary or TweetData object with text field
            
        Returns:
            Relevance score between 0 and 1
        """
        tweet_dict = self._tweet_to_dict(tweet)
        text = tweet_dict.get('text', '')
        if not text.strip():
            return 0.0
        
        # Primary method: Semantic search against reference texts
        if self.embedder and self.gold_ref_embeds:
            return self._semantic_relevance_with_references(text)
        
        # Secondary method: Vector database search if available
        elif self.semantic_enabled and self.vector_searcher:
            return self._semantic_relevance_score(text)
        
        # If no semantic search is available, return 0 to prevent data pollution
        else:
            print("Warning: No semantic search available for relevance scoring. Returning 0 to prevent data pollution.")
            return 0.0
    
    def _semantic_relevance_with_references(self, text: str) -> float:
        """
        Calculate semantic relevance using embedding similarity to reference texts and narratives.
        This is the primary method for relevance scoring.
        
        Args:
            text: Tweet text
            
        Returns:
            Semantic relevance score between 0 and 1
        """
        tweet_embed = self.embedder.embed_text(text)
        
        # Compute max similarity to each narrative group
        gold_sim = self._max_cosine_similarity(tweet_embed, self.gold_ref_embeds)
        btc_sim = self._max_cosine_similarity(tweet_embed, self.btc_ref_embeds)
        defi_sim = self._max_cosine_similarity(tweet_embed, self.defi_ref_embeds)
        
        # Take the highest as the base relevance score
        base_relevance = max(gold_sim, btc_sim, defi_sim)
        
        # Add semantic search component if enabled for additional context
        if self.semantic_enabled:
            semantic_score = self._semantic_relevance_score(text)
            semantic_weight = self.semantic_config.get('weight', 0.3)
            
            # Combine reference-based relevance with semantic search
            combined_relevance = (1 - semantic_weight) * base_relevance + semantic_weight * semantic_score
            base_relevance = combined_relevance
        
        # Apply future optimism bonus
        text_lower = text.lower()
        optimism_phrases = self.quality_config['future_optimism_phrases']
        
        if any(phrase in text_lower for phrase in optimism_phrases):
            boost_factor = self.scoring_config['relevance_boost_factor']
            base_relevance *= boost_factor
        
        return float(min(base_relevance, self.scoring_config['max_score']))
    
    def quality_score(self, tweet) -> float:
        """
        Combine engagement and relevance into a final quality score with granular levels.
        
        Args:
            tweet: Tweet dictionary or TweetData object with text and public_metrics
            
        Returns:
            Final quality score between 0 and 1 with granular differentiation
        """
        engagement = self.engagement_score(tweet)
        relevance = self.relevance_score(tweet)
        
        # Weighted combination
        relevance_weight = self.quality_config['relevance_weight']
        score = (engagement + relevance_weight * relevance) / (1 + relevance_weight)
        
        # Apply granular scoring levels for better differentiation
        granularity = self.scoring_config.get('granularity_levels', {})
        if granularity:
            score = self._apply_granular_scoring(score, granularity)
        
        return float(min(score, self.scoring_config['max_score']))
    
    def _apply_granular_scoring(self, score: float, granularity: Dict) -> float:
        """
        Apply granular scoring levels for better differentiation.
        
        Args:
            score: Raw quality score
            granularity: Granularity configuration
            
        Returns:
            Adjusted score with better granularity
        """
        # Define quality levels and their corresponding score adjustments
        levels = [
            ('excellent', granularity.get('excellent', 0.9), granularity.get('excellent_threshold', 0.85)),
            ('very_good', granularity.get('very_good', 0.8), granularity.get('very_good_threshold', 0.75)),
            ('good', granularity.get('good', 0.7), granularity.get('good_threshold', 0.65)),
            ('above_average', granularity.get('above_average', 0.6), granularity.get('above_average_threshold', 0.55)),
            ('average', granularity.get('average', 0.5), granularity.get('average_threshold', 0.45)),
            ('below_average', granularity.get('below_average', 0.4), granularity.get('below_average_threshold', 0.35)),
            ('poor', granularity.get('poor', 0.3), granularity.get('poor_threshold', 0.25)),
            ('very_poor', granularity.get('very_poor', 0.2), 0.0)
        ]
        
        # Find the appropriate level and apply adjustment
        for level_name, target_score, threshold in levels:
            if score >= threshold:
                # Apply slight adjustment to push score toward target for this level
                adjustment_factor = 0.1  # 10% adjustment
                adjusted_score = score + (target_score - score) * adjustment_factor
                return max(0.0, min(1.0, adjusted_score))
        
        return score
    
    def get_quality_level(self, score: float) -> str:
        """
        Get quality level description for a given score.
        
        Args:
            score: Quality score
            
        Returns:
            Quality level description
        """
        granularity = self.scoring_config.get('granularity_levels', {})
        
        if score >= granularity.get('excellent_threshold', 0.85):
            return "excellent"
        elif score >= granularity.get('very_good_threshold', 0.75):
            return "very_good"
        elif score >= granularity.get('good_threshold', 0.65):
            return "good"
        elif score >= granularity.get('above_average_threshold', 0.55):
            return "above_average"
        elif score >= granularity.get('average_threshold', 0.45):
            return "average"
        elif score >= granularity.get('below_average_threshold', 0.35):
            return "below_average"
        elif score >= granularity.get('poor_threshold', 0.25):
            return "poor"
        else:
            return "very_poor"
    
    def evaluate_tweet_for_training(self, tweet) -> Dict[str, Any]:
        """
        Comprehensive evaluation for training data selection.
        
        Args:
            tweet: Tweet dictionary or TweetData object
            
        Returns:
            Evaluation results with engagement and relevance scores
        """
        tweet_dict = self._tweet_to_dict(tweet)
        engagement = self.engagement_score(tweet)
        relevance = self.relevance_score(tweet)
        quality = self.quality_score(tweet)
        quality_level = self.get_quality_level(quality)
        
        # Check training criteria
        training_config = self.config['evaluation']['training']
        meets_criteria = (
            quality >= training_config['min_quality_score'] and
            engagement >= training_config['min_engagement']
        )
        
        return {
            'engagement_score': engagement,
            'relevance_score': relevance,
            'quality_score': quality,
            'quality_level': quality_level,
            'meets_training_criteria': meets_criteria,
            'text': tweet_dict.get('text', ''),
            'author': tweet_dict.get('author_id', ''),
            'created_at': tweet_dict.get('created_at', ''),
            'public_metrics': tweet_dict.get('public_metrics', {}),
            'semantic_search_used': self.semantic_enabled
        }
    
    def batch_evaluate(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple tweets for batch processing.
        
        Args:
            tweets: List of tweet dictionaries
            
        Returns:
            List of evaluation results
        """
        results = []
        for tweet in tweets:
            result = self.evaluate_tweet_for_training(tweet)
            results.append(result)
        return results
    
    def get_evaluation_stats(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistics from evaluation results.
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Statistical summary
        """
        if not evaluations:
            return {'error': 'No evaluations provided'}
        
        quality_scores = [e['quality_score'] for e in evaluations]
        engagement_scores = [e['engagement_score'] for e in evaluations]
        relevance_scores = [e['relevance_score'] for e in evaluations]
        
        # Quality level distribution
        quality_levels = [e.get('quality_level', 'unknown') for e in evaluations]
        level_counts = {}
        for level in quality_levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            'total_tweets': len(evaluations),
            'avg_quality_score': np.mean(quality_scores),
            'avg_engagement_score': np.mean(engagement_scores),
            'avg_relevance_score': np.mean(relevance_scores),
            'quality_score_std': np.std(quality_scores),
            'engagement_score_std': np.std(engagement_scores),
            'relevance_score_std': np.std(relevance_scores),
            'min_quality_score': np.min(quality_scores),
            'max_quality_score': np.max(quality_scores),
            'training_candidates': sum(1 for e in evaluations if e['meets_training_criteria']),
            'quality_level_distribution': level_counts,
            'semantic_search_usage': sum(1 for e in evaluations if e.get('semantic_search_used', False))
        } 