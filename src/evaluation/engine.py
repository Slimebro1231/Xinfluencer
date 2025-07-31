"""Core evaluation engine for measuring incremental AI improvements."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import random
import re

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Structured evaluation result."""
    prompt: str
    baseline_response: str
    experimental_response: str
    human_preference: Optional[str] = None  # 'A', 'B', or None
    human_confidence: Optional[int] = None  # 1-5 scale
    human_reasoning: Optional[str] = None
    ai_evaluation: Optional[Dict] = None
    engagement_metrics: Optional[Dict] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'prompt': self.prompt,
            'baseline_response': self.baseline_response,
            'experimental_response': self.experimental_response,
            'human_preference': self.human_preference,
            'human_confidence': self.human_confidence,
            'human_reasoning': self.human_reasoning,
            'ai_evaluation': self.ai_evaluation,
            'engagement_metrics': self.engagement_metrics,
            'timestamp': self.timestamp.isoformat()
        }

class ABTestEvaluator:
    """A/B testing evaluator for model comparison."""
    
    def __init__(self, baseline_model_path: str, experimental_model_path: str):
        """Initialize A/B test evaluator."""
        self.baseline_model_path = baseline_model_path
        self.experimental_model_path = experimental_model_path
        self.test_prompts = self.load_test_suite()
        self.results = []
        
        logger.info(f"Initialized A/B test evaluator")
        logger.info(f"Baseline model: {baseline_model_path}")
        logger.info(f"Experimental model: {experimental_model_path}")
    
    def load_test_suite(self) -> List[str]:
        """Load test prompts for evaluation."""
        # TODO: Load from file or database
        test_prompts = [
            "What is the impact of Bitcoin halving on price?",
            "How does DeFi lending work?",
            "What are the risks of investing in crypto?",
            "Explain the difference between Bitcoin and Ethereum",
            "What is the future of blockchain technology?",
            "How do smart contracts work?",
            "What is the role of miners in cryptocurrency?",
            "Explain the concept of tokenomics",
            "What are the benefits of decentralized finance?",
            "How does staking work in crypto?"
        ]
        
        logger.info(f"Loaded {len(test_prompts)} test prompts")
        return test_prompts
    
    def run_comparison(self, prompt: str, context: Dict = None) -> EvaluationResult:
        """Generate responses from both models and compare."""
        try:
            # Generate responses from both models
            baseline_response = self.generate_response(self.baseline_model_path, prompt, context)
            experimental_response = self.generate_response(self.experimental_model_path, prompt, context)
            
            # Create evaluation result
            result = EvaluationResult(
                prompt=prompt,
                baseline_response=baseline_response,
                experimental_response=experimental_response
            )
            
            # Store result
            self.results.append(result)
            
            logger.info(f"Generated comparison for prompt: {prompt[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Failed to run comparison: {e}")
            raise
    
    def generate_response(self, model_path: str, prompt: str, context: Dict = None) -> str:
        """Generate response using specified model."""
        # Use actual model generation - no mock responses allowed
        try:
            if hasattr(self, 'generator') and self.generator:
                return self.generator.generate_response(prompt)
            else:
                # Initialize proper H200 generator if not available
                from ..model.generate_h200 import H200TextGenerator
                generator = H200TextGenerator()
                return generator.generate_response(prompt)
        except Exception as e:
            logger.error(f"Model generation failed for {model_path}: {e}")
            return f"Error: Model not available. Please ensure training is complete."
    
    def run_full_evaluation(self, num_prompts: int = 10) -> List[EvaluationResult]:
        """Run full evaluation on multiple prompts."""
        selected_prompts = random.sample(self.test_prompts, min(num_prompts, len(self.test_prompts)))
        
        results = []
        for prompt in selected_prompts:
            result = self.run_comparison(prompt)
            results.append(result)
        
        logger.info(f"Completed full evaluation with {len(results)} comparisons")
        return results

class MultiDimensionalEvaluator:
    """Multi-dimensional evaluation metrics."""
    
    def __init__(self):
        """Initialize multi-dimensional evaluator."""
        self.quality_metrics = {
            'factual_accuracy': self.evaluate_factual_accuracy,
            'relevance_score': self.evaluate_relevance,
            'clarity_score': self.evaluate_clarity,
            'originality_score': self.evaluate_originality,
            'timing_relevance': self.evaluate_timing_relevance
        }
        
        logger.info("Initialized multi-dimensional evaluator")
    
    def evaluate_response(self, response: str, context: Dict = None) -> Dict[str, float]:
        """Evaluate response across multiple dimensions."""
        context = context or {}
        
        evaluation = {}
        for metric_name, metric_func in self.quality_metrics.items():
            try:
                evaluation[metric_name] = metric_func(response, context)
            except Exception as e:
                logger.warning(f"Failed to evaluate {metric_name}: {e}")
                evaluation[metric_name] = 0.0
        
        # Calculate overall score
        evaluation['overall_score'] = np.mean(list(evaluation.values()))
        
        return evaluation
    
    def evaluate_factual_accuracy(self, response: str, context: Dict) -> float:
        """Evaluate factual accuracy against trusted sources."""
        # Enhanced fact checking with multiple signals
        trusted_sources = context.get('sources', [])
        response_lower = response.lower()
        
        # Check for specific factual claims
        factual_indicators = {
            'specific_numbers': len(re.findall(r'\d+', response)),
            'technical_terms': len(re.findall(r'\b(blockchain|hash|mining|staking|defi|smart contract|tokenomics)\b', response_lower)),
            'source_mentions': sum(1 for source in trusted_sources if source.lower() in response_lower),
            'date_references': len(re.findall(r'\b(202[0-9]|202[0-9])\b', response)),
            'percentage_claims': len(re.findall(r'\d+%', response))
        }
        
        # Calculate factual score based on indicators
        total_indicators = sum(factual_indicators.values())
        if total_indicators == 0:
            return 0.5  # Neutral if no factual indicators
        
        # Weight different indicators
        weighted_score = (
            0.3 * min(factual_indicators['specific_numbers'] / 3, 1.0) +
            0.3 * min(factual_indicators['technical_terms'] / 5, 1.0) +
            0.2 * min(factual_indicators['source_mentions'] / 2, 1.0) +
            0.1 * min(factual_indicators['date_references'] / 2, 1.0) +
            0.1 * min(factual_indicators['percentage_claims'] / 2, 1.0)
        )
        
        return min(weighted_score, 1.0)
    
    def evaluate_relevance(self, response: str, context: Dict) -> float:
        """Evaluate relevance to crypto/finance domain."""
        # Enhanced relevance scoring with weighted keywords
        crypto_keywords = {
            'high_weight': ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'defi', 'token'],
            'medium_weight': ['mining', 'staking', 'smart contract', 'wallet', 'exchange'],
            'low_weight': ['price', 'market', 'investment', 'trading', 'altcoin']
        }
        
        response_lower = response.lower()
        total_score = 0
        max_possible = 0
        
        for weight, keywords in crypto_keywords.items():
            weight_value = {'high_weight': 1.0, 'medium_weight': 0.7, 'low_weight': 0.4}[weight]
            keyword_count = sum(1 for keyword in keywords if keyword in response_lower)
            total_score += keyword_count * weight_value
            max_possible += len(keywords) * weight_value
        
        if max_possible == 0:
            return 0.0
        
        return min(total_score / max_possible, 1.0)
    
    def evaluate_clarity(self, response: str, context: Dict) -> float:
        """Evaluate clarity and readability."""
        # Enhanced clarity scoring with multiple factors
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if not sentences:
            return 0.0
        
        # Sentence length analysis
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = np.mean(sentence_lengths)
        
        # Length score (prefer 10-20 words)
        if 10 <= avg_sentence_length <= 20:
            length_score = 1.0
        elif 5 <= avg_sentence_length <= 25:
            length_score = 0.8
        else:
            length_score = 0.6
        
        # Readability score (Flesch Reading Ease approximation)
        word_count = sum(sentence_lengths)
        syllable_count = sum(len(re.findall(r'[aeiouy]+', word.lower())) for word in response.split())
        
        if word_count > 0:
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * syllable_count / word_count)
            readability_score = max(0, min(1, flesch_score / 100))
        else:
            readability_score = 0.5
        
        # Structure score (paragraphs, bullet points, etc.)
        structure_indicators = len(re.findall(r'\n|•|\*|\-', response))
        structure_score = min(structure_indicators / 3, 1.0)
        
        # Combined clarity score
        clarity_score = (0.4 * length_score + 0.4 * readability_score + 0.2 * structure_score)
        
        return clarity_score
    
    def evaluate_originality(self, response: str, context: Dict) -> float:
        """Evaluate originality vs existing content."""
        # Enhanced originality scoring with multiple signals
        similar_content = context.get('similar_content', [])
        
        # Calculate similarity with existing content
        if similar_content:
            similarities = []
            for existing in similar_content:
                # Simple Jaccard similarity
                response_words = set(response.lower().split())
                existing_words = set(existing.lower().split())
                
                if response_words and existing_words:
                    intersection = response_words.intersection(existing_words)
                    union = response_words.union(existing_words)
                    similarity = len(intersection) / len(union)
                    similarities.append(similarity)
            
            max_similarity = max(similarities) if similarities else 0
            originality_score = 1.0 - max_similarity
        else:
            # If no similar content provided, use heuristics
            # Check for unique phrases and technical depth
            unique_phrases = len(set(response.split()))
            total_words = len(response.split())
            
            if total_words > 0:
                uniqueness_ratio = unique_phrases / total_words
                originality_score = min(uniqueness_ratio * 1.2, 1.0)  # Boost slightly
            else:
                originality_score = 0.5
        
        return originality_score
    
    def evaluate_timing_relevance(self, response: str, context: Dict) -> float:
        """Evaluate relevance to current market conditions."""
        # Enhanced timing relevance with market context
        market_conditions = context.get('market_conditions', {})
        current_trends = context.get('current_trends', [])
        
        response_lower = response.lower()
        timing_score = 0.5  # Base score
        
        # Check for current market references
        if market_conditions:
            # Check for current price levels, market cap, etc.
            price_indicators = ['price', 'market cap', 'volume', 'market']
            price_matches = sum(1 for indicator in price_indicators if indicator in response_lower)
            if price_matches > 0:
                timing_score += 0.2
        
        # Check for current trends
        if current_trends:
            trend_matches = sum(1 for trend in current_trends if trend.lower() in response_lower)
            if trend_matches > 0:
                timing_score += 0.2
        
        # Check for temporal indicators
        temporal_indicators = ['recent', 'latest', 'current', 'now', 'today', 'this week']
        temporal_matches = sum(1 for indicator in temporal_indicators if indicator in response_lower)
        if temporal_matches > 0:
            timing_score += 0.1
        
        return min(timing_score, 1.0)

class TrainingSignalEnhancer:
    """Enhanced training signal generation."""
    
    def __init__(self):
        """Initialize training signal enhancer."""
        self.immediate_signals = []
        self.delayed_signals = []
        self.immediate_weight = 0.7
        self.delayed_weight = 0.3
        
        logger.info("Initialized training signal enhancer")
    
    def add_immediate_feedback(self, response: str, human_score: float, context: Dict = None):
        """Add immediate human feedback."""
        signal = {
            'response': response,
            'score': human_score,
            'timestamp': datetime.now(),
            'type': 'immediate_human_feedback',
            'context': context or {}
        }
        
        self.immediate_signals.append(signal)
        logger.info(f"Added immediate feedback: {human_score}")
    
    def add_delayed_feedback(self, response: str, engagement_metrics: Dict, context: Dict = None):
        """Add delayed engagement feedback."""
        signal = {
            'response': response,
            'engagement': engagement_metrics,
            'timestamp': datetime.now(),
            'type': 'delayed_engagement_feedback',
            'context': context or {}
        }
        
        self.delayed_signals.append(signal)
        logger.info(f"Added delayed feedback: {engagement_metrics}")
    
    def calculate_engagement_score(self, engagement_metrics: Dict) -> float:
        """Calculate normalized engagement score."""
        likes = engagement_metrics.get('likes', 0)
        retweets = engagement_metrics.get('retweets', 0)
        replies = engagement_metrics.get('replies', 0)
        followers = engagement_metrics.get('followers', 1000)  # Default follower count
        
        # Calculate engagement rate
        engagement_rate = (likes + retweets + replies) / max(followers, 1)
        
        # Normalize to 0-1 scale (typical engagement rates are 0.01-0.05)
        normalized_score = min(engagement_rate / 0.05, 1.0)
        
        return normalized_score
    
    def generate_training_signal(self, window_size: int = 10) -> float:
        """Generate combined training signal."""
        if not self.immediate_signals and not self.delayed_signals:
            return 0.5  # Neutral signal if no feedback
        
        # Calculate immediate signal
        immediate_score = 0.5
        if self.immediate_signals:
            recent_immediate = self.immediate_signals[-window_size:]
            immediate_score = np.mean([s['score'] for s in recent_immediate])
        
        # Calculate delayed signal
        delayed_score = 0.5
        if self.delayed_signals:
            recent_delayed = self.delayed_signals[-window_size:]
            engagement_scores = [self.calculate_engagement_score(s['engagement']) for s in recent_delayed]
            delayed_score = np.mean(engagement_scores)
        
        # Combine signals
        combined_signal = (
            self.immediate_weight * immediate_score +
            self.delayed_weight * delayed_score
        )
        
        logger.info(f"Generated training signal: {combined_signal:.3f} "
                   f"(immediate: {immediate_score:.3f}, delayed: {delayed_score:.3f})")
        
        return combined_signal
    
    def get_signal_history(self, days: int = 7) -> Dict:
        """Get signal history for analysis."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        recent_immediate = [s for s in self.immediate_signals if s['timestamp'] > cutoff_time]
        recent_delayed = [s for s in self.delayed_signals if s['timestamp'] > cutoff_time]
        
        return {
            'immediate_signals': recent_immediate,
            'delayed_signals': recent_delayed,
            'total_immediate': len(recent_immediate),
            'total_delayed': len(recent_delayed)
        }

class StatisticalAnalyzer:
    """Statistical analysis for evaluation results."""
    
    def __init__(self):
        """Initialize statistical analyzer."""
        logger.info("Initialized statistical analyzer")
    
    def analyze_preference_distribution(self, results: List[EvaluationResult]) -> Dict:
        """Analyze human preference distribution."""
        preferences = [r.human_preference for r in results if r.human_preference]
        
        if not preferences:
            return {'error': 'No human preferences available'}
        
        preference_counts = pd.Series(preferences).value_counts()
        total = len(preferences)
        
        analysis = {
            'total_comparisons': total,
            'preference_counts': preference_counts.to_dict(),
            'preference_percentages': (preference_counts / total * 100).to_dict(),
            'statistical_significance': self.calculate_significance(preferences)
        }
        
        return analysis
    
    def calculate_significance(self, preferences: List[str], alpha: float = 0.05) -> Dict:
        """Calculate statistical significance of preference distribution."""
        if len(preferences) < 10:
            return {'significant': False, 'reason': 'Insufficient sample size'}
        
        # Simple binomial test for preference distribution
        a_count = preferences.count('A')
        b_count = preferences.count('B')
        total = len(preferences)
        
        # Expected distribution under null hypothesis (50/50)
        expected_a = total * 0.5
        expected_b = total * 0.5
        
        # Calculate chi-square statistic
        chi_square = ((a_count - expected_a) ** 2 / expected_a + 
                     (b_count - expected_b) ** 2 / expected_b)
        
        # For 1 degree of freedom, chi-square critical value at alpha=0.05 is 3.841
        significant = chi_square > 3.841
        
        return {
            'significant': significant,
            'chi_square': chi_square,
            'critical_value': 3.841,
            'p_value': self.chi_square_p_value(chi_square, 1)
        }
    
    def chi_square_p_value(self, chi_square: float, df: int) -> float:
        """Calculate p-value for chi-square statistic."""
        # Simplified approximation
        if chi_square < 1:
            return 0.5
        elif chi_square < 3.841:
            return 0.1
        else:
            return 0.01
    
    def analyze_improvement_trend(self, results: List[EvaluationResult]) -> Dict:
        """Analyze improvement trend over time."""
        if len(results) < 5:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Sort by timestamp
        sorted_results = sorted(results, key=lambda x: x.timestamp)
        
        # Calculate moving average of AI evaluation scores
        ai_scores = []
        timestamps = []
        
        for result in sorted_results:
            if result.ai_evaluation and 'overall_score' in result.ai_evaluation:
                ai_scores.append(result.ai_evaluation['overall_score'])
                timestamps.append(result.timestamp)
        
        if len(ai_scores) < 3:
            return {'error': 'Insufficient AI evaluation data'}
        
        # Calculate trend
        x = np.arange(len(ai_scores))
        slope, intercept = np.polyfit(x, ai_scores, 1)
        
        trend_analysis = {
            'slope': slope,
            'intercept': intercept,
            'improving': slope > 0,
            'improvement_rate': slope,
            'data_points': len(ai_scores),
            'time_span_days': (timestamps[-1] - timestamps[0]).days
        }
        
        return trend_analysis

class EvaluationEngine:
    """Main evaluation engine orchestrating all components."""
    
    def __init__(self, baseline_model_path: str, experimental_model_path: str):
        """Initialize evaluation engine."""
        self.ab_evaluator = ABTestEvaluator(baseline_model_path, experimental_model_path)
        self.multi_evaluator = MultiDimensionalEvaluator()
        self.signal_enhancer = TrainingSignalEnhancer()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        self.results_file = Path("evaluation_results.json")
        self.load_results()
        
        logger.info("Initialized evaluation engine")
    
    def load_results(self):
        """Load previous evaluation results."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    results = []
                    for r in data:
                        # Convert timestamp string back to datetime
                        if 'timestamp' in r and isinstance(r['timestamp'], str):
                            r['timestamp'] = datetime.fromisoformat(r['timestamp'])
                        results.append(EvaluationResult(**r))
                    self.ab_evaluator.results = results
                logger.info(f"Loaded {len(self.ab_evaluator.results)} previous results")
            except Exception as e:
                logger.error(f"Failed to load results: {e}")
    
    def save_results(self):
        """Save evaluation results to file."""
        try:
            data = [r.to_dict() for r in self.ab_evaluator.results]
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(data)} results to file")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def run_evaluation_cycle(self, num_prompts: int = 10) -> Dict:
        """Run complete evaluation cycle."""
        logger.info(f"Starting evaluation cycle with {num_prompts} prompts")
        
        # Run A/B comparisons
        results = self.ab_evaluator.run_full_evaluation(num_prompts)
        
        # Add AI evaluations
        for result in results:
            baseline_eval = self.multi_evaluator.evaluate_response(result.baseline_response)
            experimental_eval = self.multi_evaluator.evaluate_response(result.experimental_response)
            
            result.ai_evaluation = {
                'baseline': baseline_eval,
                'experimental': experimental_eval,
                'improvement': experimental_eval['overall_score'] - baseline_eval['overall_score']
            }
        
        # Save results
        self.save_results()
        
        # Generate analysis
        analysis = self.generate_analysis()
        
        logger.info("Completed evaluation cycle")
        return analysis
    
    def generate_analysis(self) -> Dict:
        """Generate comprehensive analysis of results."""
        results = self.ab_evaluator.results
        
        if not results:
            return {'error': 'No results available for analysis'}
        
        analysis = {
            'summary': {
                'total_comparisons': len(results),
                'with_human_feedback': len([r for r in results if r.human_preference]),
                'with_ai_evaluation': len([r for r in results if r.ai_evaluation])
            },
            'preference_analysis': self.statistical_analyzer.analyze_preference_distribution(results),
            'improvement_trend': self.statistical_analyzer.analyze_improvement_trend(results),
            'signal_analysis': self.signal_enhancer.get_signal_history()
        }
        
        return analysis
    
    def add_human_feedback(self, result_index: int, preference: str, confidence: int, reasoning: str):
        """Add human feedback to specific result."""
        if 0 <= result_index < len(self.ab_evaluator.results):
            result = self.ab_evaluator.results[result_index]
            result.human_preference = preference
            result.human_confidence = confidence
            result.human_reasoning = reasoning
            
            # Add to training signal
            preferred_response = (result.baseline_response if preference == 'A' 
                                else result.experimental_response)
            self.signal_enhancer.add_immediate_feedback(preferred_response, confidence / 5.0)
            
            self.save_results()
            logger.info(f"Added human feedback for result {result_index}")
        else:
            logger.error(f"Invalid result index: {result_index}")
    
    def get_training_signal(self) -> float:
        """Get current training signal."""
        return self.signal_enhancer.generate_training_signal() 