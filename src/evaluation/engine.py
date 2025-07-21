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
        # TODO: Implement actual model loading and generation
        # For now, return mock responses
        if "baseline" in model_path:
            return f"Baseline response to: {prompt}"
        else:
            return f"Experimental response to: {prompt}"
    
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
        # TODO: Implement fact checking against trusted sources
        # For now, return random score between 0.7-0.9
        return random.uniform(0.7, 0.9)
    
    def evaluate_relevance(self, response: str, context: Dict) -> float:
        """Evaluate relevance to crypto/finance domain."""
        crypto_keywords = ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'defi', 'token', 'mining', 'staking']
        response_lower = response.lower()
        
        keyword_count = sum(1 for keyword in crypto_keywords if keyword in response_lower)
        return min(keyword_count / len(crypto_keywords), 1.0)
    
    def evaluate_clarity(self, response: str, context: Dict) -> float:
        """Evaluate clarity and readability."""
        # Simple heuristics for clarity
        sentences = response.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Prefer sentences between 10-20 words
        if 10 <= avg_sentence_length <= 20:
            return 1.0
        elif 5 <= avg_sentence_length <= 25:
            return 0.8
        else:
            return 0.6
    
    def evaluate_originality(self, response: str, context: Dict) -> float:
        """Evaluate originality vs existing content."""
        # TODO: Compare against existing content database
        # For now, return random score
        return random.uniform(0.6, 0.9)
    
    def evaluate_timing_relevance(self, response: str, context: Dict) -> float:
        """Evaluate relevance to current market conditions."""
        # TODO: Compare against current market data
        # For now, return random score
        return random.uniform(0.7, 0.9)

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
                    self.ab_evaluator.results = [EvaluationResult(**r) for r in data]
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