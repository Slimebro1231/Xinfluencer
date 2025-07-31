"""
A/B Testing Module for Evaluation System
Provides statistical A/B testing capabilities for model evaluation and training.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import json
from pathlib import Path

from .tweet_quality import TweetQualityEvaluator

class ABTestingEngine:
    """
    A/B testing engine for evaluating model performance and training effectiveness.
    Integrates with TweetQualityEvaluator for comprehensive evaluation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize A/B testing engine.
        
        Args:
            config_path: Path to evaluation config file
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.ab_config = self.config['evaluation']['ab_testing']
        self.quality_evaluator = TweetQualityEvaluator(config_path)
    
    def run_quality_ab_test(self, 
                           group_a_tweets: List[Dict[str, Any]], 
                           group_b_tweets: List[Dict[str, Any]],
                           metric: str = 'quality_score') -> Dict[str, Any]:
        """
        Run A/B test comparing quality scores between two groups of tweets.
        
        Args:
            group_a_tweets: Control group tweets
            group_b_tweets: Treatment group tweets
            metric: Metric to compare ('quality_score', 'engagement_score', 'relevance_score')
            
        Returns:
            A/B test results with statistical significance
        """
        # Evaluate both groups
        group_a_evaluations = self.quality_evaluator.batch_evaluate(group_a_tweets)
        group_b_evaluations = self.quality_evaluator.batch_evaluate(group_b_tweets)
        
        # Extract metrics
        group_a_scores = [e[metric] for e in group_a_evaluations]
        group_b_scores = [e[metric] for e in group_b_evaluations]
        
        return self._statistical_test(group_a_scores, group_b_scores, metric)
    
    def run_engagement_ab_test(self, 
                              group_a_tweets: List[Dict[str, Any]], 
                              group_b_tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run A/B test comparing engagement metrics between two groups.
        
        Args:
            group_a_tweets: Control group tweets
            group_b_tweets: Treatment group tweets
            
        Returns:
            A/B test results for engagement comparison
        """
        # Calculate engagement scores
        group_a_engagement = [self.quality_evaluator.engagement_score(t) for t in group_a_tweets]
        group_b_engagement = [self.quality_evaluator.engagement_score(t) for t in group_b_tweets]
        
        return self._statistical_test(group_a_engagement, group_b_engagement, 'engagement_score')
    
    def run_training_effectiveness_test(self, 
                                       before_tweets: List[Dict[str, Any]], 
                                       after_tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test the effectiveness of training by comparing before/after quality.
        
        Args:
            before_tweets: Tweets before training
            after_tweets: Tweets after training
            
        Returns:
            Training effectiveness test results
        """
        before_evaluations = self.quality_evaluator.batch_evaluate(before_tweets)
        after_evaluations = self.quality_evaluator.batch_evaluate(after_tweets)
        
        before_quality = [e['quality_score'] for e in before_evaluations]
        after_quality = [e['quality_score'] for e in after_evaluations]
        
        results = self._statistical_test(before_quality, after_quality, 'quality_score')
        results['test_type'] = 'training_effectiveness'
        
        # Additional training-specific metrics
        before_candidates = sum(1 for e in before_evaluations if e['meets_training_criteria'])
        after_candidates = sum(1 for e in after_evaluations if e['meets_training_criteria'])
        
        results['training_candidates_improvement'] = {
            'before': before_candidates,
            'after': after_candidates,
            'improvement_pct': ((after_candidates - before_candidates) / max(before_candidates, 1)) * 100
        }
        
        return results
    
    def _statistical_test(self, 
                         group_a_scores: List[float], 
                         group_b_scores: List[float], 
                         metric_name: str) -> Dict[str, Any]:
        """
        Perform statistical test between two groups.
        
        Args:
            group_a_scores: Scores for group A
            group_b_scores: Scores for group B
            metric_name: Name of the metric being tested
            
        Returns:
            Statistical test results
        """
        if len(group_a_scores) < 2 or len(group_b_scores) < 2:
            return {
                'error': 'Insufficient data for statistical testing',
                'group_a_size': len(group_a_scores),
                'group_b_size': len(group_b_scores)
            }
        
        # Convert to numpy arrays
        a_scores = np.array(group_a_scores)
        b_scores = np.array(group_b_scores)
        
        # Basic statistics
        a_mean, a_std = np.mean(a_scores), np.std(a_scores)
        b_mean, b_std = np.mean(b_scores), np.std(b_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(a_scores) - 1) * a_std**2 + (len(b_scores) - 1) * b_std**2) / 
                           (len(a_scores) + len(b_scores) - 2))
        cohens_d = (b_mean - a_mean) / pooled_std if pooled_std > 0 else 0
        
        # T-test
        t_stat, p_value = stats.ttest_ind(a_scores, b_scores)
        
        # Confidence interval
        confidence_level = self.ab_config['confidence_level']
        alpha = 1 - confidence_level
        
        # Calculate confidence interval for difference
        diff = b_mean - a_mean
        diff_std = np.sqrt(a_std**2 / len(a_scores) + b_std**2 / len(b_scores))
        t_critical = stats.t.ppf(1 - alpha/2, len(a_scores) + len(b_scores) - 2)
        ci_lower = diff - t_critical * diff_std
        ci_upper = diff + t_critical * diff_std
        
        # Determine significance
        min_effect_size = self.ab_config['min_effect_size']
        is_significant = p_value < alpha and abs(cohens_d) >= min_effect_size
        
        return {
            'metric': metric_name,
            'group_a': {
                'size': len(a_scores),
                'mean': float(a_mean),
                'std': float(a_std),
                'min': float(np.min(a_scores)),
                'max': float(np.max(a_scores))
            },
            'group_b': {
                'size': len(b_scores),
                'mean': float(b_mean),
                'std': float(b_std),
                'min': float(np.min(b_scores)),
                'max': float(np.max(b_scores))
            },
            'statistical_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'confidence_interval': [float(ci_lower), float(ci_upper)],
                'confidence_level': confidence_level,
                'is_significant': is_significant,
                'effect_size_interpretation': self._interpret_effect_size(cohens_d)
            },
            'practical_significance': {
                'mean_difference': float(diff),
                'percent_improvement': float((diff / a_mean) * 100) if a_mean > 0 else 0,
                'meets_minimum_effect': abs(cohens_d) >= min_effect_size
            }
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def run_model_comparison_test(self, 
                                 model_a_tweets: List[Dict[str, Any]], 
                                 model_b_tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare two different models' outputs.
        
        Args:
            model_a_tweets: Tweets from model A
            model_b_tweets: Tweets from model B
            
        Returns:
            Model comparison results
        """
        # Run comprehensive evaluation
        quality_results = self.run_quality_ab_test(model_a_tweets, model_b_tweets, 'quality_score')
        engagement_results = self.run_engagement_ab_test(model_a_tweets, model_b_tweets)
        
        # Additional model-specific metrics
        a_evaluations = self.quality_evaluator.batch_evaluate(model_a_tweets)
        b_evaluations = self.quality_evaluator.batch_evaluate(model_b_tweets)
        
        a_training_candidates = sum(1 for e in a_evaluations if e['meets_training_criteria'])
        b_training_candidates = sum(1 for e in b_evaluations if e['meets_training_criteria'])
        
        return {
            'test_type': 'model_comparison',
            'quality_comparison': quality_results,
            'engagement_comparison': engagement_results,
            'training_candidates': {
                'model_a': a_training_candidates,
                'model_b': b_training_candidates,
                'difference': b_training_candidates - a_training_candidates
            },
            'overall_winner': self._determine_winner(quality_results, engagement_results)
        }
    
    def _determine_winner(self, quality_results: Dict[str, Any], engagement_results: Dict[str, Any]) -> str:
        """Determine overall winner based on quality and engagement results."""
        if 'error' in quality_results or 'error' in engagement_results:
            return 'inconclusive'
        
        quality_significant = quality_results['statistical_test']['is_significant']
        engagement_significant = engagement_results['statistical_test']['is_significant']
        
        if not quality_significant and not engagement_significant:
            return 'no_significant_difference'
        
        # Weight quality more heavily than engagement
        quality_weight = 0.7
        engagement_weight = 0.3
        
        quality_score = quality_results['statistical_test']['cohens_d'] * quality_weight
        engagement_score = engagement_results['statistical_test']['cohens_d'] * engagement_weight
        
        total_score = quality_score + engagement_score
        
        if total_score > 0.1:
            return 'model_b'
        elif total_score < -0.1:
            return 'model_a'
        else:
            return 'inconclusive'
    
    def generate_ab_test_report(self, test_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable report from A/B test results.
        
        Args:
            test_results: Results from any A/B test method
            
        Returns:
            Formatted report string
        """
        if 'error' in test_results:
            return f"Error: {test_results['error']}"
        
        report = []
        report.append("=" * 60)
        report.append("A/B TEST RESULTS")
        report.append("=" * 60)
        
        # Basic info
        report.append(f"Metric: {test_results.get('metric', 'N/A')}")
        report.append(f"Test Type: {test_results.get('test_type', 'standard')}")
        report.append("")
        
        # Group statistics
        group_a = test_results['group_a']
        group_b = test_results['group_b']
        
        report.append("GROUP STATISTICS:")
        report.append(f"  Group A (Control): n={group_a['size']}, mean={group_a['mean']:.3f} ± {group_a['std']:.3f}")
        report.append(f"  Group B (Treatment): n={group_b['size']}, mean={group_b['mean']:.3f} ± {group_b['std']:.3f}")
        report.append("")
        
        # Statistical results
        stats = test_results['statistical_test']
        report.append("STATISTICAL RESULTS:")
        report.append(f"  T-statistic: {stats['t_statistic']:.3f}")
        report.append(f"  P-value: {stats['p_value']:.4f}")
        report.append(f"  Cohen's d: {stats['cohens_d']:.3f} ({stats['effect_size_interpretation']})")
        report.append(f"  Confidence Interval: [{stats['confidence_interval'][0]:.3f}, {stats['confidence_interval'][1]:.3f}]")
        report.append(f"  Significant: {'Yes' if stats['is_significant'] else 'No'}")
        report.append("")
        
        # Practical significance
        practical = test_results['practical_significance']
        report.append("PRACTICAL SIGNIFICANCE:")
        report.append(f"  Mean Difference: {practical['mean_difference']:.3f}")
        report.append(f"  Percent Improvement: {practical['percent_improvement']:.1f}%")
        report.append(f"  Meets Minimum Effect: {'Yes' if practical['meets_minimum_effect'] else 'No'}")
        
        return "\n".join(report) 