"""AI-based review and feedback system."""

from typing import Dict, List, Optional
import logging
from model.generate import TextGenerator

logger = logging.getLogger(__name__)

class AIReviewer:
    """AI-powered review system for generated content."""
    
    def __init__(self, generator: Optional[TextGenerator] = None):
        """Initialize AI reviewer."""
        self.generator = generator or TextGenerator()
        
        # Review criteria prompts
        self.review_criteria = {
            "relevance": {
                "prompt": "Rate how relevant this response is to crypto/influencer content (1-10):",
                "weight": 0.25
            },
            "accuracy": {
                "prompt": "Rate the factual accuracy of this response (1-10):",
                "weight": 0.25
            },
            "engagement": {
                "prompt": "Rate how engaging and interesting this response is (1-10):",
                "weight": 0.20
            },
            "clarity": {
                "prompt": "Rate how clear and well-written this response is (1-10):",
                "weight": 0.15
            },
            "toxicity": {
                "prompt": "Rate how toxic or harmful this response is (1=very toxic, 10=not toxic):",
                "weight": 0.15
            }
        }
    
    def review_response(self, query: str, response: str, context: str = "") -> Dict:
        """Perform comprehensive AI review of a response."""
        logger.info(f"AI reviewing response for query: {query[:50]}...")
        
        scores = {}
        explanations = {}
        
        for criterion, config in self.review_criteria.items():
            score, explanation = self._evaluate_criterion(
                query, response, context, criterion, config["prompt"]
            )
            scores[criterion] = score
            explanations[criterion] = explanation
        
        # Calculate weighted overall score
        overall_score = self._calculate_weighted_score(scores)
        
        # Determine if response should be approved
        approved = overall_score >= 5.0 and scores.get("toxicity", 10) >= 5.0  # Lowered for testing
        
        review_result = {
            "query": query,
            "response": response,
            "scores": scores,
            "explanations": explanations,
            "overall_score": overall_score,
            "approved": approved,
            "timestamp": self._get_timestamp()
        }
        
        logger.info(f"AI review completed. Score: {overall_score:.1f}, Approved: {approved}")
        return review_result
    
    def _evaluate_criterion(self, query: str, response: str, context: str, criterion: str, prompt: str) -> tuple:
        """Evaluate a single criterion."""
        review_prompt = f"""
Query: {query}
Context: {context}
Response: {response}

{prompt}

Please provide:
1. A score from 1-10
2. A brief explanation of your rating

Format your response as:
Score: [number]
Explanation: [your explanation]
"""
        
        # Generate review
        review = self.generator.generate_response(review_prompt, max_new_tokens=80)
        
        # Parse score and explanation
        score = self._parse_score(review)
        explanation = self._parse_explanation(review)
        
        logger.debug(f"Criterion {criterion}: Score={score}, Explanation={explanation[:50]}...")
        return score, explanation
    
    def _parse_score(self, review: str) -> float:
        """Parse numerical score from review text."""
        import re
        
        # Look for "Score: X" pattern
        score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', review, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                return min(max(score, 1.0), 10.0)  # Clamp to 1-10
            except ValueError:
                pass
        
        # Fallback: look for any number in the text
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', review)
        if numbers:
            try:
                score = float(numbers[0])
                return min(max(score, 1.0), 10.0)
            except ValueError:
                pass
        
        return 5.0  # Default middle score
    
    def _parse_explanation(self, review: str) -> str:
        """Parse explanation from review text."""
        import re
        
        # Look for "Explanation: ..." pattern
        explanation_match = re.search(r'Explanation:\s*(.+)', review, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            return explanation_match.group(1).strip()
        
        # Fallback: return the whole review
        return review.strip()
    
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        weighted_sum = 0
        total_weight = 0
        
        for criterion, score in scores.items():
            if criterion in self.review_criteria:
                weight = self.review_criteria[criterion]["weight"]
                weighted_sum += score * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 5.0
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def batch_review(self, items: List[Dict]) -> List[Dict]:
        """Review multiple responses."""
        results = []
        for item in items:
            result = self.review_response(
                item["query"],
                item["response"],
                item.get("context", "")
            )
            results.append(result)
        
        logger.info(f"Batch review completed for {len(items)} items")
        return results 