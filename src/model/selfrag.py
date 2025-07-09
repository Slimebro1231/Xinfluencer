"""Self-RAG implementation for reflection and self-critique."""

from typing import List, Dict, Optional, Tuple
import logging
from model.generate import TextGenerator
from vector.search import VectorSearcher
from review.ai import AIReviewer

logger = logging.getLogger(__name__)

class SelfRAGGenerator:
    """A generator that uses Self-RAG to improve responses."""

    def __init__(self, generator: TextGenerator, searcher: VectorSearcher, reviewer: AIReviewer):
        """Initialize the Self-RAG generator."""
        self.generator = generator
        self.searcher = searcher
        self.reviewer = reviewer
        self.max_iterations = 3
        
        # Self-reflection prompts
        self.reflection_prompts = {
            "relevance": "Is this response relevant to the crypto/influencer question? Answer Yes/No:",
            "accuracy": "Does this response contain accurate information? Answer Yes/No:",
            "completeness": "Is this response complete and helpful? Answer Yes/No:",
            "confidence": "Rate your confidence in this response from 1-10:"
        }
    
    def generate_with_self_rag(self, query: str, max_iterations: int = 3) -> Dict:
        """Generate response using Self-RAG approach."""
        logger.info(f"Starting Self-RAG generation for: {query[:50]}...")
        
        best_response = None
        best_score = 0
        iterations = []
        
        for iteration in range(max_iterations):
            logger.info(f"Self-RAG iteration {iteration + 1}/{max_iterations}")
            
            # Step 1: Retrieve context
            context_chunks = self.searcher.search_text(query, limit=3)
            context = " ".join([chunk['text'] for chunk in context_chunks])

            # Step 2: Generate with context
            prompt_with_context = f"Query: {query}\n\nContext: {context}\n\nAnswer:"
            response = self.generator.generate(prompt_with_context, max_length=150)

            # Step 3: Evaluate response
            review = self.reviewer.review_response(query, response, context)
            score = review["overall_score"]
            
            # Step 4: Store results
            iterations.append({
                "iteration": iteration + 1,
                "response": response,
                "context": context,
                "scores": review["scores"],
                "overall_score": score
            })
            
            # Step 5: Check if this is the best response so far
            if score > best_score:
                best_response = response
                best_score = score
            
            # Step 6: Early stopping if score is high enough
            if score >= 8.0:
                logger.info(f"Early stopping - high quality response achieved (score: {score})")
                break
        
        result = {
            "query": query,
            "best_response": best_response,
            "best_score": best_score,
            "iterations": iterations,
            "total_iterations": len(iterations)
        }
        
        logger.info(f"Self-RAG completed. Best score: {best_score}")
        return result
    
    def _self_reflect(self, query: str, response: str, context: str) -> Dict[str, float]:
        """Perform self-reflection on the generated response."""
        reflection_scores = {}
        
        for aspect, prompt in self.reflection_prompts.items():
            # Create reflection prompt
            reflection_input = f"""
Query: {query}
Context: {context}
Response: {response}

{prompt}
"""
            
            # Generate self-reflection
            reflection = self.generator.generate_response(reflection_input, max_new_tokens=30)
            
            # Parse reflection score
            score = self._parse_reflection_score(reflection, aspect)
            reflection_scores[aspect] = score
            
            logger.debug(f"Reflection {aspect}: {score} (raw: {reflection})")
        
        return reflection_scores
    
    def _parse_reflection_score(self, reflection: str, aspect: str) -> float:
        """Parse reflection response into numerical score."""
        reflection = reflection.lower().strip()
        
        if aspect == "confidence":
            # Extract number from 1-10 scale
            try:
                # Look for numbers in the response
                import re
                numbers = re.findall(r'\b(\d+)\b', reflection)
                if numbers:
                    score = int(numbers[0])
                    return min(max(score, 1), 10)  # Clamp to 1-10
            except:
                pass
            return 5.0  # Default middle score
        else:
            # Yes/No questions - convert to binary score
            if "yes" in reflection:
                return 10.0
            elif "no" in reflection:
                return 1.0
            else:
                return 5.0  # Uncertain
    
    def _calculate_overall_score(self, reflection_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score from reflection scores."""
        # Weights for different aspects
        weights = {
            "relevance": 0.3,
            "accuracy": 0.3,
            "completeness": 0.2,
            "confidence": 0.2
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for aspect, score in reflection_scores.items():
            if aspect in weights:
                weighted_sum += score * weights[aspect]
                total_weight += weights[aspect]
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 5.0  # Default score 