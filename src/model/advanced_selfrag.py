"""Advanced Self-RAG implementation with multi-iteration refinement and contradiction detection."""

from typing import List, Dict, Optional, Tuple
import logging
import re
from model.generate_h200 import H200TextGenerator
from vector.hybrid_search import HybridSearch
from model.cot_rag import ChainOfThoughtRAG

logger = logging.getLogger(__name__)

class AdvancedSelfRAG:
    """Advanced Self-Reflective Retrieval-Augmented Generation with multi-iteration refinement."""
    
    def __init__(self, generator: Optional[H200TextGenerator] = None, 
                 hybrid_search: Optional[HybridSearch] = None,
                 cot_rag: Optional[ChainOfThoughtRAG] = None):
        """Initialize Advanced Self-RAG components."""
        self.generator = generator or H200TextGenerator()
        self.hybrid_search = hybrid_search
        self.cot_rag = cot_rag
        
        # Advanced reflection prompts
        self.reflection_prompts = {
            "relevance": "Is this response directly relevant to the crypto/influencer question? Answer Yes/No with brief explanation:",
            "accuracy": "Does this response contain accurate and up-to-date information? Answer Yes/No with brief explanation:",
            "completeness": "Is this response complete and addresses all aspects of the question? Answer Yes/No with brief explanation:",
            "confidence": "Rate your confidence in this response from 1-10 and explain why:",
            "evidence_support": "How well does the evidence support this response? Rate 1-10:",
            "contradictions": "Are there any contradictions or inconsistencies in this response? Answer Yes/No with details:"
        }
        
        # Evidence weighting factors
        self.evidence_weights = {
            "recent_tweets": 1.2,  # Recent tweets weighted higher
            "verified_users": 1.3,  # Verified users weighted higher
            "high_engagement": 1.1,  # High engagement tweets weighted higher
            "official_sources": 1.4,  # Official sources weighted highest
            "contradictory": 0.3,  # Contradictory evidence weighted lower
        }
    
    def generate_with_advanced_selfrag(self, query: str, max_iterations: int = 5) -> Dict:
        """Generate response using Advanced Self-RAG with multi-iteration refinement."""
        logger.info(f"Starting Advanced Self-RAG generation for: {query[:50]}...")
        
        best_response = None
        best_score = 0
        best_evidence = None
        iterations = []
        contradiction_detected = False
        
        try:
            for iteration in range(max_iterations):
                logger.info(f"Advanced Self-RAG iteration {iteration + 1}/{max_iterations}")
                
                # Step 1: Retrieve relevant context with evidence weighting
                context_data = self._retrieve_weighted_context(query, iteration)
                
                # Step 2: Check for contradictions in evidence
                contradictions = self._detect_contradictions(context_data['evidence'])
                if contradictions and not contradiction_detected:
                    logger.warning(f"Contradictions detected: {contradictions}")
                    contradiction_detected = True
                
                # Step 3: Generate response with Chain-of-Thought reasoning
                if self.cot_rag:
                    response_data = self.cot_rag.generate_with_reasoning(query)
                    response = response_data['final_answer']
                    reasoning_steps = response_data['reasoning_steps']
                else:
                    # Fallback to simple generation
                    response = self.generator.generate_text(query, max_tokens=200)
                    reasoning_steps = []
                
                # Step 4: Advanced self-reflection and critique
                reflection_scores = self._advanced_self_reflect(query, response, context_data['evidence'])
                
                # Step 5: Calculate weighted overall score
                overall_score = self._calculate_weighted_score(reflection_scores, context_data['evidence_quality'])
                
                # Step 6: Check for contradictions in response
                response_contradictions = self._detect_response_contradictions(response, context_data['evidence'])
                if response_contradictions:
                    reflection_scores['contradictions'] = 2.0  # Penalize contradictions
                    overall_score *= 0.7  # Reduce overall score
                
                iteration_data = {
                    "iteration": iteration + 1,
                    "response": response,
                    "reasoning_steps": reasoning_steps,
                    "evidence": context_data['evidence'],
                    "evidence_quality": context_data['evidence_quality'],
                    "reflection_scores": reflection_scores,
                    "overall_score": overall_score,
                    "contradictions": contradictions,
                    "response_contradictions": response_contradictions
                }
                iterations.append(iteration_data)
                
                # Step 7: Check if this is the best response so far
                if overall_score > best_score:
                    best_response = response
                    best_score = overall_score
                    best_evidence = context_data['evidence']
                
                # Step 8: Early stopping conditions
                if self._should_stop_early(overall_score, iteration, contradiction_detected):
                    logger.info(f"Early stopping - conditions met (score: {overall_score}, iteration: {iteration + 1})")
                    break
            
            result = {
                "query": query,
                "best_response": best_response,
                "best_score": best_score,
                "best_evidence": best_evidence,
                "iterations": iterations,
                "total_iterations": len(iterations),
                "contradiction_detected": contradiction_detected,
                "final_confidence": self._calculate_final_confidence(best_score, len(iterations))
            }
            
            logger.info(f"Advanced Self-RAG completed. Best score: {best_score}, Confidence: {result['final_confidence']}")
            return result
        except Exception as e:
            logger.error(f"Advanced Self-RAG failed: {e}")
            raise RuntimeError(f"Advanced Self-RAG failed: {e}")
    
    def _retrieve_weighted_context(self, query: str, iteration: int) -> Dict:
        """Retrieve context with evidence weighting and quality assessment."""
        if not self.hybrid_search:
            # Fallback to simple retrieval
            return {
                "evidence": [{"text": "No evidence available", "weight": 1.0}],
                "evidence_quality": 5.0
            }
        
        # Retrieve evidence with hybrid search
        results = self.hybrid_search.search(query, top_k=5, rerank=True)
        
        # Weight evidence based on various factors
        weighted_evidence = []
        total_weight = 0
        
        for result in results:
            evidence = {
                "text": result['text'],
                "score": result['score'],
                "weight": 1.0,  # Base weight
                "factors": []
            }
            
            # Apply weighting factors
            text_lower = result['text'].lower()
            
            # Check for recent content (mentions of recent years)
            if any(year in text_lower for year in ['2024', '2025', 'recent', 'latest']):
                evidence['weight'] *= self.evidence_weights['recent_tweets']
                evidence['factors'].append('recent')
            
            # Check for verified users or official sources
            if any(source in text_lower for source in ['official', 'verified', 'confirmed', 'announcement']):
                evidence['weight'] *= self.evidence_weights['verified_users']
                evidence['factors'].append('verified')
            
            # Check for high engagement indicators
            if any(term in text_lower for term in ['viral', 'trending', 'popular', 'breaking']):
                evidence['weight'] *= self.evidence_weights['high_engagement']
                evidence['factors'].append('high_engagement')
            
            # Check for official sources
            if any(source in text_lower for source in ['government', 'regulatory', 'sec', 'federal']):
                evidence['weight'] *= self.evidence_weights['official_sources']
                evidence['factors'].append('official_source')
            
            weighted_evidence.append(evidence)
            total_weight += evidence['weight']
        
        # Normalize weights
        if total_weight > 0:
            for evidence in weighted_evidence:
                evidence['weight'] /= total_weight
        
        # Calculate overall evidence quality
        evidence_quality = sum(evidence['score'] * evidence['weight'] for evidence in weighted_evidence)
        
        return {
            "evidence": weighted_evidence,
            "evidence_quality": evidence_quality
        }
    
    def _detect_contradictions(self, evidence: List[Dict]) -> List[str]:
        """Detect contradictions in the evidence."""
        contradictions = []
        
        if len(evidence) < 2:
            return contradictions
        
        # Extract key claims from evidence
        claims = []
        for item in evidence:
            text = item['text'].lower()
            # Extract potential claims (simplified)
            if 'bitcoin' in text and ('up' in text or 'down' in text):
                direction = 'up' if 'up' in text else 'down'
                claims.append(('bitcoin', direction))
            if 'ethereum' in text and ('up' in text or 'down' in text):
                direction = 'up' if 'up' in text else 'down'
                claims.append(('ethereum', direction))
        
        # Check for contradictory claims
        for i, (asset1, direction1) in enumerate(claims):
            for j, (asset2, direction2) in enumerate(claims[i+1:], i+1):
                if asset1 == asset2 and direction1 != direction2:
                    contradiction = f"Contradictory claims about {asset1}: {direction1} vs {direction2}"
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _advanced_self_reflect(self, query: str, response: str, evidence: List[Dict]) -> Dict[str, float]:
        """Perform advanced self-reflection on the generated response."""
        reflection_scores = {}
        
        for aspect, prompt in self.reflection_prompts.items():
            # Create enhanced reflection prompt with evidence
            evidence_text = "\n".join([f"- {item['text'][:100]}..." for item in evidence[:3]])
            
            reflection_input = f"""
Query: {query}

Evidence:
{evidence_text}

Response: {response}

{prompt}
"""
            
            # Generate self-reflection
            reflection = self.generator.generate_text(reflection_input, max_tokens=50)
            
            # Parse reflection score
            score = self._parse_advanced_reflection_score(reflection, aspect)
            reflection_scores[aspect] = score
            
            logger.debug(f"Advanced reflection {aspect}: {score} (raw: {reflection})")
        
        return reflection_scores
    
    def _parse_advanced_reflection_score(self, reflection: str, aspect: str) -> float:
        """Parse advanced reflection response into numerical score."""
        reflection = reflection.lower().strip()
        
        if aspect == "confidence":
            # Extract number from 1-10 scale
            try:
                numbers = re.findall(r'\b(\d+)\b', reflection)
                if numbers:
                    score = int(numbers[0])
                    return min(max(score, 1), 10)
            except:
                pass
            return 5.0
        elif aspect == "evidence_support":
            # Extract number from 1-10 scale
            try:
                numbers = re.findall(r'\b(\d+)\b', reflection)
                if numbers:
                    score = int(numbers[0])
                    return min(max(score, 1), 10)
            except:
                pass
            return 5.0
        else:
            # Yes/No questions with explanation
            if "yes" in reflection and "no" not in reflection:
                return 10.0
            elif "no" in reflection and "yes" not in reflection:
                return 1.0
            else:
                return 5.0  # Uncertain
    
    def _calculate_weighted_score(self, reflection_scores: Dict[str, float], evidence_quality: float) -> float:
        """Calculate weighted overall score from reflection scores and evidence quality."""
        # Weights for different aspects
        weights = {
            "relevance": 0.25,
            "accuracy": 0.25,
            "completeness": 0.15,
            "confidence": 0.15,
            "evidence_support": 0.15,
            "contradictions": 0.05
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for aspect, score in reflection_scores.items():
            if aspect in weights:
                weighted_sum += score * weights[aspect]
                total_weight += weights[aspect]
        
        # Incorporate evidence quality
        evidence_factor = min(evidence_quality / 10.0, 1.0)  # Normalize to [0, 1]
        
        if total_weight > 0:
            base_score = weighted_sum / total_weight
            # Combine base score with evidence quality
            final_score = 0.8 * base_score + 0.2 * (evidence_factor * 10)
            return final_score
        else:
            return 5.0
    
    def _detect_response_contradictions(self, response: str, evidence: List[Dict]) -> List[str]:
        """Detect contradictions between response and evidence."""
        contradictions = []
        
        response_lower = response.lower()
        
        # Check for contradictions with evidence
        for item in evidence:
            evidence_text = item['text'].lower()
            
            # Simple contradiction detection (can be enhanced)
            if 'bitcoin' in response_lower and 'bitcoin' in evidence_text:
                if ('up' in response_lower and 'down' in evidence_text) or \
                   ('down' in response_lower and 'up' in evidence_text):
                    contradictions.append("Contradiction in Bitcoin direction")
            
            if 'ethereum' in response_lower and 'ethereum' in evidence_text:
                if ('up' in response_lower and 'down' in evidence_text) or \
                   ('down' in response_lower and 'up' in evidence_text):
                    contradictions.append("Contradiction in Ethereum direction")
        
        return contradictions
    
    def _should_stop_early(self, score: float, iteration: int, contradiction_detected: bool) -> bool:
        """Determine if we should stop early based on various conditions."""
        # Stop if score is very high
        if score >= 8.5:
            return True
        
        # Stop if we've done enough iterations and score is acceptable
        if iteration >= 3 and score >= 7.0:
            return True
        
        # Stop if contradictions detected and we've done at least 2 iterations
        if contradiction_detected and iteration >= 2:
            return True
        
        return False
    
    def _calculate_final_confidence(self, best_score: float, total_iterations: int) -> str:
        """Calculate final confidence level based on score and iterations."""
        if best_score >= 8.5:
            return "very_high"
        elif best_score >= 7.0:
            return "high"
        elif best_score >= 5.5:
            return "medium"
        else:
            return "low" 