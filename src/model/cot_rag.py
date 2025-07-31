"""
Chain-of-Thought RAG implementation for advanced reasoning.
Extracted and enhanced from deployment scripts.
"""

from typing import List, Dict, Any, Optional
import logging
from src.model.generate_h200 import H200TextGenerator

logger = logging.getLogger(__name__)

class ChainOfThoughtRAG:
    """Chain-of-Thought RAG for step-by-step reasoning."""
    
    def __init__(self, generator: Optional[H200TextGenerator] = None):
        """Initialize Chain-of-Thought RAG."""
        self.generator = generator or H200TextGenerator()
        
        # Reasoning prompts for Soju's personality learning
        self.reasoning_prompts = {
            "style_analysis": """
Analyze the following influential tweets to understand the author's style:

Context tweets:
{context}

Step 1: What is the author's personality and tone?
Step 2: What makes these tweets engaging and influential?
Step 3: What patterns do you see in their writing style?
Step 4: How do they structure their thoughts and arguments?

Provide your analysis:
""",
            "engagement_patterns": """
Analyze engagement patterns in these tweets:

Tweets with metrics:
{context}

Step 1: Which tweets got the most engagement and why?
Step 2: What emotional triggers or hooks do they use?
Step 3: How do they balance humor, insight, and controversy?
Step 4: What timing or context factors contribute to success?

Provide your analysis:
""",
            "personality_synthesis": """
Based on the analysis above, synthesize the author's personality:

Style analysis: {style_analysis}
Engagement patterns: {engagement_patterns}

Step 1: What is the core personality trait that makes them influential?
Step 2: How do they balance being informative vs entertaining?
Step 3: What is their unique voice or perspective?
Step 4: How can you emulate their style while being authentic?

Provide your synthesis:
"""
        }
        
        logger.info("ChainOfThoughtRAG initialized")
    
    def analyze_personality_from_tweets(self, context_tweets: str = "") -> Dict[str, Any]:
        """
        Analyze personality and style from influential tweets.
        
        Args:
            context_tweets: Influential tweets to analyze
            
        Returns:
            Dictionary with personality analysis and style synthesis
        """
        logger.info(f"Starting personality analysis from tweets...")
        
        try:
            # Step 1: Style analysis
            style_prompt = self.reasoning_prompts["style_analysis"].format(
                context=context_tweets or "No tweets provided"
            )
            
            style_analysis = self.generator.generate_text(
                style_prompt, 
                max_tokens=400,
                temperature=0.3
            )
            
            # Step 2: Engagement patterns analysis
            engagement_prompt = self.reasoning_prompts["engagement_patterns"].format(
                context=context_tweets or "No tweets provided"
            )
            
            engagement_analysis = self.generator.generate_text(
                engagement_prompt,
                max_tokens=300,
                temperature=0.3
            )
            
            # Step 3: Personality synthesis
            synthesis_prompt = self.reasoning_prompts["personality_synthesis"].format(
                style_analysis=style_analysis,
                engagement_patterns=engagement_analysis
            )
            
            personality_synthesis = self.generator.generate_text(
                synthesis_prompt,
                max_tokens=300,
                temperature=0.3
            )
            
            # Compile analysis steps
            analysis_steps = [
                {
                    "step": "style_analysis",
                    "prompt": style_prompt,
                    "response": style_analysis
                },
                {
                    "step": "engagement_patterns",
                    "prompt": engagement_prompt,
                    "response": engagement_analysis
                },
                {
                    "step": "personality_synthesis",
                    "prompt": synthesis_prompt,
                    "response": personality_synthesis
                }
            ]
            
            result = {
                "context_tweets": context_tweets,
                "analysis_steps": analysis_steps,
                "personality_synthesis": personality_synthesis,
                "style_analysis": style_analysis,
                "engagement_analysis": engagement_analysis,
                "confidence": self._estimate_confidence(analysis_steps)
            }
            
            logger.info(f"Personality analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Personality analysis failed: {e}")
            return {
                "context_tweets": context_tweets,
                "analysis_steps": [],
                "personality_synthesis": f"Error in analysis: {e}",
                "style_analysis": "",
                "engagement_analysis": "",
                "confidence": 0.0
            }
    
    def _estimate_confidence(self, reasoning_steps: List[Dict]) -> float:
        """Estimate confidence based on reasoning quality."""
        if not reasoning_steps:
            return 0.0
        
        # Simple confidence estimation
        confidence_indicators = [
            "confident", "certain", "clear", "definite", "strong evidence",
            "high quality", "reliable", "verified", "consistent"
        ]
        
        uncertainty_indicators = [
            "uncertain", "unclear", "ambiguous", "conflicting", "weak evidence",
            "low quality", "unreliable", "inconsistent", "doubtful"
        ]
        
        total_steps = len(reasoning_steps)
        confidence_score = 0.5  # Base confidence
        
        for step in reasoning_steps:
            response = step.get("response", "").lower()
            
            # Count confidence indicators
            confidence_count = sum(1 for indicator in confidence_indicators if indicator in response)
            uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in response)
            
            # Adjust confidence based on indicators
            if confidence_count > uncertainty_count:
                confidence_score += 0.1
            elif uncertainty_count > confidence_count:
                confidence_score -= 0.1
        
        # Normalize to [0, 1]
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        return confidence_score
    
    def get_personality_summary(self, analysis_steps: List[Dict]) -> str:
        """Get a summary of the personality analysis."""
        if not analysis_steps:
            return "No analysis steps available"
        
        summary_parts = []
        for i, step in enumerate(analysis_steps, 1):
            step_name = step.get("step", "unknown").replace("_", " ").title()
            response = step.get("response", "")[:100] + "..." if len(step.get("response", "")) > 100 else step.get("response", "")
            summary_parts.append(f"{i}. {step_name}: {response}")
        
        return "\n".join(summary_parts) 