#!/usr/bin/env python3
"""
Soju Identity Training Script for H200
Trains Soju persona using real Twitter data with intermediate testing checkpoints.
"""

import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.identity_pipeline import IdentityTrainingPipeline
from src.model.lora import LoRAFineTuner
from src.model.generate_h200 import H200TextGenerator
from src.evaluation.engine import EvaluationEngine
from src.utils.logger import setup_logger

logger = setup_logger("soju_training", level="INFO")

class SojuTrainingManager:
    """Manages Soju training with testing checkpoints."""
    
    def __init__(self, test_interval: int = 50):
        """Initialize training manager."""
        self.test_interval = test_interval
        self.identity_pipeline = IdentityTrainingPipeline()
        self.lora_trainer = None
        self.h200_generator = None
        self.evaluation_engine = None
        
        # Test queries for checking learning progress
        self.test_queries = [
            "What is RWA in crypto?",
            "Why are you bullish on tokenized real estate?", 
            "What's your take on the latest DeFi regulations?",
            "How do you see the future of Real World Assets?",
            "What makes a good crypto investment?",
            "Who created you and why?",
            "Why don't you use emojis in your posts?"
        ]
        
        # Expected Soju-style characteristics
        self.expected_characteristics = [
            "professional tone",
            "no emojis", 
            "educational focus",
            "RWA expertise",
            "created by Max",
            "crypto influencer AI"
        ]
        
        logger.info("Soju training manager initialized")

    def run_pre_training_tests(self) -> Dict[str, Any]:
        """Test baseline model before training."""
        logger.info("Running pre-training baseline tests")
        
        try:
            # Initialize base model for testing
            if self.h200_generator is None:
                self.h200_generator = H200TextGenerator()
            
            baseline_results = []
            for query in self.test_queries:
                # Generate with base model
                response = self.h200_generator.generate_response(query, max_length=100)
                
                # Check Soju characteristics in baseline
                characteristics_found = []
                for char in self.expected_characteristics:
                    if self._check_characteristic(response, char):
                        characteristics_found.append(char)
                
                baseline_results.append({
                    "query": query,
                    "response": response,
                    "characteristics_found": characteristics_found,
                    "soju_score": len(characteristics_found) / len(self.expected_characteristics)
                })
            
            avg_baseline_score = sum(r["soju_score"] for r in baseline_results) / len(baseline_results)
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "test_type": "baseline",
                "avg_soju_score": avg_baseline_score,
                "results": baseline_results
            }
            
            logger.info(f"Baseline Soju score: {avg_baseline_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Pre-training tests failed: {e}")
            return {"error": str(e)}
    
    def run_checkpoint_test(self, adapter_path: str, checkpoint_num: int) -> Dict[str, Any]:
        """Test model at training checkpoint."""
        logger.info(f"Running checkpoint test #{checkpoint_num}")
        
        try:
            # Load model with LoRA adapter
            if self.lora_trainer is None:
                self.lora_trainer = LoRAFineTuner()
            
            self.lora_trainer.load_adapter(adapter_path)
            
            checkpoint_results = []
            for query in self.test_queries:
                # Generate with LoRA adapter
                prompt = f"Query: {query}\nResponse:"
                response = self.lora_trainer.generate_with_lora(prompt, max_length=100)
                
                # Check Soju characteristics
                characteristics_found = []
                for char in self.expected_characteristics:
                    if self._check_characteristic(response, char):
                        characteristics_found.append(char)
                
                checkpoint_results.append({
                    "query": query,
                    "response": response,
                    "characteristics_found": characteristics_found,
                    "soju_score": len(characteristics_found) / len(self.expected_characteristics)
                })
            
            avg_checkpoint_score = sum(r["soju_score"] for r in checkpoint_results) / len(checkpoint_results)
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "test_type": f"checkpoint_{checkpoint_num}",
                "adapter_path": adapter_path,
                "avg_soju_score": avg_checkpoint_score,
                "results": checkpoint_results
            }
            
            logger.info(f"Checkpoint #{checkpoint_num} Soju score: {avg_checkpoint_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Checkpoint test failed: {e}")
            return {"error": str(e)}
    
    def _check_characteristic(self, response: str, characteristic: str) -> bool:
        """Check if response exhibits a Soju characteristic."""
        response_lower = response.lower()
        
        if characteristic == "professional tone":
            # Check for professional language, avoid slang
            unprofessional_words = ["dude", "bro", "yolo", "sick", "lit"]
            return not any(word in response_lower for word in unprofessional_words)
        
        elif characteristic == "no emojis":
            # Check for absence of emoji characters
            emoji_chars = ["😀", "🚀", "💎", "🔥", "📈", "💰", "🌙"]
            return not any(emoji in response for emoji in emoji_chars)
        
        elif characteristic == "educational focus":
            # Check for educational language
            educational_words = ["explain", "understand", "learn", "means", "definition"]
            return any(word in response_lower for word in educational_words)
        
        elif characteristic == "RWA expertise":
            # Check for RWA-related content
            rwa_words = ["real world assets", "tokenization", "rwa", "traditional assets"]
            return any(word in response_lower for word in rwa_words)
        
        elif characteristic == "created by Max":
            # Check for creator mention
            creator_words = ["max", "created by", "my creator"]
            return any(word in response_lower for word in creator_words)
        
        elif characteristic == "crypto influencer AI":
            # Check for AI identity acknowledgment
            ai_words = ["ai", "artificial intelligence", "bot", "influencer"]
            return any(word in response_lower for word in ai_words)
        
        return False
    
    def train_with_testing(self, output_dir: str = "lora_checkpoints/soju_training") -> Dict[str, Any]:
        """Run full training with testing checkpoints."""
        logger.info("Starting Soju training with testing checkpoints...")
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Pre-training baseline
            baseline_results = self.run_pre_training_tests()
            
            # Start training
            training_results = self.identity_pipeline.train_identity_model(output_dir)
            
            if training_results is None:
                raise RuntimeError("Training failed to produce results")
            
            # Post-training test
            final_results = self.run_checkpoint_test(training_results, 999)  # Final checkpoint
            
            # Compile full results
            full_results = {
                "training_started": datetime.now().isoformat(),
                "training_completed": datetime.now().isoformat(),
                "adapter_path": training_results,
                "baseline_test": baseline_results,
                "final_test": final_results,
                "improvement": final_results["avg_soju_score"] - baseline_results["avg_soju_score"]
            }
            
            # Save results
            results_file = f"{output_dir}/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(full_results, f, indent=2)
            
            logger.info(f"Training completed! Results saved to {results_file}")
            logger.info(f"Soju score improvement: {full_results['improvement']:.3f}")
            
            return full_results
            
        except Exception as e:
            logger.error(f"Training with testing failed: {e}")
            raise e

def main():
    """Main training execution."""
    print("🤖 Soju Identity Training on H200")
    print("=" * 50)
    
    try:
        # Initialize training manager
        trainer = SojuTrainingManager()
        
        # Run complete training with testing
        results = trainer.train_with_testing()
        
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Baseline score: {results['baseline_test']['avg_soju_score']:.3f}")
        print(f"📊 Final score: {results['final_test']['avg_soju_score']:.3f}")
        print(f"📈 Improvement: {results['improvement']:.3f}")
        print(f"🎯 LoRA adapter: {results['adapter_path']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 