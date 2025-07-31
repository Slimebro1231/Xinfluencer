#!/usr/bin/env python3
"""
Official Soju Generator - Integrated LoRA-powered crypto content generation.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.generate import TextGenerator
from src.model.lora import LoRAFineTuner

logger = logging.getLogger(__name__)

class SojuGenerator:
    """Official Soju generator with integrated LoRA support."""
    
    def __init__(self, use_lora: bool = True, lora_adapter_path: str = "lora_checkpoints/proper_identity/final_adapter"):
        """Initialize Soju generator with optional LoRA support."""
        self.use_lora = use_lora
        self.lora_adapter_path = lora_adapter_path
        
        # Initialize base generator
        self.generator = TextGenerator(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            use_quantization=True,
            use_lora=use_lora,
            lora_adapter_path=lora_adapter_path if use_lora else None
        )
        
        # Initialize LoRA trainer for direct access
        self.lora_trainer = None
        if use_lora:
            try:
                self.lora_trainer = LoRAFineTuner()
                if Path(lora_adapter_path).exists():
                    self.lora_trainer.load_adapter(lora_adapter_path)
                    logger.info(f"Loaded LoRA adapter: {lora_adapter_path}")
                else:
                    logger.warning(f"LoRA adapter not found: {lora_adapter_path}")
            except Exception as e:
                logger.error(f"Failed to initialize LoRA trainer: {e}")
                self.use_lora = False
        
        logger.info(f"Soju generator initialized (LoRA: {self.use_lora})")
    
    def generate_tweet(self, topic: str, style: str = "professional") -> str:
        """Generate a Soju-style crypto tweet."""
        try:
            if self.use_lora and self.lora_trainer:
                return self.lora_trainer.generate_soju_tweet(topic, style)
            else:
                return self.generator.generate_soju_tweet(topic, style)
        except Exception as e:
            logger.error(f"Tweet generation failed: {e}")
            return f"Error generating tweet about {topic}: {str(e)}"
    
    def generate_content(self, content_type: str, topic: str) -> str:
        """Generate different types of crypto content."""
        try:
            if self.use_lora and self.lora_trainer:
                return self.lora_trainer.generate_crypto_content(content_type, topic)
            else:
                return self.generator.generate_crypto_content(content_type, topic)
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return f"Error generating {content_type} about {topic}: {str(e)}"
    
    def generate_tweet_batch(self, topics: List[str], style: str = "professional") -> List[Dict[str, str]]:
        """Generate multiple tweets in batch."""
        results = []
        for topic in topics:
            try:
                tweet = self.generate_tweet(topic, style)
                results.append({
                    "topic": topic,
                    "tweet": tweet,
                    "style": style,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to generate tweet for {topic}: {e}")
                results.append({
                    "topic": topic,
                    "tweet": f"Error generating tweet: {str(e)}",
                    "style": style,
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def generate_daily_content(self) -> Dict[str, List[str]]:
        """Generate a daily content package with various crypto topics."""
        daily_topics = [
            "Bitcoin market sentiment",
            "DeFi yield farming opportunities",
            "RWA tokenization trends",
            "Crypto regulation updates",
            "Blockchain innovation",
            "Portfolio diversification strategies",
            "Smart contract security",
            "Bitcoin adoption in emerging markets"
        ]
        
        content_package = {
            "tweets": [],
            "explanations": [],
            "analyses": [],
            "opinions": []
        }
        
        # Generate tweets
        for topic in daily_topics[:4]:  # First 4 topics as tweets
            tweet = self.generate_tweet(topic, "professional")
            content_package["tweets"].append({
                "topic": topic,
                "content": tweet
            })
        
        # Generate explanations
        for topic in daily_topics[4:6]:  # Next 2 as explanations
            explanation = self.generate_content("explanation", topic)
            content_package["explanations"].append({
                "topic": topic,
                "content": explanation
            })
        
        # Generate analyses
        for topic in daily_topics[6:7]:  # Next 1 as analysis
            analysis = self.generate_content("analysis", topic)
            content_package["analyses"].append({
                "topic": topic,
                "content": analysis
            })
        
        # Generate opinions
        for topic in daily_topics[7:]:  # Last 1 as opinion
            opinion = self.generate_content("opinion", topic)
            content_package["opinions"].append({
                "topic": topic,
                "content": opinion
            })
        
        return content_package
    
    def save_content(self, content: Dict, output_file: str = None) -> str:
        """Save generated content to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"soju_content_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(content, f, indent=2)
        
        logger.info(f"Content saved to: {output_file}")
        return output_file
    
    def get_status(self) -> Dict[str, any]:
        """Get generator status and capabilities."""
        return {
            "lora_enabled": self.use_lora,
            "lora_adapter_path": self.lora_adapter_path,
            "lora_available": self.lora_trainer is not None,
            "base_model": self.generator.model_name,
            "device": str(self.generator.device),
            "memory_usage": self.generator.get_memory_usage()
        }


def main():
    """Main function for testing Soju generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Soju Generator CLI")
    parser.add_argument("--topic", type=str, help="Topic for tweet generation")
    parser.add_argument("--style", type=str, default="professional", 
                       choices=["professional", "casual", "educational", "analytical"],
                       help="Tweet style")
    parser.add_argument("--content-type", type=str, 
                       choices=["tweet", "explanation", "analysis", "opinion"],
                       help="Content type to generate")
    parser.add_argument("--batch", action="store_true", help="Generate batch of tweets")
    parser.add_argument("--daily", action="store_true", help="Generate daily content package")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA and use base model only")
    parser.add_argument("--output", type=str, help="Output file for saving content")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SojuGenerator(use_lora=not args.no_lora)
    
    # Print status
    status = generator.get_status()
    print(f"\nSoju Generator Status:")
    print(f"  LoRA Enabled: {status['lora_enabled']}")
    print(f"  LoRA Available: {status['lora_available']}")
    print(f"  Base Model: {status['base_model']}")
    print(f"  Device: {status['device']}")
    
    if args.topic:
        if args.content_type:
            content = generator.generate_content(args.content_type, args.topic)
            print(f"\n{args.content_type.title()} about {args.topic}:")
            print(f"{content}")
        else:
            tweet = generator.generate_tweet(args.topic, args.style)
            print(f"\nTweet about {args.topic} ({args.style} style):")
            print(f"{tweet}")
    
    elif args.batch:
        topics = [
            "Bitcoin price action",
            "DeFi protocols",
            "RWA tokenization",
            "Crypto regulation"
        ]
        results = generator.generate_tweet_batch(topics, args.style)
        print(f"\nGenerated {len(results)} tweets:")
        for result in results:
            print(f"\n{result['topic']}:")
            print(f"{result['tweet']}")
        
        if args.output:
            generator.save_content({"batch_tweets": results}, args.output)
    
    elif args.daily:
        content = generator.generate_daily_content()
        print(f"\nDaily Content Package Generated:")
        print(f"  Tweets: {len(content['tweets'])}")
        print(f"  Explanations: {len(content['explanations'])}")
        print(f"  Analyses: {len(content['analyses'])}")
        print(f"  Opinions: {len(content['opinions'])}")
        
        if args.output:
            generator.save_content(content, args.output)
    
    else:
        # Default: generate a sample tweet
        tweet = generator.generate_tweet("Bitcoin adoption", "professional")
        print(f"\nSample tweet:")
        print(f"{tweet}")


if __name__ == "__main__":
    main() 