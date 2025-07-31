#!/usr/bin/env python3
"""
Generate high-quality Soju-style crypto tweets.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.model.lora import LoRAFineTuner

def generate_soju_tweets():
    """Generate high-quality Soju-style crypto tweets."""
    print("Generating Soju-style crypto tweets...")
    
    # Initialize LoRA trainer
    lora_trainer = LoRAFineTuner()
    
    # Load the trained adapter
    adapter_path = "lora_checkpoints/proper_identity/final_adapter"
    if Path(adapter_path).exists():
        print(f"Loading adapter from: {adapter_path}")
        lora_trainer.load_adapter(adapter_path)
    else:
        print("No adapter found, using base model")
    
    # High-quality Soju-style prompts
    soju_prompts = [
        "As a crypto influencer, write a tweet about Bitcoin's current market sentiment:",
        "As a crypto expert, share your thoughts on DeFi yield farming:",
        "As a blockchain analyst, explain RWA tokenization benefits:",
        "As a crypto commentator, give your take on recent regulatory developments:",
        "As a crypto investor, share your thoughts on portfolio diversification:",
        "As a crypto educator, explain smart contracts to beginners:",
        "As a crypto trader, share your thoughts on market volatility:",
        "As a crypto advocate, discuss Bitcoin adoption in emerging markets:"
    ]
    
    print("\n" + "="*70)
    print("SOJU-STYLE CRYPTO TWEETS")
    print("="*70)
    
    generated_tweets = []
    
    for i, prompt in enumerate(soju_prompts, 1):
        print(f"\n{i}. {prompt}")
        print("-" * 60)
        
        try:
            # Generate with LoRA
            response = lora_trainer.generate_with_lora(prompt, max_length=150)
            
            # Clean up response
            clean_response = response.strip()
            if clean_response.startswith('"') and clean_response.endswith('"'):
                clean_response = clean_response[1:-1]
            
            # Remove any remaining artifacts
            if "##" in clean_response or "Step" in clean_response:
                lines = clean_response.split('\n')
                clean_lines = []
                for line in lines:
                    if not any(x in line for x in ['##', 'Step', '//', 'const', 'function', 'A.', 'B.', 'C.']):
                        clean_lines.append(line)
                clean_response = ' '.join(clean_lines).strip()
            
            print(f"Tweet: {clean_response}")
            generated_tweets.append({
                "prompt": prompt,
                "tweet": clean_response
            })
            
        except Exception as e:
            print(f"Generation failed: {e}")
        
        print()
    
    # Save generated tweets
    output_file = "generated_soju_tweets.json"
    with open(output_file, 'w') as f:
        json.dump(generated_tweets, f, indent=2)
    
    print(f"\nGenerated {len(generated_tweets)} tweets")
    print(f"Saved to: {output_file}")
    
    return generated_tweets

if __name__ == "__main__":
    generate_soju_tweets() 