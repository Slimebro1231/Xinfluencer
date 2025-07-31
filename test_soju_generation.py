#!/usr/bin/env python3
"""
Test script to generate Soju-style crypto tweets.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.model.lora import LoRAFineTuner

def test_soju_generation():
    """Test generating Soju-style crypto tweets."""
    print("Testing Soju-style crypto tweet generation...")
    
    # Initialize LoRA trainer
    lora_trainer = LoRAFineTuner()
    
    # Load the trained adapter
    adapter_path = "lora_checkpoints/proper_identity/final_adapter"
    if Path(adapter_path).exists():
        print(f"Loading adapter from: {adapter_path}")
        lora_trainer.load_adapter(adapter_path)
    else:
        print("No adapter found, using base model")
    
    # Soju-style prompts that should generate more personality-driven responses
    soju_prompts = [
        "As Soju, write a tweet about Bitcoin's recent price action:",
        "As Soju, share your thoughts on DeFi protocols:",
        "As Soju, explain RWA tokenization to your followers:",
        "As Soju, give your take on crypto regulation:",
        "As Soju, tweet about blockchain innovation:",
        "As Soju, share your crypto investment philosophy:",
        "As Soju, respond to someone asking about Bitcoin adoption:",
        "As Soju, explain smart contracts in simple terms:"
    ]
    
    print("\n" + "="*60)
    print("SOJU-STYLE CRYPTO TWEET GENERATION")
    print("="*60)
    
    for i, prompt in enumerate(soju_prompts, 1):
        print(f"\n{i}. {prompt}")
        print("-" * 50)
        
        try:
            # Generate with LoRA
            response = lora_trainer.generate_with_lora(prompt, max_length=120)
            print(f"Response: {response}")
            
            # Clean up any artifacts
            if "##" in response or "Step" in response or "//" in response:
                # Remove code-like artifacts
                lines = response.split('\n')
                clean_lines = []
                for line in lines:
                    if not any(x in line for x in ['##', 'Step', '//', 'const', 'function']):
                        clean_lines.append(line)
                response = ' '.join(clean_lines).strip()
                print(f"Cleaned: {response}")
            
        except Exception as e:
            print(f"Generation failed: {e}")
        
        print()

if __name__ == "__main__":
    test_soju_generation() 