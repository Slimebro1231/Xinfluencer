#!/usr/bin/env python3
"""
Test script to compare LoRA vs base model outputs and verify progressive training.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.model.soju_generator import SojuGenerator

def test_lora_vs_base():
    """Test LoRA vs base model outputs to verify training effectiveness."""
    print("Testing LoRA vs Base Model Outputs")
    print("="*60)
    
    # Test prompts
    test_prompts = [
        "Bitcoin price action",
        "DeFi protocols", 
        "RWA tokenization",
        "Crypto regulation"
    ]
    
    # Test 1: Base model only
    print("\n1. BASE MODEL ONLY:")
    print("-" * 40)
    base_generator = SojuGenerator(use_lora=False)
    
    base_responses = {}
    for topic in test_prompts:
        response = base_generator.generate_tweet(topic, "professional")
        base_responses[topic] = response
        print(f"\n{topic}:")
        print(f"{response}")
    
    # Test 2: LoRA model
    print("\n\n2. LORA MODEL:")
    print("-" * 40)
    lora_generator = SojuGenerator(use_lora=True)
    
    lora_responses = {}
    for topic in test_prompts:
        response = lora_generator.generate_tweet(topic, "professional")
        lora_responses[topic] = response
        print(f"\n{topic}:")
        print(f"{response}")
    
    # Test 3: Comparison
    print("\n\n3. COMPARISON:")
    print("-" * 40)
    
    for topic in test_prompts:
        print(f"\n{topic}:")
        print(f"Base:  {base_responses[topic][:100]}...")
        print(f"LoRA:  {lora_responses[topic][:100]}...")
        
        # Check if responses are different
        if base_responses[topic] != lora_responses[topic]:
            print("✅ DIFFERENT - LoRA is working!")
        else:
            print("❌ IDENTICAL - LoRA may not be working")
    
    # Save results
    results = {
        "base_model": base_responses,
        "lora_model": lora_responses,
        "test_prompts": test_prompts
    }
    
    with open("lora_vs_base_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: lora_vs_base_comparison.json")


def test_progressive_training():
    """Test that progressive training (continuing from checkpoints) works."""
    print("\n\nTesting Progressive Training")
    print("="*60)
    
    # Check if existing adapter exists
    adapter_path = "lora_checkpoints/proper_identity/final_adapter"
    if Path(adapter_path).exists():
        print(f"✅ Existing adapter found: {adapter_path}")
        
        # Test loading the adapter
        try:
            generator = SojuGenerator(use_lora=True, lora_adapter_path=adapter_path)
            status = generator.get_status()
            print(f"✅ Adapter loaded successfully")
            print(f"   LoRA Available: {status['lora_available']}")
            print(f"   Adapter Path: {status['lora_adapter_path']}")
            
            # Test generation with loaded adapter
            response = generator.generate_tweet("Bitcoin adoption", "professional")
            print(f"✅ Generation with loaded adapter works:")
            print(f"   {response[:100]}...")
            
        except Exception as e:
            print(f"❌ Failed to load adapter: {e}")
    else:
        print(f"❌ No existing adapter found at: {adapter_path}")
        print("   Progressive training test skipped")


def main():
    """Main function."""
    try:
        test_lora_vs_base()
        test_progressive_training()
        
        print("\n" + "="*60)
        print("TESTING COMPLETED!")
        print("="*60)
        print("Check the output above to verify:")
        print("1. LoRA outputs differ from base model outputs")
        print("2. Progressive training works with existing adapters")
        
    except Exception as e:
        print(f"Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 