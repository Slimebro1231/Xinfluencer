#!/usr/bin/env python3
"""
Test script to generate crypto tweets using the trained LoRA model.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.model.lora import LoRAFineTuner

def test_tweet_generation():
    """Test generating crypto tweets with the trained model."""
    print("Testing crypto tweet generation...")
    
    # Initialize LoRA trainer
    lora_trainer = LoRAFineTuner()
    
    # Load the trained adapter
    adapter_path = "lora_checkpoints/proper_identity/final_adapter"
    if Path(adapter_path).exists():
        print(f"Loading adapter from: {adapter_path}")
        lora_trainer.load_adapter(adapter_path)
    else:
        print("No adapter found, using base model")
    
    # Test prompts for crypto tweet generation
    test_prompts = [
        "Generate a tweet about Bitcoin:",
        "Write a tweet about DeFi:",
        "Create a tweet about RWA tokenization:",
        "Generate a tweet about crypto regulation:",
        "Write a tweet about blockchain technology:",
        "Create a tweet about the future of crypto:",
        "Generate a tweet about Bitcoin adoption:",
        "Write a tweet about smart contracts:"
    ]
    
    print("\n" + "="*60)
    print("CRYPTO TWEET GENERATION TEST")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. {prompt}")
        print("-" * 40)
        
        try:
            # Try to generate with LoRA first
            response = lora_trainer.generate_with_lora(prompt, max_length=100)
            print(f"LoRA Response: {response}")
        except Exception as e:
            print(f"LoRA generation failed: {e}")
            
            # Fallback to base model
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                # Load base model for fallback
                model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HUGGINGFACE_TOKEN"))
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    token=os.getenv("HUGGINGFACE_TOKEN"),
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                # Generate with base model
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                base_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if base_response.startswith(prompt):
                    base_response = base_response[len(prompt):].strip()
                
                print(f"Base Model Response: {base_response}")
                
            except Exception as base_error:
                print(f"Base model generation also failed: {base_error}")
                print("No response generated")
        
        print()

if __name__ == "__main__":
    test_tweet_generation() 