#!/usr/bin/env python3
"""
Test script to verify LoRA adapter effectiveness.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.model.lora import LoRAFineTuner

def test_lora_effectiveness():
    """Test if LoRA adapter is actually working."""
    print("Testing LoRA adapter effectiveness...")
    
    # Initialize LoRA trainer
    lora_trainer = LoRAFineTuner()
    
    # Test prompt
    test_prompt = "Write a tweet about Bitcoin:"
    
    print(f"\nTest prompt: {test_prompt}")
    print("="*60)
    
    # Test 1: Base model only
    print("\n1. BASE MODEL ONLY:")
    print("-" * 30)
    try:
        # Disable LoRA temporarily
        original_peft_model = lora_trainer.peft_model
        lora_trainer.peft_model = None
        
        base_response = lora_trainer.generate_with_lora(test_prompt, max_length=100)
        print(f"Base response: {base_response}")
        
        # Restore LoRA
        lora_trainer.peft_model = original_peft_model
        
    except Exception as e:
        print(f"Base model test failed: {e}")
    
    # Test 2: With LoRA adapter
    print("\n2. WITH LORA ADAPTER:")
    print("-" * 30)
    try:
        # Load adapter
        adapter_path = "lora_checkpoints/proper_identity/final_adapter"
        if Path(adapter_path).exists():
            lora_trainer.load_adapter(adapter_path)
            print("Adapter loaded")
        
        lora_response = lora_trainer.generate_with_lora(test_prompt, max_length=100)
        print(f"LoRA response: {lora_response}")
        
    except Exception as e:
        print(f"LoRA test failed: {e}")
    
    # Test 3: Check if responses are different
    print("\n3. COMPARISON:")
    print("-" * 30)
    if 'base_response' in locals() and 'lora_response' in locals():
        if base_response == lora_response:
            print("❌ LoRA is NOT working - responses are identical")
        else:
            print("✅ LoRA is working - responses are different")
            print(f"Base: {base_response[:50]}...")
            print(f"LoRA: {lora_response[:50]}...")
    else:
        print("⚠️  Could not compare responses")
    
    # Test 4: Check LoRA weights
    print("\n4. LORA WEIGHT ANALYSIS:")
    print("-" * 30)
    try:
        if lora_trainer.peft_model is not None:
            # Check if LoRA weights are non-zero
            total_params = 0
            non_zero_params = 0
            
            for name, param in lora_trainer.peft_model.named_parameters():
                if 'lora' in name.lower():
                    total_params += param.numel()
                    non_zero_params += (param != 0).sum().item()
            
            print(f"Total LoRA parameters: {total_params}")
            print(f"Non-zero LoRA parameters: {non_zero_params}")
            print(f"LoRA sparsity: {1 - (non_zero_params / total_params):.2%}")
            
            if non_zero_params > 0:
                print("✅ LoRA weights are non-zero")
            else:
                print("❌ LoRA weights are all zero")
        else:
            print("❌ No PEFT model loaded")
            
    except Exception as e:
        print(f"LoRA weight analysis failed: {e}")

if __name__ == "__main__":
    test_lora_effectiveness() 