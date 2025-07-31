#!/usr/bin/env python3
"""Minimal LoRA test to isolate gradient issues."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

def test_minimal_lora():
    """Test minimal LoRA setup."""
    print("Testing minimal LoRA setup...")
    
    # Use a smaller model for testing
    model_name = "microsoft/DialoGPT-medium"  # Much smaller model for testing
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        print("✓ Base model loaded")
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["c_attn"]
        )
        
        # Create PEFT model
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.to("cuda" if torch.cuda.is_available() else "cpu")
        print("✓ PEFT model created and moved to device")
        
        # Print trainable parameters
        peft_model.print_trainable_parameters()
        
        # Test forward pass
        peft_model.train()
        inputs = tokenizer("Hello world", return_tensors="pt")
        inputs = {k: v.to(peft_model.device) for k, v in inputs.items()}
        
        outputs = peft_model(**inputs)
        loss = outputs.logits.mean()
        print(f"✓ Forward pass successful, loss: {loss.item()}")
        
        # Test backward pass
        loss.backward()
        print("✓ Backward pass successful")
        
        print("✓ Minimal LoRA test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Minimal LoRA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_lora()
    if success:
        print("\nMinimal LoRA setup works! The issue is with the Llama model configuration.")
    else:
        print("\nMinimal LoRA setup failed. There's a fundamental issue.") 