#!/usr/bin/env python3
"""
H200 Setup Test Script
Tests the H200 GPU setup and model loading capabilities
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
import torch
import json
import gc
from transformers import AutoTokenizer, BitsAndBytesConfig, GPT2LMHeadModel, MistralForCausalLM

def test_cuda_setup():
    """Test CUDA and PyTorch setup."""
    print("=== Testing CUDA Setup ===")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            print(f"✅ GPU name: {torch.cuda.get_device_name()}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Test GPU computation
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print(f"✅ GPU computation test passed: {z.shape}")
            
            return True
        else:
            print("❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_model_loading():
    """Test model loading capabilities."""
    print("\n=== Testing Model Loading ===")
    
    try:
        from transformers import AutoTokenizer, BitsAndBytesConfig, GPT2LMHeadModel, MistralForCausalLM
        import torch
        
        # Test with a smaller model first
        model_name = "microsoft/DialoGPT-medium"
        print(f"Testing with: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer loaded")
        
        # Test quantization config
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print("✅ Quantization config created")
        else:
            quantization_config = None
        
        try:
            if "mistral" in model_name.lower():
                model = MistralForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                model = GPT2LMHeadModel.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            print("✅ Model loaded successfully")
            return model
        except Exception as e:
            print(f"❌ Model loading test failed: {e}")
            return None
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def test_mistral_loading():
    """Test Mistral-7B loading (will be slow)."""
    print("\n=== Testing Mistral-7B Loading ===")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        
        model_name = "mistralai/Mistral-7B-v0.1"
        print(f"Testing with: {model_name}")
        print("⚠️  This may take several minutes...")
        
        start_time = time.time()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Mistral tokenizer loaded")
        
        # Test quantization config
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print("✅ Mistral quantization config created")
        else:
            quantization_config = None
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Mistral model loaded successfully in {load_time:.1f} seconds")
        
        # Test generation
        prompt = "<s>[INST] What is Bitcoin? [/INST]"
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=50, temperature=0.7)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Mistral generation test passed: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Mistral loading test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage monitoring."""
    print("\n=== Testing Memory Usage ===")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"✅ Memory monitoring:")
            print(f"   Allocated: {allocated:.2f} GB")
            print(f"   Reserved: {reserved:.2f} GB")
            print(f"   Total: {total:.2f} GB")
            print(f"   Free: {total - reserved:.2f} GB")
            
            return True
        else:
            print("❌ CUDA not available for memory test")
            return False
            
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 H200 Setup Test Suite")
    print("==================================================")
    print("Python Path:")
    print(sys.path)
    print("==================================================")
    
    results = {}
    
    # Run tests
    results['cuda'] = test_cuda_setup()
    results['model_loading'] = test_model_loading()
    results['memory'] = test_memory_usage()
    
    # Only test Mistral if other tests pass
    if all(results.values()):
        results['mistral'] = test_mistral_loading()
    else:
        results['mistral'] = False
        print("\n⚠️  Skipping Mistral test due to previous failures")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("🎉 All tests passed! H200 setup is ready for production.")
    else:
        print("⚠️  Some tests failed. Please check the setup.")
    
    # Save results
    with open("h200_test_results.json", "w") as f:
        json.dump({
            "timestamp": time.time(),
            "results": results,
            "summary": {
                "passed": passed_count,
                "total": total_count,
                "success_rate": passed_count / total_count
            }
        }, f, indent=2)
    
    print(f"\n📄 Results saved to: h200_test_results.json")

if __name__ == "__main__":
    main()