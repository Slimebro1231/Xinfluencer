#!/usr/bin/env python3
"""
H200 Setup Test Script
Tests the H200 GPU setup and model loading capabilities
"""

import torch
import json
import gc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.generate import TextGenerator

def run_test(test_fn, name):
    print(f"\n=== Testing {name} ===")
    try:
        result = test_fn()
        print(f"✅ {name} test passed")
        return True, result
    except Exception as e:
        print(f"❌ {name} test failed: {e}")
        return False, str(e)

def test_cuda_setup():
    assert torch.cuda.is_available(), "CUDA is not available"
    assert torch.cuda.device_count() > 0, "No GPU found"
    return {"cuda_available": True, "gpu_name": torch.cuda.get_device_name(0)}

def test_model_loading():
    generator = TextGenerator()
    assert generator.model is not None, "Model failed to load"
    assert generator.tokenizer is not None, "Tokenizer failed to load"
    return {"model_name": generator.model_name}

def test_generation():
    generator = TextGenerator()
    prompt = "Hello, world!"
    response = generator.generate(prompt, max_length=20)
    assert isinstance(response, str) and len(response) > len(prompt)
    return {"prompt": prompt, "response": response}

def main():
    results = {}
    passed_all = True

    tests = {
        "CUDA Setup": test_cuda_setup,
        "Model Loading": test_model_loading,
        "Generation": test_generation,
    }

    for name, test_fn in tests.items():
        passed, result = run_test(test_fn, name)
        results[name] = {"passed": passed, "details": result}
        if not passed:
            passed_all = False

    print("\n====================")
    print("  Test Summary")
    print("====================")
    for name, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{status} - {name}")

    with open("h200_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if not passed_all:
        print("\n⚠️  Some tests failed.")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")

if __name__ == "__main__":
    main()