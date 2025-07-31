#!/usr/bin/env python3
"""
Test script to verify data cleaning fixes work properly.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.identity_pipeline import IdentityTrainingPipeline
from src.model.lora import LoRAFineTuner

def test_data_cleaning():
    """Test the data cleaning functionality."""
    print("Testing data cleaning functionality...")
    
    # Test the cleaning function directly
    lora_trainer = LoRAFineTuner()
    
    # Test cases with problematic content
    test_cases = [
        {
            "input": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWho are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI'm Soju, a crypto influencer AI created by Max.",
            "expected_clean": "I'm Soju, a crypto influencer AI created by Max."
        },
        {
            "input": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nCreate an influential crypto tweet.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nShort Bitcoin if you hate money.",
            "expected_clean": "Short Bitcoin if you hate money."
        },
        {
            "input": "Bitcoin is Freedom https://t.co/iQuKYin3Jb",
            "expected_clean": "Bitcoin is Freedom"
        },
        {
            "input": "You are Soju, an AI crypto influencer. Your mission is to become a leading voice in RWA. Bitcoin is the future.",
            "expected_clean": "Bitcoin is the future."
        }
    ]
    
    print("\nTesting individual cleaning cases:")
    for i, case in enumerate(test_cases):
        cleaned = lora_trainer._clean_tweet_content(case["input"])
        print(f"\nTest {i+1}:")
        print(f"  Input: {case['input'][:100]}...")
        print(f"  Cleaned: {cleaned}")
        print(f"  Expected: {case['expected_clean']}")
        print(f"  Match: {cleaned == case['expected_clean']}")
    
    # Test the full pipeline
    print("\nTesting full pipeline data preparation...")
    pipeline = IdentityTrainingPipeline()
    
    # Load existing training data
    training_data_file = "lora_checkpoints/proper_identity/training_data.json"
    if Path(training_data_file).exists():
        with open(training_data_file, 'r') as f:
            old_data = json.load(f)
        
        print(f"Loaded {len(old_data)} examples from existing training data")
        
        # Test cleaning on first few examples
        print("\nTesting cleaning on existing data:")
        for i, item in enumerate(old_data[:5]):
            original_text = item.get('text', '')
            cleaned_text = lora_trainer._clean_tweet_content(original_text)
            print(f"\nExample {i+1}:")
            print(f"  Original: {original_text[:100]}...")
            print(f"  Cleaned: {cleaned_text}")
            print(f"  Length: {len(cleaned_text)}")
            print(f"  Valid: {len(cleaned_text) > 10 and len(cleaned_text) < 280}")
    
    print("\nData cleaning test completed!")

if __name__ == "__main__":
    test_data_cleaning() 