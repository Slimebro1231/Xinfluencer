#!/usr/bin/env python3
import json

# Load training data
with open('lora_checkpoints/proper_identity/training_data.json', 'r') as f:
    data = json.load(f)

print(f"Total training examples: {len(data)}")
print("\nTraining examples:")
for i, item in enumerate(data):
    print(f"{i+1}. {item['text']}") 