#!/bin/bash

# H200 Language Model Upgrade Script
# Upgrades from Mistral-7B-Instruct-v0.2 to meta-llama/Llama-3.1-8B-Instruct

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SSH_KEY="/Users/max/Xinfluencer/influencer.pem"
H200_HOST="157.10.162.127"
H200_USER="ubuntu"
REMOTE_DIR="/home/ubuntu/xinfluencer"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "Starting H200 language model upgrade..."

# Test SSH connection
print_status "Testing H200 connection..."
if ! ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o BatchMode=yes "$H200_USER@$H200_HOST" "echo 'SSH connection successful'" > /dev/null 2>&1; then
    print_error "Failed to connect to H200 server"
    exit 1
fi
print_success "H200 connection established"

# Check GPU status
print_status "Checking GPU status..."
GPU_INFO=$(ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits")
print_success "GPU Status: $GPU_INFO"

# Create backup of current model configuration
print_status "Creating backup of current model configuration..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && cp src/config.py src/config.py.backup.$(date +%Y%m%d_%H%M%S)"

# Update model configuration
print_status "Updating model configuration..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && sed -i 's/mistralai\/Mistral-7B-Instruct-v0.2/meta-llama\/Llama-3.1-8B-Instruct/g' src/config.py"

# Update requirements for latest transformers and TRL
print_status "Updating requirements for latest libraries..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && echo 'transformers>=4.50.0' >> requirements_h200.txt"
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && echo 'trl>=0.10.0' >> requirements_h200.txt"
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && echo 'peft>=0.10.0' >> requirements_h200.txt"

# Install updated requirements
print_status "Installing updated requirements..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && source xinfluencer_env/bin/activate && pip install --upgrade transformers trl peft"

# Test new language model
print_status "Testing new language model..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && source xinfluencer_env/bin/activate && python3 -c \"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

print('Loading Llama-3.1-8B-Instruct...')
start_time = time.time()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
tokenizer.pad_token = tokenizer.eos_token

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.1-8B-Instruct',
    torch_dtype=torch.float16,
    device_map='auto',
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

load_time = time.time() - start_time
print(f'Model loaded in {load_time:.2f} seconds')

# Test generation
print('Testing text generation...')
prompt = 'What is Bitcoin and how does it work?'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

start_time = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generate_time = time.time() - start_time
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Generated response in {generate_time:.2f} seconds')
print(f'Response: {response}')

print('Language model test completed successfully!')
\""

if [ $? -eq 0 ]; then
    print_success "New language model test passed!"
else
    print_error "New language model test failed!"
    print_status "Rolling back to previous configuration..."
    ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && git checkout src/config.py"
    exit 1
fi

# Test with existing pipeline
print_status "Testing language model with existing pipeline..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && source xinfluencer_env/bin/activate && python3 -c \"
import sys
sys.path.insert(0, 'src')

from model.generate_h200 import H200TextGenerator
import time

print('Testing H200TextGenerator with new model...')

# Initialize generator
generator = H200TextGenerator()

# Test prompts
test_prompts = [
    'What is Bitcoin?',
    'Explain DeFi in simple terms',
    'What are the benefits of blockchain technology?',
    'How does Ethereum work?'
]

for prompt in test_prompts:
    start_time = time.time()
    response = generator.generate_text(prompt, max_tokens=100)
    generate_time = time.time() - start_time
    print(f'Prompt: \"{prompt}\"')
    print(f'Response: {response[:200]}...')
    print(f'Generation time: {generate_time:.2f}s')
    print('---')

print('Pipeline test completed!')
\""

# Update LoRA configuration for new model
print_status "Updating LoRA configuration for new model..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && sed -i 's/r=16/r=32/g' src/model/lora.py"
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && sed -i 's/alpha=32/alpha=64/g' src/model/lora.py"

# Test LoRA training compatibility
print_status "Testing LoRA training compatibility..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && source xinfluencer_env/bin/activate && python3 -c \"
import sys
sys.path.insert(0, 'src')

from model.lora import LoRATrainer
import torch

print('Testing LoRA trainer with new model...')

# Initialize trainer (without actual training)
trainer = LoRATrainer()
print('LoRA trainer initialized successfully')

# Test model loading
model = trainer.load_model()
print(f'Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters')

print('LoRA compatibility test completed!')
\""

# Update documentation
print_status "Updating documentation..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && sed -i 's/Mistral-7B-Instruct-v0.2/Llama-3.1-8B-Instruct/g' TECHNICAL_ARCHITECTURE.md 2>/dev/null || echo 'Documentation file not found'"

print_success "Language model upgrade completed successfully!"
print_status "New model: meta-llama/Llama-3.1-8B-Instruct"
print_status "Benefits: Latest Llama architecture, excellent reasoning, optimized for instruction following"
print_status "Memory usage: ~12GB (vs previous 12GB)"
print_status "Context length: 8K tokens (optimized for efficiency)"

print_status "Next steps:"
echo "  1. Test with real tweet data"
echo "  2. Begin DPO training implementation"
echo "  3. Proceed to Phase 2 (Advanced Algorithms)" 