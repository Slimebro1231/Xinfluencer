#!/bin/bash

# H200 Mistral Deployment Script - Fixed Version
# This script deploys the Mistral model to H200 with proper error handling

set -e

# Configuration
H200_SERVER="157.10.162.127"
H200_USER="ubuntu"
PEM_FILE="$HOME/.ssh/id_rsa"
REMOTE_DIR="/home/ubuntu/xinfluencer"
PROJECT_NAME="xinfluencer"
VENV_NAME="xinfluencer_env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if SSH key exists
if [ ! -f "$PEM_FILE" ]; then
    print_error "SSH key not found at $PEM_FILE"
    print_status "Please ensure your SSH key is properly configured"
    exit 1
fi

print_status "Starting H200 Mistral deployment (Fixed Version)..."

# Step 1: Test H200 connection
print_status "Testing H200 server connection..."
if ssh -i "$PEM_FILE" -o ConnectTimeout=10 -o BatchMode=yes "$H200_USER@$H200_SERVER" "echo 'Connection successful'" > /dev/null 2>&1; then
    print_success "H200 server connection established"
else
    print_error "Failed to connect to H200 server"
    print_status "Please check your SSH configuration and server availability"
    exit 1
fi

# Step 2: Check H200 GPU status
print_status "Checking H200 GPU status..."
GPU_STATUS=$(ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits" 2>/dev/null || echo "GPU_ERROR")

if [[ "$GPU_STATUS" == "GPU_ERROR" ]]; then
    print_error "Failed to get GPU status. Is nvidia-smi available?"
    exit 1
else
    print_success "GPU Status: $GPU_STATUS"
fi

# Step 3: Setup remote directory and virtual environment
print_status "Setting up remote environment..."

ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" << 'EOF'
set -e

# Create directory structure
mkdir -p /home/ubuntu/xinfluencer/{src,scripts,logs,models,data}

# Check if virtual environment exists
if [ ! -d "/home/ubuntu/xinfluencer/xinfluencer_env" ]; then
    echo "Creating new virtual environment..."
    cd /home/ubuntu/xinfluencer
    python3 -m venv xinfluencer_env
    echo "Virtual environment created successfully"
else
    echo "Virtual environment already exists, skipping creation"
fi

# Activate and upgrade pip
source /home/ubuntu/xinfluencer/xinfluencer_env/bin/activate
pip install --upgrade pip wheel setuptools

echo "Environment setup completed"
EOF

print_success "Remote environment setup completed"

# Step 4: Sync project files
print_status "Syncing project files to H200..."
rsync -avz --delete -e "ssh -i $PEM_FILE" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='logs/*' \
    --exclude='models/*' \
    --exclude='data/*' \
    --exclude='xinfluencer_env' \
    ./ "$H200_USER@$H200_SERVER:$REMOTE_DIR/"

print_success "Project files synced"

# Step 5: Install dependencies with proper error handling
print_status "Installing Python dependencies..."

ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" << 'EOF'
set -e

cd /home/ubuntu/xinfluencer
source xinfluencer_env/bin/activate

echo "Installing PyTorch with CUDA support..."
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing transformers and related packages..."
pip install transformers>=4.30.0 accelerate>=0.20.0 bitsandbytes>=0.41.0

echo "Installing other dependencies..."
pip install sentence-transformers>=2.2.0 peft>=0.4.0

echo "Installing vector database and utilities..."
pip install qdrant-client>=1.3.0 pandas>=2.0.0 numpy>=1.24.0

echo "Installing configuration packages..."
pip install pydantic>=2.0.0 pydantic-settings>=2.0.0 python-dotenv>=1.0.0

echo "Installing additional utilities..."
pip install tqdm>=4.65.0 requests>=2.31.0

# Verify installations
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}');"

echo "Verifying transformers installation..."
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}');"

echo "Dependencies installed successfully"
EOF

print_success "Dependencies installed successfully"

# Step 6: Create optimized startup script
print_status "Creating optimized startup script..."

cat > /tmp/start_mistral_h200.sh << 'EOF'
#!/bin/bash

# Mistral H200 Startup Script - Optimized
cd /home/ubuntu/xinfluencer
source xinfluencer_env/bin/activate

# Set environment variables for H200 optimization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export TRANSFORMERS_CACHE=/home/ubuntu/xinfluencer/models
export HF_HOME=/home/ubuntu/xinfluencer/models
export TOKENIZERS_PARALLELISM=false

# Create models directory
mkdir -p models

echo "=== H200 GPU Status ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu --format=csv

echo "=== Testing PyTorch CUDA ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    # Test CUDA computation
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.mm(x, y)
    print(f'CUDA computation test: OK ({z.shape})')
    del x, y, z
    torch.cuda.empty_cache()
else:
    print('CUDA not available')
"

echo "=== Starting Mistral Model Test ==="
python3 -c "
import sys
sys.path.insert(0, 'src')

try:
    from model.generate import TextGenerator
    print('Testing Mistral model loading...')
    
    # Initialize with quantization for memory efficiency
    generator = TextGenerator(model_name='mistralai/Mistral-7B-v0.1', use_quantization=True)
    print('✅ Mistral model loaded successfully')
    
    # Test generation
    response = generator.generate_response('What is Bitcoin?', max_new_tokens=50)
    print(f'✅ Test generation: {response[:100]}...')
    
    # Check memory usage
    memory = generator.get_memory_usage()
    if 'allocated_gb' in memory:
        print(f'✅ Memory usage: {memory[\"allocated_gb\"]:.1f}/{memory[\"total_gb\"]:.1f} GB')
    
    print('🎉 Mistral deployment successful!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"

echo "=== Mistral H200 System Ready ==="
EOF

scp -i "$PEM_FILE" /tmp/start_mistral_h200.sh "$H200_USER@$H200_SERVER:$REMOTE_DIR/"
ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "chmod +x $REMOTE_DIR/start_mistral_h200.sh"

# Step 7: Create model testing script
print_status "Creating model testing script..."

cat > /tmp/test_mistral.py << 'EOF'
#!/usr/bin/env python3
"""Test Mistral model on H200."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_mistral_basic():
    """Test basic Mistral functionality."""
    print("🧪 Testing Mistral Model on H200")
    print("=" * 50)
    
    try:
        # Import and initialize
        from model.generate import TextGenerator
        
        print("1. Loading Mistral-7B model...")
        start_time = time.time()
        
        generator = TextGenerator(
            model_name="mistralai/Mistral-7B-v0.1",
            use_quantization=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        
        # Test memory usage
        memory = generator.get_memory_usage()
        if 'error' not in str(memory):
            print(f"💾 GPU Memory: {memory['allocated_gb']:.1f}/{memory['total_gb']:.1f} GB")
        
        # Test generation
        print("\n2. Testing generation...")
        test_prompts = [
            "What is Bitcoin?",
            "Explain DeFi in simple terms.",
            "What are the risks of cryptocurrency investment?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt}")
            start_time = time.time()
            
            response = generator.generate_response(prompt, max_new_tokens=100)
            
            gen_time = time.time() - start_time
            print(f"Generated in {gen_time:.1f}s: {response[:150]}...")
        
        print("\n✅ All tests passed! Mistral is ready for production.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mistral_basic()
    sys.exit(0 if success else 1)
EOF

scp -i "$PEM_FILE" /tmp/test_mistral.py "$H200_USER@$H200_SERVER:$REMOTE_DIR/"
ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "chmod +x $REMOTE_DIR/test_mistral.py"

# Step 8: Create CLI wrapper script
print_status "Creating CLI wrapper script..."

cat > /tmp/mistral_cli.sh << 'EOF'
#!/bin/bash

# Mistral CLI Wrapper
cd /home/ubuntu/xinfluencer
source xinfluencer_env/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=/home/ubuntu/xinfluencer/models
export HF_HOME=/home/ubuntu/xinfluencer/models

# Run CLI with arguments
python3 src/cli.py "$@"
EOF

scp -i "$PEM_FILE" /tmp/mistral_cli.sh "$H200_USER@$H200_SERVER:$REMOTE_DIR/"
ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "chmod +x $REMOTE_DIR/mistral_cli.sh"

# Cleanup temp files
rm -f /tmp/start_mistral_h200.sh /tmp/test_mistral.py /tmp/mistral_cli.sh

print_success "H200 Mistral deployment completed successfully!"
echo ""
print_status "Next steps:"
echo "1. SSH to H200: ssh -i $PEM_FILE $H200_USER@$H200_SERVER"
echo "2. Test deployment: cd $REMOTE_DIR && ./start_mistral_h200.sh"
echo "3. Run model test: python3 test_mistral.py"
echo "4. Use CLI: ./mistral_cli.sh interactive"
echo ""
print_status "🎉 Mistral-7B is now ready on H200 with proper isolation and error handling!"