
#!/bin/bash

# H200 Mistral Deployment Script - Fixed Version 3
# This script deploys the Mistral model to H200 with proper error handling,
# correct SSH key path, and uses python3 for remote commands.

set -e

# Configuration
H200_SERVER="157.10.162.127"
H200_USER="ubuntu"
PEM_FILE="/Users/max/Xinfluencer/influencer.pem" # CORRECTED PATH
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
    print_status "Please ensure your 'influencer.pem' key is in the project root"
    exit 1
fi

print_status "Starting H200 Mistral deployment (Fixed Version 3)..."

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
mkdir -p /home/ubuntu/xinfluencer/{src,scripts,logs,models,data/seed_tweets}

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
    --exclude='xinfluencer_env' \
    ./ "$H200_USER@$H200_SERVER:$REMOTE_DIR/"

print_success "Project files synced"

# Step 5: Install dependencies with proper error handling
print_status "Installing Python dependencies..."

ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "
set -e
cd $REMOTE_DIR
source $VENV_NAME/bin/activate
python3 -m pip install -r requirements.txt
"

print_success "Dependencies installed successfully"

# Step 6: Verify installations
print_status "Verifying remote installations..."
ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "
set -e
cd $REMOTE_DIR
source $VENV_NAME/bin/activate
echo '--- Verifying PyTorch ---'
python3 -c 'import torch; print(f\"PyTorch Version: {torch.__version__}\"); print(f\"CUDA Available: {torch.cuda.is_available()}\")'
echo '--- Verifying Pydantic ---'
python3 -c 'import pydantic; print(f\"Pydantic Version: {pydantic.__version__}\")'
"
print_success "Installations verified."

print_status "Deployment script finished successfully."
echo "You can now SSH into the server to run tests:"
echo "ssh -i $PEM_FILE $H200_USER@$H200_SERVER" 