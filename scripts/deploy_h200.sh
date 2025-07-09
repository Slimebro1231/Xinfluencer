#!/bin/bash

# H200 Deployment Script for Xinfluencer AI
# This script deploys the complete AI system to the H200 server

set -e

# Configuration
H200_SERVER="157.10.162.127"
H200_USER="ubuntu"
PEM_FILE="$HOME/.ssh/id_rsa"
REMOTE_DIR="/home/ubuntu/xinfluencer"
PROJECT_NAME="xinfluencer"

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

print_status "Starting H200 deployment for $PROJECT_NAME..."

# Step 1: Test H200 connection
print_status "Testing H200 server connection..."
if ssh -i "$PEM_FILE" -o ConnectTimeout=10 -o BatchMode=yes "$H200_USER@$H200_SERVER" "echo 'Connection successful'"; then
    print_success "H200 server connection established"
else
    print_error "Failed to connect to H200 server"
    exit 1
fi

# Step 2: Check H200 GPU status
print_status "Checking H200 GPU status..."
ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "nvidia-smi"

# Step 3: Create remote directory structure
print_status "Setting up remote directory structure..."
ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "mkdir -p $REMOTE_DIR/{src,scripts,logs,models,data}"

# Step 4: Sync project files
print_status "Syncing project files to H200..."
rsync -avz --delete -e "ssh -i $PEM_FILE" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='logs/*' \
    --exclude='models/*' \
    --exclude='data/*' \
    ./ "$H200_USER@$H200_SERVER:$REMOTE_DIR/"

# Step 5: Install Python dependencies on H200
print_status "Installing Python dependencies on H200..."
ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "cd $REMOTE_DIR && pip3 install --upgrade pip"
ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "cd $REMOTE_DIR && pip3 install -r requirements.txt"

# Step 6: Install additional H200-specific packages
print_status "Installing H200-specific packages..."
ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "pip3 install flash-attn --no-build-isolation"

# Step 7: Set up environment variables
print_status "Setting up environment variables..."
ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "cd $REMOTE_DIR && touch .env"

# Step 8: Create startup script
print_status "Creating startup script..."
cat > /tmp/start_xinfluencer.sh << 'EOF'
#!/bin/bash

# Xinfluencer AI Startup Script for H200
cd /home/ubuntu/xinfluencer

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=/home/ubuntu/xinfluencer/models
export HF_HOME=/home/ubuntu/xinfluencer/models

# Check GPU status
echo "=== H200 GPU Status ==="
nvidia-smi

# Start the AI system
echo "=== Starting Xinfluencer AI ==="
python3 src/main.py

EOF

scp -i "$PEM_FILE" /tmp/start_xinfluencer.sh "$H200_USER@$H200_SERVER:$REMOTE_DIR/"
ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "chmod +x $REMOTE_DIR/start_xinfluencer.sh"

# Step 9: Create monitoring script
print_status "Creating monitoring script..."
cat > /tmp/monitor_h200.sh << 'EOF'
#!/bin/bash

# H200 Monitoring Script
echo "=== H200 System Status ==="
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv

echo -e "\n=== Process Status ==="
ps aux | grep python | grep -v grep

echo -e "\n=== Disk Usage ==="
df -h

echo -e "\n=== Memory Usage ==="
free -h

EOF

scp -i "$PEM_FILE" /tmp/monitor_h200.sh "$H200_USER@$H200_SERVER:$REMOTE_DIR/"
ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "chmod +x $REMOTE_DIR/monitor_h200.sh"

# Step 10: Test the deployment
print_status "Testing deployment..."
ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "cd $REMOTE_DIR && python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}\")'"

# Step 11: Create SSH access script
print_status "Creating SSH access script..."
cat > /tmp/ssh_h200.sh << EOF
#!/bin/bash

# SSH Access Script for H200
echo "Connecting to H200 server..."
echo "Server: $H200_SERVER"
echo "User: $H200_USER"
echo "Project Directory: $REMOTE_DIR"
echo ""
echo "Useful commands:"
echo "  cd $REMOTE_DIR                    # Go to project directory"
echo "  ./start_xinfluencer.sh           # Start the AI system"
echo "  ./monitor_h200.sh                # Monitor system status"
echo "  python3 src/main.py              # Run the main pipeline"
echo "  nvidia-smi                       # Check GPU status"
echo ""

ssh -i $PEM_FILE $H200_USER@$H200_SERVER
EOF

chmod +x /tmp/ssh_h200.sh
mv /tmp/ssh_h200.sh ./ssh_h200.sh

# Cleanup
rm -f /tmp/start_xinfluencer.sh /tmp/monitor_h200.sh

print_success "H200 deployment completed successfully!"
echo ""
print_status "Next steps:"
echo "1. SSH to H200: ./ssh_h200.sh"
echo "2. Start AI system: cd $REMOTE_DIR && ./start_xinfluencer.sh"
echo "3. Monitor system: ./monitor_h200.sh"
echo ""
print_status "The AI system is now ready to run on H200 with Mistral-7B!"