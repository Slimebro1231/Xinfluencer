#!/bin/bash

# Deploy and test evaluation framework on H200
# Excludes X API integration as requested

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
# See .ai_notes/h200_connection.md for H200 connection details
# Public IP: 157.10.162.127
# Private IP: 10.11.0.203
H200_IP="157.10.162.127"
H200_USER="ubuntu"
SSH_KEY="/Users/max/Xinfluencer/influencer.pem"
REMOTE_DIR="/home/ubuntu/Xinfluencer"
LOCAL_DIR="$(pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  H200 Evaluation Framework Deployment${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to print status
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

# Check if H200 IP is configured
if [ "$H200_IP" = "your_h200_server_ip" ]; then
    print_error "Please update H200_IP in this script with your actual H200 server IP"
    exit 1
fi

# Test H200 connection
print_status "Testing H200 connection..."
if ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$H200_USER@$H200_IP" "echo 'Connection successful'" 2>/dev/null; then
    print_success "H200 connection established"
else
    print_error "Failed to connect to H200 server"
    print_error "Please check:"
    print_error "  - H200 server IP: $H200_IP"
    print_error "  - SSH key: $SSH_KEY"
    print_error "  - Network connectivity"
    exit 1
fi

# Get H200 system info
print_status "Getting H200 system information..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_IP" << 'EOF'
echo "=== H200 System Information ==="
echo "Hostname: $(hostname)"
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "CPU: $(nproc) cores"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
echo "CUDA Version: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
echo "================================="
EOF

# Create remote directory structure
print_status "Setting up remote directory structure..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_IP" << EOF
mkdir -p $REMOTE_DIR
mkdir -p $REMOTE_DIR/src
mkdir -p $REMOTE_DIR/scripts
mkdir -p $REMOTE_DIR/evaluation_results
mkdir -p $REMOTE_DIR/logs
EOF

# Copy source code to H200
print_status "Copying source code to H200..."
rsync -avz --progress \
    --exclude '*.pyc' \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude 'xinfluencer_env' \
    --exclude 'data/scraped' \
    --exclude 'logs' \
    --exclude 'lora_checkpoints_test' \
    -e "ssh -i $SSH_KEY" \
    "$LOCAL_DIR/src/" "$H200_USER@$H200_IP:$REMOTE_DIR/src/"

# Copy scripts
print_status "Copying deployment scripts..."
rsync -avz --progress \
    -e "ssh -i $SSH_KEY" \
    "$LOCAL_DIR/scripts/deploy_evaluation_h200.py" "$H200_USER@$H200_IP:$REMOTE_DIR/scripts/"

# Copy requirements
print_status "Copying requirements..."
rsync -avz --progress \
    -e "ssh -i $SSH_KEY" \
    "$LOCAL_DIR/requirements_h200.txt" "$H200_USER@$H200_IP:$REMOTE_DIR/"

# Install dependencies on H200
print_status "Installing Python dependencies on H200..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_IP" << 'EOF'
cd /home/ubuntu/Xinfluencer

# Check if virtual environment exists
if [ ! -d "xinfluencer_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv xinfluencer_env
fi

# Activate virtual environment and install dependencies
source xinfluencer_env/bin/activate
pip install --upgrade pip
pip install -r requirements_h200.txt

echo "Dependencies installed successfully"
EOF

# Run evaluation framework tests
print_status "Running evaluation framework tests on H200..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_IP" << 'EOF'
cd /home/ubuntu/Xinfluencer
source xinfluencer_env/bin/activate

echo "Starting evaluation framework deployment test..."
python scripts/deploy_evaluation_h200.py

echo "Test completed. Check evaluation_results/ directory for results."
EOF

# Download test results
print_status "Downloading test results..."
rsync -avz --progress \
    -e "ssh -i $SSH_KEY" \
    "$H200_USER@$H200_IP:$REMOTE_DIR/evaluation_results/" "$LOCAL_DIR/evaluation_results/"

# Display results summary
print_status "Test Results Summary:"
if [ -f "evaluation_results/h200_evaluation_summary_*.json" ]; then
    latest_summary=$(ls -t evaluation_results/h200_evaluation_summary_*.json | head -1)
    if [ -n "$latest_summary" ]; then
        echo "Latest summary: $latest_summary"
        cat "$latest_summary" | python3 -m json.tool
    fi
else
    print_warning "No summary file found. Check evaluation_results/ directory."
fi

# Check for any error logs
print_status "Checking for error logs..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_IP" << 'EOF'
cd /home/ubuntu/Xinfluencer
if [ -f "logs/evaluation_deployment.log" ]; then
    echo "=== Recent Log Entries ==="
    tail -20 logs/evaluation_deployment.log
    echo "=========================="
fi
EOF

print_success "H200 evaluation framework deployment completed!"
print_status "Next steps:"
print_status "1. Review test results in evaluation_results/ directory"
print_status "2. Check logs for any issues"
print_status "3. Proceed with production deployment if all tests pass"
print_status "4. Integrate X API when ready"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Deployment Complete${NC}"
echo -e "${BLUE}========================================${NC}" 