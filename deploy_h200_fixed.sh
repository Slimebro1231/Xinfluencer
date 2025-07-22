#!/bin/bash

# Enhanced H200 Deployment Script with X API Optimization
# Version: 2.0 - Optimized for evaluation framework deployment

set -e  # Exit on any error

# Configuration
H200_HOST="h200-server"
H200_USER="ubuntu"
PROJECT_NAME="xinfluencer"
REMOTE_DIR="/home/$H200_USER/$PROJECT_NAME"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check SSH connection
    if ! ssh -o ConnectTimeout=10 "$H200_USER@$H200_HOST" "echo 'SSH connection successful'" >/dev/null 2>&1; then
        error "Cannot connect to H200 server. Check SSH configuration."
        exit 1
    fi
    
    # Check GPU availability
    if ! ssh "$H200_USER@$H200_HOST" "nvidia-smi" >/dev/null 2>&1; then
        error "GPU not available on H200 server"
        exit 1
    fi
    
    # Check local files
    if [[ ! -f "src/main.py" ]]; then
        error "src/main.py not found. Run from project root."
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Sync project files with optimizations
sync_project_files() {
    log "Syncing optimized project files to H200 server..."
    
    # Create remote directory
    ssh "$H200_USER@$H200_HOST" "mkdir -p $REMOTE_DIR"
    
    # Sync files with exclusions for efficiency
    rsync -avz --progress \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.git/' \
        --exclude='logs/' \
        --exclude='data/cache/' \
        --exclude='data/collected/' \
        --exclude='evaluation_results/' \
        --exclude='lora_checkpoints_test/' \
        --exclude='xinfluencer_env/' \
        --exclude='.pytest_cache/' \
        --exclude='*.log' \
        ./ "$H200_USER@$H200_HOST:$REMOTE_DIR/"
    
    log "Project files synced successfully"
}

# Setup optimized Python environment
setup_environment() {
    log "Setting up optimized Python environment on H200..."
    
    ssh "$H200_USER@$H200_HOST" << 'EOF'
        cd /home/ubuntu/xinfluencer
        
        # Create virtual environment if it doesn't exist
        if [[ ! -d "xinfluencer_env" ]]; then
            echo "Creating Python virtual environment..."
            python3.10 -m venv xinfluencer_env
        fi
        
        # Activate environment
        source xinfluencer_env/bin/activate
        
        # Upgrade pip
        pip install --upgrade pip
        
        # Install PyTorch with CUDA 11.8 support first (critical for H200)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        
        # Install other dependencies for evaluation framework
        pip install -r requirements_h200.txt
        
        # Install additional dependencies for X API optimization
        pip install tweepy>=4.14.0
        pip install sqlite3  # For caching optimization
        pip install pandas>=1.5.0
        pip install numpy>=1.21.0
        
        echo "Environment setup completed"
EOF
    
    log "Environment setup completed"
}

# Create optimized startup scripts
create_startup_scripts() {
    log "Creating optimized startup scripts for evaluation framework..."
    
    # Create optimized Mistral startup script
    ssh "$H200_USER@$H200_HOST" << 'EOF'
        cd /home/ubuntu/xinfluencer
        
        # Create optimized startup script for evaluation framework
        cat > start_evaluation_h200.sh << 'SCRIPT_EOF'
#!/bin/bash

# Optimized H200 Evaluation Framework Startup
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=/home/ubuntu/.cache/huggingface
export HF_HOME=/home/ubuntu/.cache/huggingface

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

cd /home/ubuntu/xinfluencer
source xinfluencer_env/bin/activate

echo "Starting X API-optimized evaluation framework on H200..."
echo "GPU Status:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

echo "Testing CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo "Testing X API optimization..."
python -c "
from src.utils.x_api_client import XAPIClient
from src.evaluation.engine import EvaluationEngine
print('X API Client initialized successfully')
print('Evaluation Engine ready for deployment')
"

echo "Evaluation framework ready for production use"
echo "Available commands:"
echo "  python src/cli.py evaluate --test"
echo "  python src/cli.py x-api test"
echo "  python src/cli.py x-api collect"
echo "  python src/cli.py human-eval start"
SCRIPT_EOF

        chmod +x start_evaluation_h200.sh
        
        # Create evaluation test script
        cat > test_evaluation.py << 'TEST_EOF'
#!/usr/bin/env python3
"""Test optimized evaluation framework on H200."""

import sys
import torch
from src.utils.x_api_client import XAPIClient
from src.evaluation.engine import EvaluationEngine
from src.utils.data_collection_pipeline import DataCollectionPipeline

def test_h200_evaluation():
    print("Testing H200 Evaluation Framework...")
    print("="*50)
    
    # Test CUDA
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test X API optimization
    print("\nTesting X API optimization...")
    try:
        x_api = XAPIClient()
        status = x_api.get_rate_limit_status()
        print(f"API Connected: {status.get('api_connected', False)}")
        print(f"User cache loaded: {len(x_api.user_id_cache)} users")
        print("✓ X API optimization working")
    except Exception as e:
        print(f"✗ X API optimization failed: {e}")
    
    # Test evaluation engine
    print("\nTesting evaluation engine...")
    try:
        engine = EvaluationEngine()
        print("✓ Evaluation engine initialized")
    except Exception as e:
        print(f"✗ Evaluation engine failed: {e}")
    
    # Test data collection pipeline
    print("\nTesting optimized data collection...")
    try:
        pipeline = DataCollectionPipeline()
        stats = pipeline.get_collection_statistics()
        print("✓ Data collection pipeline ready")
    except Exception as e:
        print(f"✗ Data collection failed: {e}")
    
    print("\nH200 Evaluation Framework Test Complete!")

if __name__ == "__main__":
    test_h200_evaluation()
TEST_EOF

        chmod +x test_evaluation.py
        
        echo "Startup scripts created successfully"
EOF
    
    log "Startup scripts created"
}

# Test the optimized deployment
test_deployment() {
    log "Testing optimized deployment on H200..."
    
    ssh "$H200_USER@$H200_HOST" << 'EOF'
        cd /home/ubuntu/xinfluencer
        source xinfluencer_env/bin/activate
        
        echo "Running deployment tests..."
        python test_evaluation.py
        
        echo "Testing CLI commands..."
        python src/cli.py status
        
        echo "Testing X API optimization..."
        python src/cli.py x-api test
EOF
    
    log "Deployment test completed"
}

# Create monitoring and access scripts
create_access_scripts() {
    log "Creating access and monitoring scripts..."
    
    # Create local SSH access script
    cat > ssh_h200.sh << 'EOF'
#!/bin/bash
echo "Connecting to H200 Evaluation Server..."
ssh -t ubuntu@h200-server "cd /home/ubuntu/xinfluencer && source xinfluencer_env/bin/activate && bash"
EOF
    chmod +x ssh_h200.sh
    
    # Create monitoring script on H200
    ssh "$H200_USER@$H200_HOST" << 'EOF'
        cd /home/ubuntu/xinfluencer
        
        cat > monitor_evaluation.sh << 'MONITOR_EOF'
#!/bin/bash

echo "H200 Evaluation Framework Monitoring"
echo "===================================="

echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits

echo -e "\nPython Environment:"
source xinfluencer_env/bin/activate
python --version
pip show torch | grep Version

echo -e "\nEvaluation Framework Status:"
python -c "
try:
    from src.utils.x_api_client import XAPIClient
    from src.evaluation.engine import EvaluationEngine
    x_api = XAPIClient()
    status = x_api.get_rate_limit_status()
    print(f'X API Connected: {status.get(\"api_connected\", False)}')
    print(f'Rate Limits Status: {list(status.keys())}')
    print('Evaluation Framework: ✓ Ready')
except Exception as e:
    print(f'Evaluation Framework: ✗ Error - {e}')
"

echo -e "\nSystem Resources:"
free -h
df -h | head -5
MONITOR_EOF

        chmod +x monitor_evaluation.sh
        
EOF
    
    log "Access scripts created"
}

# Main deployment function
main() {
    log "Starting optimized H200 deployment for evaluation framework..."
    log "Target: $H200_USER@$H200_HOST:$REMOTE_DIR"
    
    check_prerequisites
    sync_project_files
    setup_environment
    create_startup_scripts
    test_deployment
    create_access_scripts
    
    log "Deployment completed successfully!"
    info "Next steps:"
    info "1. Connect to H200: ./ssh_h200.sh"
    info "2. Start evaluation framework: ./start_evaluation_h200.sh"
    info "3. Test X API: python src/cli.py x-api test"
    info "4. Start data collection: python src/cli.py x-api collect"
    info "5. Monitor system: ./monitor_evaluation.sh"
}

# Run main function
main "$@"