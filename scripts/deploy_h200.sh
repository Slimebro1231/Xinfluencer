#!/bin/bash

# H200 Deployment Script for Xinfluencer AI
# This script sets up the complete environment on an H200 GPU server

set -e  # Exit on any error

echo "Starting H200 deployment for Xinfluencer AI..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if we're on the right system
check_system() {
    print_status "Checking system requirements..."
    
    # Check if CUDA is available
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "NVIDIA GPU not detected. Please ensure CUDA drivers are installed."
        exit 1
    fi
    
    # Check GPU memory
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$GPU_MEMORY" -lt 80000 ]; then  # Less than 80GB
        print_warning "GPU memory is ${GPU_MEMORY}MB. H200 should have 80GB+ for optimal performance."
    else
        print_success "GPU memory: ${GPU_MEMORY}MB (sufficient for H200)"
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_status "Python version: $PYTHON_VERSION"
    
    # Check if we're in a virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_success "Running in virtual environment: $VIRTUAL_ENV"
    else
        print_warning "Not in a virtual environment. Consider creating one."
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Update package list
    sudo apt-get update
    
    # Install essential packages
    sudo apt-get install -y \
        python3-pip \
        python3-venv \
        git \
        curl \
        wget \
        build-essential \
        cmake \
        pkg-config \
        libssl-dev \
        libffi-dev \
        python3-dev
    
    print_success "System dependencies installed"
}

# Setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment if not exists
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        python3 -m venv xinfluencer_env
        source xinfluencer_env/bin/activate
        print_success "Created virtual environment: xinfluencer_env"
    fi
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch with CUDA support
    print_status "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install other requirements
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    print_success "Python environment setup complete"
}

# Setup configuration
setup_config() {
    print_status "Setting up configuration..."
    
    # Check if .env exists
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from template..."
        cp env.example .env
        print_warning "Please update .env with your Twitter API credentials and other settings"
    else
        print_success ".env file found"
    fi
    
    # Create necessary directories
    mkdir -p logs
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p models/checkpoints
    mkdir -p models/lora
    
    print_success "Configuration setup complete"
}

# Test GPU setup
test_gpu_setup() {
    print_status "Testing GPU setup..."
    
    # Test CUDA availability
    python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.current_device()}')
    print(f'GPU name: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
    
    print_success "GPU setup test complete"
}

# Run performance tests
run_performance_tests() {
    print_status "Running H200 performance tests..."
    
    # Run the performance test suite
    python3 tests/test_h200_performance.py
    
    print_success "Performance tests completed"
}

# Setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring..."
    
    # Create monitoring directory
    mkdir -p monitoring/metrics
    
    # Create basic monitoring script
    cat > monitoring/gpu_monitor.py << 'EOF'
#!/usr/bin/env python3
"""
GPU monitoring script for H200
"""

import psutil
import torch
import time
import json
from datetime import datetime

def get_gpu_stats():
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated(0) / 1e9
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_utilization = torch.cuda.utilization(0)
        
        return {
            "gpu_memory_used_gb": gpu_memory_used,
            "gpu_memory_total_gb": gpu_memory_total,
            "gpu_memory_percent": (gpu_memory_used / gpu_memory_total) * 100,
            "gpu_utilization_percent": gpu_utilization
        }
    return {}

def get_system_stats():
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }

def main():
    while True:
        stats = {
            "timestamp": datetime.now().isoformat(),
            "system": get_system_stats(),
            "gpu": get_gpu_stats()
        }
        
        # Save to file
        with open("monitoring/metrics/current_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        time.sleep(30)  # Update every 30 seconds

if __name__ == "__main__":
    main()
EOF
    
    print_success "Monitoring setup complete"
}

# Create startup script
create_startup_script() {
    print_status "Creating startup script..."
    
    cat > start_xinfluencer.sh << 'EOF'
#!/bin/bash

# Startup script for Xinfluencer AI on H200

echo "Starting Xinfluencer AI..."

# Activate virtual environment
source xinfluencer_env/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Start monitoring in background
python3 monitoring/gpu_monitor.py &
MONITOR_PID=$!

# Start main application
python3 src/main.py

# Cleanup
kill $MONITOR_PID
EOF
    
    chmod +x start_xinfluencer.sh
    print_success "Startup script created: start_xinfluencer.sh"
}

# Main deployment function
main() {
    echo "H200 Deployment for Xinfluencer AI"
    echo "====================================="
    
    check_system
    install_system_deps
    setup_python_env
    setup_config
    test_gpu_setup
    run_performance_tests
    setup_monitoring
    create_startup_script
    
    echo ""
    echo "Deployment completed successfully"
    echo ""
    echo "Next steps:"
    echo "1. Update .env with your Twitter API credentials"
    echo "2. Run: ./start_xinfluencer.sh"
    echo "3. Monitor performance with: tail -f monitoring/metrics/current_stats.json"
    echo ""
    echo "For performance results, check:"
    echo "- h200_performance_results_*.json"
    echo "- h200_performance_summary_*.txt"
}

# Run main function
main "$@" 