#!/bin/bash

# H200 Deployment Validation Script
# Run this before deploying to catch issues early

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[CHECK]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

echo "🔍 H200 Deployment Validation"
echo "=============================="

ISSUES=0

# Check 1: SSH key exists
print_status "Checking SSH key..."
if [ -f "$HOME/.ssh/id_rsa" ]; then
    print_success "SSH key found at ~/.ssh/id_rsa"
else
    print_error "SSH key not found at ~/.ssh/id_rsa"
    echo "       Please ensure your SSH key is configured for H200 access"
    ISSUES=$((ISSUES + 1))
fi

# Check 2: Required files exist
print_status "Checking required files..."
REQUIRED_FILES=(
    "deploy_h200_fixed.sh"
    "requirements_h200.txt"
    "src/model/generate_h200.py"
    "src/config.py"
    "src/cli.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "Found: $file"
    else
        print_error "Missing: $file"
        ISSUES=$((ISSUES + 1))
    fi
done

# Check 3: Deployment script is executable
print_status "Checking deployment script permissions..."
if [ -x "deploy_h200_fixed.sh" ]; then
    print_success "Deployment script is executable"
else
    print_warning "Making deployment script executable..."
    chmod +x deploy_h200_fixed.sh
    print_success "Fixed: deployment script permissions"
fi

# Check 4: H200 connectivity (optional test)
print_status "Testing H200 connectivity..."
H200_SERVER="157.10.162.127"
H200_USER="ubuntu"
PEM_FILE="$HOME/.ssh/id_rsa"

if timeout 10 ssh -i "$PEM_FILE" -o ConnectTimeout=5 -o BatchMode=yes "$H200_USER@$H200_SERVER" "echo 'Connection test'" >/dev/null 2>&1; then
    print_success "H200 server is reachable"
    
    # Quick GPU check
    GPU_STATUS=$(ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "nvidia-smi --query-gpu=name --format=csv,noheader" 2>/dev/null || echo "UNKNOWN")
    if [[ "$GPU_STATUS" == "UNKNOWN" ]]; then
        print_warning "Could not verify GPU status"
    else
        print_success "GPU detected: $GPU_STATUS"
    fi
else
    print_warning "H200 server not reachable (check network/VPN)"
    echo "          This is not critical - you can still deploy later"
fi

# Check 5: Local utilities
print_status "Checking local utilities..."
REQUIRED_UTILS=("rsync" "ssh" "scp")

for util in "${REQUIRED_UTILS[@]}"; do
    if command -v "$util" >/dev/null 2>&1; then
        print_success "Found: $util"
    else
        print_error "Missing: $util"
        ISSUES=$((ISSUES + 1))
    fi
done

# Check 6: Disk space estimation
print_status "Checking available disk space..."
AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
print_success "Available disk space: $AVAILABLE_SPACE"

# Summary
echo ""
echo "=============================="
echo "📋 Validation Summary"
echo "=============================="

if [ $ISSUES -eq 0 ]; then
    print_success "All checks passed! Ready for H200 deployment"
    echo ""
    echo "🚀 Next steps:"
    echo "1. Run: ./deploy_h200_fixed.sh"
    echo "2. Wait for deployment to complete (~10-15 minutes)"
    echo "3. SSH to H200 and test: ./start_mistral_h200.sh"
    echo ""
    exit 0
else
    print_error "Found $ISSUES issues that need to be resolved"
    echo ""
    echo "❌ Please fix the issues above before deploying"
    echo ""
    exit 1
fi