#!/bin/bash

# Run H200 Verification Script
# This script runs the verification on the H200 server

set -e

# Configuration
H200_SERVER="157.10.162.127"
H200_USER="ubuntu"
PEM_FILE="/Users/max/Xinfluencer/influencer.pem"
REMOTE_DIR="/home/ubuntu/Xinfluencer"

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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Sync verification script
sync_verification_script() {
    print_status "Syncing verification script to H200 server..."
    
    scp -i "$PEM_FILE" scripts/test_h200_setup.py "$H200_USER@$H200_SERVER:$REMOTE_DIR/scripts/"
    
    print_success "Verification script synced"
}

# Run verification
run_verification() {
    print_status "Running verification on H200 server..."
    
    ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "bash -c '
        cd $REMOTE_DIR
        source xinfluencer_env/bin/activate
        
        echo \"Running H200 verification...\"
        python3 scripts/test_h200_setup.py
    '"
    
    print_success "Verification completed"
}

# Main execution
main() {
    echo "H200 Verification Runner"
    echo "======================="
    
    sync_verification_script
    run_verification
    
    echo ""
    print_success "H200 verification process completed!"
    echo ""
    echo "If verification passed, you can now:"
    echo "1. Start development: ssh -i $PEM_FILE $H200_USER@$H200_SERVER"
    echo "2. Run the AI: ssh -i $PEM_FILE $H200_USER@$H200_SERVER 'cd $REMOTE_DIR && python3 src/main.py'"
    echo ""
}

# Run main function
main "$@" 