#!/bin/bash

# Update Dependencies on H200 Server
# This script syncs requirements.txt and updates dependencies on the H200 server

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

# Sync requirements file
sync_requirements() {
    print_status "Syncing requirements.txt to H200 server..."
    
    scp -i "$PEM_FILE" requirements.txt "$H200_USER@$H200_SERVER:$REMOTE_DIR/"
    
    print_success "requirements.txt synced"
}

# Run pip upgrade
update_dependencies() {
    print_status "Updating dependencies on H200 server..."
    
    ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "bash -c '
        cd $REMOTE_DIR
        source xinfluencer_env/bin/activate
        
        echo \"Upgrading pip and installing requirements...\"
        pip install --upgrade pip
        pip install --upgrade -r requirements.txt
    '"
    
    print_success "Dependencies updated"
}

# Main execution
main() {
    echo "H200 Dependency Updater"
    echo "======================="
    
    sync_requirements
    update_dependencies
    
    echo ""
    print_success "H200 dependency update process completed!"
}

# Run main function
main "$@" 