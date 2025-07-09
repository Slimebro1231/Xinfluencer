#!/bin/bash

# Sync Code to H200 Server
# This script syncs the src directory to the H200 server

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

# Sync src directory
sync_src() {
    print_status "Syncing src directory to H200 server..."
    
    rsync -avz --delete --exclude='.git' --exclude='__pycache__' -e "ssh -i $PEM_FILE" src/ "$H200_USER@$H200_SERVER:$REMOTE_DIR/src/"
    
    print_success "src directory synced"
}

# Sync scripts directory
sync_scripts() {
    print_status "Syncing scripts directory to H200 server..."
    rsync -avz --delete -e "ssh -i $PEM_FILE" scripts/ "$H200_USER@$H200_SERVER:$REMOTE_DIR/scripts/"
    
    print_status "Setting execute permissions on remote scripts..."
    ssh -i "$PEM_FILE" "$H200_USER@$H200_SERVER" "chmod +x $REMOTE_DIR/scripts/*.sh"
    
    print_success "scripts directory synced and permissions set"
}

# Sync requirements.txt
sync_requirements() {
    print_status "Syncing requirements.txt to H200 server..."
    rsync -avz -e "ssh -i $PEM_FILE" requirements.txt "$H200_USER@$H200_SERVER:$REMOTE_DIR/requirements.txt"
    print_success "requirements.txt synced"
}

# Sync .env file
sync_dotenv() {
    print_status "Syncing .env file to H200 server..."
    if [ -f .env ]; then
        rsync -avz -e "ssh -i $PEM_FILE" .env "$H200_USER@$H200_SERVER:$REMOTE_DIR/.env"
        print_success ".env file synced"
    else
        print_status ".env file not found locally, skipping sync."
    fi
}

# Main execution
main() {
    echo "H200 Code Syncer"
    echo "================="
    
    sync_src
    sync_scripts
    sync_requirements
    sync_dotenv
    
    echo ""
    print_success "H200 code sync process completed!"
}

# Run main function
main "$@" 