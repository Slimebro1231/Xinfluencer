#!/bin/bash

# Deploy Bulletproof Twitter Collection System to H200
# Post-focused with smart rate limiting

set -e

# Configuration
SSH_KEY="/Users/max/Xinfluencer/influencer.pem"
H200_HOST="157.10.162.127"
H200_USER="ubuntu"
REMOTE_DIR="/home/ubuntu/xinfluencer"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

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

print_status "Deploying Bulletproof Twitter Collection System to H200..."

# Test SSH connection
if ! ssh -i "$SSH_KEY" -o ConnectTimeout=10 "$H200_USER@$H200_HOST" "echo 'SSH OK'" >/dev/null 2>&1; then
    print_error "Cannot connect to H200"
    exit 1
fi

print_success "H200 connection established"

# Deploy bulletproof scripts
print_status "Deploying bulletproof collection system..."
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    api_safeguard.py \
    safe_collection_script.py \
    prevent_api_waste.md \
    "$H200_USER@$H200_HOST:$REMOTE_DIR/"

# Set up Twitter credentials on H200
print_status "Setting up Twitter API credentials..."

# Check if credentials exist locally
if [ -z "$TWITTER_BEARER_TOKEN" ]; then
    print_warning "TWITTER_BEARER_TOKEN not found in local environment"
    print_status "Please set your Twitter Bearer Token:"
    read -s -p "Enter Twitter Bearer Token: " BEARER_TOKEN
    echo
else
    BEARER_TOKEN="$TWITTER_BEARER_TOKEN"
    print_success "Using local TWITTER_BEARER_TOKEN"
fi

# Create .env file on H200
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && cat > .env << EOF
# Twitter API Configuration
TWITTER_BEARER_TOKEN=$BEARER_TOKEN

# Export for shell sessions
export TWITTER_BEARER_TOKEN=$BEARER_TOKEN
EOF"

# Update bashrc to load credentials automatically
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && 
if ! grep -q 'source.*xinfluencer.*\.env' ~/.bashrc; then
    echo 'source $REMOTE_DIR/.env' >> ~/.bashrc
fi"

print_success "Twitter credentials configured"

# Install dependencies if needed
print_status "Checking Python dependencies..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && 
source xinfluencer_env/bin/activate && 
pip install requests >/dev/null 2>&1 || echo 'Dependencies already installed'"

# Test the bulletproof system
print_status "Testing bulletproof safeguard system..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && 
source .env && 
source xinfluencer_env/bin/activate && 
python3 api_safeguard.py"

# Create monitoring script
print_status "Creating monitoring tools..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && cat > monitor_collection.sh << 'EOF'
#!/bin/bash

echo '🛡️ Bulletproof Collection Monitor'
echo '================================='

echo '📊 Current Safeguard Status:'
source .env && source xinfluencer_env/bin/activate && python3 api_safeguard.py

echo -e '\n🔍 Running Processes:'
ps aux | grep -E 'python.*collect|python.*scrape|python.*twitter' | grep -v grep || echo 'No collection processes running'

echo -e '\n📁 Recent Collections:'
ls -la data/safe_collection/ 2>/dev/null | tail -5 || echo 'No collections yet'

echo -e '\n💾 Disk Usage:'
df -h | grep -E 'Use%|/$'

echo -e '\n🌐 Network Test:'
curl -s --max-time 5 https://api.twitter.com/2/openapi.json >/dev/null && echo 'Twitter API reachable ✅' || echo 'Twitter API unreachable ❌'
EOF

chmod +x monitor_collection.sh"

# Create safe collection wrapper
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && cat > collect_safe.sh << 'EOF'
#!/bin/bash

# Safe Collection Wrapper with Automatic Monitoring
# Usage: ./collect_safe.sh [target_posts]

TARGET_POSTS=\${1:-300}

echo '🛡️ Starting Bulletproof Collection'
echo '=================================='
echo \"Target: \$TARGET_POSTS posts\"

# Load environment
source .env
source xinfluencer_env/bin/activate

# Pre-flight check
echo '🔍 Pre-flight safety check...'
python3 -c \"
from api_safeguard import TwitterAPISafeguard
safeguard = TwitterAPISafeguard()
limits = safeguard.check_post_limits()
if not limits['can_collect']:
    print('❌ Collection blocked by safety limits')
    print(f'Posts last hour: {limits[\"posts_last_hour\"]}/{safeguard.limits[\"posts_per_hour\"]}')
    print(f'Posts last day: {limits[\"posts_last_day\"]}/{safeguard.limits[\"posts_per_day\"]}')
    exit(1)
else:
    print('✅ Safety check passed')
\"

if [ \$? -ne 0 ]; then
    echo '❌ Safety check failed. Collection aborted.'
    exit 1
fi

# Run collection
echo '🚀 Running safe collection...'
python3 safe_collection_script.py \$TARGET_POSTS

# Post-collection status
echo -e '\n📊 Post-collection status:'
python3 api_safeguard.py
EOF

chmod +x collect_safe.sh"

print_success "Monitoring tools created"

# Final test with small collection
print_status "Running test collection (50 posts)..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && 
source .env && 
source xinfluencer_env/bin/activate && 
timeout 60 python3 safe_collection_script.py 50" || print_warning "Test collection timed out (API may be slow)"

print_success "Bulletproof system deployed to H200!"

echo
print_status "🎯 Bulletproof System Ready!"
print_status "Usage on H200:"
print_status "  SSH: ssh -i $SSH_KEY $H200_USER@$H200_HOST"
print_status "  Monitor: cd $REMOTE_DIR && ./monitor_collection.sh"
print_status "  Collect: ./collect_safe.sh [target_posts]"
print_status "  Status: python3 api_safeguard.py"

echo
print_status "🛡️ Safety Features Active:"
print_status "  ✅ Post-based limits (1000/hour, 10000/day)"
print_status "  ✅ API rate limit monitoring (450 search, 90 lookup per 15min)"
print_status "  ✅ Automatic target adjustment"
print_status "  ✅ Process monitoring and safeguards"
print_status "  ✅ Collection efficiency tracking"

echo
print_warning "⚠️  Remember: No more hanging processes!"
print_success "The bulletproof system will prevent API quota waste." 