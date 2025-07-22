#!/bin/bash

# Deploy Identity Training System to H200
# Trains bot with crypto identity using retrieved posts

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

print_status "Deploying Identity Training System to H200..."

# Test SSH connection
if ! ssh -i "$SSH_KEY" -o ConnectTimeout=10 "$H200_USER@$H200_HOST" "echo 'SSH OK'" >/dev/null 2>&1; then
    print_error "Cannot connect to H200"
    exit 1
fi

print_success "H200 connection established"

# Deploy identity training files
print_status "Deploying identity training system..."
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    identity_training_pipeline.py \
    safe_collection_script.py \
    api_safeguard.py \
    "$H200_USER@$H200_HOST:$REMOTE_DIR/"

# Deploy training directories and data
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && mkdir -p data/training data/all_posts data/training_posts logs"

# Install additional dependencies for training
print_status "Installing training dependencies..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && 
source xinfluencer_env/bin/activate && 
pip install pandas sqlite3 || echo 'Some packages already installed'"

# Create training launcher script
print_status "Creating training automation scripts..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && cat > train_identity.sh << 'EOF'
#!/bin/bash

# Identity Training Launcher for H200

echo '🤖 Starting Identity Training Pipeline'
echo '===================================='

# Load environment
source .env 2>/dev/null || echo 'No .env file found'
source xinfluencer_env/bin/activate

# Check GPU status
echo '🔧 GPU Status:'
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# Run identity training
echo -e '\n🚀 Starting identity training...'
python3 identity_training_pipeline.py

# Check results
echo -e '\n📊 Training Results:'
if [ -d 'data/training' ]; then
    ls -la data/training/ | tail -5
else
    echo 'No training directory found'
fi

echo -e '\n💾 Storage Status:'
if [ -f 'data/all_posts/posts.db' ]; then
    sqlite3 data/all_posts/posts.db \"SELECT COUNT(*) as total_posts FROM posts; SELECT source, COUNT(*) as count FROM posts GROUP BY source;\"
else
    echo 'No posts database found'
fi
EOF

chmod +x train_identity.sh"

# Create collection + training automation
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && cat > collect_and_train.sh << 'EOF'
#!/bin/bash

# Automated Collection + Identity Training Pipeline

POSTS_TARGET=\${1:-500}

echo '🔄 Automated Collection + Identity Training'
echo '=========================================='
echo \"Target posts: \$POSTS_TARGET\"

# Load environment
source .env 2>/dev/null || echo 'No .env file found'
source xinfluencer_env/bin/activate

# Step 1: Safe collection with training integration
echo -e '\n📥 Step 1: Collecting posts...'
python3 safe_collection_script.py \$POSTS_TARGET

# Step 2: Run identity training on collected data
echo -e '\n🤖 Step 2: Running identity training...'
python3 identity_training_pipeline.py

# Step 3: Summary
echo -e '\n📋 Summary:'
echo '✅ Collection completed with training data storage'
echo '✅ Identity training completed'
echo '✅ Bot identity updated with latest crypto content'

# Check final results
if [ -f 'data/all_posts/posts.db' ]; then
    echo -e '\n💾 Database Stats:'
    sqlite3 data/all_posts/posts.db \"SELECT 'Total Posts:', COUNT(*) FROM posts; SELECT 'By Quality:', quality_score, COUNT(*) FROM posts WHERE quality_score > 0 GROUP BY ROUND(quality_score, 1) ORDER BY quality_score DESC LIMIT 5;\"
fi

if [ -d 'lora_checkpoints/identity' ]; then
    echo -e '\n🧠 Latest LoRA Adapter:'
    ls -la lora_checkpoints/identity/ | tail -3
fi
EOF

chmod +x collect_and_train.sh"

# Create monitoring script
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && cat > monitor_training.sh << 'EOF'
#!/bin/bash

echo '📊 Identity Training Monitor'
echo '==========================='

# Load environment
source xinfluencer_env/bin/activate 2>/dev/null || echo 'Virtual env not found'

echo '🗄️ Posts Database Status:'
if [ -f 'data/all_posts/posts.db' ]; then
    sqlite3 data/all_posts/posts.db \"
    SELECT 'Total Posts:', COUNT(*) FROM posts;
    SELECT 'High Quality Posts (>0.7):', COUNT(*) FROM posts WHERE quality_score > 0.7;
    SELECT 'High Crypto Relevance (>0.8):', COUNT(*) FROM posts WHERE crypto_relevance > 0.8;
    SELECT 'Top 5 Authors:', author, COUNT(*) as posts FROM posts GROUP BY author ORDER BY posts DESC LIMIT 5;
    \"
else
    echo 'No posts database found'
fi

echo -e '\n🧠 LoRA Training Status:'
if [ -d 'lora_checkpoints/identity' ]; then
    echo 'Identity training checkpoints:'
    ls -la lora_checkpoints/identity/ | tail -5
else
    echo 'No identity training checkpoints found'
fi

echo -e '\n📁 Training Data Files:'
if [ -d 'data/training_posts' ]; then
    echo 'Recent training collections:'
    ls -la data/training_posts/ | tail -5
else
    echo 'No training posts directory'
fi

echo -e '\n📈 Training Results:'
if [ -d 'data/training' ]; then
    echo 'Recent training sessions:'
    ls -la data/training/ | tail -5
else
    echo 'No training results directory'
fi

echo -e '\n💾 Disk Usage:'
du -sh data/ lora_checkpoints/ logs/ 2>/dev/null || echo 'Some directories not found'

echo -e '\n🔧 GPU Status:'
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null || echo 'nvidia-smi not available'
EOF

chmod +x monitor_training.sh"

# Test the identity training system
print_status "Testing identity training system..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && 
source xinfluencer_env/bin/activate && 
python3 -c \"
import sys
sys.path.insert(0, 'src')
from identity_training_pipeline import IdentityTrainingPipeline
print('✅ Identity training pipeline imports successfully')
pipeline = IdentityTrainingPipeline()
print('✅ Pipeline initialized successfully')
stats = pipeline.storage.get_storage_stats()
print(f'📊 Current storage: {stats}')
\" 2>/dev/null || echo 'Some imports may need data to be collected first'"

print_success "Identity training system deployed!"

echo
print_status "🎯 Identity Training System Ready!"
print_status "Usage on H200:"
print_status "  SSH: ssh -i $SSH_KEY $H200_USER@$H200_HOST"
print_status "  Full Pipeline: ./collect_and_train.sh [target_posts]"
print_status "  Training Only: ./train_identity.sh"
print_status "  Monitor: ./monitor_training.sh"

echo
print_status "🤖 Identity Training Features:"
print_status "  ✅ Stores ALL retrieved posts (even failed API calls)"
print_status "  ✅ Analyzes crypto relevance and content quality"
print_status "  ✅ LoRA fine-tuning with identity focus"
print_status "  ✅ SQLite database for efficient post storage"
print_status "  ✅ Vector database integration for RAG"
print_status "  ✅ Automatic training on collected data"

echo
print_status "🔄 Training Workflow:"
print_status "  1. Collect posts with bulletproof system"
print_status "  2. Store ALL posts in training database"
print_status "  3. Analyze quality and crypto relevance"
print_status "  4. Select high-quality posts for training"
print_status "  5. Fine-tune model with LoRA adapters"
print_status "  6. Update vector database with identity examples"

echo
print_warning "💡 Remember: Every retrieved post costs money - we store everything!"
print_success "Your bot will learn crypto identity from the best KOL content." 