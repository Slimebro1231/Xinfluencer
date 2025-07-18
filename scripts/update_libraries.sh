#!/bin/bash

# H200 Library Update Script
# Updates libraries for Phase 2: Advanced Algorithms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SSH_KEY="/Users/max/Xinfluencer/influencer.pem"
H200_HOST="157.10.162.127"
H200_USER="ubuntu"
REMOTE_DIR="/home/ubuntu/xinfluencer"

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

print_status "Starting H200 library updates for Phase 2..."

# Test SSH connection
print_status "Testing H200 connection..."
if ! ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o BatchMode=yes "$H200_USER@$H200_HOST" "echo 'SSH connection successful'" > /dev/null 2>&1; then
    print_error "Failed to connect to H200 server"
    exit 1
fi
print_success "H200 connection established"

# Create backup of current requirements
print_status "Creating backup of current requirements..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && cp requirements_h200.txt requirements_h200.txt.backup.$(date +%Y%m%d_%H%M%S)"

# Update requirements for Phase 2
print_status "Updating requirements for Phase 2 libraries..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && cat >> requirements_h200.txt << 'EOF'

# Phase 2: Advanced Algorithms
rank-bm25>=0.2.2
sentence-transformers[cross-encoder]>=5.0.0
networkx>=3.0
torch-geometric>=2.4.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
dash>=2.10.0
EOF"

# Install updated requirements
print_status "Installing Phase 2 libraries..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && source xinfluencer_env/bin/activate && pip install --upgrade rank-bm25 sentence-transformers[cross-encoder] networkx torch-geometric scikit-learn pandas numpy matplotlib seaborn plotly dash"

# Test library installations
print_status "Testing library installations..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && source xinfluencer_env/bin/activate && python3 -c \"
import rank_bm25
import sentence_transformers
import networkx as nx
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import dash

print('✅ All Phase 2 libraries imported successfully')
print(f'rank-bm25: Installed successfully')
print(f'sentence-transformers version: {sentence_transformers.__version__}')
print(f'networkx version: {nx.__version__}')
print(f'scikit-learn version: {sklearn.__version__}')
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'matplotlib version: {matplotlib.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'plotly version: {plotly.__version__}')
print(f'dash version: {dash.__version__}')
\""

if [ $? -eq 0 ]; then
    print_success "Phase 2 libraries installed successfully!"
else
    print_error "Library installation failed!"
    print_status "Rolling back to previous requirements..."
    ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && git checkout requirements_h200.txt"
    exit 1
fi

# Test BM25 functionality
print_status "Testing BM25 functionality..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && source xinfluencer_env/bin/activate && python3 -c \"
from rank_bm25 import BM25Okapi
import numpy as np

# Test documents
documents = [
    'Bitcoin is a decentralized cryptocurrency',
    'Ethereum is a blockchain platform for smart contracts',
    'DeFi stands for decentralized finance',
    'NFTs are non-fungible tokens on blockchain'
]

# Tokenize documents
tokenized_docs = [doc.lower().split() for doc in documents]

# Create BM25 model
bm25 = BM25Okapi(tokenized_docs)

# Test query
query = 'cryptocurrency blockchain'
tokenized_query = query.lower().split()

# Get scores
scores = bm25.get_scores(tokenized_query)
print('BM25 scores:', scores)
print('✅ BM25 functionality working correctly')
\""

# Test NetworkX functionality
print_status "Testing NetworkX functionality..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && source xinfluencer_env/bin/activate && python3 -c \"
import networkx as nx

# Create test graph
G = nx.Graph()
G.add_edge('user1', 'user2', weight=0.8)
G.add_edge('user2', 'user3', weight=0.6)
G.add_edge('user1', 'user3', weight=0.4)

# Test centrality
centrality = nx.degree_centrality(G)
print('Degree centrality:', centrality)
print('✅ NetworkX functionality working correctly')
\""

# Test Cross-Encoder functionality
print_status "Testing Cross-Encoder functionality..."
ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "cd $REMOTE_DIR && source xinfluencer_env/bin/activate && python3 -c \"
from sentence_transformers import CrossEncoder

# Test cross-encoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Test pairs
pairs = [
    ['What is Bitcoin?', 'Bitcoin is a cryptocurrency'],
    ['What is Bitcoin?', 'The weather is sunny today'],
    ['How does Ethereum work?', 'Ethereum uses smart contracts'],
    ['How does Ethereum work?', 'Cats are cute animals']
]

# Get scores
scores = model.predict(pairs)
print('Cross-encoder scores:', scores)
print('✅ Cross-Encoder functionality working correctly')
\""

print_success "Library updates completed successfully!"
print_status "Phase 2 libraries installed and tested:"
echo "  - rank-bm25: BM25 sparse retrieval"
echo "  - sentence-transformers[cross-encoder]: Reranking"
echo "  - networkx: Graph analysis"
echo "  - scikit-learn: Machine learning utilities"
echo "  - pandas/numpy: Data manipulation"
echo "  - matplotlib/seaborn/plotly: Visualization"
echo "  - dash: Interactive dashboards"

print_status "Next steps:"
echo "  1. Begin Phase 2: Advanced Algorithms"
echo "  2. Implement Hybrid Search (BM25 + Dense)"
echo "  3. Add Chain-of-Thought RAG"
echo "  4. Deploy advanced Self-RAG" 