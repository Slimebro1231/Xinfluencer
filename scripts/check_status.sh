#!/bin/bash

# H200 Status Check Script
# Quick status check for the Xinfluencer AI system

echo "🔍 Xinfluencer AI H200 Status Check"
echo "=================================="

# Check if we're on H200 server
if [ -f "/home/ubuntu/xinfluencer/start_xinfluencer.sh" ]; then
    echo "✅ Running on H200 server"
    cd /home/ubuntu/xinfluencer
    
    echo ""
    echo "📊 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv
    
    echo ""
    echo "🐍 Python Environment:"
    python3 --version
    
    echo ""
    echo "📦 Key Packages:"
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: Not installed"
    python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "Transformers: Not installed"
    
    echo ""
    echo "📁 Project Status:"
    if [ -f "src/main.py" ]; then
        echo "✅ Main script exists"
    else
        echo "❌ Main script missing"
    fi
    
    if [ -f "requirements.txt" ]; then
        echo "✅ Requirements file exists"
    else
        echo "❌ Requirements file missing"
    fi
    
    echo ""
    echo "🚀 Quick Test:"
    python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from model.generate import TextGenerator
    print('✅ Model module imports successfully')
except Exception as e:
    print(f'❌ Model module import failed: {e}')
" 2>/dev/null
    
else
    echo "❌ Not running on H200 server"
    echo "Run this script on the H200 server or use: ./ssh_h200.sh"
fi

echo ""
echo "=================================="
echo "For detailed testing, run: python3 scripts/test_h200_setup.py"