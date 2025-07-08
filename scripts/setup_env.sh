#!/bin/bash
# Setup script for Xinfluencer AI development environment

set -e

echo "🚀 Setting up Xinfluencer AI development environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Install pytest for testing
echo "🧪 Installing test dependencies..."
pip install pytest

echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python src/main.py"
echo ""
echo "To run tests:"
echo "  python -m pytest tests/" 