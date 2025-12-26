#!/bin/bash
# Bash Setup Script for Advanced Multi-Source RAG (Linux/Mac)
# Run this script to set up the project environment

set -e  # Exit on error

echo "========================================"
echo "Advanced Multi-Source RAG - Setup"
echo "========================================"
echo ""

# Check Python version
echo "[1/7] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Please install Python 3.9-3.12."
    exit 1
fi

# Check if version is in supported range (3.9 - 3.12)
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
major=$(echo $python_version | cut -d. -f1)
minor=$(echo $python_version | cut -d. -f2)

if [ "$major" -ne 3 ] || [ "$minor" -lt 9 ] || [ "$minor" -gt 12 ]; then
    echo "ERROR: Python $python_version is not supported!"
    echo "This project requires Python 3.9, 3.10, 3.11, or 3.12"
    echo "Reason: spaCy (required dependency) doesn't support Python 3.13+ yet"
    echo ""
    echo "Please install Python 3.12 from: https://www.python.org/downloads/"
    exit 1
fi

python3 --version
echo "✓ Python $python_version is supported"

# Create virtual environment
echo ""
echo "[2/7] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "[3/7] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "[4/7] Upgrading pip..."
pip install --upgrade pip
echo "✓ Pip upgraded"

# Install dependencies
echo ""
echo "[5/7] Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Download spaCy model
echo ""
echo "[6/7] Downloading spaCy NER model..."
python -m spacy download en_core_web_sm || echo "WARNING: Failed to download spaCy model"

# Create directory structure
echo ""
echo "[7/7] Creating project directories..."
mkdir -p data/documents data/cache data/faiss_index
mkdir -p models logs tests/sample_data docs
mkdir -p src/ingestion src/processing src/indexing
mkdir -p src/retrieval src/reranking src/llm src/ui
echo "✓ Directories created"

# Create .env from .env.example
echo ""
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ Created .env file from .env.example"
    echo "Please review and update .env with your configuration."
else
    echo ".env file already exists."
fi

# Create __init__.py files
touch src/ingestion/__init__.py
touch src/processing/__init__.py
touch src/indexing/__init__.py
touch src/retrieval/__init__.py
touch src/reranking/__init__.py
touch src/llm/__init__.py
touch src/ui/__init__.py

# Summary
echo ""
echo "========================================"
echo "✓ Setup completed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Review and update .env file"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Test configuration: python -c 'from src.config import config; print(config)'"
echo ""
echo "To run the application (once implemented):"
echo "  streamlit run src/app.py"
echo ""
