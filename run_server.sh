#!/bin/bash

# DataBrain AI Agent - Server Startup Script
# This script starts the FastAPI backend server

set -e  # Exit on error

cd "$(dirname "$0")"

echo "🧠 Starting DataBrain AI Agent..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv .venv
    echo "📦 Installing dependencies..."
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    # Activate virtual environment
    source .venv/bin/activate
fi

# Check for API keys
if [ -z "$OPENAI_API_KEY" ] && [ -z "$DEEPSEEK_API_KEY" ]; then
    if [ ! -f ".env" ]; then
        echo "⚠️  Warning: No API keys found and no .env file!"
        echo "Please create a .env file with at least one API key:"
        echo "  OPENAI_API_KEY=your-key-here"
        echo "  DEEPSEEK_API_KEY=your-key-here"
        echo ""
        echo "See .env.example for a template."
    fi
fi

# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "🚀 Starting backend server on http://localhost:8000"
echo "📊 Open frontend/index.html in your browser to use the agent"
echo ""

# Start the server
cd databrain_agent/backend
python main.py
