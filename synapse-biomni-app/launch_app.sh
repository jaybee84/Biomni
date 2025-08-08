#!/bin/bash

# Biomni Streamlit App Launcher
# This script helps launch the Biomni Streamlit application with proper setup

echo "🧬 Starting Biomni Streamlit Application..."
echo "============================================"

# Check if we're in the right directory
if [ ! -f "biomni_streamlit_app.py" ]; then
    echo "❌ Error: biomni_streamlit_app.py not found in current directory"
    echo "Please navigate to the directory containing the Streamlit app"
    exit 1
fi

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Error: Streamlit is not installed"
    echo "Please install it with: pip install streamlit"
    exit 1
fi

# Check if requirements are met
echo "📋 Checking requirements..."
if [ -f "requirements_streamlit.txt" ]; then
    echo "   Installing/updating dependencies..."
    pip install -r requirements_streamlit.txt
else
    echo "⚠️  Warning: requirements_streamlit.txt not found"
fi

# Set up environment
echo "🔧 Setting up environment..."

# Check for .env file
if [ -f ".env" ]; then
    echo "   Found .env file - loading environment variables"
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  Warning: ANTHROPIC_API_KEY not found in environment"
    echo "   You'll need to configure it in the Streamlit interface"
fi

# Launch the app
echo "🚀 Launching Streamlit application..."
echo "   The app will open in your default browser"
echo "   If it doesn't open automatically, visit: http://localhost:8501"
echo ""
echo "   To stop the app, press Ctrl+C in this terminal"
echo ""

# Run Streamlit with custom configuration
streamlit run biomni_streamlit_app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false \
    --server.enableXsrfProtection false
