#!/bin/bash
echo "===================================================="
echo " Installing Used Car Price Prediction Dependencies"
echo "===================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment (venv)..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

# Upgrade pip and install requirements
echo "Upgrading pip and installing dependencies from requirements.txt..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "===================================================="
echo " All dependencies installed successfully!"
echo " To start the server, run:"
echo "   source venv/bin/activate"
echo "   uvicorn app.main:app --reload"
echo "===================================================="
