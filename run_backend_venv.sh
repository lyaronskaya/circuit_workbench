#!/bin/bash

IP_ADDRESS=${1:-localhost}
BIND_HOST=${BIND_HOST:-0.0.0.0}

# Create virtual environment if it doesn't exist
if [ ! -d "backend_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv backend_venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source backend_venv/bin/activate

cd backend

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run uvicorn
echo "Starting uvicorn server..."
echo "Backend will be available at: http://${IP_ADDRESS}:8000"
uvicorn attention.api:app --reload --host "$BIND_HOST" --port 8000
