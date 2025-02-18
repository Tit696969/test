#!/bin/bash
# Install ffmpeg if not already installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    sudo apt update
    sudo apt install -y ffmpeg
fi

# Start the FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000
