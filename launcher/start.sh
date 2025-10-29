#!/bin/bash
# Railway startup script
# This script handles the directory navigation properly

set -e  # Exit on error

echo "🚀 Starting Railway deployment..."
echo "📍 Current directory: $(pwd)"
echo "📂 Directory contents:"
ls -la

# Navigate to backend directory
cd backend

echo "📍 Changed to: $(pwd)"
echo "📂 Backend directory contents:"
ls -la

# Start the server
echo "🚀 Starting uvicorn server..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
