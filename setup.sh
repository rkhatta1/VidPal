#!/bin/bash
echo "🚀 Setting up VidPalAI with pgvector and Gemini embeddings..."

echo "🐘 Starting PostgreSQL with pgvector..."
docker compose up -d

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 15

# Test connection
docker exec vidpalai_postgres pg_isready -U vidpalai

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Verify Google Cloud authentication (if using Vertex AI)
echo "🔐 Checking Google Cloud authentication..."
if command -v gcloud &> /dev/null; then
    gcloud auth application-default login
    echo "✅ Google Cloud authentication configured"
else
    echo "⚠️  gcloud CLI not found. Install it for Vertex AI access:"
    echo "   https://cloud.google.com/sdk/docs/install"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Configure .env with your credentials"
echo "2. If using Vertex AI, ensure you're authenticated:"
echo "   gcloud auth application-default login"
echo "3. Place your video files in input/"
echo "4. Run: python pipeline.py"
echo ""
echo "📊 Database info:"
echo "   Host: localhost:5432"
echo "   Database: vidpalai"
echo "   User: vidpalai"
