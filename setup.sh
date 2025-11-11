#!/bin/bash

# CV Parser & Job Matcher - Setup Script for WSL + Docker

echo "======================================================================="
echo "🚀 CV Parser & Job Matcher - WSL/Docker Setup"
echo "======================================================================="
echo ""

# Check if running in WSL
if grep -qi microsoft /proc/version; then
    echo "✅ Detected WSL environment"
    IN_WSL=true
else
    echo "⚠️  Not running in WSL"
    IN_WSL=false
fi

# Step 1: Create .env file from example
echo "📝 Step 1: Setting up environment variables..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
    echo "⚠️  Please edit .env and add your GEMINI_API_KEY"
else
    echo "✅ .env file already exists"
fi

# Step 2: Get Windows host IP for LM Studio (WSL only)
if [ "$IN_WSL" = true ]; then
    echo ""
    echo "🔧 Step 2: Detecting Windows host IP for LM Studio..."
    WINDOWS_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
    echo "   Windows host IP: $WINDOWS_IP"
    echo "   Update your .env file with: LM_STUDIO_HOST=http://$WINDOWS_IP:1234"
    echo "   OR use: LM_STUDIO_HOST=http://host.docker.internal:1234 (in Docker)"
fi

# Step 3: Check Docker
echo ""
echo "🐳 Step 3: Checking Docker installation..."
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed"
    docker --version
else
    echo "❌ Docker is not installed"
    echo "   Install Docker Desktop for Windows with WSL2 backend"
    echo "   https://docs.docker.com/desktop/install/windows-install/"
    exit 1
fi

# Step 4: Check Docker Compose
echo ""
echo "🐳 Step 4: Checking Docker Compose..."
if command -v docker compose &> /dev/null || command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose is available"
else
    echo "❌ Docker Compose is not available"
    exit 1
fi

# Step 5: Build Docker image
echo ""
echo "🏗️  Step 5: Building Docker image..."
read -p "Do you want to build the Docker image now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker compose build
    echo "✅ Docker image built successfully"
fi

# Step 6: Instructions
echo ""
echo "======================================================================="
echo "✅ Setup Complete!"
echo "======================================================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Edit .env file and add your GEMINI_API_KEY"
echo "   nano .env"
echo ""
echo "2. Make sure LM Studio is running on Windows (if using Qwen):"
echo "   - Open LM Studio on Windows"
echo "   - Load Qwen2.5-VL-7B model"
echo "   - Start server on port 1234"
echo "   - Enable 'Serve on Local Network' in settings"
echo ""
echo "3. Test LM Studio connectivity from WSL:"
if [ "$IN_WSL" = true ]; then
    echo "   curl http://$WINDOWS_IP:1234/v1/models"
fi
echo ""
echo "4. Start the application:"
echo "   docker compose up"
echo ""
echo "   OR run without Docker:"
echo "   python main.py"
echo ""
echo "5. Access the application:"
echo "   http://localhost:7861"
echo ""
echo "======================================================================="
