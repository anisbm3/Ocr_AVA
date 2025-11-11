#!/bin/bash
# WSL Setup Script for CV Parser
# This script helps configure LM Studio connectivity from WSL to Windows host

echo "🔧 CV Parser - WSL Setup Script"
echo "================================="

# Check if running in WSL
if [[ ! -f /proc/version ]] || ! grep -q "Microsoft" /proc/version; then
    echo "❌ This script is designed for WSL environments only"
    exit 1
fi

echo "✅ Running in WSL environment"

# Find Windows host IP
echo "🔍 Finding Windows host IP..."
WINDOWS_IP=$(ip route | grep default | awk '{print $3}')

if [[ -n "$WINDOWS_IP" ]]; then
    echo "✅ Windows host IP: $WINDOWS_IP"
    echo "💡 You can use: LM_STUDIO_HOST=http://$WINDOWS_IP:1234"
else
    echo "❌ Could not determine Windows host IP"
    echo "💡 Try: ip route | grep default"
fi

# Check if .env exists
if [[ -f ".env" ]]; then
    echo "✅ .env file exists"
    echo "🔄 Updating LM_STUDIO_HOST..."

    # Update .env file
    if [[ -n "$WINDOWS_IP" ]]; then
        sed -i "s|LM_STUDIO_HOST=.*|LM_STUDIO_HOST=http://$WINDOWS_IP:1234|" .env
        echo "✅ Updated .env with Windows host IP"
    else
        sed -i "s|LM_STUDIO_HOST=.*|LM_STUDIO_HOST=http://host.docker.internal:1234|" .env
        echo "✅ Updated .env with host.docker.internal (for Docker)"
    fi
else
    echo "❌ .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "✅ Created .env from .env.example"
fi

echo ""
echo "📋 Next Steps:"
echo "1. Make sure LM Studio is running on Windows host"
echo "2. Load Qwen2.5-VL-7B model in LM Studio"
echo "3. Start LM Studio server on port 1234"
echo "4. Run: python main.py"
echo ""
echo "🔗 Alternative: Use Docker (recommended)"
echo "   docker compose up --build"