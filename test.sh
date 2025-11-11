#!/bin/bash

# Quick Test Script for CV Parser

echo "======================================"
echo "CV Parser - Quick Test"
echo "======================================"
echo ""

# Test 1: Check environment
echo "✓ Checking environment..."
if [ -f .env ]; then
    echo "  ✓ .env file exists"
    
    # Check if GEMINI_API_KEY is set
    source .env
    if [ -n "$GEMINI_API_KEY" ] && [ "$GEMINI_API_KEY" != "your_gemini_api_key_here" ]; then
        echo "  ✓ GEMINI_API_KEY is configured"
    else
        echo "  ⚠️  GEMINI_API_KEY not configured"
    fi
    
    if [ -n "$LM_STUDIO_HOST" ]; then
        echo "  ✓ LM_STUDIO_HOST: $LM_STUDIO_HOST"
    fi
else
    echo "  ❌ .env file not found - run ./setup.sh first"
    exit 1
fi

echo ""

# Test 2: Check Python dependencies
echo "✓ Checking Python dependencies..."
python3 -c "
import sys
try:
    import gradio
    print('  ✓ gradio installed')
except ImportError:
    print('  ❌ gradio not installed')
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print('  ✓ python-dotenv installed')
except ImportError:
    print('  ❌ python-dotenv not installed')
    sys.exit(1)

try:
    import fitz  # PyMuPDF
    print('  ✓ PyMuPDF installed')
except ImportError:
    print('  ⚠️  PyMuPDF not installed (optional, for PDF support)')
"

echo ""

# Test 3: Check LM Studio connectivity (if configured)
if [ -n "$LM_STUDIO_HOST" ] && [ "$LM_STUDIO_HOST" != "http://host.docker.internal:1234" ]; then
    echo "✓ Testing LM Studio connectivity..."
    if curl -s --max-time 3 "$LM_STUDIO_HOST/v1/models" > /dev/null 2>&1; then
        echo "  ✓ LM Studio is reachable"
        echo "  Models available:"
        curl -s "$LM_STUDIO_HOST/v1/models" | python3 -m json.tool 2>/dev/null | grep '"id"' | head -3
    else
        echo "  ⚠️  LM Studio not reachable at $LM_STUDIO_HOST"
        echo "     Make sure LM Studio is running on Windows"
        echo "     and 'Serve on Local Network' is enabled"
    fi
else
    echo "✓ LM Studio check skipped (using Docker or not configured)"
fi

echo ""

# Test 4: Check if CV template exists
echo "✓ Checking CV template..."
if [ -f cv_template_camelCase.json ]; then
    echo "  ✓ CV template found"
else
    echo "  ❌ CV template not found: cv_template_camelCase.json"
    exit 1
fi

echo ""

# Test 5: Try importing main modules
echo "✓ Testing Python modules..."
python3 -c "
import sys
sys.path.insert(0, '.')

try:
    from cv_extractor_gemini import GeminiCVExtractor
    print('  ✓ cv_extractor_gemini imports successfully')
except Exception as e:
    print(f'  ⚠️  cv_extractor_gemini import warning: {e}')

try:
    from cv_extractor_qwen import CVParserAgent
    print('  ✓ cv_extractor_qwen imports successfully')
except Exception as e:
    print(f'  ❌ cv_extractor_qwen import failed: {e}')
    sys.exit(1)

try:
    from job_matcher import JobMatcherAgent
    print('  ✓ job_matcher imports successfully')
except Exception as e:
    print(f'  ❌ job_matcher import failed: {e}')
    sys.exit(1)
"

echo ""
echo "======================================"
echo "✅ Test Complete!"
echo "======================================"
echo ""
echo "To start the application:"
echo "  python3 main.py"
echo ""
echo "Or with Docker:"
echo "  docker compose up"
echo ""
