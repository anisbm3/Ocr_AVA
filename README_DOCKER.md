# CV Parser & Job Matcher

AI-powered CV parsing and job matching system with dual extraction methods (Google Gemini & Qwen Vision) and intelligent job matching.

## 🚀 Features

- **Dual CV Extraction**:
  - 🌟 Google Gemini: Fast, cloud-based extraction
  - 🤖 Qwen Vision: Local, privacy-focused extraction via LM Studio
- **Intelligent Job Matching**: LLM-powered scoring and recommendations
- **Web Scraping**: GitHub/LinkedIn profile analysis
- **Beautiful UI**: Gradio interface with structured output

## 📋 Prerequisites

### For Windows + WSL Setup

1. **Windows 11/10 with WSL2**
   ```bash
   wsl --install
   ```

2. **Docker Desktop for Windows**
   - Download from: https://docs.docker.com/desktop/install/windows-install/
   - Enable WSL2 backend
   - Enable "Expose daemon on tcp://localhost:2375"

3. **LM Studio (Optional - for Qwen Vision)**
   - Download from: https://lmstudio.ai/
   - Install on Windows (not in WSL)
   - Load Qwen2.5-VL-7B model
   - Enable "Serve on Local Network" in settings

4. **Google Gemini API Key**
   - Get from: https://ai.google.dev/

## 🛠️ Installation

### Option 1: Docker (Recommended for WSL)

1. **Clone and setup**:
   ```bash
   cd /home/yourusername/python
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Edit environment variables**:
   ```bash
   nano .env
   ```
   
   Update these values:
   ```bash
   GEMINI_API_KEY=your_actual_gemini_api_key
   LM_STUDIO_HOST=http://host.docker.internal:1234  # For Docker
   # OR
   LM_STUDIO_HOST=http://172.X.X.X:1234  # Your Windows IP from WSL
   ```

3. **Start LM Studio on Windows** (if using Qwen):
   - Open LM Studio
   - Load Qwen2.5-VL-7B model
   - Click "Start Server" (port 1234)
   - Enable "Serve on Local Network"

4. **Test LM Studio connectivity from WSL**:
   ```bash
   # Get Windows IP
   cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
   
   # Test connection (replace with your Windows IP)
   curl http://172.X.X.X:1234/v1/models
   ```

5. **Run with Docker**:
   ```bash
   docker compose up
   ```

6. **Access the app**:
   Open browser: `http://localhost:7861`

### Option 2: Direct Python (Without Docker)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup environment**:
   ```bash
   cp .env.example .env
   nano .env
   ```

3. **For WSL + LM Studio on Windows**:
   ```bash
   # Get your Windows host IP
   export WINDOWS_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
   echo $WINDOWS_IP
   
   # Update .env with your Windows IP
   # LM_STUDIO_HOST=http://172.X.X.X:1234
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# LM Studio Configuration
# For Docker:
LM_STUDIO_HOST=http://host.docker.internal:1234

# For WSL without Docker (use your Windows IP):
# LM_STUDIO_HOST=http://172.X.X.X:1234

# Model Configuration
QWEN_MODEL=qwen2.5-vl-7b

# Application Settings
APP_HOST=0.0.0.0
APP_PORT=7861
DEBUG=true
```

### WSL + Windows LM Studio Connectivity

When running the app in WSL and LM Studio on Windows:

1. **Find your Windows host IP**:
   ```bash
   cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
   ```

2. **Configure LM Studio on Windows**:
   - Open LM Studio settings
   - Enable "Serve on Local Network"
   - Make sure Windows Firewall allows port 1234

3. **Test from WSL**:
   ```bash
   curl http://YOUR_WINDOWS_IP:1234/v1/models
   ```

4. **In Docker**: Use `host.docker.internal:1234`
5. **Without Docker**: Use your Windows IP address

## 📖 Usage

### 1. Extract CV

Choose either extraction method:

- **Gemini Tab**: Upload CV → Click "Extract with Gemini"
- **Qwen Tab**: Upload CV → Click "Extract with Qwen"

Supported formats: PDF, DOC, DOCX, JPG, PNG

### 2. Copy JSON Output

After extraction, copy the JSON from the output panel.

### 3. Match with Job

- Go to "Job Matching" tab
- Paste the CV JSON
- Enter job title and requirements
- Optionally add GitHub/LinkedIn URLs
- Click "Match Job"

### 4. Review Results

Get detailed matching analysis:
- Overall compatibility score
- Skills match/gap analysis
- Experience evaluation
- Recommendations

## 🐳 Docker Commands

```bash
# Build image
docker compose build

# Start services
docker compose up

# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Rebuild and start
docker compose up --build
```

## 🔍 Troubleshooting

### LM Studio not connecting from WSL

1. **Check Windows Firewall**:
   ```powershell
   # Run in PowerShell as Administrator
   New-NetFirewallRule -DisplayName "LM Studio" -Direction Inbound -Port 1234 -Protocol TCP -Action Allow
   ```

2. **Verify LM Studio is serving on network**:
   - LM Studio → Settings → Enable "Serve on Local Network"

3. **Test from WSL**:
   ```bash
   # Get Windows IP
   ip route | grep default | awk '{print $3}'
   
   # Test connection
   curl http://WINDOWS_IP:1234/v1/models
   ```

### Docker can't reach LM Studio

- Make sure you're using `host.docker.internal:1234` in .env
- Check Docker Desktop WSL2 integration is enabled

### Gemini API errors

- Verify your API key in .env
- Check API quota: https://ai.google.dev/

### Port 7861 already in use

```bash
# Change port in .env
APP_PORT=7862

# Or kill existing process
lsof -ti:7861 | xargs kill -9
```

## 📁 Project Structure

```
.
├── main.py                      # Main Gradio application
├── cv_extractor_gemini/         # Gemini CV extractor
├── cv_extractor_qwen/           # Qwen CV extractor
├── job_matcher/                 # Job matching agent
├── llm_clients/                 # LLM client wrappers
├── web_scraper/                 # Web scraping utilities
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
├── .env.example                 # Environment template
├── requirements.txt             # Python dependencies
└── setup.sh                     # Setup script
```

## 🤝 Contributing

Contributions welcome! Please submit issues and pull requests.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API
- Qwen Vision (Alibaba Cloud)
- LM Studio
- Gradio
