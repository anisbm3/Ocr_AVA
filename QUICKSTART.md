# Quick Start Guide - CV Parser & Job Matcher

## 🚀 Quick Start (WSL + Docker)

### 1. Initial Setup
```bash
# Run setup script
./setup.sh

# OR manually:
cp .env.example .env
nano .env  # Add your GEMINI_API_KEY
```

### 2. Configure LM Studio (Optional - for Qwen)

On Windows (not in WSL):
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load `Qwen2.5-VL-7B` model
3. Settings → Enable "Serve on Local Network"
4. Start server on port 1234

### 3. Find Your Windows IP (from WSL)
```bash
# Get Windows host IP
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'

# Example output: 172.24.80.1
```

### 4. Update .env File
```bash
nano .env
```

Change:
```bash
# For Docker
LM_STUDIO_HOST=http://host.docker.internal:1234

# For direct Python (without Docker)
LM_STUDIO_HOST=http://172.24.80.1:1234  # Use your actual Windows IP
```

### 5. Test LM Studio Connection
```bash
# Test from WSL (use your Windows IP)
curl http://172.24.80.1:1234/v1/models

# Should see list of loaded models
```

### 6. Run the Application

**Option A: Docker (Recommended)**
```bash
# Build and start
docker compose up --build

# Or in background
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

**Option B: Direct Python**
```bash
# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

### 7. Access the App
Open browser: **http://localhost:7861**

---

## 🔧 Common Issues

### Issue: LM Studio not connecting

**Solution**:
```bash
# 1. Check Windows Firewall (Run in PowerShell as Admin)
New-NetFirewallRule -DisplayName "LM Studio" -Direction Inbound -Port 1234 -Protocol TCP -Action Allow

# 2. Test from WSL
WINDOWS_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
curl http://$WINDOWS_IP:1234/v1/models

# 3. Make sure LM Studio has "Serve on Local Network" enabled
```

### Issue: Docker can't find LM Studio

**Solution**: Use `host.docker.internal` in .env:
```bash
LM_STUDIO_HOST=http://host.docker.internal:1234
```

### Issue: Port 7861 in use

**Solution**:
```bash
# Change port in .env
APP_PORT=7862

# Or kill existing process
lsof -ti:7861 | xargs kill -9
```

---

## 📋 Using Make Commands

```bash
# View all commands
make help

# Initial setup
make setup

# Install dependencies
make install

# Run tests
make test

# Start app (without Docker)
make run

# Docker commands
make docker-build
make docker-up
make docker-logs
make docker-down

# Clean cache
make clean
```

---

## 🎯 Usage Workflow

1. **Start the app** → Go to http://localhost:7861

2. **Upload CV**:
   - Gemini tab: Fast cloud extraction
   - Qwen tab: Local private extraction

3. **Copy JSON** from extraction output

4. **Job Matching**:
   - Paste CV JSON
   - Enter job details
   - Add GitHub/LinkedIn URLs (optional)
   - Click "Match Job"

5. **Review results** → Get scores and recommendations

---

## 🌐 URLs

- **Application**: http://localhost:7861
- **LM Studio API**: http://localhost:1234 (on Windows)
- **From WSL**: http://YOUR_WINDOWS_IP:1234

---

## 📝 Environment Variables

```bash
# Required
GEMINI_API_KEY=your_key_here

# Optional (for Qwen)
LM_STUDIO_HOST=http://host.docker.internal:1234  # Docker
# OR
LM_STUDIO_HOST=http://172.X.X.X:1234  # Your Windows IP

# Optional
QWEN_MODEL=qwen2.5-vl-7b
APP_HOST=0.0.0.0
APP_PORT=7861
DEBUG=true
```

---

## 🆘 Need Help?

1. Run diagnostics: `./test.sh`
2. Check logs: `docker compose logs -f`
3. Verify LM Studio: Check it's running on Windows
4. Test connectivity: `curl http://YOUR_WINDOWS_IP:1234/v1/models`

---

**Ready to go!** 🚀
