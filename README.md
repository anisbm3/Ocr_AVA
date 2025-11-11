# AI-Powered CV Parser & Job Matcher

Intelligent CV parsing and job matching platform with dual extraction methods (Gemini & Qwen Vision), LLM-powered analysis, web scraping, and modern Gradio interface.

## 🏗️ Project Structure

```
python/
├── cv_extractor_gemini/         # Gemini-based CV extraction
├── cv_extractor_qwen/           # Qwen-based CV extraction
├── job_database/                # Job storage and database operations
├── job_matcher/                 # Intelligent job matching with LLM
├── web_scraper/                 # GitHub/LinkedIn profile scraping
├── llm_clients/                 # LLM client implementations
├── main.py                      # Main Gradio web interface
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker container configuration
├── docker-compose.yml           # Docker Compose setup
├── .env                         # Environment variables
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+ or Docker
- **For Qwen Vision**: LM Studio running with Qwen2.5-VL-7B model
- **For Gemini**: Google Gemini API key

### OpenAI-Compatible Endpoints

The application uses these OpenAI-compatible endpoints via LM Studio:

- **GET** `/v1/models` - List available models
- **POST** `/v1/chat/completions` - Text generation and vision analysis

### WSL Setup (Windows Subsystem for Linux)

If running in WSL, LM Studio runs on Windows host. Configure connectivity:

```bash
# Find Windows host IP from WSL
ip route | grep default | awk '{print $3}'

# Or use host.docker.internal for Docker
echo "LM_STUDIO_HOST=http://host.docker.internal:1234" >> .env
```

### Installation

#### Option 1: Native Python
```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

#### Option 2: Docker (Recommended)
```bash
# Build and run with Docker Compose
docker compose up --build
```

### WSL Setup (Windows Subsystem for Linux)

If running in WSL, LM Studio runs on Windows host. The application automatically handles this:

```bash
# Automatic WSL setup (recommended)
./setup_wsl.sh

# Or manually configure:
# Find Windows host IP
ip route | grep default | awk '{print $3}'

# Update .env file
echo "LM_STUDIO_HOST=http://YOUR_WINDOWS_IP:1234" >> .env
```

**WSL Troubleshooting:**
- Use `host.docker.internal` for Docker containers
- Use Windows IP (e.g., `172.XX.X.X`) for native Python
- Ensure LM Studio server is running on `0.0.0.0:1234` (not localhost)

### Launch the Application

```bash
# Native Python
python main.py

# Docker (handles WSL networking automatically)
docker compose up
```

### Access the Application
- **Gradio Interface**: <http://localhost:7861> or <http://0.0.0.0:7861>

## 🤖 AI Features

### CV Parsing & Analysis
- **Qwen Vision**: Advanced text extraction from CV images/PDFs using Qwen3VL-4B
- **Structured Parsing**: Extracts personal info, work experience, education, skills
- **LLM Enhancement**: Intelligent analysis with LM Studio integration

### Intelligent Job Matching
- **LLM-Powered Skills Analysis**: Advanced skill matching with synonym recognition
- **Experience Evaluation**: Years of experience and role relevance assessment
- **Education Verification**: Degree level and qualification matching
- **Web Scraping**: GitHub and LinkedIn profile analysis for enhanced verification
- **Comprehensive Scoring**: Weighted scoring across skills (50%), experience (30%), education (20%)

### Video CV Analysis (Optional)
- **Multi-modal Analysis**: Visual assessment using Qwen Vision
- **Audio Transcription**: Speech-to-text using Whisper AI
- **Combined Insights**: Integrated visual and audio candidate evaluation

### Web Scraping Integration
- **GitHub Analysis**: Repository count, programming languages, project activity
- **LinkedIn Scraping**: Profile verification and additional skill discovery
- **Score Boosting**: Online presence verification improves matching scores

## 🔄 Three-Phase Workflow

The Gradio interface provides a streamlined three-phase process:

### Phase 1: CV Parsing
- Upload CV documents (PDF/Image)
- AI extracts structured data (personal info, experience, skills, education)
- Outputs JSON for next phase

### Phase 2: Job Matching
- Input job requirements and description
- Optional: Provide GitHub/LinkedIn URLs for enhanced analysis
- LLM-powered matching with detailed scoring and recommendations
- Web scraping integration for additional verification

### Phase 3: Video CV Analysis (Optional)
- Upload video CV/resume
- Multi-modal analysis with visual assessment and audio transcription
- Combined insights for comprehensive candidate evaluation

## 🔧 Configuration

The system auto-detects LM Studio:
- **LM Studio URL**: http://0.0.0.0:1234
- **Model**: Qwen3VL-4B (recommended)
- **Fallback**: Basic OCR if LLM unavailable

## 📝 API Endpoints

### Jobs
- `POST /api/jobs` - Create job posting
- `GET /api/jobs` - List all jobs

### Applications
- `POST /api/applications` - Submit CV application
- `GET /api/applications` - List all applications
- `GET /api/applications/{job_id}` - Get applications for specific job

### Candidates
- `GET /api/candidates` - List all candidates

### Assessments
- `GET /api/assessments` - List all assessments
- `GET /api/assessments/{candidate_id}` - Get candidate's assessments

### Stats
- `GET /api/stats` - Get system statistics

## 🎯 Usage Example

### Web Interface (Recommended)
1. **Launch the interface**: `python main.py`
2. **Phase 1**: Upload a CV document and click "PARSE CV"
3. **Phase 2**: Copy the JSON output, paste in job matching tab, add job details and optional GitHub/LinkedIn URLs
4. **Phase 3**: Optionally upload a video CV for comprehensive analysis

### Programmatic Usage (API)

```python
import requests

# 1. Create a job posting
job = {
    "title": "Senior Python Developer",
    "department": "Engineering",
    "requirements": "5+ years Python, FastAPI, AI/ML experience"
}
response = requests.post("http://0.0.0.0:8000/api/jobs", json=job)
job_id = response.json()["job_id"]

# 2. Submit a CV
files = {"cv": open("candidate_cv.pdf", "rb")}
data = {
    "job_id": job_id,
    "candidate_name": "John Doe",
    "candidate_email": "john@example.com"
}
response = requests.post("http://0.0.0.0:8000/api/applications", files=files, data=data)

# 3. Get analysis results
print(response.json())
```

## 🧹 Clean Architecture

- **Gradio Interface**: Modern web UI for easy CV processing and job matching
- **No Database**: JSON file storage only - portable and human-readable
- **No Email**: Removed SMTP dependencies for simplicity
- **AI-First**: Focus on advanced CV analysis and intelligent matching
- **Web Scraping**: Enhanced candidate verification via GitHub/LinkedIn
- **Minimal Dependencies**: Only essential packages for AI functionality

## 📦 Version

**v5.0** - Advanced AI Job Matching Edition
- Added: Gradio web interface with three-phase workflow
- Added: LLM-powered intelligent skill matching and job analysis
- Added: GitHub and LinkedIn web scraping for enhanced verification
- Added: Video CV analysis with Qwen Vision and Whisper
- Focus: Complete AI-powered CV processing pipeline

## 🛠️ Development

```bash
# Run in development mode with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 📄 License

MIT License - Feel free to use and modify
