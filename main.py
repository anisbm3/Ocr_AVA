#!/usr/bin/env python3
"""
Enhanced CV Parsing & Job Matching Interface  
Dual extraction methods: Google Gemini & Qwen Vision
"""
import gradio as gr
import json
import os
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import CV extractors
from cv_extractor_gemini import GeminiCVExtractor
from cv_extractor_qwen import CVParserAgent
from llm_clients.qwen_vision_agent import QwenVisionAgent
from llm_clients.llm_client import LLMClient
from job_matcher import JobMatcherAgent

# Initialize components lazily
gemini_extractor = None
qwen_agent = None
llm_client = None
qwen_parser = None
job_matcher = None


def get_gemini_extractor():
    """Lazy initialization of Gemini CV extractor"""
    global gemini_extractor
    if gemini_extractor is None:
        try:
            gemini_extractor = GeminiCVExtractor()
            print("✓ Gemini extractor initialized")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize Gemini: {e}")
            print("   Make sure GEMINI_API_KEY is set in environment")
            gemini_extractor = None
    return gemini_extractor


def get_qwen_parser():
    """Lazy initialization of Qwen CV parser with Qwen2.5-VL-7B"""
    global qwen_agent, llm_client, qwen_parser
    if qwen_parser is None:
        try:
            lm_studio_host = os.getenv("LM_STUDIO_HOST", "http://host.docker.internal:1234")
            qwen_model = os.getenv("QWEN_MODEL", "qwen2.5-vl-7b")

            print(f"🔧 Connecting to LM Studio at: {lm_studio_host}")
            print("   If running in WSL, make sure LM Studio is running on Windows host")
            print("   If using Docker, LM Studio should be accessible via host.docker.internal")

            qwen_agent = QwenVisionAgent(lm_studio_host=lm_studio_host)
            llm_client = LLMClient(lm_studio_host=lm_studio_host, preferred_model=qwen_model)
            qwen_parser = CVParserAgent(qwen_agent, llm_client)
            print(f"✅ Qwen parser initialized with {qwen_model}")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize Qwen with LLM: {e}")
            print("🔄 Falling back to PyMuPDF-only extraction")
            print("   To enable full LLM features:")
            print("   1. Install LM Studio on Windows host")
            print("   2. Load Qwen2.5-VL-7B model")
            print("   3. Start LM Studio server on port 1234")
            print("   4. For WSL: Use host.docker.internal in LM_STUDIO_HOST")
            # Initialize without LLM components for basic extraction
            qwen_agent = None
            llm_client = None
            qwen_parser = CVParserAgent(qwen_agent, llm_client)
    return qwen_parser


def get_job_matcher():
    """Lazy initialization of job matcher with Qwen2.5-VL-7B"""
    global llm_client, job_matcher
    if job_matcher is None:
        try:
            if llm_client is None:
                lm_studio_host = os.getenv("LM_STUDIO_HOST", "http://host.docker.internal:1234")
                qwen_model = os.getenv("QWEN_MODEL", "qwen2.5-vl-7b")
                llm_client = LLMClient(lm_studio_host=lm_studio_host, preferred_model=qwen_model)
            job_matcher = JobMatcherAgent(llm_client)
            print(f"✅ Job matcher initialized with {qwen_model}")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize job matcher: {e}")
            job_matcher = JobMatcherAgent(None)
    return job_matcher


# ============================================================================
# CV PARSING - GEMINI
# ============================================================================

def parse_cv_gemini(cv_file):
    """Parse CV using Google Gemini"""
    try:
        if not cv_file:
            return "❌ Error: Please upload a resume!", None
        
        print(f"📄 Processing CV with Gemini: {cv_file.name}")
        
        # Get Gemini extractor
        extractor = get_gemini_extractor()
        if extractor is None:
            return "❌ Error: Gemini API not available. Please set GEMINI_API_KEY environment variable.", None
        
        # Extract CV data
        resume_data = extractor.extract_cv_to_json(cv_file.name)
        
        # Format the result
        display_md = format_cv_display(resume_data, "Google Gemini")
        
        # Return markdown display and JSON
        return display_md, json.dumps(resume_data, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"❌ Error parsing CV with Gemini: {str(e)}"
        print(error_msg)
        return error_msg, None


# ============================================================================
# CV PARSING - QWEN
# ============================================================================

def parse_cv_qwen(cv_file):
    """Parse CV using Qwen Vision"""
    try:
        if not cv_file:
            return "❌ Error: Please upload a resume!", None
        
        # Handle data URL input from frontend
        if isinstance(cv_file, str) and cv_file.startswith('data:'):
            import base64
            import tempfile
            import os
            
            # Extract base64 data from data URL
            header, encoded = cv_file.split(',', 1)
            file_data = base64.b64decode(encoded)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_data)
                cv_file_path = temp_file.name
            
            print(f"📄 Processing CV from data URL: {cv_file_path}")
        else:
            # Handle file object from Gradio interface
            cv_file_path = cv_file.name
            print(f"📄 Processing CV with Qwen: {cv_file_path}")
        
        # Parse CV with Qwen
        parser = get_qwen_parser()
        resume_data = asyncio.run(parser.extract_and_parse_cv(cv_file_path))
        
        # Clean up temp file if created
        if isinstance(cv_file, str) and cv_file.startswith('data:'):
            try:
                os.unlink(cv_file_path)
            except:
                pass
        
        # Format the result
        display_md = format_cv_display(resume_data, "Qwen3VL-4B")
        
        # Return markdown display and JSON
        return display_md, json.dumps(resume_data, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"❌ Error parsing CV with Qwen: {str(e)}"
        print(error_msg)
        return error_msg, None

def format_cv_display(resume_data: Dict[str, Any], model_name: str) -> str:
    """Format CV data for beautiful display"""
    personal_info = resume_data.get("personalInformation", {})
    
    display_md = f"""
# ✅ CV PARSING COMPLETE! ({model_name})

---

## 👤 Personal Information
**Name:** {personal_info.get('fullName', 'N/A')}  
**Email:** {personal_info.get('email', 'N/A')}  
**Phone:** {personal_info.get('phone', 'N/A')}  
**Location:** {personal_info.get('location', 'N/A')}  
**Links:** {', '.join(personal_info.get('links', [])) or 'None'}

**Summary:**  
{personal_info.get('summary', 'N/A')}

---

## 💼 Work Experience ({len(resume_data.get('workExperience', []))})
"""
    for exp in resume_data.get('workExperience', []):
        display_md += f"""
### {exp.get('jobTitle', 'N/A')} at {exp.get('company', 'N/A')}
📍 {exp.get('location', 'N/A')} | 📅 {exp.get('startDate', 'N/A')} - {exp.get('endDate', 'N/A')}

**Responsibilities:**
"""
        for desc in exp.get('description', []):
            display_md += f"- {desc}\n"
        
        if exp.get('tags'):
            display_md += f"\n**Technologies:** {', '.join(exp.get('tags', []))}\n"

    display_md += "\n---\n\n## 🎓 Education\n"
    for edu in resume_data.get('education', []):
        display_md += f"""
### {edu.get('degree', 'N/A')} in {edu.get('major', 'N/A')}
🏫 {edu.get('institution', 'N/A')} | 📍 {edu.get('location', 'N/A')}  
📅 {edu.get('startDate', 'N/A')} - {edu.get('endDate', 'N/A')}
"""
        if edu.get('gpa'):
            display_md += f"**GPA:** {edu.get('gpa')}\n"

    display_md += "\n---\n\n## 🛠️ Skills\n"
    skills = resume_data.get('skills', [])
    if skills:
        # Display in columns
        for i in range(0, len(skills), 5):
            skill_row = skills[i:i+5]
            display_md += "• " + " • ".join(skill_row) + "\n"
    else:
        display_md += "No skills extracted\n"

    # Projects
    if resume_data.get('projects'):
        display_md += "\n---\n\n## �� Projects\n"
        for proj in resume_data.get('projects', []):
            display_md += f"""
### {proj.get('name', 'N/A')}
{proj.get('description', 'N/A')}

**Technologies:** {', '.join(proj.get('technologies', []))}
"""
            if proj.get('url'):
                display_md += f"**Link:** {proj.get('url')}\n"

    # Languages
    if resume_data.get('languages'):
        display_md += "\n---\n\n## 🌍 Languages\n"
        for lang in resume_data.get('languages', []):
            display_md += f"• **{lang.get('language', 'N/A')}:** {lang.get('proficiency', 'N/A')}\n"

    # Certifications
    if resume_data.get('certifications'):
        display_md += "\n---\n\n## 🏆 Certifications\n"
        for cert in resume_data.get('certifications', []):
            display_md += f"• {cert.get('name', 'N/A')} - {cert.get('issuer', 'N/A')} ({cert.get('date', 'N/A')})\n"

    display_md += "\n---\n\n✨ **Ready for job matching!** Copy the JSON output to the Job Matching tab."
    
    return display_md


# ============================================================================
# JOB MATCHING
# ============================================================================

def match_job(cv_json: str, job_title: str, job_requirements: str, job_description: str, 
              github_url: str = "", linkedin_url: str = "") -> str:
    """Match CV with job requirements using LLM"""
    try:
        if not cv_json or not job_title:
            return "❌ Error: Please provide CV JSON and job title"
        
        print(f"🔍 Matching CV with job: {job_title}")
        
        # Parse CV JSON
        try:
            cv_data = json.loads(cv_json)
        except json.JSONDecodeError:
            return "❌ Error: Invalid CV JSON format"
        
        # Get job matcher
        matcher = get_job_matcher()
        
        # Prepare job data
        job_data = {
            "title": job_title,
            "requirements": job_requirements,
            "description": job_description
        }
        
        # Add URLs if provided
        urls = []
        if github_url:
            urls.append(github_url)
        if linkedin_url:
            urls.append(linkedin_url)
        
        # Run matching
        result = asyncio.run(matcher.match_cv_to_job(
            cv_data,
            job_data["title"],
            job_data["description"],
            job_data["requirements"],
            github_url=github_url if github_url else "",
            linkedin_url=linkedin_url if linkedin_url else ""
        ))
        
        # Format result
        display_md = f"""
# 🎯 JOB MATCHING RESULTS

---

## 📊 Overall Score: {result.get('overall_score', 0):.1f}/100

### Score Breakdown:
- **Skills Match:** {result.get('skills_score', 0):.1f}/100 (Weight: 50%)
- **Experience Match:** {result.get('experience_score', 0):.1f}/100 (Weight: 30%)
- **Education Match:** {result.get('education_score', 0):.1f}/100 (Weight: 20%)

---

## ✅ Matching Skills
{format_list(result.get('matching_skills', []))}

## ⚠️ Missing Skills
{format_list(result.get('missing_skills', []))}

---

## 💼 Experience Analysis
**Years of Experience:** {result.get('years_of_experience', 'N/A')}  
**Relevant Experience:** {result.get('relevant_experience', 'N/A')}

---

## 🎓 Education Match
{result.get('education_match', 'N/A')}

---

## 📝 Recommendation
{result.get('recommendation', 'N/A')}

---

## 🌐 Web Scraping Results
"""
        if result.get('web_scraping_data'):
            for url, data in result.get('web_scraping_data', {}).items():
                display_md += f"\n**{url}:**\n{json.dumps(data, indent=2)}\n"
        else:
            display_md += "No web scraping data available\n"

        return display_md
        
    except Exception as e:
        error_msg = f"❌ Error matching job: {str(e)}"
        print(error_msg)
        return error_msg


def format_list(items):
    """Format list items for display"""
    if not items:
        return "None"
    return "\n".join([f"• {item}" for item in items])


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="CV Parser & Job Matcher", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🎯 AI-Powered CV Parser & Job Matcher
        
        **Two extraction methods available:**
        - 🌟 **Google Gemini**: Fast, cloud-based extraction with Google's latest AI
        - 🤖 **Qwen3VL-4B**: Local extraction using LM Studio (requires setup)
        
        ---
        """)
        
        with gr.Tabs():
            # ====== TAB 1: CV EXTRACTION - GEMINI ======
            with gr.Tab("�� CV Extraction (Gemini)"):
                gr.Markdown("""
                ## Extract CV Data using Google Gemini
                
                **Requirements:** GEMINI_API_KEY environment variable must be set
                
                Upload your CV (PDF, DOC, DOCX, or image) and get structured JSON output.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gemini_file_input = gr.File(
                            label="Upload CV/Resume",
                            file_types=[".pdf", ".doc", ".docx", ".jpg", ".jpeg", ".png", ".txt", ".rtf"]
                        )
                        gemini_parse_btn = gr.Button("🚀 Extract with Gemini", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        gemini_display_output = gr.Markdown(label="Parsed CV")
                
                with gr.Row():
                    gemini_json_output = gr.Code(
                        label="CV JSON (Copy this for job matching)",
                        language="json",
                        lines=15
                    )
                
                gemini_parse_btn.click(
                    fn=parse_cv_gemini,
                    inputs=[gemini_file_input],
                    outputs=[gemini_display_output, gemini_json_output]
                )
            
            # ====== TAB 2: CV EXTRACTION - QWEN ======
            with gr.Tab("📄 CV Extraction (Qwen)"):
                gr.Markdown("""
                ## Extract CV Data using Qwen3VL-4B
                
                **Requirements:** LM Studio running on 0.0.0.0:1234 with Qwen3VL-4B model
                
                Upload your CV (PDF, DOC, DOCX, or image) and get structured JSON output.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        qwen_file_input = gr.File(
                            label="Upload CV/Resume",
                            file_types=[".pdf", ".doc", ".docx", ".jpg", ".jpeg", ".png", ".txt", ".rtf"]
                        )
                        qwen_parse_btn = gr.Button("🚀 Extract with Qwen", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        qwen_display_output = gr.Markdown(label="Parsed CV")
                
                with gr.Row():
                    qwen_json_output = gr.Code(
                        label="CV JSON (Copy this for job matching)",
                        language="json",
                        lines=15
                    )
                
                qwen_parse_btn.click(
                    fn=parse_cv_qwen,
                    inputs=[qwen_file_input],
                    outputs=[qwen_display_output, qwen_json_output]
                )
            
            # ====== TAB 3: JOB MATCHING ======
            with gr.Tab("🎯 Job Matching"):
                gr.Markdown("""
                ## Match CV with Job Requirements
                
                Paste the CV JSON from the extraction tab and provide job details for intelligent matching.
                """)
                
                with gr.Row():
                    with gr.Column():
                        cv_json_input = gr.Code(
                            label="CV JSON (from extraction tab)",
                            language="json",
                            lines=10
                        )
                        
                        job_title_input = gr.Textbox(
                            label="Job Title",
                            placeholder="e.g., Senior Python Developer"
                        )
                        
                        job_requirements_input = gr.TextArea(
                            label="Job Requirements",
                            placeholder="e.g., 5+ years Python, FastAPI, AI/ML experience",
                            lines=5
                        )
                        
                        job_description_input = gr.TextArea(
                            label="Job Description (Optional)",
                            placeholder="Full job description...",
                            lines=5
                        )
                        
                        with gr.Row():
                            github_url_input = gr.Textbox(
                                label="GitHub URL (Optional)",
                                placeholder="https://github.com/username"
                            )
                            linkedin_url_input = gr.Textbox(
                                label="LinkedIn URL (Optional)",
                                placeholder="https://linkedin.com/in/username"
                            )
                        
                        match_btn = gr.Button("🔍 Match Job", variant="primary", size="lg")
                    
                    with gr.Column():
                        match_output = gr.Markdown(label="Matching Results")
                
                match_btn.click(
                    fn=match_job,
                    inputs=[
                        cv_json_input,
                        job_title_input,
                        job_requirements_input,
                        job_description_input,
                        github_url_input,
                        linkedin_url_input
                    ],
                    outputs=[match_output]
                )
        
        gr.Markdown("""
        ---
        ### 📚 Usage Instructions:
        
        1. **Extract CV**: Choose either Gemini or Qwen tab and upload your CV
        2. **Copy JSON**: Copy the JSON output from the extraction
        3. **Match Job**: Paste the JSON in the Job Matching tab and provide job details
        4. **Review Results**: Get detailed matching scores and recommendations
        
        ### 🔧 Setup:
        - **Gemini**: Set `GEMINI_API_KEY` environment variable
        - **Qwen**: Run LM Studio with Qwen3VL-4B model on 0.0.0.0:1234
        - **PyMuPDF**: Install with `pip install PyMuPDF` for better PDF extraction
        """)
    
    return app


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    # Get configuration from environment
    app_host = os.getenv("APP_HOST", "0.0.0.0")
    app_port = int(os.getenv("APP_PORT", "7861"))
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    lm_studio_host = os.getenv("LM_STUDIO_HOST", "http://host.docker.internal:1234")
    qwen_model = os.getenv("QWEN_MODEL", "qwen2.5-vl-7b")

    print(f"""
    ======================================================================
    Starting Enhanced CV Parser & Job Matcher
    ======================================================================

    Features:
      - Dual CV extraction: Gemini & Qwen Vision
      - Intelligent job matching with LLM scoring
      - Web scraping for GitHub/LinkedIn profiles
      - Beautiful UI with structured output

    Configuration:
      - App Host: {app_host}:{app_port}
      - LM Studio: {lm_studio_host}
      - Qwen Model: {qwen_model}
      - Debug Mode: {debug_mode}

    OpenAI-Compatible Endpoints Used:
      - GET  /v1/models (model listing)
      - POST /v1/chat/completions (text generation & vision)

    ======================================================================
    """)

    app = create_interface()
    app.launch(
        server_name=app_host,
        server_port=app_port,
        share=False,
        debug=debug_mode
    )
