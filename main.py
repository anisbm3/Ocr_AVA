#!/usr/bin/env python3
"""
Enhanced CV Parsing & Job Matching Interface  
Dual extraction methods: Google Gemini & Qwen Vision
"""
import gradio as gr
import json
import os
import shutil
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

# Additional imports for integrated features
from carreer_advisor import (
    remove_personal_info,
    load_prompt_files,
    combine_prompt_parts,
    query_model,
    run_advisor,
    apply_feedback,
)
from cv_reviewer.cv_review import review_cv, prepare_ats_prompt, prepare_ats_prompt_multilingual, review_cv_multilingual
from cv_reviewer.cv_rewriter import rewrite_cv

# Initialize components lazily
gemini_extractor = None
qwen_agent = None
llm_client = None
qwen_parser = None
job_matcher = None

# Define default career paths
DEFAULT_PATHS = ["Data Science", "Software Engineer", "Product Manager", "DevOps", "Research", "AI/ML Engineer"]

def career_advisor_fn(cv_json_str: str, desired_paths: list, intentions: str, temperature: float = 0.7, max_tokens: int = 8192):
    """
    Main career advisor function:
    1. Remove personal info from CV
    2. Assemble advisor input with desired paths and intentions
    3. Build prompt using prompt file + example + template
    4. Call model and return response as parsed JSON
    """
    # Load prompt files if not already loaded
    load_prompt_files()
    
    # Remove personal info
    cv_anonymized = remove_personal_info(cv_json_str)
    if "error" in cv_anonymized:
        return cv_anonymized
    
    # Assemble advisor input
    advisor_input = {
        "cv_anonymized": cv_anonymized,
        "careerIntentions": intentions or "",
        "desiredPaths": desired_paths or []
    }
    
    # Build prompt
    prompt = combine_prompt_parts(advisor_input)
    
    # Query model
    response = query_model(prompt, temperature=temperature, max_tokens=max_tokens)
    
    # Try to parse response as JSON for gr.JSON output
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # If response is not valid JSON, return error dict
        return {
            "error": "Model response was not valid JSON",
            "raw_response": response[:1000] + ("..." if len(response) > 1000 else "")
        }

def apply_feedback_fn(original_output: str, step_identifier: str, feedback_json_str: str, temperature: float = 0.7, max_tokens: int = 8192):
    """
    Apply feedback to a specific step in the advisor output. 
    Accepts feedback as JSON string and returns the updated step as parsed JSON.
    
    Expected feedback format:
    {
        "clarityScore": <int 1-5 or null>,
        "relevanceScore": <int 1-5 or null>,
        "difficultyLevel": <"too easy"|"appropriate"|"too hard" or null>,
        "userComment": <string or null>
    }
    """
    response = apply_feedback(original_output, step_identifier, feedback_json_str, temperature, max_tokens)
    
    # Try to parse response as JSON for gr.JSON output
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "error": "Model response was not valid JSON",
            "raw_response": response[:1000] + ("..." if len(response) > 1000 else "")
        }


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


def get_qwen_parser(llm_host=None, qwen_model=None):
    """Lazy initialization of Qwen CV parser with Qwen2.5-VL-7B"""
    global qwen_agent, llm_client, qwen_parser
    if qwen_parser is None:
        try:
            lm_studio_host = llm_host or os.getenv("LM_STUDIO_HOST", "http://host.docker.internal:1234")
            qwen_model = qwen_model or os.getenv("QWEN_MODEL", "qwen2.5-vl-7b")

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


def get_job_matcher(llm_host=None, qwen_model=None, scrape_enabled=True, scrape_timeout=10):
    """Lazy initialization of job matcher with Qwen2.5-VL-7B"""
    global llm_client, job_matcher
    if job_matcher is None:
        try:
            if llm_client is None:
                lm_studio_host = llm_host or os.getenv("LM_STUDIO_HOST", "http://host.docker.internal:1234")
                qwen_model = qwen_model or os.getenv("QWEN_MODEL", "qwen2.5-vl-7b")
                llm_client = LLMClient(lm_studio_host=lm_studio_host, preferred_model=qwen_model)
            job_matcher = JobMatcherAgent(llm_client, scrape_enabled, scrape_timeout)
            print(f"✅ Job matcher initialized with {qwen_model}")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize job matcher: {e}")
            job_matcher = JobMatcherAgent(None, scrape_enabled, scrape_timeout)
    return job_matcher


# ============================================================================
# CV PARSING - GEMINI
# ============================================================================

def parse_cv_gemini(cv_file, max_upload_mb):
    """Parse CV using Google Gemini"""
    try:
        if not cv_file:
            return "❌ Error: Please upload a resume!", None
        
        # Check file size
        if hasattr(cv_file, 'size'):
            file_size_mb = cv_file.size / (1024 * 1024)
            if file_size_mb > max_upload_mb:
                return f"❌ Error: File size ({file_size_mb:.1f}MB) exceeds limit ({max_upload_mb}MB)!", None
        
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

def parse_cv_qwen(cv_file, llm_host, qwen_model, max_upload_mb):
    """Parse CV using Qwen Vision"""
    try:
        print(f"🔍 Received cv_file type: {type(cv_file)}")
        print(f"🔍 cv_file value: {cv_file}")
        
        if not cv_file:
            return "❌ Error: Please upload a resume!", None
        
        # Check file size
        if hasattr(cv_file, 'size'):
            file_size_mb = cv_file.size / (1024 * 1024)
            if file_size_mb > max_upload_mb:
                return f"❌ Error: File size ({file_size_mb:.1f}MB) exceeds limit ({max_upload_mb}MB)!", None
        
        # Handle data URL input from frontend
        if isinstance(cv_file, str) and cv_file.startswith('data:'):
            import base64
            import tempfile
            
            # Extract base64 data from data URL
            header, encoded = cv_file.split(',', 1)
            file_data = base64.b64decode(encoded)
            
            # Determine file extension from MIME type
            mime_type = header.split(';')[0].split(':')[1]
            if mime_type == 'application/pdf':
                ext = '.pdf'
            elif mime_type in ['image/jpeg', 'image/jpg']:
                ext = '.jpg'
            elif mime_type == 'image/png':
                ext = '.png'
            elif mime_type == 'text/plain':
                ext = '.txt'
            else:
                ext = '.pdf'  # Default fallback
            
            # Create temporary file with proper extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                temp_file.write(file_data)
                cv_file_path = temp_file.name
            
            print(f"📄 Processing CV from data URL: {cv_file_path} (MIME: {mime_type})")
        elif hasattr(cv_file, 'name'):
            # Handle file object from Gradio interface
            cv_file_path = cv_file.name
            print(f"📄 Processing CV: {cv_file_path}")
            
            # Check if file has no extension (Gradio temporary files)
            if not os.path.splitext(cv_file_path)[1]:
                # Copy the file to a new path with .pdf extension instead of renaming
                # This avoids the file locking issue with Gradio
                new_path = cv_file_path + '.pdf'
                try:
                    shutil.copy2(cv_file_path, new_path)
                    cv_file_path = new_path
                    print(f"📄 Copied to PDF: {cv_file_path}")
                except Exception as e:
                    print(f"⚠️ Could not copy file: {e}")
                    print(f"📄 Using original path: {cv_file_path}")
        else:
            # Handle other file formats
            cv_file_path = str(cv_file)
            print(f"📄 Processing CV (string path): {cv_file_path}")
        
        # Parse CV with Qwen
        parser = get_qwen_parser(llm_host, qwen_model)
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
        import traceback
        traceback.print_exc()
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
              github_url: str = "", linkedin_url: str = "", llm_host=None, qwen_model=None, 
              scrape_enabled=True, scrape_timeout=10) -> str:
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
        matcher = get_job_matcher(llm_host, qwen_model, scrape_enabled, scrape_timeout)
        
        # Prepare job data
        job_data = {
            "title": job_title,
            "requirements": job_requirements,
            "description": job_description
        }
        
        # Add URLs if provided and scraping enabled
        urls = []
        if scrape_enabled:
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
            github_url=github_url if scrape_enabled and github_url else "",
            linkedin_url=linkedin_url if scrape_enabled and linkedin_url else ""
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
        if scrape_enabled and result.get('web_scraping_data'):
            for url, data in result.get('web_scraping_data', {}).items():
                display_md += f"\n**{url}:**\n{json.dumps(data, indent=2)}\n"
        elif scrape_enabled:
            display_md += "No web scraping data available\n"
        else:
            display_md += "Web scraping disabled\n"

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
    
    # Initialize settings from environment
    default_llm_host = os.getenv("LM_STUDIO_HOST", "http://host.docker.internal:1234")
    default_qwen_model = os.getenv("QWEN_MODEL", "qwen/qwen3-vl-4b")
    default_scrape_enabled = True
    default_scrape_timeout = int(os.getenv("WEB_SCRAPE_TIMEOUT", "10"))
    default_max_upload_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    default_cv_template_path = os.getenv("CV_TEMPLATE_PATH", "./cv_template_camelCase.json")
    default_job_desc_path = os.getenv("JOB_DESCRIPTIONS_PATH", "./job_descriptions.json")
    
    with gr.Blocks(title="AI-Powered CV Tools: Parser, Matcher, Reviewer & Advisor", theme=gr.themes.Soft()) as app:
        
        # State management for settings
        llm_host_state = gr.State(default_llm_host)
        qwen_model_state = gr.State(default_qwen_model)
        scrape_enabled_state = gr.State(default_scrape_enabled)
        scrape_timeout_state = gr.State(default_scrape_timeout)
        max_upload_mb_state = gr.State(default_max_upload_mb)
        cv_template_path_state = gr.State(default_cv_template_path)
        job_desc_path_state = gr.State(default_job_desc_path)
        
        gr.Markdown("""
        # 🎯 AI-Powered CV Tools: Parser, Matcher, Reviewer & Advisor
        
        **Comprehensive CV processing suite:**
        - 🌟 **CV Extraction**: Google Gemini & Qwen Vision for accurate parsing
        - 🎯 **Job Matching**: Intelligent scoring with LLM analysis
        - 📝 **CV Review**: ATS-optimized feedback in multiple languages
        - ✍️ **CV Rewrite**: Automatic optimization with strong action verbs
        - 🎓 **Career Advisor**: Personalized guidance and learning paths
        - 🔄 **Feedback System**: Iterative improvement of advice
        
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
                            file_types=[".pdf", ".doc", ".docx", ".jpg", ".jpeg", ".png", ".txt"]
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
                    inputs=[gemini_file_input, max_upload_mb_state],
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
                            file_types=[".pdf", ".doc", ".docx", ".jpg", ".jpeg", ".png", ".txt"]
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
                    inputs=[qwen_file_input, llm_host_state, qwen_model_state, max_upload_mb_state],
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
                        linkedin_url_input,
                        llm_host_state,
                        qwen_model_state,
                        scrape_enabled_state,
                        scrape_timeout_state
                    ],
                    outputs=[match_output]
                )
            
            # ====== TAB 4: SETTINGS & CONFIGURATION ======
            with gr.Tab("⚙️ Settings & Configuration"):
                gr.Markdown("""
                ## Advanced Configuration
                
                Configure LLM models, web scraping settings, and file paths.
                Changes take effect immediately for new operations.
                """)
                
                with gr.Accordion("🤖 LLM Configuration", open=True):
                    gr.Markdown("Configure LLM client settings for Qwen and other models.")
                    
                    llm_host_input = gr.Textbox(
                        label="LM Studio Host",
                        value=default_llm_host,
                        placeholder="http://host.docker.internal:1234"
                    )
                    
                    qwen_model_input = gr.Dropdown(
                        label="Qwen Model",
                        choices=["qwen/qwen3-vl-4b", "qwen2.5-vl-7b", "qwen2-vl-7b", "other"],
                        value=default_qwen_model,
                        allow_custom_value=True
                    )
                    
                    test_llm_btn = gr.Button("🧪 Test LLM Connection", variant="secondary")
                    test_llm_output = gr.Textbox(label="Test Result", interactive=False)
                    
                    def update_llm_settings(host, model):
                        llm_host_state.value = host
                        qwen_model_state.value = model
                        return f"✅ Settings updated: Host={host}, Model={model}"
                    
                    def test_llm_connection(host, model):
                        try:
                            # Test connection by initializing LLM client
                            test_client = LLMClient(lm_studio_host=host, preferred_model=model)
                            # Try a simple request
                            import asyncio
                            response = asyncio.run(test_client.generate("Hello", max_tokens=10))
                            return f"✅ Connection successful! Response: {response[:50]}..."
                        except Exception as e:
                            return f"❌ Connection failed: {str(e)}"
                    
                    llm_host_input.change(
                        fn=update_llm_settings,
                        inputs=[llm_host_input, qwen_model_input],
                        outputs=[]
                    )
                    qwen_model_input.change(
                        fn=update_llm_settings,
                        inputs=[llm_host_input, qwen_model_input],
                        outputs=[]
                    )
                    
                    test_llm_btn.click(
                        fn=test_llm_connection,
                        inputs=[llm_host_input, qwen_model_input],
                        outputs=[test_llm_output]
                    )
                
                with gr.Accordion("🌐 Web Scraping Controls", open=False):
                    gr.Markdown("Control web scraping behavior for GitHub/LinkedIn profiles.")
                    
                    scrape_enabled_input = gr.Checkbox(
                        label="Enable Web Scraping",
                        value=default_scrape_enabled
                    )
                    
                    scrape_timeout_input = gr.Number(
                        label="Scraping Timeout (seconds)",
                        value=default_scrape_timeout,
                        minimum=1,
                        maximum=60
                    )
                    
                    def update_scrape_settings(enabled, timeout):
                        scrape_enabled_state.value = enabled
                        scrape_timeout_state.value = timeout
                        return f"✅ Web scraping settings updated: Enabled={enabled}, Timeout={timeout}s"
                    
                    scrape_enabled_input.change(
                        fn=update_scrape_settings,
                        inputs=[scrape_enabled_input, scrape_timeout_input],
                        outputs=[]
                    )
                    scrape_timeout_input.change(
                        fn=update_scrape_settings,
                        inputs=[scrape_enabled_input, scrape_timeout_input],
                        outputs=[]
                    )
                
                with gr.Accordion("📁 File Configuration", open=False):
                    gr.Markdown("Configure file paths and upload limits.")
                    
                    max_upload_mb_input = gr.Number(
                        label="Maximum Upload Size (MB)",
                        value=default_max_upload_mb,
                        minimum=1,
                        maximum=100
                    )
                    
                    cv_template_path_input = gr.Textbox(
                        label="CV Template Path",
                        value=default_cv_template_path,
                        placeholder="./cv_template_camelCase.json"
                    )
                    
                    job_desc_path_input = gr.Textbox(
                        label="Job Descriptions Path",
                        value=default_job_desc_path,
                        placeholder="./job_descriptions.json"
                    )
                    
                    def update_file_settings(max_mb, cv_path, job_path):
                        max_upload_mb_state.value = max_mb
                        cv_template_path_state.value = cv_path
                        job_desc_path_state.value = job_path
                        return f"✅ File settings updated: Max={max_mb}MB, CV={cv_path}, Jobs={job_path}"
                    
                    max_upload_mb_input.change(
                        fn=update_file_settings,
                        inputs=[max_upload_mb_input, cv_template_path_input, job_desc_path_input],
                        outputs=[]
                    )
                    cv_template_path_input.change(
                        fn=update_file_settings,
                        inputs=[max_upload_mb_input, cv_template_path_input, job_desc_path_input],
                        outputs=[]
                    )
                    job_desc_path_input.change(
                        fn=update_file_settings,
                        inputs=[max_upload_mb_input, cv_template_path_input, job_desc_path_input],
                        outputs=[]
                    )
            
            # ====== TAB 5: CV REVIEWER ======
            with gr.Tab("📝 CV Reviewer"):
                gr.Markdown("""
                ## CV Reviewer (English)
                
                Reviews your CV using OpenRouter (primary) or LMStudio (fallback). Provides detailed ATS-optimized feedback in English.
                """)
                
                cv_json_reviewer = gr.Textbox(lines=20, label="Paste CV JSON", placeholder='{"personalInformation": {...}, "experience": [...], ...}')
                temp_reviewer = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="Temperature")
                max_tokens_reviewer = gr.Slider(minimum=128, maximum=16384, value=2048, step=128, label="Max Tokens")
                reviewer_output = gr.Textbox(label="Review Output")
                
                reviewer_btn = gr.Button("Review CV")
                reviewer_btn.click(
                    fn=review_cv,
                    inputs=[cv_json_reviewer, temp_reviewer, max_tokens_reviewer],
                    outputs=[reviewer_output]
                )
            
            # ====== TAB 6: CV REVIEWER MULTILINGUAL ======
            with gr.Tab("📝 CV Reviewer (Multilingual)"):
                gr.Markdown("""
                ## CV Reviewer (Multilingual)
                
                Reviews your CV with automatic language detection (English/French/Arabic). Responds in the same language as your CV. Uses OpenRouter (primary) or LMStudio (fallback).
                """)
                
                cv_json_reviewer_multi = gr.Textbox(lines=20, label="Paste CV JSON", placeholder='{"personalInformation": {...}, "experience": [...], ...}')
                temp_reviewer_multi = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="Temperature")
                max_tokens_reviewer_multi = gr.Slider(minimum=128, maximum=16384, value=4000, step=128, label="Max Tokens")
                reviewer_multi_output = gr.Textbox(label="Review Output")
                
                reviewer_multi_btn = gr.Button("Review CV")
                reviewer_multi_btn.click(
                    fn=review_cv_multilingual,
                    inputs=[cv_json_reviewer_multi, temp_reviewer_multi, max_tokens_reviewer_multi],
                    outputs=[reviewer_multi_output]
                )
            
            # ====== TAB 7: CV REWRITER ======
            with gr.Tab("✍️ CV Rewriter"):
                gr.Markdown("""
                ## CV Rewriter
                
                Rewrites your CV to be ATS-optimized. Applies XYZ pattern (Accomplished X, measured by Y, by doing Z) to all bullets. Adds quantifiable metrics and uses strong action verbs. Uses OpenRouter (primary) or LMStudio (fallback).
                """)
                
                cv_json_rewriter = gr.Textbox(lines=20, label="Paste CV JSON", placeholder='{"personalInformation": {...}, "experience": [...], ...}')
                temp_rewriter = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="Temperature")
                max_tokens_rewriter = gr.Slider(minimum=128, maximum=16384, value=8192, step=128, label="Max Tokens")
                rewriter_output = gr.Textbox(label="Rewritten CV")
                
                rewriter_btn = gr.Button("Rewrite CV")
                rewriter_btn.click(
                    fn=rewrite_cv,
                    inputs=[cv_json_rewriter, temp_rewriter, max_tokens_rewriter],
                    outputs=[rewriter_output]
                )
            
            # ====== TAB 8: CAREER ADVISOR ======
            with gr.Tab("🎓 Career Advisor"):
                gr.Markdown("""
                ## Career Advisor
                
                Provides personalized career guidance based on your CV, desired paths, and intentions.
                """)
                
                cv_json_advisor = gr.Textbox(lines=20, label="Paste Full CV JSON", placeholder='{"skills": [...], "experience": [...], ...}')
                desired_paths_advisor = gr.CheckboxGroup(choices=DEFAULT_PATHS, label="Desired Career Paths (select one or more)", value=[])
                intentions_advisor = gr.Textbox(lines=3, label="Career Intentions / Goals", placeholder="What are your career goals?")
                temp_advisor = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="Temperature")
                max_tokens_advisor = gr.Slider(minimum=128, maximum=16384, value=8192, step=128, label="Max Tokens")
                advisor_output = gr.JSON(label="Career Advisor Output (JSON)")
                
                advisor_btn = gr.Button("Get Career Advice")
                advisor_btn.click(
                    fn=career_advisor_fn,
                    inputs=[cv_json_advisor, desired_paths_advisor, intentions_advisor, temp_advisor, max_tokens_advisor],
                    outputs=[advisor_output]
                )
            
            # ====== TAB 9: APPLY FEEDBACK ======
            with gr.Tab("🔄 Apply Feedback"):
                gr.Markdown("""
                ## Apply Feedback to Learning Path Step
                
                Provide feedback on a specific step. Fill in the feedback JSON with your scores (1-5), difficulty level ("too easy", "appropriate", "too hard"), and comments.
                """)
                
                original_output_feedback = gr.Textbox(lines=20, label="Original Advisor Output (Full JSON)", placeholder="Paste the complete JSON output from Career Advisor")
                step_identifier_feedback = gr.Textbox(label="Step Number", placeholder="e.g., '1', '2', '3'")
                feedback_json_feedback = gr.Textbox(
                    lines=8, 
                    label="Feedback JSON", 
                    placeholder='{\n  "clarityScore": null,\n  "relevanceScore": null,\n  "difficultyLevel": null,\n  "userComment": null\n}',
                    value='{\n  "clarityScore": null,\n  "relevanceScore": null,\n  "difficultyLevel": null,\n  "userComment": null\n}'
                )
                temp_feedback = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="Temperature")
                max_tokens_feedback = gr.Slider(minimum=128, maximum=16384, value=8192, step=128, label="Max Tokens")
                feedback_output = gr.JSON(label="Updated Step (JSON)")
                
                feedback_btn = gr.Button("Apply Feedback")
                feedback_btn.click(
                    fn=apply_feedback_fn,
                    inputs=[original_output_feedback, step_identifier_feedback, feedback_json_feedback, temp_feedback, max_tokens_feedback],
                    outputs=[feedback_output]
                )
        
        gr.Markdown("""
        ---
        ### 📚 Usage Instructions:
        
        1. **Extract CV**: Choose either Gemini or Qwen tab and upload your CV
        2. **Copy JSON**: Copy the JSON output from the extraction
        3. **Match Job**: Paste the JSON in the Job Matching tab and provide job details
        4. **Review Results**: Get detailed matching scores and recommendations
        5. **Review CV**: Use CV Reviewer tabs for ATS-optimized feedback
        6. **Rewrite CV**: Optimize your CV with the Rewriter tab
        7. **Get Career Advice**: Use Career Advisor for personalized guidance
        8. **Apply Feedback**: Refine learning paths with the Feedback tab
        
        ### 🔧 Setup:
        - **Gemini**: Set `GEMINI_API_KEY` environment variable
        - **Qwen**: Run LM Studio with Qwen3VL-4B model on 0.0.0.0:1234
        - **PyMuPDF**: Install with `pip install PyMuPDF` for better PDF extraction
        - **OpenRouter/LMStudio**: For CV review/rewrite/advisor features
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
    Starting AI-Powered CV Tools Suite
    ======================================================================

    Features:
      - Dual CV extraction: Gemini & Qwen Vision
      - Intelligent job matching with LLM scoring
      - Web scraping for GitHub/LinkedIn profiles
      - CV review and ATS optimization (English/Multilingual)
      - CV rewriting with strong action verbs
      - Career advisor with personalized guidance
      - Feedback system for iterative improvement
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

    # Add CORS headers for frontend integration
    @app.app.middleware("http")
    async def add_cors_headers(request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    app.launch(
        server_name=app_host,
        server_port=app_port,
        share=False,
        debug=debug_mode,
        allowed_paths=["/"]  # Allow all paths for CORS
    )
