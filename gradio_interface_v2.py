#!/usr/bin/env python3
"""
CV Parsing Gradio Interface
Direct integration with CVParserAgent for extraction and parsing
"""
import gradio as gr
import json
import os
import asyncio
from datetime import datetime

# Import CV Parser components directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ai.qwen_vision_agent import QwenVisionAgent
from ai.llm_client import LLMClient
from ai.cv_parser_agent import CVParserAgent
from ai.job_matcher_agent import JobMatcherAgent
from ai.video_analyzer import QwenVideoCVAgent

# Initialize components lazily
qwen_agent = None
llm_client = None
cv_parser = None
job_matcher = None
video_agent = None

def get_cv_parser():
    """Lazy initialization of CV parser components"""
    global qwen_agent, llm_client, cv_parser
    if cv_parser is None:
        try:
            qwen_agent = QwenVisionAgent(lm_studio_host="http://localhost:1234")
            llm_client = LLMClient()
            cv_parser = CVParserAgent(qwen_agent, llm_client)
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize LLM components: {e}")
            print("🔄 Falling back to PyMuPDF-only extraction")
            # Initialize without LLM components for basic extraction
            qwen_agent = None
            llm_client = None
            cv_parser = CVParserAgent(qwen_agent, llm_client)
    return cv_parser

def get_job_matcher():
    """Lazy initialization of job matcher components"""
    global llm_client, job_matcher
    if job_matcher is None:
        try:
            if llm_client is None:
                llm_client = LLMClient()
            job_matcher = JobMatcherAgent(llm_client)
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize job matcher: {e}")
            job_matcher = JobMatcherAgent(None)
    return job_matcher


def get_video_agent():
    """Lazy initialization of video analyzer"""
    global video_agent
    if video_agent is None:
        try:
            video_agent = QwenVideoCVAgent()
            print("✓ Video analyzer initialized")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize video analyzer: {e}")
            video_agent = None
    return video_agent


# ============================================================================
# PHASE 1: CV PARSING
# ============================================================================

def parse_cv(cv_file):
    """Parse CV directly using CVParserAgent"""
    try:
        if not cv_file:
            return "❌ Error: Please upload a resume!", None
        
        print(f"📄 Processing CV: {cv_file.name}")
        
        # Parse CV directly with CVParserAgent
        parser = get_cv_parser()
        resume_data = asyncio.run(parser.extract_and_parse_cv(cv_file.name))
        
        # Format the result beautifully
        personal_info = resume_data.get("personalInformation", {})
        
        display_md = f"""
# ✅ CV PARSING COMPLETE!

---

## 👤 Personal Information
**Name:** {personal_info.get('fullName', 'N/A')}  
**Email:** {personal_info.get('email', 'N/A')}  
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
            display_md += "\n---\n\n## 🚀 Projects\n"
            for proj in resume_data.get('projects', []):
                display_md += f"""
### {proj.get('projectName', 'N/A')}
**Role:** {proj.get('role', 'N/A')}  
**Duration:** {proj.get('startDate', 'N/A')} - {proj.get('endDate', 'N/A')}

{proj.get('description', 'N/A')}

**Technologies:** {', '.join(proj.get('tags', []))}
"""
                if proj.get('link'):
                    display_md += f"**Link:** {proj.get('link')}\n"

        # Certifications
        if resume_data.get('certifications'):
            display_md += "\n---\n\n## 📜 Certifications\n"
            for cert in resume_data.get('certifications', []):
                display_md += f"• **{cert.get('certificationName', 'N/A')}** - {cert.get('dateObtained', 'N/A')}\n"

        # Languages
        if resume_data.get('languages'):
            display_md += "\n---\n\n## 🌍 Languages\n"
            for lang in resume_data.get('languages', []):
                display_md += f"• {lang.get('language', 'N/A')}: {lang.get('proficiency', 'N/A')}\n"

        # Metadata
        metadata = resume_data.get("extractionMetadata", {})
        display_md += f"""
---

## 📊 Extraction Metadata
**Method:** {metadata.get('extractionMethod', 'N/A')}  
**Confidence:** {metadata.get('confidence', 0):.2%}  
**Text Length:** {metadata.get('rawTextLength', 0)} characters  
**Extracted At:** {metadata.get('extractedAt', 'N/A')}
"""

        # Return formatted display + JSON for next phase
        return display_md, json.dumps(resume_data, indent=2)
        
    except Exception as e:
        error_msg = f"❌ Error parsing CV: {str(e)}"
        print(error_msg)
        return error_msg, None


# ============================================================================
# PHASE 2: JOB MATCHING
# ============================================================================

def match_resume_to_job(resume_json, job_title, job_description, job_requirements, github_url="", linkedin_url=""):
    """Phase 2: Match parsed resume to job description"""
    try:
        if not resume_json:
            return "❌ Error: Please paste the resume JSON from Phase 1!"
        
        if not job_title or not job_description or not job_requirements:
            return "❌ Error: Please fill in all job details!"
        
        print(f"🎯 Matching resume to job: {job_title}")
        
        # Parse resume JSON
        try:
            resume_data = json.loads(resume_json)
        except json.JSONDecodeError as e:
            return f"❌ Error: Invalid JSON format. Please copy the JSON output from Phase 1.\n\nError: {str(e)}"
        
        # Get job matcher and perform matching
        matcher = get_job_matcher()
        result = asyncio.run(matcher.match_cv_to_job(
            resume_data,
            job_title,
            job_description,
            job_requirements,
            github_url=github_url,
            linkedin_url=linkedin_url
        ))
        
        # Format results beautifully
        candidate_name = result["candidate_name"]
        overall_score = result["overall_score"]
        decision = result["decision"]
        skills_analysis = result["skills_analysis"]
        experience_analysis = result["experience_analysis"]
        education_analysis = result["education_analysis"]
        recommendations = result["recommendations"]
        
        # Build display markdown
        display_md = f"""
# 🎯 JOB MATCHING RESULTS

---

## 👤 Candidate: {candidate_name}
### 📋 Position: {job_title}

---

## 📊 OVERALL MATCHING SCORE

# {overall_score:.1f}%

### Decision: **{decision}**

---

## 🛠️ SKILLS ANALYSIS

**Skills Score:** {skills_analysis['skills_score']:.1f}%  
**Candidate Skills:** {skills_analysis['skills_count']} skills identified  
**Match Rate:** {skills_analysis['match_percentage']:.1f}%

### ✅ Matched Skills ({len(skills_analysis['matched_skills'])}):
"""
        if skills_analysis['matched_skills']:
            for skill in skills_analysis['matched_skills']:
                display_md += f"- {skill.title()}\n"
        else:
            display_md += "- None\n"
        
        display_md += f"\n### ❌ Missing Skills ({len(skills_analysis['missing_skills'])}):\n"
        if skills_analysis['missing_skills']:
            for skill in skills_analysis['missing_skills']:
                display_md += f"- {skill.title()}\n"
        else:
            display_md += "- None\n"
        
        display_md += f"""
---

## 💼 EXPERIENCE ANALYSIS

**Experience Score:** {experience_analysis['experience_score']:.1f}%  
**Candidate Experience:** {experience_analysis['total_years']:.1f} years  
**Required Experience:** {experience_analysis['required_years']:.1f} years  
**Experience Gap:** {experience_analysis['experience_gap']:.1f} years

### Work History:
{experience_analysis['work_entries']} position(s) listed

### Relevant Roles:
"""
        if experience_analysis['relevant_roles']:
            for role in experience_analysis['relevant_roles']:
                display_md += f"- {role}\n"
        else:
            display_md += "- No relevant roles identified\n"
        
        display_md += f"""
---

## 🎓 EDUCATION ANALYSIS

**Education Score:** {education_analysis['education_score']:.1f}%  
**Candidate Degree:** {education_analysis['candidate_degree']}  
**Required Degree:** {education_analysis['required_degree']}  
**Meets Requirement:** {'✅ Yes' if education_analysis['meets_requirement'] else '❌ No'}

**Education Entries:** {education_analysis['education_entries']}

---

## 💡 RECOMMENDATIONS

"""
        for rec in recommendations:
            display_md += f"{rec}\n\n"
        
        # Add LLM insights if available
        if result.get("llm_insights") and "unavailable" not in result["llm_insights"].lower():
            display_md += f"""
---

## 🤖 AI INSIGHTS

{result['llm_insights']}
"""
        
        # Add web scraping results if available
        web_results = result.get("web_scraping_results")
        if web_results and web_results.get("successful_scrapes", 0) > 0:
            display_md += f"""
---

## 🌐 WEB SCRAPING RESULTS

**Online Presence Verified:** {web_results['successful_scrapes']} profile(s) scraped  
**Web Score Boost:** +{result.get('web_score_boost', 0):.1f}%

### Profiles Found:
"""
            for profile in web_results.get("profiles", []):
                platform = profile.get("platform", "Unknown")
                url = profile.get("url", "")
                display_md += f"- **{platform}:** {url}\n"
                
                if platform == "GitHub":
                    data = profile.get("data", {})
                    repos = data.get("public_repos", 0)
                    if repos > 0:
                        display_md += f"  - Public Repositories: {repos}\n"
            
            # Additional skills found
            additional_skills = web_results.get("additional_skills", [])
            if additional_skills:
                display_md += f"\n### Additional Skills Discovered:\n"
                for skill in additional_skills[:10]:
                    display_md += f"- {skill}\n"
        
        display_md += f"""
---

## 📈 SCORE BREAKDOWN

| Category | Score | Weight |
|----------|-------|--------|
| Skills Match | {skills_analysis['skills_score']:.1f}% | 50% |
| Experience Match | {experience_analysis['experience_score']:.1f}% | 30% |
| Education Match | {education_analysis['education_score']:.1f}% | 20% |
| **Base Score** | **{overall_score - result.get('web_score_boost', 0):.1f}%** | **100%** |
| Web Presence Boost | +{result.get('web_score_boost', 0):.1f}% | Bonus |
| **Final Score** | **{overall_score:.1f}%** | **Total** |

---

**Matched at:** {result['matched_at']}
"""
        
        return display_md
        
    except Exception as e:
        error_msg = f"❌ Error matching resume to job: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


# ============================================================================
#  VIDEO CV ANALYSIS
# ============================================================================

async def analyze_video_cv(video_file, job_description=""):
    """Analyze uploaded video CV"""
    if not video_file:
        return "❌ Error: Please upload a video file!", None

    agent = get_video_agent()
    if not agent:
        return "❌ Error: Video analyzer not available. Check LM Studio connection.", None

    try:
        print(f"📹 Processing video: {video_file}")

        # Analyze video
        result = await agent.execute_task({
            "video_path": video_file,
            "job_description": job_description or "General job candidate analysis"
        })

        # Format results
        if hasattr(result, 'to_dict'):
            analysis_data = result.to_dict()
        else:
            analysis_data = result

        # Create readable output
        output = f"""
# 📹 Video CV Analysis Results

## 📊 Basic Info
- **File:** {os.path.basename(video_file)}
- **Size:** {os.path.getsize(video_file) / (1024*1024):.1f} MB
- **Confidence:** {analysis_data.get('confidence', 0.0) * 100:.1f}%

## 🎬 Video Metadata
- **Duration:** {analysis_data.get('duration_seconds', 0):.1f} seconds
- **Resolution:** {analysis_data.get('resolution', 'Unknown')}
- **FPS:** {analysis_data.get('fps', 0):.1f}
- **Format:** {analysis_data.get('format', 'Unknown')}

## 👁️ Visual Analysis
{analysis_data.get('frames_analysis', {}).get('summary', 'No visual analysis available')}

## 🎤 Audio Transcript
{analysis_data.get('transcript', 'No transcript available')}

## 🧠 Combined Assessment
{analysis_data.get('overall_assessment', 'No assessment available')}

## 🛠️ Extracted Skills
{', '.join(analysis_data.get('skills', [])) or 'No skills detected'}

## 📈 Experience Estimate
{analysis_data.get('experience_years', 0):.1f} years
"""

        # JSON output for advanced users
        json_output = json.dumps(analysis_data, indent=2, default=str)

        return output, json_output

    except Exception as e:
        error_msg = f"❌ Analysis failed: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, None


def analyze_video_cv_sync(video_file, job_description=""):
    """Synchronous wrapper for video CV analysis"""
    return asyncio.run(analyze_video_cv(video_file, job_description))


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(title="CV Matcher - Two Phase System", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("""
    # 🎯 AI-Powered CV Matcher
    ### Multi-Phase System: Parse CV → Match to Job → Analyze Video CV
    
    **✅ Phase 1:** Extract structured data from CV (PDF/Image)  
    **✅ Phase 2:** Intelligent job matching with detailed analysis  
    **✅ ** Video CV analysis with visual and audio insights
    """)
    
    with gr.Tabs():
        # TAB 1: CV PARSING
        with gr.Tab("📄 Phase 1: Parse CV"):
            gr.Markdown("""
            ## Upload your resume to extract structured data
            The AI will extract personal info, work experience, education, skills, and more.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    cv_upload = gr.File(
                        label="Upload Resume (PDF or Image)",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg"]
                    )
                    parse_btn = gr.Button("🔍 PARSE CV", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    parse_result = gr.Markdown(label="Parsed Resume")
                    resume_json_output = gr.Textbox(
                        label="Structured JSON (for Phase 2)",
                        lines=10,
                        visible=True
                    )
            
            parse_btn.click(
                fn=parse_cv,
                inputs=[cv_upload],
                outputs=[parse_result, resume_json_output]
            )
        
        # TAB 2: JOB MATCHING
        with gr.Tab("🎯 Phase 2: Match to Job"):
            gr.Markdown("""
            ## Match your parsed resume to a job description
            Use the structured resume JSON from Phase 1 to get intelligent matching analysis.
            """)
            
            with gr.Row():
                with gr.Column():
                    resume_json_input = gr.Textbox(
                        label="Resume JSON (from Phase 1)",
                        placeholder="Paste the JSON output from Phase 1 here...",
                        lines=10
                    )
                    
                    job_title_input = gr.Textbox(
                        label="Job Title",
                        placeholder="e.g., Senior Software Engineer"
                    )
                    
                    job_desc_input = gr.Textbox(
                        label="Job Description",
                        placeholder="Describe the job role...",
                        lines=5
                    )
                    
                    job_req_input = gr.Textbox(
                        label="Requirements",
                        placeholder="List required skills and qualifications...",
                        lines=5
                    )
                    
                    github_url_input = gr.Textbox(
                        label="GitHub Profile URL (Optional)",
                        placeholder="https://github.com/username - for enhanced web scraping",
                        info="Enter candidate's GitHub URL to scrape repositories and skills"
                    )
                    
                    linkedin_url_input = gr.Textbox(
                        label="LinkedIn Profile URL (Optional)",
                        placeholder="https://www.linkedin.com/in/username - for enhanced web scraping",
                        info="Enter candidate's LinkedIn URL to scrape profile information"
                    )
                    
                    match_btn = gr.Button("🎯 MATCH TO JOB", variant="primary", size="lg")
                
                with gr.Column():
                    match_result = gr.Markdown(label="Matching Results")
            
            match_btn.click(
                fn=match_resume_to_job,
                inputs=[resume_json_input, job_title_input, job_desc_input, job_req_input, github_url_input, linkedin_url_input],
                outputs=[match_result]
            )
        
        # TAB 3: VIDEO CV ANALYSIS
        with gr.Tab("🎬 Video CV Analysis"):
            gr.Markdown("""
            ## Upload a video CV/resume for AI-powered analysis
            Analyze video presentations using **Qwen Vision** for visual analysis and **Whisper** for audio transcription.
            
            ### Features:
            - **Visual Analysis:** Analyzes appearance, body language, and setting
            - **Audio Transcription:** Transcribes spoken content using Whisper
            - **Skill Extraction:** Identifies technical skills mentioned
            - **Experience Estimation:** Estimates years of experience
            - **Combined Assessment:** Overall candidate evaluation
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Upload Video")
                    video_upload = gr.Video(
                        label="Video CV"
                    )
                    
                    video_job_desc = gr.Textbox(
                        label="Job Description (Optional)",
                        placeholder="Describe the target job role for better analysis...",
                        lines=3,
                        info="Helps tailor the analysis to specific job requirements"
                    )
                    
                    analyze_video_btn = gr.Button(
                        "🎬 ANALYZE VIDEO CV",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Analysis Results")
                    video_analysis_output = gr.Markdown(
                        label="Analysis Report",
                        value="Upload a video and click 'ANALYZE VIDEO CV' to begin..."
                    )
                    
                    with gr.Accordion("🔧 Raw JSON Data", open=False):
                        video_json_output = gr.Textbox(
                            label="Technical Details",
                            lines=15,
                            show_copy_button=True
                        )
            
            analyze_video_btn.click(
                fn=analyze_video_cv_sync,
                inputs=[video_upload, video_job_desc],
                outputs=[video_analysis_output, video_json_output]
            )
    
    # Footer
    gr.Markdown("""
    ---
    **Instructions:**
    1. **Phase 1:** Upload your CV and click "PARSE CV" to extract structured data
    2. **Phase 2:** Copy the JSON output, paste it in Phase 2 tab, add job details, and click "MATCH TO JOB"
    3. **        ** Upload a video CV for comprehensive analysis including visual and audio insights
    
    **Features:**
    - ✅ Intelligent CV parsing with PyMuPDF + Regex fallback
    - ✅ Comprehensive job matching (Skills, Experience, Education)
    - ✅ Video CV analysis with Qwen Vision + Whisper
    - ✅ Detailed recommendations and scoring
    - ✅ Works offline (LM Studio optional for enhanced analysis)
    
    *Powered by Qwen2.5-VL + Local LLM (LM Studio) + Whisper*
    """)


if __name__ == "__main__":
    import sys
    
    # Check for port argument
    port = 7861
    if len(sys.argv) > 1 and sys.argv[1].startswith('--server_port='):
        try:
            port = int(sys.argv[1].split('=')[1])
        except:
            pass
    
    print("\n" + "="*70)
    print("🚀 Starting Three-Phase CV Matcher Interface")
    print("="*70)
    print("\n📊 Features:")
    print("  ✓ Phase 1: CV Parsing with structured JSON output")
    print("  ✓ Phase 2: Job Matching with intelligent scoring")
    print("  ✓ Phase 3: Video CV Analysis with visual + audio insights")
    print(f"\n🌐 Interface will open at: http://localhost:{port}")
    print("="*70 + "\n")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False
    )
