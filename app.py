import os
import sys
import json
import dotenv
import gradio as gr
import requests
import importlib
import importlib.util
from pathlib import Path
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

dotenv.load_dotenv()

# Career Advisor Gradio interface
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

# Define default career paths
DEFAULT_PATHS = ["Data Science", "Software Engineer", "Product Manager", "DevOps", "Research", "AI/ML Engineer"]



CV_reviewer = gr.Interface(
    fn=review_cv,
    inputs=[
        gr.Textbox(lines=20, label="Paste CV JSON", placeholder='{"personalInformation": {...}, "experience": [...], ...}'),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(minimum=128, maximum=16384, value=2048, step=128, label="Max Tokens"),
    ],
    outputs="text",
    title="CV Reviewer (English)",
    description="Reviews your CV using OpenRouter (primary) or LMStudio (fallback). Provides detailed ATS-optimized feedback in English.",
    api_name="cv_reviewer",
)

CV_reviewer_multilingual = gr.Interface(
    fn=review_cv_multilingual,
    inputs=[
        gr.Textbox(lines=20, label="Paste CV JSON", placeholder='{"personalInformation": {...}, "experience": [...], ...}'),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(minimum=128, maximum=16384, value=4000, step=128, label="Max Tokens"),
    ],
    outputs="text",
    title="CV Reviewer (Multilingual)",
    description="Reviews your CV with automatic language detection (English/French/Arabic). Responds in the same language as your CV. Uses OpenRouter (primary) or LMStudio (fallback).",
    api_name="cv_reviewer_multilingual",
)

CV_rewriter = gr.Interface(
    fn=rewrite_cv,
    inputs=[
        gr.Textbox(lines=20, label="Paste CV JSON", placeholder='{"personalInformation": {...}, "experience": [...], ...}'),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(minimum=128, maximum=16384, value=8192, step=128, label="Max Tokens"),
    ],
    outputs="text",
    title="CV Rewriter",
    description="Rewrites your CV to be ATS-optimized. Applies XYZ pattern (Accomplished X, measured by Y, by doing Z) to all bullets. Adds quantifiable metrics and uses strong action verbs. Uses OpenRouter (primary) or LMStudio (fallback).",
    api_name="cv_rewriter",
)

Carreer_advisor = gr.Interface(
   
    fn=career_advisor_fn,
    inputs=[
        gr.Textbox(lines=20, label="Paste Full CV JSON", placeholder='{"skills": [...], "experience": [...], ...}'),
        gr.CheckboxGroup(choices=DEFAULT_PATHS, label="Desired Career Paths (select one or more)", value=[]),
        gr.Textbox(lines=3, label="Career Intentions / Goals", placeholder="What are your career goals?"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(minimum=128, maximum=16384, value=8192, step=128, label="Max Tokens"),
    ],
    outputs=gr.JSON(label="Career Advisor Output (JSON)"),
    title="Career Advisor",
    description="Provides personalized career guidance based on your CV, desired paths, and intentions.",
    api_name="career_advisor",
)
Feedback_interface = gr.Interface(
    fn=apply_feedback_fn,
    inputs=[
        gr.Textbox(lines=20, label="Original Advisor Output (Full JSON)", placeholder="Paste the complete JSON output from Career Advisor"),
        gr.Textbox(label="Step Number", placeholder="e.g., '1', '2', '3'"),
        gr.Textbox(
            lines=8, 
            label="Feedback JSON", 
            placeholder='{\n  "clarityScore": null,\n  "relevanceScore": null,\n  "difficultyLevel": null,\n  "userComment": null\n}',
            value='{\n  "clarityScore": null,\n  "relevanceScore": null,\n  "difficultyLevel": null,\n  "userComment": null\n}'
        ),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(minimum=128, maximum=16384, value=8192, step=128, label="Max Tokens"),
    ],
    outputs=gr.JSON(label="Updated Step (JSON)"),
    title="Apply Feedback to Learning Path Step",
    description="Provide feedback on a specific step. Fill in the feedback JSON with your scores (1-5), difficulty level (\"too easy\", \"appropriate\", \"too hard\"), and comments.",
    api_name="apply_feedback",
)

demo = gr.TabbedInterface(
    [CV_reviewer, CV_reviewer_multilingual, CV_rewriter, Carreer_advisor, Feedback_interface],
    ["CV Reviewer", "CV Reviewer (Multilingual)", "CV Rewriter", "Career Advisor", "Apply Feedback"],
)

demo.launch(debug=True, mcp_server=True)
