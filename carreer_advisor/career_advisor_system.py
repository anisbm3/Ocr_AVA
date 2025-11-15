import json
import os
import re
import dotenv
from typing import Any, Dict
import requests
dotenv.load_dotenv()

# OpenRouter config (primary)
OPENROUTER_API_KEY = os.getenv("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("openrouter_API_URL") or os.getenv("OPENROUTER_URL") or "https://api.openrouter.ai/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("MISTRAL_NAME") or os.getenv("OPENROUTER_MODEL") or "mistralai/mistral-saba"

# LMStudio config (fallback)
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL") or "http://localhost:1234/v1/chat/completions"
LMSTUDIO_MODEL = os.getenv("MODEL_NAME") or os.getenv("LMSTUDIO_MODEL") or "qwen3-vl-4b"

DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", 8192))  # Increased default to avoid truncation
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(HERE, "career_advisor_prompt.txt")
EXAMPLE_OUTPUT_FILE = os.path.join(HERE, "CareerAdvisor_OutputExample_regenerated.json")
OUTPUT_TEMPLATE_FILE = os.path.join(HERE, "CareerAdvisor_OutputTemplate.JSON")

# Debug: Print configuration on startup
print("\n" + "="*60)
print("CAREER ADVISOR CONFIGURATION")
print("="*60)
print(f"OpenRouter API Key: {'✓ Set' if OPENROUTER_API_KEY else '✗ Missing'}")
if OPENROUTER_API_KEY:
    print(f"  Key preview: {OPENROUTER_API_KEY[:20]}...")
print(f"OpenRouter URL: {OPENROUTER_URL}")
print(f"OpenRouter Model: {OPENROUTER_MODEL}")
print(f"LMStudio URL: {LMSTUDIO_URL}")
print(f"LMStudio Model: {LMSTUDIO_MODEL}")
print(f"Max Tokens: {DEFAULT_MAX_TOKENS}")
print("="*60 + "\n")

def remove_personal_info(cv_json_str: str) -> dict:
    """Receive CV JSON string and return as json but without personal_information section, or error message."""
    try:
        cv = json.loads(cv_json_str)
        cv.pop("personal_information", None)
        return cv
    except Exception as e:
        return {"error": f"Invalid CV JSON: {e}"}

def _read_text(filepath: str) -> str:
    """Read and return the contents of a text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return ""

def load_prompt_files():
    global PROMPT_INSTRUCTIONS, EXAMPLE_OUTPUT, OUTPUT_TEMPLATE
    PROMPT_INSTRUCTIONS = _read_text(PROMPT_FILE)
    EXAMPLE_OUTPUT = _read_text(EXAMPLE_OUTPUT_FILE)
    OUTPUT_TEMPLATE = _read_text(OUTPUT_TEMPLATE_FILE)
    
def combine_prompt_parts(cv_anonymized: dict) -> str:
    parts = []
    if PROMPT_INSTRUCTIONS:
        parts.append(PROMPT_INSTRUCTIONS)
    parts.append("\n-- INPUT JSON (use this as the candidate profile):\n")
    parts.append(json.dumps(cv_anonymized, ensure_ascii=False, indent=2))
    if EXAMPLE_OUTPUT:
        parts.append("\n-- EXAMPLE OUTPUT (style reference):\n")
        parts.append(EXAMPLE_OUTPUT)
    if OUTPUT_TEMPLATE:
        parts.append("\n-- OUTPUT TEMPLATE (must match this schema):\n")
        parts.append(OUTPUT_TEMPLATE)
    parts.append("\nPlease produce a single JSON object that conforms to the output template and addresses the candidate above.")
    return "\n".join(parts)

def query_model(prompt: str, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Query the language model with the given prompt. Tries OpenRouter first, falls back to LMStudio.
    Returns validated JSON output or error message."""
    
    # Set high max_tokens to avoid truncation (use max available or user-specified)
    # For most models, 16k-32k is safe; we'll use the passed value or a high default
    if max_tokens < 8192:
        max_tokens = 8192  # Ensure we get complete responses
    
    # Try OpenRouter (Mistral Saba) first
    if OPENROUTER_API_KEY:
        # Use the configured URL, or default to OpenRouter endpoint
        url = OPENROUTER_URL or "https://openrouter.ai/api/v1/chat/completions"
        
        # Ensure URL is properly formatted
        if not url.startswith("http"):
            url = "https://openrouter.ai/api/v1/chat/completions"
        
        print(f"\n{'='*60}")
        print(f"ATTEMPTING OPENROUTER API CALL")
        print(f"{'='*60}")
        print(f"URL: {url}")
        print(f"Model: {OPENROUTER_MODEL}")
        print(f"Temperature: {temperature}")
        print(f"Max Tokens: {max_tokens}")
        print(f"Prompt Length: {len(prompt)} characters")
        print(f"API Key (first 20 chars): {OPENROUTER_API_KEY[:20]}...")
        
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:7860",  # Optional: for OpenRouter analytics
            "X-Title": "Career Advisor"  # Optional: for OpenRouter analytics
        }
        
        try:
            print(f"Sending request...")
            r = requests.post(url, json=payload, headers=headers, timeout=600)
            
            print(f"Response Status Code: {r.status_code}")
            print(f"Response Headers: {dict(r.headers)}")
            
            if r.status_code < 400:
                print(f"✓ OpenRouter SUCCESS!")
                data = r.json()
                if isinstance(data, dict) and "choices" in data:
                    try:
                        content = data["choices"][0]["message"]["content"]
                        # Validate that response is valid JSON
                        try:
                            json.loads(content)
                            print(f"✓ Valid JSON response received")
                            return content
                        except json.JSONDecodeError as je:
                            print(f"Warning: OpenRouter returned invalid JSON: {je}")
                            # Try to extract JSON from markdown code blocks
                            content = _extract_json_from_response(content)
                            return content
                    except Exception as e:
                        print(f"Error extracting OpenRouter response: {e}")
            else:
                # OpenRouter failed, log detailed error and fall through to LMStudio
                print(f"\n{'!'*60}")
                print(f"✗ OPENROUTER FAILED - Status {r.status_code}")
                print(f"{'!'*60}")
                try:
                    error_detail = r.json()
                    print(f"Error Response:")
                    print(json.dumps(error_detail, indent=2))
                except:
                    print(f"Error Response (raw text):")
                    print(r.text[:1000])
                print(f"{'!'*60}\n")
        except requests.exceptions.Timeout as e:
            print(f"\n✗ OpenRouter TIMEOUT after 600s: {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"\n✗ OpenRouter CONNECTION ERROR: {e}")
        except Exception as e:
            # Network error with OpenRouter, fall through to LMStudio
            print(f"\n✗ OpenRouter EXCEPTION: {type(e).__name__}: {e}")
    
    # Fallback to LMStudio (Qwen model)
    print(f"Falling back to LMStudio at {LMSTUDIO_URL} with model {LMSTUDIO_MODEL}")
    print(f"Requesting max_tokens: {max_tokens}")
    
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "max_completion_tokens": int(max_tokens),  # Some servers use this variant
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        r = requests.post(LMSTUDIO_URL, json=payload, headers=headers, timeout=600)
        
        if r.status_code >= 400:
            try:
                return f"LMStudio error {r.status_code}: {json.dumps(r.json(), ensure_ascii=False, indent=2)}"
            except Exception:
                return f"LMStudio error {r.status_code}: {r.text}"
        
        data = r.json()
        print(f"LMStudio response finish_reason: {data.get('choices', [{}])[0].get('finish_reason', 'unknown')}")
        print(f"LMStudio completion_tokens used: {data.get('usage', {}).get('completion_tokens', 'unknown')}")
        
        if isinstance(data, dict) and "choices" in data:
            try:
                content = data["choices"][0]["message"]["content"]
                # Validate JSON
                try:
                    json.loads(content)
                    return content
                except json.JSONDecodeError:
                    print(f"Warning: LMStudio returned invalid JSON, attempting to extract...")
                    content = _extract_json_from_response(content)
                    return content
            except Exception as e:
                print(f"Error extracting LMStudio response: {e}")
                return json.dumps(data, ensure_ascii=False, indent=2)
        return json.dumps(data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"LMStudio network error: {e}"

def _extract_json_from_response(text: str) -> str:
    """Extract JSON from response that may contain markdown code blocks or extra text."""
    # Try to find JSON in markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        potential_json = match.group(1)
        try:
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON object (look for outermost braces)
    brace_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    match = re.search(brace_pattern, text, re.DOTALL)
    if match:
        potential_json = match.group(0)
        try:
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            pass
    
    # If no valid JSON found, return original text with warning
    return json.dumps({
        "error": "Could not extract valid JSON from model response",
        "raw_response": text[:1000] + ("..." if len(text) > 1000 else "")
    }, ensure_ascii=False, indent=2)

def run_advisor(cv_json_str: str, desired_paths: list, intentions: str, temperature: float, max_tokens: int):
    prompt = combine_prompt_parts(cv_json_str, desired_paths, intentions)
    if "error" in prompt:
        return json.dumps(prompt, ensure_ascii=False, indent=2)

    return query_model(prompt, temperature=temperature, max_tokens=max_tokens)


 
def apply_feedback(original_output: str, step_identifier: str, feedback_json: str, temperature: float, max_tokens: int):
    """
    Apply user feedback to a specific step in the learning path.
    
    Args:
        original_output: The full career advisor JSON output
        step_identifier: The step number or identifier to modify (e.g., "1", "2", "step_1")
        feedback_json: JSON string with feedback structure: 
            {"clarityScore": <1-5 or null>, "relevanceScore": <1-5 or null>, 
             "difficultyLevel": <"easy"|"medium"|"hard" or null>, "userComment": "<text or null>"}
        temperature: Model temperature
        max_tokens: Maximum tokens for response
    
    Returns:
        JSON string of the updated step only, in the format:
        {
            "stepNumber": <int>,
            "title": "<string>",
            "description": "<string>",
            "duration": "<string>",
            "skillsTargeted": [<list of strings>],
            "recommendedResources": [{type, title, url, explanation}, ...],
            "feedback": {"clarityScore": <int or null>, "relevanceScore": <int or null>, 
                        "difficultyLevel": <string or null>, "userComment": <string or null>},
            "explainabilityNote": "<string>"
        }
    """
    if not original_output:
        return json.dumps({"error": "No original output provided."}, ensure_ascii=False)
    
    # Parse feedback to ensure it's valid
    try:
        feedback_data = json.loads(feedback_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid feedback JSON format. Expected: {\"clarityScore\": null, \"relevanceScore\": null, \"difficultyLevel\": null, \"userComment\": null}"}, ensure_ascii=False)
    
    prompt = (
        "You are a career advisor AI. You previously generated a career development plan. "
        "The user has now provided feedback on a specific step.\n\n"
        "# ORIGINAL CAREER ADVISOR OUTPUT:\n"
        f"{original_output}\n\n"
        f"# TARGET STEP TO MODIFY:\n"
        f"Step identifier: {step_identifier or 'Not specified - use step 1'}\n\n"
        f"# USER FEEDBACK:\n"
        f"{json.dumps(feedback_data, ensure_ascii=False, indent=2)}\n\n"
        "# YOUR TASK:\n"
        "1. Locate the step identified by the step number or identifier\n"
        "2. Analyze the user feedback (scores and comment)\n"
        "3. Improve the step based on the feedback:\n"
        "   - If clarityScore is low (1-2), make the description clearer\n"
        "   - If relevanceScore is low, better align with user's career goals\n"
        "   - If difficultyLevel feedback suggests it's too hard/easy, adjust resources\n"
        "   - Incorporate the userComment suggestions\n"
        "4. Return ONLY the modified step as a JSON object (not the entire plan)\n\n"
        "# OUTPUT FORMAT (return ONLY this structure):\n"
        "{\n"
        '  "stepNumber": <int>,\n'
        '  "title": "<improved title>",\n'
        '  "description": "<improved description>",\n'
        '  "duration": "<time estimate>",\n'
        '  "skillsTargeted": ["<skill1>", "<skill2>", ...],\n'
        '  "recommendedResources": [\n'
        '    {\n'
        '      "type": "<YouTube|Coursera|Udemy|Book|Project|etc>",\n'
        '      "title": "<resource title>",\n'
        '      "url": "<url or null>",\n'
        '      "explanation": "<why this resource is recommended>"\n'
        '    }\n'
        '  ],\n'
        '  "feedback": ' + json.dumps(feedback_data, ensure_ascii=False) + ',\n'
        '  "explainabilityNote": "<explanation of why this step is important and how it addresses gaps>"\n'
        "}\n\n"
        "Return ONLY the JSON object for the updated step, no additional text."
    )
    
    return query_model(prompt, temperature=temperature, max_tokens=max_tokens)
