import json
import os
import requests
from pathlib import Path
from typing import Any, Dict, Union

PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW_TEMPLATE_PATH = os.path.join(PYTHON_DIR, "Review_template.JSON")
def prepare_ats_prompt(
    cv: Union[Dict[str, Any], str],
    template_path: Union[str, Path] = REVIEW_TEMPLATE_PATH,
) -> Dict[str, Any]:
    """
    Prepare the ATS model prompt and return components:
      - system_prompt: the full rules/prompt text (exactly as provided)
      - input_json: the CV as a Python dict
      - output_template: the contents of Review_template.JSON as a string
      - final_prompt: the composed prompt string ready to send to an LLM

    Usage:
      cv_dict = json.loads(cv_json_string)  # or pass a dict
      result = prepare_ats_prompt(cv_dict)
      final_prompt = result["final_prompt"]
      # send final_prompt to your LLM

    Notes:
      - This function does NOT call any LLM; it only prepares the prompt and template variables.
      - It preserves the prompt text exactly as provided by you.
    """
    # Ensure CV is a dict
    if isinstance(cv, str):
        try:
            input_json = json.loads(cv)
        except json.JSONDecodeError as e:
            raise ValueError("cv is a string but not valid JSON") from e
    else:
        input_json = cv

    # Load the Review_template.JSON into a variable
    tpl_path = Path(template_path)
    if not tpl_path.exists():
        # If the file is missing, raise a clear error
        raise FileNotFoundError(f"Template file not found: {tpl_path.resolve()}")

    output_template = tpl_path.read_text(encoding="utf-8")

    # Load system prompt from file
    prompt_path = Path(__file__).parent / "ATS_system_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {prompt_path.resolve()}")
    
    system_prompt = prompt_path.read_text(encoding="utf-8")

    # Compose final prompt for structured output
    final_prompt = (
        system_prompt
        + "\n\nInput CV JSON (as parsed object):\n"
        + json.dumps(input_json, ensure_ascii=False, indent=2)
        + "\n\nAnalyze this CV according to the rules above. Your response will be automatically structured according to the CV review schema."
    )

    return {
        "system_prompt": system_prompt,
        "input_json": input_json,
        "output_template": output_template,
        "final_prompt": final_prompt,
    }


def review_cv(cv_json: Union[Dict[str, Any], str], 
              temperature: float = 0.7,
              max_tokens: int = 2048) -> str:
    """
    Review a CV using OpenRouter (primary) or LM Studio (fallback).
    
    Args:
        cv_json: Either a JSON string or a dict containing the CV data
        temperature: Model temperature (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 2048 for longer reviews)
    
    Returns:
        str: The model's CV review and feedback
    
    Example:
        >>> cv_text = '''{"personalInformation": {"fullName": "John Doe"...}'''
        >>> review = review_cv(cv_text)
        >>> print(review)  # Prints the model's analysis
    """
    # 1. Prepare the ATS prompt
    try:
        payload = prepare_ats_prompt(cv_json)
        final_prompt = payload["final_prompt"]
    except Exception as e:
        return f"⚠️ Error preparing prompt: {str(e)}"

    # Load the JSON Schema for structured output
    schema_path = Path(__file__).parent / "Review_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        review_schema = json.load(f)

    # 2. Try OpenRouter first
    openrouter_api_key = os.getenv("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")
    openrouter_url = os.getenv("openrouter_API_URL") or os.getenv("OPENROUTER_URL") or "https://openrouter.ai/api/v1/chat/completions"
    openrouter_model = os.getenv("MISTRAL_NAME") or os.getenv("OPENROUTER_MODEL") or "mistralai/mistral-saba"
    
    if openrouter_api_key:
        print(f"\n{'='*60}")
        print(f"CV REVIEWER - ATTEMPTING OPENROUTER")
        print(f"{'='*60}")
        print(f"URL: {openrouter_url}")
        print(f"Model: {openrouter_model}")
        
        try:
            request_payload = {
                "model": openrouter_model,
                "messages": [{"role": "user", "content": final_prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openrouter_api_key}",
                "HTTP-Referer": "http://localhost:7860",
                "X-Title": "CV Reviewer"
            }
            
            response = requests.post(
                openrouter_url,
                json=request_payload,
                headers=headers,
                timeout=300
            )
            
            print(f"Response Status: {response.status_code}")
            
            if response.ok:
                print(f"✓ OpenRouter SUCCESS!")
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                # Check if response was truncated
                finish_reason = data["choices"][0].get("finish_reason", "")
                truncation_warning = ""
                if finish_reason == "length":
                    truncation_warning = "\n\n⚠️ WARNING: Response was truncated due to token limit. Consider increasing max_tokens."
                
                # Try to parse and format as JSON
                try:
                    # Clean markdown formatting if present
                    cleaned_content = content.strip()
                    if cleaned_content.startswith('```json'):
                        cleaned_content = cleaned_content.replace('```json', '').strip()
                    if cleaned_content.startswith('```'):
                        cleaned_content = cleaned_content.replace('```', '').strip()
                    if '```' in cleaned_content:
                        cleaned_content = cleaned_content[:cleaned_content.index('```')].strip()
                    
                    review_json = json.loads(cleaned_content)
                    # Return clean JSON without markdown formatting
                    formatted_content = json.dumps(review_json, ensure_ascii=False, indent=2)
                    return formatted_content + truncation_warning
                except json.JSONDecodeError:
                    # Try to extract JSON from the content
                    try:
                        json_start = content.find('{')
                        json_end = content.rfind('}')
                        if json_start != -1 and json_end != -1 and json_end > json_start:
                            extracted_json = content[json_start:json_end + 1]
                            review_json = json.loads(extracted_json)
                            return json.dumps(review_json, ensure_ascii=False, indent=2) + truncation_warning
                    except:
                        pass
                    return content + truncation_warning
            else:
                print(f"✗ OpenRouter failed with status {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"Error: {json.dumps(error_detail, indent=2)}")
                except:
                    print(f"Error: {response.text[:500]}")
        
        except Exception as e:
            print(f"✗ OpenRouter exception: {e}")

    # 3. Fallback to LM Studio
    print(f"\nFalling back to LMStudio for CV review...")
    
    try:
        lmstudio_url = os.getenv("LMSTUDIO_URL")
        model_name = os.getenv("MODEL_NAME")

        if not lmstudio_url:
            return "⚠️ Error: LMSTUDIO_URL environment variable not set"

        if not model_name:
            return "⚠️ Error: MODEL_NAME environment variable not set"

        request_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "cv_review_response",
                    "strict": True,
                    "schema": review_schema
                }
            }
        }

        response = requests.post(
            lmstudio_url,
            json=request_payload,
            timeout=300  # 5-minute timeout for longer reviews
        )
        
        # If error, try to get detailed message
        if not response.ok:
            try:
                error_detail = response.json()
                return f"⚠️ Error from LM Studio ({response.status_code}): {error_detail}"
            except:
                return f"⚠️ Error from LM Studio ({response.status_code}): {response.text}"
        
        # Extract the model's response
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Check if response was truncated
        finish_reason = data["choices"][0].get("finish_reason", "")
        truncation_warning = ""
        if finish_reason == "length":
            truncation_warning = "\n\n⚠️ WARNING: Response was truncated due to token limit. Consider increasing max_tokens."
        
        # Structured output should already be valid JSON, but let's verify and format it nicely
        try:
            # Clean markdown formatting if present
            cleaned_content = content.strip()
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content.replace('```json', '').strip()
            if cleaned_content.startswith('```'):
                cleaned_content = cleaned_content.replace('```', '').strip()
            if '```' in cleaned_content:
                cleaned_content = cleaned_content[:cleaned_content.index('```')].strip()
            
            review_json = json.loads(cleaned_content)
            # Return clean JSON without markdown formatting
            formatted_content = json.dumps(review_json, ensure_ascii=False, indent=2)
            return formatted_content + truncation_warning
        except json.JSONDecodeError:
            # Try to extract JSON from the content
            try:
                json_start = content.find('{')
                json_end = content.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    extracted_json = content[json_start:json_end + 1]
                    review_json = json.loads(extracted_json)
                    return json.dumps(review_json, ensure_ascii=False, indent=2) + truncation_warning
            except:
                pass
            # If parsing fails, return as-is with a warning
            return content + truncation_warning + "\n\n⚠️ Note: Response may not be valid JSON."

    except requests.RequestException as e:
        return f"⚠️ Error calling LM Studio: {str(e)}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def prepare_ats_prompt_multilingual(
    cv: Union[Dict[str, Any], str],
    template_path: Union[str, Path] = "Review_template.JSON",
) -> Dict[str, Any]:
    """
    Prepare the ATS prompt with multilingual support (FR/AR/EN detection).
    Uses ATS_system_prompt_multilingual.txt instead of the default prompt.
    
    Returns the same structure as prepare_ats_prompt() but with language-aware instructions.
    """
    # Ensure CV is a dict
    if isinstance(cv, str):
        try:
            input_json = json.loads(cv)
        except json.JSONDecodeError as e:
            raise ValueError("cv is a string but not valid JSON") from e
    else:
        input_json = cv

    # Load the Review_template.JSON into a variable
    tpl_path = Path(template_path)
    if not tpl_path.exists():
        raise FileNotFoundError(f"Template file not found: {tpl_path.resolve()}")

    output_template = tpl_path.read_text(encoding="utf-8")

    # Load MULTILINGUAL system prompt from file
    prompt_path = Path(__file__).parent / "ATS_system_prompt_multilingual.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Multilingual system prompt file not found: {prompt_path.resolve()}")
    
    system_prompt = prompt_path.read_text(encoding="utf-8")

    # Compose final prompt
    final_prompt = (
        system_prompt
        + "\n\nInput CV JSON (as parsed object):\n"
        + json.dumps(input_json, ensure_ascii=False, indent=2)
        + "\n\nAnalyze this CV according to the rules above. Detect the language and respond accordingly."
    )

    return {
        "system_prompt": system_prompt,
        "input_json": input_json,
        "output_template": output_template,
        "final_prompt": final_prompt,
    }


def review_cv_multilingual(
    cv_json: Union[Dict[str, Any], str], 
    temperature: float = 0.7,
    max_tokens: int = 4000
) -> str:
    """
    Review a CV using OpenRouter (primary) or LM Studio (fallback) with multilingual support.
    Automatically detects CV language (EN/FR/AR) and responds in the same language.
    
    Args:
        cv_json: Either a JSON string or a dict containing the CV data
        temperature: Model temperature (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 4000 for longer reviews)
    
    Returns:
        str: The model's CV review and feedback in the CV's language
    
    Example:
        >>> cv_text = '''{"personalInformation": {"fullName": "Jean Dupont"...}'''
        >>> review = review_cv_multilingual(cv_text)
        >>> print(review)  # Prints analysis in French if CV is in French
    """
    # Prepare the ATS prompt with multilingual support
    try:
        payload = prepare_ats_prompt_multilingual(cv_json)
        final_prompt = payload["final_prompt"]
    except Exception as e:
        return f"⚠️ Error preparing prompt: {str(e)}"

    # Load schema
    schema_path = Path(__file__).parent / "Review_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        review_schema = json.load(f)

    # Try OpenRouter first
    openrouter_api_key = os.getenv("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")
    openrouter_url = os.getenv("openrouter_API_URL") or os.getenv("OPENROUTER_URL") or "https://openrouter.ai/api/v1/chat/completions"
    openrouter_model = os.getenv("MISTRAL_NAME") or os.getenv("OPENROUTER_MODEL") or "mistralai/mistral-saba"
    
    if openrouter_api_key:
        print(f"\n{'='*60}")
        print(f"CV REVIEWER (MULTILINGUAL) - ATTEMPTING OPENROUTER")
        print(f"{'='*60}")
        print(f"URL: {openrouter_url}")
        print(f"Model: {openrouter_model}")
        
        try:
            request_payload = {
                "model": openrouter_model,
                "messages": [{"role": "user", "content": final_prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openrouter_api_key}",
                "HTTP-Referer": "http://localhost:7860",
                "X-Title": "CV Reviewer Multilingual"
            }
            
            response = requests.post(
                openrouter_url,
                json=request_payload,
                headers=headers,
                timeout=300
            )
            
            print(f"Response Status: {response.status_code}")
            
            if response.ok:
                print(f"✓ OpenRouter SUCCESS!")
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                finish_reason = data["choices"][0].get("finish_reason", "")
                truncation_warning = ""
                if finish_reason == "length":
                    truncation_warning = "\n\n⚠️ WARNING: Response was truncated due to token limit. Consider increasing max_tokens."
                
                try:
                    # Clean markdown formatting if present
                    cleaned_content = content.strip()
                    if cleaned_content.startswith('```json'):
                        cleaned_content = cleaned_content.replace('```json', '').strip()
                    if cleaned_content.startswith('```'):
                        cleaned_content = cleaned_content.replace('```', '').strip()
                    if '```' in cleaned_content:
                        cleaned_content = cleaned_content[:cleaned_content.index('```')].strip()
                    
                    review_json = json.loads(cleaned_content)
                    # Return clean JSON without markdown formatting
                    formatted_content = json.dumps(review_json, ensure_ascii=False, indent=2)
                    return formatted_content + truncation_warning
                except json.JSONDecodeError:
                    # Try to extract JSON from the content
                    try:
                        json_start = content.find('{')
                        json_end = content.rfind('}')
                        if json_start != -1 and json_end != -1 and json_end > json_start:
                            extracted_json = content[json_start:json_end + 1]
                            review_json = json.loads(extracted_json)
                            return json.dumps(review_json, ensure_ascii=False, indent=2) + truncation_warning
                    except:
                        pass
                    return content + truncation_warning
            else:
                print(f"✗ OpenRouter failed with status {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"Error: {json.dumps(error_detail, indent=2)}")
                except:
                    print(f"Error: {response.text[:500]}")
        
        except Exception as e:
            print(f"✗ OpenRouter exception: {e}")

    # Fallback to LM Studio
    print(f"\nFalling back to LMStudio for multilingual CV review...")
    
    try:
        lmstudio_api_key = os.getenv("LMSTUDIO_API_KEY")
        lm_studio_url = os.getenv("LMSTUDIO_URL")
        model_name = os.getenv("MODEL_NAME")

        if not lm_studio_url:
            return "⚠️ Error: LMSTUDIO_URL environment variable not set"

        if not model_name:
            return "⚠️ Error: MODEL_NAME environment variable not set"

        headers = {
            "Content-Type": "application/json"
        }
        
        # Add auth header if API key is set
        if lmstudio_api_key:
            headers["Authorization"] = f"Bearer {lmstudio_api_key}"

        request_payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "cv_review_response",
                    "strict": True,
                    "schema": review_schema
                }
            }
        }

        response = requests.post(
            lm_studio_url,
            json=request_payload,
            headers=headers,
            timeout=300,
        )

        if not response.ok:
            try:
                error_detail = response.json()
                return f"⚠️ Error from LM Studio ({response.status_code}): {error_detail}"
            except:
                return f"⚠️ Error from LM Studio ({response.status_code}): {response.text}"
        
        # Extract the model's response
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Check if response was truncated
        finish_reason = data["choices"][0].get("finish_reason", "")
        truncation_warning = ""
        if finish_reason == "length":
            truncation_warning = "\n\n⚠️ WARNING: Response was truncated due to token limit. Consider increasing max_tokens."
        
        # Structured output should already be valid JSON
        try:
            # Clean markdown formatting if present
            cleaned_content = content.strip()
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content.replace('```json', '').strip()
            if cleaned_content.startswith('```'):
                cleaned_content = cleaned_content.replace('```', '').strip()
            if '```' in cleaned_content:
                cleaned_content = cleaned_content[:cleaned_content.index('```')].strip()
            
            review_json = json.loads(cleaned_content)
            # Return clean JSON without markdown formatting
            formatted_content = json.dumps(review_json, ensure_ascii=False, indent=2)
            return formatted_content + truncation_warning
        except json.JSONDecodeError:
            # Try to extract JSON from the content
            try:
                json_start = content.find('{')
                json_end = content.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    extracted_json = content[json_start:json_end + 1]
                    review_json = json.loads(extracted_json)
                    return json.dumps(review_json, ensure_ascii=False, indent=2) + truncation_warning
            except:
                pass
            # If parsing fails, return as-is with a warning
            return content + truncation_warning + "\n\n⚠️ Note: Response may not be valid JSON."

    except requests.RequestException as e:
        return f"⚠️ Error calling LM Studio: {str(e)}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
