import json
import os
import requests
from pathlib import Path
from typing import Any, Dict, Union


def rewrite_cv(
    cv_json: Union[Dict[str, Any], str],
    temperature: float = 0.7,
    max_tokens: int = 8192
) -> str:
    """
    Rewrite a CV to be ATS-optimized with XYZ pattern and quantified achievements.
    Uses OpenRouter (primary) or LMStudio (fallback).
    
    Args:
        cv_json: Either a JSON string or a dict containing the CV data
        temperature: Model temperature (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 8192 for complete CV)
    
    Returns:
        str: The rewritten CV in JSON format
    """
    # Ensure CV is a dict
    if isinstance(cv_json, str):
        try:
            input_cv = json.loads(cv_json)
        except json.JSONDecodeError as e:
            return f"⚠️ Error: Invalid CV JSON - {str(e)}"
    else:
        input_cv = cv_json
    
    # Load rewriter prompt
    prompt_path = Path(__file__).parent / "CV_rewriter_prompt.txt"
    if not prompt_path.exists():
        return f"⚠️ Error: Rewriter prompt file not found at {prompt_path}"
    
    rewriter_rules = prompt_path.read_text(encoding="utf-8")
    
    # Compose final prompt
    final_prompt = (
        rewriter_rules
        + "\n\n# INPUT CV (to be rewritten):\n"
        + json.dumps(input_cv, ensure_ascii=False, indent=2)
        + "\n\n# YOUR TASK:\n"
        + "Rewrite this CV following all the rules above.\n"
        + "Apply XYZ pattern to ALL experience bullets.\n"
        + "Add quantifiable metrics wherever possible.\n"
        + "Return ONLY the rewritten CV as valid JSON in the same structure.\n"
        + "Do NOT add explanations. Do NOT wrap in markdown code blocks."
    )
    
    # Try OpenRouter first
    openrouter_api_key = os.getenv("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")
    openrouter_url = os.getenv("openrouter_API_URL") or os.getenv("OPENROUTER_URL") or "https://openrouter.ai/api/v1/chat/completions"
    openrouter_model = os.getenv("MISTRAL_NAME") or os.getenv("OPENROUTER_MODEL") or "mistralai/mistral-saba"
    
    if openrouter_api_key:
        print(f"\n{'='*60}")
        print(f"CV REWRITER - ATTEMPTING OPENROUTER")
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
                "X-Title": "CV Rewriter"
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
                    truncation_warning = "\n\n⚠️ WARNING: Response was truncated. Increase max_tokens."
                
                # Try to parse and format as JSON
                try:
                    cv_dict = json.loads(content)
                    formatted_content = json.dumps(cv_dict, ensure_ascii=False, indent=2)
                    return formatted_content + truncation_warning
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown
                    import re
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        try:
                            cv_dict = json.loads(json_match.group(1))
                            return json.dumps(cv_dict, ensure_ascii=False, indent=2) + truncation_warning
                        except:
                            pass
                    # Try to find JSON between first { and last }
                    first_brace = content.find('{')
                    last_brace = content.rfind('}')
                    if first_brace != -1 and last_brace != -1:
                        try:
                            cv_dict = json.loads(content[first_brace:last_brace + 1])
                            return json.dumps(cv_dict, ensure_ascii=False, indent=2) + truncation_warning
                        except:
                            pass
                    return content + truncation_warning + "\n\n⚠️ Note: Response may not be valid JSON."
            else:
                print(f"✗ OpenRouter failed with status {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"Error: {json.dumps(error_detail, indent=2)}")
                except:
                    print(f"Error: {response.text[:500]}")
        
        except Exception as e:
            print(f"✗ OpenRouter exception: {e}")

    # Fallback to LMStudio
    print(f"\nFalling back to LMStudio for CV rewriting...")
    
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
        }

        response = requests.post(
            lmstudio_url,
            json=request_payload,
            timeout=300
        )
        
        if not response.ok:
            try:
                error_detail = response.json()
                return f"⚠️ Error from LM Studio ({response.status_code}): {error_detail}"
            except:
                return f"⚠️ Error from LM Studio ({response.status_code}): {response.text}"
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        finish_reason = data["choices"][0].get("finish_reason", "")
        truncation_warning = ""
        if finish_reason == "length":
            truncation_warning = "\n\n⚠️ WARNING: Response was truncated. Increase max_tokens."
        
        # Try to parse and format as JSON
        try:
            cv_dict = json.loads(content)
            formatted_content = json.dumps(cv_dict, ensure_ascii=False, indent=2)
            return formatted_content + truncation_warning
        except json.JSONDecodeError:
            # Try to extract JSON
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    cv_dict = json.loads(json_match.group(1))
                    return json.dumps(cv_dict, ensure_ascii=False, indent=2) + truncation_warning
                except:
                    pass
            first_brace = content.find('{')
            last_brace = content.rfind('}')
            if first_brace != -1 and last_brace != -1:
                try:
                    cv_dict = json.loads(content[first_brace:last_brace + 1])
                    return json.dumps(cv_dict, ensure_ascii=False, indent=2) + truncation_warning
                except:
                    pass
            return content + truncation_warning + "\n\n⚠️ Note: Response may not be valid JSON."

    except requests.RequestException as e:
        return f"⚠️ Error calling LM Studio: {str(e)}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
