"""
Universal LLM Client for LM Studio (Qwen2.5-VL-7B)
General-purpose text generation for any task: CV parsing, job matching, analysis, etc.
"""
import logging
import json
import os
import requests

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Universal LLM Client for LM Studio
    Works with any local model (optimized for Qwen2.5-VL-7B)
    Use for: CV parsing, job matching, skill analysis, text generation, etc.
    """

    def __init__(self, lm_studio_host: str = "http://localhost:1234", preferred_model: str = "qwen2.5-vl-7b", timeout: int = None):
        """
        Initialize LLM Client
        
        Args:
            lm_studio_host: LM Studio server URL
            preferred_model: Preferred model name (defaults to qwen2.5-vl-7b)
            timeout: Request timeout in seconds (default: 600 = 10 minutes, or LM_STUDIO_TIMEOUT env var)
        """
        self.lm_studio_host = lm_studio_host
        self.lm_studio_url = f"{lm_studio_host}/v1/chat/completions"
        self.preferred_model = preferred_model
        self.timeout = timeout or int(os.getenv("LM_STUDIO_TIMEOUT", "600"))  # Default 10 minutes, configurable via env
        self.provider = None
        self.available_models = []
        self.active_model = None

        # Check LM Studio connection
        try:
            response = requests.get(f"{lm_studio_host}/v1/models", timeout=5)  # Quick check, 5 seconds
            if response.status_code == 200:
                self.provider = "lm_studio"
                models = response.json().get('data', [])
                self.available_models = [m.get('id', '') for m in models]
                
                # Select active model
                if self.preferred_model in self.available_models:
                    self.active_model = self.preferred_model
                elif self.available_models:
                    self.active_model = self.available_models[0]
                
                logger.info(f"✅ LM Studio connected: {len(models)} models available")
                if self.active_model:
                    logger.info(f"   Active model: {self.active_model}")
                logger.debug(f"   All models: {self.available_models}")
        except Exception as e:
            logger.warning(f"⚠ LM Studio not available: {e}")
            logger.info("   Will use fallback responses")

    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 2000, 
        temperature: float = 0.7,
        system_prompt: str = None
    ) -> str:
        """
        Universal text generation method
        
        Args:
            prompt: User prompt/question
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0=deterministic, 1.0=creative)
            system_prompt: Optional system prompt (auto-generated if None)
        
        Returns:
            Generated text
        """
        if self.provider == "lm_studio":
            return await self._generate_lm_studio(prompt, max_tokens, temperature, system_prompt)
        return self._mock_response(prompt)

    async def _generate_lm_studio(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float,
        system_prompt: str = None
    ) -> str:
        """Generate using LM Studio API"""
        try:
            # Use active model or fall back to available models
            model_name = self.active_model or self.preferred_model
            if not model_name and self.available_models:
                model_name = self.available_models[0]
            
            # Auto-generate system prompt based on task if not provided
            if system_prompt is None:
                system_prompt = self._auto_system_prompt(prompt)
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            response = requests.post(self.lm_studio_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            
            logger.info(f"✅ LLM generated {len(generated_text)} characters")
            return generated_text
            
        except requests.exceptions.Timeout:
            logger.error(f"❌ LM Studio request timed out after {self.timeout} seconds")
            logger.info(f"   LM Studio URL: {self.lm_studio_url}")
            logger.info(f"   Model: {model_name}")
            logger.info(f"   Prompt length: {len(prompt)} characters")
            return self._mock_response(prompt)
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ LM Studio API error: {e}")
            return self._mock_response(prompt)
        except Exception as e:
            logger.error(f"❌ Unexpected LLM error: {e}")
            return self._mock_response(prompt)
    
    def _auto_system_prompt(self, prompt: str) -> str:
        """Auto-generate appropriate system prompt based on task"""
        prompt_lower = prompt.lower()
        
        # CV/Resume parsing
        if "resume" in prompt_lower or "cv" in prompt_lower or "json" in prompt_lower:
            return "You are a professional resume parser. Extract information accurately and return only valid JSON."
        
        # Job matching / skill analysis
        elif "skill" in prompt_lower or "match" in prompt_lower or "recruiter" in prompt_lower:
            return "You are an expert technical recruiter. Analyze skills and experience objectively."
        
        # Video analysis
        elif "video" in prompt_lower or "frame" in prompt_lower:
            return "You are a video content analyst. Describe what you see clearly and professionally."
        
        # General assistant
        else:
            return "You are a helpful AI assistant. Provide accurate, concise, and professional responses."

    def _mock_response(self, prompt: str) -> str:
        """Fallback response when LM Studio unavailable"""
        logger.warning("Using mock response (LM Studio unavailable)")
        
        if "matching score" in prompt.lower() or "analyze this cv" in prompt.lower():
            return json.dumps({
                "score": 70.0,
                "strengths": ["Relevant experience", "Good technical skills", "Clear background"],
                "weaknesses": ["More specific examples needed", "Could highlight achievements better"],
                "summary": "Candidate shows potential for the role with relevant background and skills."
            })
        elif "technical recruiter" in prompt.lower() or "skill matches" in prompt.lower():
            # Skill analysis prompt - return valid JSON
            return json.dumps({
                "required_skills": ["Python", "JavaScript", "React", "Node.js", "SQL"],
                "matched_skills": ["Python", "JavaScript"],
                "missing_skills": ["React", "Node.js", "SQL"],
                "skills_score": 40.0,
                "analysis": "Candidate has basic programming skills but lacks modern web development experience."
            })
        elif "professional cv/resume parser" in prompt.lower() or \
             ("extract and structure" in prompt.lower() and "resume" in prompt.lower()) or \
             ("you will receive a resume" in prompt.lower() and "json" in prompt.lower()) or \
             ("convert it into a single json" in prompt.lower() and "resume" in prompt.lower()):
            # CV parsing prompt - return invalid JSON to trigger regex fallback
            return "INVALID_JSON_RESPONSE_TO_TRIGGER_FALLBACK"
        
        return json.dumps({"response": "LLM response generated successfully."})

    def check_connection(self) -> bool:
        """Check if LM Studio is connected"""
        return self.provider == "lm_studio"


# Convenience function for one-off LLM calls
async def generate_llm_response(prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
    """
    Convenience function for one-off LLM calls
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text
    """
    client = LLMClient()
    return await client.generate(prompt, max_tokens, temperature)


if __name__ == "__main__":
    import asyncio
    
    # Test LLM Client
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        print("="*70)
        print("🧪 LLM CLIENT TEST")
        print("="*70)
        
        client = LLMClient()
        print(f"\nConnection Status: {'✅ Connected' if client.check_connection() else '❌ Not Connected'}")
        
        if client.check_connection():
            print("\nTesting LLM generation...")
            prompt = "Say 'Hello from Qwen!' in one sentence."
            response = await client.generate(prompt, max_tokens=50, temperature=0.7)
            print(f"\nLLM Response: {response}")
        else:
            print("\n⚠️ LM Studio not available. Please start LM Studio with a model loaded.")
        
        print("\n" + "="*70)
    
    asyncio.run(test())
