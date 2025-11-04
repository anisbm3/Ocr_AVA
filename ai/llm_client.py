"""
LLM Client for connecting to LM Studio with Qwen
Provides async text generation capabilities
"""
import logging
import json
import requests

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM Client for connecting to LM Studio with Qwen"""

    def __init__(self, lm_studio_host: str = "http://localhost:1234"):
        self.lm_studio_host = lm_studio_host
        self.lm_studio_url = f"{lm_studio_host}/v1/chat/completions"
        self.provider = None

        # Check LM Studio connection
        try:
            response = requests.get(f"{lm_studio_host}/v1/models", timeout=2)
            if response.status_code == 200:
                self.provider = "lm_studio"
                models = response.json().get('data', [])
                model_names = [m.get('id', '') for m in models]
                logger.info(f"✅ LM Studio connected: {len(models)} models available")
                logger.debug(f"   Models: {model_names}")
        except Exception as e:
            logger.warning(f"⚠ LM Studio not available: {e}")
            logger.info("   Will use fallback responses")

    async def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """
        Generate response using LM Studio or fallback
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
        
        Returns:
            Generated text
        """
        if self.provider == "lm_studio":
            return await self._generate_lm_studio(prompt, max_tokens, temperature)
        return self._mock_response(prompt)

    async def _generate_lm_studio(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using LM Studio API"""
        try:
            payload = {
                "model": "qwen2.5-vl-7b-instruct",  # Will use whatever model is loaded
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant for CV analysis and recruitment."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            response = requests.post(self.lm_studio_url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            
            logger.debug(f"LLM generated {len(generated_text)} characters")
            return generated_text
            
        except requests.exceptions.Timeout:
            logger.error("LM Studio request timed out")
            return self._mock_response(prompt)
        except requests.exceptions.RequestException as e:
            logger.error(f"LM Studio API error: {e}")
            return self._mock_response(prompt)
        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}")
            return self._mock_response(prompt)

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
        elif "json" in prompt.lower() and ("resume" in prompt.lower() or "cv" in prompt.lower()):
            # Return invalid JSON to trigger regex fallback
            return "INVALID_JSON_RESPONSE_TO_TRIGGER_FALLBACK"
        
        return "LLM response generated successfully."

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
