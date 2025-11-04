"""
Qwen2.5-VL-7B-Instruct Vision Agent using LM Studio
Replaces OCR functionality with advanced vision-language model
"""
import os
import json
import logging
import base64
import requests
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class QwenVisionAgent:
    """
    Vision Agent using Qwen2.5-VL-7B-Instruct via LM Studio
    Replaces traditional OCR with vision-language model capabilities
    """

    def __init__(self, lm_studio_host: str = "http://localhost:1234"):
        self.lm_studio_host = lm_studio_host
        self.api_url = f"{lm_studio_host}/v1/chat/completions"
        self.model_name = "qwen2.5-vl-7b-instruct"  # Adjust based on your LM Studio model name

        # Check connection
        self.is_connected = self.check_connection()

    def check_connection(self) -> bool:
        """Check if LM Studio is running and accessible"""
        try:
            response = requests.get(f"{self.lm_studio_host}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                logger.info(f"✓ LM Studio connected. Available models: {models.get('data', [])}")
                return True
            else:
                logger.warning(f"⚠ LM Studio returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Cannot connect to LM Studio at {self.lm_studio_host}: {e}")
            logger.info("Make sure LM Studio is running with Qwen2.5-VL-7B-Instruct loaded")
            return False

    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode image to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None

    def analyze_image(
        self,
        image_path: str,
        prompt: str = "Extract all text from this image. Include names, contact information, education, work experience, skills, and any other relevant details.",
        max_tokens: int = 2048,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Analyze image using Qwen2.5-VL through LM Studio API

        Args:
            image_path: Path to image file
            prompt: Analysis prompt
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (lower = more focused)

        Returns:
            Dictionary with extracted text and confidence
        """
        if not self.is_connected:
            return {
                "text": "",
                "confidence": 0.0,
                "error": "LM Studio not connected",
                "method": "qwen_vision_failed"
            }

        try:
            # Encode image to base64
            image_base64 = self.encode_image_to_base64(image_path)
            if not image_base64:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "error": "Failed to encode image",
                    "method": "qwen_vision_failed"
                }

            # Prepare the API request for vision model
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }

            # Make API request
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120  # Vision models may take longer
            )
            response.raise_for_status()

            # Parse response
            data = response.json()
            extracted_text = data["choices"][0]["message"]["content"]

            return {
                "text": extracted_text,
                "confidence": 0.95,  # Qwen2.5-VL is highly accurate
                "method": "qwen2.5-vl",
                "model": self.model_name,
                "timestamp": datetime.utcnow().isoformat()
            }

        except requests.exceptions.Timeout:
            logger.error("LM Studio request timed out")
            return {
                "text": "",
                "confidence": 0.0,
                "error": "Request timeout",
                "method": "qwen_vision_failed"
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"LM Studio API error: {e}")
            logger.warning("⚠️ Qwen Vision failed, falling back to Tesseract OCR...")
            return self._fallback_to_ocr(image_path)
        except Exception as e:
            logger.error(f"Unexpected error in Qwen vision analysis: {e}")
            logger.warning("⚠️ Qwen Vision failed, falling back to Tesseract OCR...")
            return self._fallback_to_ocr(image_path)
    
    def _fallback_to_ocr(self, image_path: str) -> Dict[str, Any]:
        """Fallback to Tesseract OCR when Qwen Vision fails"""
        try:
            from PIL import Image
            import pytesseract
            
            # Open image
            img = Image.open(image_path)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(img)
            
            logger.info("✓ Fallback OCR extraction successful")
            return {
                "text": text,
                "confidence": 0.7,  # OCR is less accurate
                "method": "tesseract_ocr",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as ocr_error:
            logger.error(f"OCR fallback also failed: {ocr_error}")
            
            # Last resort: Try PyMuPDF if it's a PDF
            if str(image_path).lower().endswith('.pdf'):
                try:
                    import fitz  # PyMuPDF
                    logger.info("Trying PyMuPDF for PDF text extraction...")
                    
                    doc = fitz.open(image_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    doc.close()
                    
                    logger.info("✓ PyMuPDF extraction successful")
                    return {
                        "text": text,
                        "confidence": 0.8,
                        "method": "pymupdf",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                except Exception as pdf_error:
                    logger.error(f"PyMuPDF fallback also failed: {pdf_error}")
            
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(ocr_error),
                "method": "all_methods_failed"
            }

    def extract_cv_data(self, image_path: str) -> Dict[str, Any]:
        """
        Extract structured CV data from image using Qwen2.5-VL

        Args:
            image_path: Path to CV image

        Returns:
            Structured CV data
        """
        prompt = """Analyze this CV/resume image and extract the following information in a structured format:

1. Personal Information:
   - Full Name
   - Email address
   - Phone number
   - Location/Address
   - LinkedIn URL
   - GitHub URL
   - Portfolio URL

2. Professional Summary or Objective

3. Work Experience:
   - For each position: Company, Job Title, Duration, Responsibilities

4. Education:
   - For each degree: Institution, Degree, Field of Study, Year

5. Skills:
   - Technical skills
   - Soft skills
   - Languages
   - Certifications

6. Projects (if any)

7. Additional Information (awards, publications, etc.)

Please provide the information in a clear, structured format. If any information is not present, mention "Not found"."""

        result = self.analyze_image(image_path, prompt, max_tokens=3000, temperature=0.1)

        if result.get("text"):
            # Parse the structured response
            try:
                structured_data = self.parse_cv_text(result["text"])
                result["structured_data"] = structured_data
            except Exception as e:
                logger.error(f"Error parsing CV text: {e}")
                result["structured_data"] = {}

        return result

    def parse_cv_text(self, text: str) -> Dict[str, Any]:
        """
        Parse extracted CV text into structured format

        Args:
            text: Raw text from Qwen2.5-VL

        Returns:
            Structured dictionary
        """
        # Basic parsing - you can enhance this based on Qwen's output format
        structured = {
            "name": "",
            "email": "",
            "phone": "",
            "skills": [],
            "experience": [],
            "education": [],
            "raw_text": text
        }

        # Extract email using regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            structured["email"] = emails[0]

        # Extract phone using regex
        phone_pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,6}[-\s\.]?[0-9]{1,6}'
        phones = re.findall(phone_pattern, text)
        if phones:
            structured["phone"] = phones[0]

        # The rest can be extracted by Qwen2.5-VL's structured output
        # Since Qwen is good at following instructions, it will provide structured data

        return structured

    def analyze_video_frame(
        self,
        frame_path: str,
        prompt: str = "Describe what you see in this video frame. Focus on the person's appearance, emotions, setting, and any text visible."
    ) -> Dict[str, Any]:
        """
        Analyze video frame using Qwen2.5-VL

        Args:
            frame_path: Path to video frame image
            prompt: Analysis prompt

        Returns:
            Frame analysis results
        """
        return self.analyze_image(frame_path, prompt, max_tokens=1024, temperature=0.3)

    def analyze_document_page(
        self,
        page_image_path: str,
        document_type: str = "cv"
    ) -> Dict[str, Any]:
        """
        Analyze document page (CV, certificate, etc.)

        Args:
            page_image_path: Path to document page image
            document_type: Type of document (cv, certificate, letter, etc.)

        Returns:
            Extracted text and analysis
        """
        prompts = {
            "cv": "Extract all text from this CV/resume. Preserve the structure and formatting as much as possible.",
            "certificate": "Extract all text from this certificate, including issuing organization, recipient name, date, and certification details.",
            "letter": "Extract all text from this letter, preserving paragraph structure.",
            "form": "Extract all fields and values from this form."
        }

        prompt = prompts.get(document_type, prompts["cv"])
        return self.analyze_image(page_image_path, prompt, max_tokens=2048, temperature=0.1)


# Enhanced OCR Processor that uses Qwen2.5-VL
class EnhancedOCRProcessor:
    """
    Enhanced OCR processor that primarily uses Qwen2.5-VL
    Falls back to traditional OCR if needed
    """

    def __init__(self, lm_studio_host: str = "http://localhost:1234"):
        self.qwen_agent = QwenVisionAgent(lm_studio_host)
        self.engines = {}

        logger.info("🎯 Using Qwen2.5-VL as primary text extraction engine")

        # Initialize traditional OCR as fallback only if needed
        self._init_fallback_ocr()

    def _init_fallback_ocr(self):
        """Initialize traditional OCR engines as fallback"""
        try:
            import easyocr
            self.engines['easyocr'] = easyocr.Reader(['en'], gpu=False)
            logger.info("✓ EasyOCR available as fallback")
        except ImportError:
            pass

        try:
            from paddleocr import PaddleOCR
            self.engines['paddleocr'] = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            logger.info("✓ PaddleOCR available as fallback")
        except Exception:
            pass

    def extract_text(self, image_path: str) -> Tuple[str, float, str]:
        """
        Extract text from image using Qwen2.5-VL (primary) or traditional OCR (fallback)

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (extracted_text, confidence_score, method_used)
        """
        # Try Qwen2.5-VL first (primary method)
        if self.qwen_agent.is_connected:
            result = self.qwen_agent.extract_cv_data(image_path)
            if result.get("text") and result.get("confidence", 0) > 0.5:
                return result["text"], result["confidence"], result["method"]
            else:
                logger.warning("Qwen2.5-VL extraction failed or low confidence, trying fallback OCR")

        # Fallback to traditional OCR
        if 'easyocr' in self.engines:
            return self._extract_easyocr(image_path)
        elif 'paddleocr' in self.engines:
            return self._extract_paddleocr(image_path)
        else:
            return "", 0.0, "no_ocr_available"

    def _extract_easyocr(self, image_path: str) -> Tuple[str, float, str]:
        """Fallback: Extract text using EasyOCR"""
        try:
            results = self.engines['easyocr'].readtext(image_path)
            full_text = ' '.join([text for (bbox, text, confidence) in results])
            avg_confidence = np.mean([confidence for (bbox, text, confidence) in results]) if results else 0.0
            return full_text, avg_confidence, "easyocr_fallback"
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return "", 0.0, "easyocr_failed"

    def _extract_paddleocr(self, image_path: str) -> Tuple[str, float, str]:
        """Fallback: Extract text using PaddleOCR"""
        try:
            result = self.engines['paddleocr'].ocr(image_path, cls=True)
            texts = []
            confidences = []

            if result and result[0]:
                for line in result[0]:
                    texts.append(line[1][0])
                    confidences.append(line[1][1])

            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            return full_text, avg_confidence, "paddleocr_fallback"
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return "", 0.0, "paddleocr_failed"


# Test function
def test_qwen_vision():
    """Test Qwen2.5-VL vision agent"""
    print("=" * 70)
    print("Testing Qwen2.5-VL Vision Agent")
    print("=" * 70)

    agent = QwenVisionAgent()

    if not agent.is_connected:
        print("❌ LM Studio not connected!")
        print("Please:")
        print("1. Open LM Studio")
        print("2. Load Qwen2.5-VL-7B-Instruct model")
        print("3. Start the local server")
        return

    print("✅ Qwen2.5-VL connected!")

    # Test with a sample image (you need to provide)
    test_image = "test_cv.jpg"  # Replace with actual path

    if os.path.exists(test_image):
        print(f"\n📄 Analyzing: {test_image}")
        result = agent.extract_cv_data(test_image)

        print("\n=== Results ===")
        print(f"Method: {result.get('method')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"\nExtracted Text:\n{result.get('text')[:500]}...")
    else:
        print(f"\n⚠ Test image not found: {test_image}")
        print("Create a test image or provide correct path")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    import re  # Import for parsing
    test_qwen_vision()