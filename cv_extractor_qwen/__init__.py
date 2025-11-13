"""
CV Parser Agent - Structured Resume Extraction
Extracts CV data into structured JSON format using Local LLM (Qwen2.5-VL-7B via LM Studio)
Privacy-focused alternative to cloud-based solutions
"""
import json
import logging
import re
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class CVParserAgent:
    """
    Privacy-focused CV Parser using Local LLM
    Matches Gemini's quality while keeping data local via LM Studio
    """

    def __init__(self, qwen_agent=None, llm_client=None, cv_template_path: str = "./cv_template_camelCase.json"):
        """
        Initialize CV Parser

        Args:
            qwen_agent: Optional QwenVisionAgent for image processing
            llm_client: Optional LLMClient for text processing
            cv_template_path: Path to CV template JSON
        """
        self.qwen_agent = qwen_agent
        self.llm_client = llm_client
        
        # Load CV template
        try:
            with open(cv_template_path, "r") as f:
                self.cv_template = f.read()
            logger.info("✅ CV template loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load CV template: {e}")
            self.cv_template = json.dumps(self.get_empty_resume_template(), indent=2)
        
        logger.info("✅ CV Parser Agent initialized")

    def get_empty_resume_template(self) -> Dict[str, Any]:
        """Return empty resume template matching the standard format"""
        return {
            "personalInformation": {
                "fullName": "",
                "email": "",
                "links": [],
                "summary": ""
            },
            "education": [],
            "workExperience": [],
            "projects": [],
            "skills": [],
            "languages": [],
            "certifications": [],
            "awards": [],
            "volunteerExperience": [],
            "extractionMetadata": {
                "extractionMethod": "",
                "confidence": 0.0,
                "extractedAt": "",
                "rawTextLength": 0
            }
        }

    async def extract_and_parse_cv(self, cv_path: str) -> Dict[str, Any]:
        """
        Complete CV extraction and parsing pipeline

        Args:
            cv_path: Path to CV file (PDF or image)

        Returns:
            Structured resume JSON
        """
        logger.info(f"📄 Starting CV extraction: {cv_path}")

        # Step 1: Extract raw text using PyMuPDF for PDFs
        # Check if it's a PDF file or has no extension (assume PDF from Gradio)
        file_ext = str(cv_path).lower()
        if file_ext.endswith('.pdf') or not Path(cv_path).suffix:
            extraction_result = self._extract_pdf_with_pymupdf(cv_path)
        else:
            # For non-PDF files, return error
            logger.error(f"❌ Unsupported file format: {cv_path}. Only PDF files are supported.")
            return {
                "personalInformation": {
                    "fullName": "",
                    "email": "",
                    "phone": "",
                    "location": "",
                    "summary": ""
                },
                "education": [],
                "workExperience": [],
                "projects": [],
                "skills": [],
                "languages": [],
                "certifications": [],
                "awards": [],
                "volunteerExperience": [],
                "extractionMetadata": {
                    "extractionMethod": "unsupported_format",
                    "confidence": 0.0,
                    "extractedAt": datetime.utcnow().isoformat(),
                    "rawTextLength": 0,
                    "error": f"Unsupported file format: {Path(cv_path).suffix}. Only PDF files are supported."
                }
            }

        raw_text = extraction_result.get("text", "")
        extraction_method = extraction_result.get("method", "unknown")
        confidence = extraction_result.get("confidence", 0.0)

        if not raw_text or len(raw_text.strip()) < 50:
            logger.warning("⚠️ Insufficient text extracted from CV")
            resume = self.get_empty_resume_template()
            resume["extractionMetadata"] = {
                "extractionMethod": extraction_method,
                "confidence": 0.0,
                "extractedAt": datetime.utcnow().isoformat(),
                "rawTextLength": len(raw_text),
                "error": "Insufficient text extracted"
            }
            return resume

        logger.info(f"✅ Text extracted ({len(raw_text)} chars) using {extraction_method}")

        # Step 2: Parse text into structured JSON using LLM
        try:
            structured_resume = await self.parse_cv_with_llm(raw_text)

            # Add metadata
            structured_resume["extractionMetadata"] = {
                "extractionMethod": extraction_method,
                "confidence": confidence,
                "extractedAt": datetime.utcnow().isoformat(),
                "rawTextLength": len(raw_text)
            }

            # Quick validation
            if not structured_resume.get("personalInformation", {}).get("fullName"):
                logger.warning("⚠️ No name found in CV - attempting regex extraction")
                structured_resume["personalInformation"]["fullName"] = self._extract_name_fallback(raw_text)

            if not structured_resume.get("personalInformation", {}).get("email"):
                logger.warning("⚠️ No email found - attempting regex extraction")
                structured_resume["personalInformation"]["email"] = self._extract_email_fallback(raw_text)

            logger.info(f"✅ CV parsed successfully - Found {len(structured_resume.get('skills', []))} skills")
            return structured_resume

        except Exception as e:
            logger.error(f"❌ LLM parsing failed: {e}")
            # Return template with raw text embedded
            resume = self.get_empty_resume_template()
            resume["extractionMetadata"] = {
                "extractionMethod": extraction_method,
                "confidence": confidence,
                "extractedAt": datetime.utcnow().isoformat(),
                "rawTextLength": len(raw_text),
                "error": str(e),
                "rawText": raw_text[:1000]  # Include snippet for debugging
            }
            return resume

    async def parse_cv_with_llm(self, raw_text: str) -> Dict[str, Any]:
        """
        Parse raw CV text into structured JSON using Local LLM
        Mimics Gemini's approach but keeps data private via LM Studio

        Args:
            raw_text: Raw extracted text from CV

        Returns:
            Structured resume dictionary
        """
        prompt = f"""You will receive a resume as the next input. Convert it into a single JSON object that exactly matches the template below:

```json
{self.cv_template}
```

Output requirements:
- Return only one valid JSON object and nothing else (no explanations, no headings, no surrounding text or code fences).
- Preserve the template's keys and structure exactly.
- For any missing or unknown value, use the template's empty defaults: empty string "" for text fields and empty list [] for list fields.
- For list fields (e.g. skills, languages, certifications) return arrays of short strings.
- For workExperience and education entries, populate subfields (jobTitle, company, startDate, endDate, description); if a subfield is missing use an empty string.
- Ensure the output is valid, parseable JSON.

Do not include any additional text before or after the JSON object.

RESUME TEXT:
{raw_text[:4000]}

"""

        # Check if LLM is available
        if not self.llm_client or not self.llm_client.provider:
            logger.warning("⚠️ LLM not available, using regex-based fallback parsing")
            return self._parse_cv_with_regex(raw_text)

        try:
            logger.info("🤖 Sending CV to Local LLM (Qwen2.5-VL-7B) for structured parsing...")
            llm_response = await self.llm_client.generate(prompt, max_tokens=4000, temperature=0.1)

            # Check if we got a mock/invalid response (indicates LLM unavailable)
            if "INVALID_JSON_RESPONSE_TO_TRIGGER_FALLBACK" in llm_response:
                logger.warning("⚠️ LLM unavailable, using regex-based fallback parsing")
                return self._parse_cv_with_regex(raw_text)
            
            # Extract JSON from response (similar to Gemini approach)
            text_output = llm_response.strip()
            
            # Find JSON object boundaries
            json_start = text_output.find("{")
            json_end = text_output.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("⚠️ No valid JSON found in LLM response, using fallback")
                return self._parse_cv_with_regex(raw_text)
            
            json_str = text_output[json_start:json_end]
            structured_data = json.loads(json_str)

            # Ensure all required keys exist
            template = self.get_empty_resume_template()
            for key in template:
                if key not in structured_data:
                    structured_data[key] = template[key]

            logger.info("✅ Local LLM parsing successful")
            return structured_data

        except json.JSONDecodeError as e:
            logger.error(f"❌ LLM response is not valid JSON: {e}")
            logger.debug(f"LLM Response: {llm_response[:500] if 'llm_response' in locals() else 'No response'}")
            # Fall back to regex parsing
            logger.info("🔄 Falling back to regex-based parsing due to JSON error")
            return self._parse_cv_with_regex(raw_text)
        except Exception as e:
            logger.error(f"❌ LLM parsing error: {e}")
            # Fall back to regex parsing
            logger.info("🔄 Falling back to regex-based parsing due to LLM error")
            return self._parse_cv_with_regex(raw_text)

    def _extract_email_fallback(self, text: str) -> str:
        """Fallback: Extract email using regex"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else ""

    def _extract_name_fallback(self, text: str) -> str:
        """Fallback: Extract likely name from first few lines"""
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            # Likely a name if it's 2-4 words, capitalized, no numbers
            words = line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w) and not any(c.isdigit() for c in line):
                return line
        return ""

    def _parse_cv_with_regex(self, raw_text: str) -> Dict[str, Any]:
        """
        Fallback CV parsing using regex and text processing when LLM is unavailable

        Args:
            raw_text: Raw extracted text from CV

        Returns:
            Structured resume dictionary with basic extracted information
        """
        logger.info("🔍 Using regex-based CV parsing fallback")

        # Initialize empty template
        resume = self.get_empty_resume_template()

        # Extract personal information
        resume["personalInformation"]["fullName"] = self._extract_name_fallback(raw_text)
        resume["personalInformation"]["email"] = self._extract_email_fallback(raw_text)
        resume["personalInformation"]["links"] = self._extract_links_fallback(raw_text)

        # Extract phone number
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})'
        phones = re.findall(phone_pattern, raw_text)
        if phones:
            resume["personalInformation"]["phone"] = phones[0][0] + phones[0][1] + "-" + phones[0][2] + "-" + phones[0][3]

        # Extract skills (common tech skills)
        tech_skills = [
            "python", "java", "javascript", "c\\+\\+", "c#", "php", "ruby", "go", "rust",
            "html", "css", "react", "angular", "vue", "node", "express", "django", "flask",
            "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "linux",
            "machine learning", "ai", "data science", "tensorflow", "pytorch", "pandas",
            "numpy", "scikit-learn", "opencv", "nlp", "computer vision"
        ]

        found_skills = []
        text_lower = raw_text.lower()
        for skill in tech_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found_skills.append(skill.title())

        resume["skills"] = found_skills

        # Extract education (basic pattern matching)
        education_keywords = ["bachelor", "master", "phd", "degree", "university", "college", "school"]
        education_lines = []
        for line in raw_text.split('\n'):
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in education_keywords):
                education_lines.append(line.strip())

        # Try to structure education entries
        for line in education_lines[:3]:  # Limit to first 3 education entries
            edu_entry = {
                "degree": "",
                "major": "",
                "institution": "",
                "location": "",
                "startDate": "",
                "endDate": ""
            }

            # Extract degree
            degree_match = re.search(r'(bachelor|master|phd|associate).*?(?:of|in)?\s*([^\n,]*)', line, re.IGNORECASE)
            if degree_match:
                edu_entry["degree"] = degree_match.group(0).strip()

            # Extract institution
            inst_match = re.search(r'(university|college|institute|school).*?([^\n,]*)', line, re.IGNORECASE)
            if inst_match:
                edu_entry["institution"] = inst_match.group(0).strip()

            if edu_entry["degree"] or edu_entry["institution"]:
                resume["education"].append(edu_entry)

        # Extract work experience (basic pattern matching)
        work_keywords = ["experience", "work", "employment", "job", "position", "role"]
        work_lines = []
        for line in raw_text.split('\n'):
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in work_keywords) or re.search(r'\b\d{4}\b', line):
                work_lines.append(line.strip())

        # Try to structure work experience entries
        for line in work_lines[:5]:  # Limit to first 5 work entries
            work_entry = {
                "jobTitle": "",
                "company": "",
                "location": "",
                "startDate": "",
                "endDate": "",
                "description": [],
                "tags": []
            }

            # Extract dates
            date_matches = re.findall(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2})/?\d{4}|\b\d{4}\b', line)
            if len(date_matches) >= 2:
                work_entry["startDate"] = date_matches[0]
                work_entry["endDate"] = date_matches[1]

            # Extract job title and company (simple heuristic)
            words = line.split()
            if len(words) > 3:
                work_entry["jobTitle"] = " ".join(words[:2])  # First 2 words as title
                work_entry["company"] = " ".join(words[2:])   # Rest as company

            if work_entry["jobTitle"] or work_entry["company"]:
                resume["workExperience"].append(work_entry)

        # Extract languages
        language_keywords = ["english", "french", "spanish", "german", "italian", "chinese", "japanese", "arabic", "hindi"]
        found_languages = []
        for lang in language_keywords:
            if re.search(r'\b' + lang + r'\b', text_lower):
                found_languages.append({"language": lang.title(), "proficiency": "Professional"})

        resume["languages"] = found_languages

        # Extract summary (first paragraph or first few lines)
        lines = raw_text.split('\n')
        summary_lines = []
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 20 and not re.search(r'\b(email|phone|address)\b', line, re.IGNORECASE):
                summary_lines.append(line)
                if len(summary_lines) >= 3:  # Take first 3 meaningful lines
                    break

        resume["personalInformation"]["summary"] = " ".join(summary_lines)

        logger.info(f"✅ Regex parsing complete - Found {len(found_skills)} skills, {len(resume['education'])} education entries, {len(resume['workExperience'])} work entries")
        return resume

    def _extract_links_fallback(self, text: str) -> list:
        """Extract URLs and social media links"""
        links = []

        # URL pattern
        url_pattern = r'https?://[^\s]+|www\.[^\s]+'
        urls = re.findall(url_pattern, text)
        links.extend(urls)

        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[^\s]+'
        linkedin_links = re.findall(linkedin_pattern, text, re.IGNORECASE)
        links.extend(linkedin_links)

        # GitHub pattern
        github_pattern = r'github\.com/[^\s]+'
        github_links = re.findall(github_pattern, text, re.IGNORECASE)
        links.extend(github_links)

        return list(set(links))  # Remove duplicates

    def _extract_pdf_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF using PyMuPDF (fitz)
        Fast and reliable, no external dependencies needed

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            import fitz  # PyMuPDF

            logger.info(f"📄 Extracting text from PDF with PyMuPDF: {pdf_path}")

            # Check if file exists
            if not Path(pdf_path).exists():
                print(f"❌ PDF file does not exist: {pdf_path}")
                logger.error(f"❌ PDF file does not exist: {pdf_path}")
                return self._pdf_to_image_fallback(pdf_path)

            # Open PDF
            doc = fitz.open(pdf_path)
            num_pages = len(doc)

            print(f"✅ PDF opened successfully: {num_pages} pages")
            logger.info(f"✅ PDF opened successfully: {num_pages} pages")

            # Extract text from all pages
            full_text = ""
            for page_num in range(num_pages):
                page = doc[page_num]
                text = page.get_text()
                full_text += text + "\n\n"

            doc.close()

            # Clean up text
            full_text = full_text.strip()

            print(f"DEBUG: PyMuPDF extracted {len(full_text)} characters from {num_pages} pages")
            logger.info(f"✅ Extracted {len(full_text)} characters")

            if not full_text or len(full_text) < 50:
                print(f"DEBUG: Text too short ({len(full_text)} chars), falling back to OCR")
                logger.warning("⚠️ PyMuPDF extracted very little text, trying OCR fallback...")
                # Try converting to image and using Qwen Vision
                return self._pdf_to_image_fallback(pdf_path)

            logger.info(f"✅ PyMuPDF extracted {len(full_text)} characters from {num_pages} pages")

            return {
                "text": full_text,
                "confidence": 0.95,  # PyMuPDF is very reliable for text-based PDFs
                "method": "pymupdf",
                "pages": num_pages,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ PyMuPDF extraction failed: {e}")
            logger.info("Trying image-based fallback...")
            return self._pdf_to_image_fallback(pdf_path)

    def _pdf_to_image_fallback(self, pdf_path: str) -> Dict[str, Any]:
        """
        Fallback: Convert PDF to image and use Qwen Vision
        Used when PyMuPDF fails or extracts no text (scanned PDFs)

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extracted text from image
        """
        try:
            import fitz
            from PIL import Image
            import tempfile

            logger.info("🔄 Converting PDF to image for Qwen Vision...")

            # Open PDF and get first page
            doc = fitz.open(pdf_path)
            first_page = doc[0]

            # Render to high-quality image
            zoom = 2.0  # 2x zoom for better OCR
            mat = fitz.Matrix(zoom, zoom)
            pix = first_page.get_pixmap(matrix=mat)

            # Save to temporary image
            temp_dir = Path("uploads/temp")
            temp_dir.mkdir(exist_ok=True, parents=True)

            image_path = temp_dir / f"{Path(pdf_path).stem}_qwen.png"
            pix.save(str(image_path))
            doc.close()

            logger.info(f"✅ PDF converted to image: {image_path}")

            # Now use Qwen Vision on the image
            return self.qwen_agent.extract_cv_data(str(image_path))

        except Exception as e:
            logger.error(f"❌ PDF to image conversion failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "method": "pdf_to_image_failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def save_resume_json(self, resume_data: Dict[str, Any], output_path: str) -> bool:
        """
        Save parsed resume to JSON file

        Args:
            resume_data: Structured resume dictionary
            output_path: Path to save JSON file

        Returns:
            True if successful
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(resume_data, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Resume JSON saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save resume JSON: {e}")
            return False


# Convenience functions for different extraction methods
async def extract_cv_with_qwen(cv_path: str) -> Dict[str, Any]:
    """
    Extract CV using Qwen Vision + LLM parsing

    Args:
        cv_path: Path to CV file

    Returns:
        Structured CV data
    """
    from llm_clients.qwen_vision_agent import QwenVisionAgent
    from llm_clients.llm_client import LLMClient

    qwen_agent = QwenVisionAgent()
    llm_client = LLMClient()
    parser = CVParserAgent(qwen_agent, llm_client)

    return await parser.extract_and_parse_cv(cv_path)


def extract_cv_simple_qwen(cv_path: str) -> Dict[str, Any]:
    """
    Simple CV extraction using Qwen Vision (text only, no LLM parsing)

    Args:
        cv_path: Path to CV file

    Returns:
        Raw extracted text
    """
    from llm_clients.qwen_vision_agent import QwenVisionAgent

    qwen_agent = QwenVisionAgent()
    result = qwen_agent.extract_cv_data(cv_path)

    return {
        "text": result.get("text", ""),
        "confidence": result.get("confidence", 0.0),
        "method": result.get("method", "qwen_vision")
    }


# Backward compatibility function
async def resume_to_json_qwen(filepath: str) -> Dict[str, Any]:
    """
    Extract CV using Qwen (backward compatibility function)

    Args:
        filepath: Path to CV file

    Returns:
        Structured CV data
    """
    return await extract_cv_with_qwen(filepath)


# Standalone test function
async def test_cv_parser():
    """Test CV parser with sample CV"""
    print("=" * 70)
    print("🧪 CV PARSER TEST")
    print("=" * 70)

    # Initialize agents
    from llm_clients.qwen_vision_agent import QwenVisionAgent
    from llm_clients.llm_client import LLMClient

    qwen_agent = QwenVisionAgent()
    llm_client = LLMClient()
    parser = CVParserAgent(qwen_agent, llm_client)

    # Test with a CV file
    cv_path = "uploads/cvs/sample_cv.pdf"  # Change to your test file

    if not Path(cv_path).exists():
        print(f"❌ Test file not found: {cv_path}")
        return

    print(f"\n📄 Parsing: {cv_path}")
    resume_data = await parser.extract_and_parse_cv(cv_path)

    print("\n📊 RESULTS:")
    print(f"Name: {resume_data['personalInformation'].get('fullName', 'N/A')}")
    print(f"Email: {resume_data['personalInformation'].get('email', 'N/A')}")
    print(f"Skills: {len(resume_data.get('skills', []))} found")
    print(f"Experience: {len(resume_data.get('workExperience', []))} positions")
    print(f"Education: {len(resume_data.get('education', []))} degrees")

    # Save result
    parser.save_resume_json(resume_data, "test_resume_output.json")
    print("\n✅ Test complete! Check test_resume_output.json")


async def extract_cv_to_json(filepath: str) -> Dict[str, Any]:
    """
    Extract CV to JSON using Local LLM (Qwen2.5-VL-7B via LM Studio)
    Privacy-focused alternative - keeps all data local
    
    Args:
        filepath: Path to CV file (PDF or image)
    
    Returns:
        Structured CV data as dict
    """
    from llm_clients.qwen_vision_agent import QwenVisionAgent
    from llm_clients.llm_client import LLMClient
    
    qwen_agent = QwenVisionAgent()
    llm_client = LLMClient()
    parser = CVParserAgent(qwen_agent, llm_client)
    
    return await parser.extract_and_parse_cv(filepath)


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_cv_parser())