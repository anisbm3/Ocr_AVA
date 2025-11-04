"""
CV Parser Agent - Structured Resume Extraction
Extracts CV data into structured JSON format using Qwen Vision + LLM
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
    Dedicated CV Parser that extracts structured resume data
    Uses Qwen Vision for text extraction + LLM for structured parsing
    """
    
    def __init__(self, qwen_agent, llm_client):
        """
        Initialize CV Parser
        
        Args:
            qwen_agent: QwenVisionAgent instance for text extraction
            llm_client: LLMClient instance for structured parsing
        """
        self.qwen_agent = qwen_agent
        self.llm_client = llm_client
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
        
        # Step 1: Extract raw text using PyMuPDF for PDFs, Qwen Vision for images
        if str(cv_path).lower().endswith('.pdf'):
            extraction_result = self._extract_pdf_with_pymupdf(cv_path)
        else:
            extraction_result = self.qwen_agent.extract_cv_data(cv_path)
        
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
        Parse raw CV text into structured JSON using LLM
        
        Args:
            raw_text: Raw extracted text from CV
        
        Returns:
            Structured resume dictionary
        """
        prompt = f"""You are a professional CV/Resume parser. Extract and structure the following resume into JSON format.

RESUME TEXT:
{raw_text[:4000]}

REQUIRED JSON STRUCTURE (return ONLY valid JSON, no markdown):
{{
  "personalInformation": {{
    "fullName": "string",
    "email": "string",
    "links": ["array of URLs/profiles"],
    "summary": "professional summary or objective"
  }},
  "education": [
    {{
      "degree": "string",
      "major": "string",
      "institution": "string",
      "location": "string",
      "startDate": "YYYY-MM or YYYY",
      "endDate": "YYYY-MM or YYYY or 'Present'",
      "gpa": "string (if available)"
    }}
  ],
  "workExperience": [
    {{
      "jobTitle": "string",
      "company": "string",
      "location": "string",
      "startDate": "YYYY-MM or YYYY",
      "endDate": "YYYY-MM or YYYY or 'Present'",
      "description": ["array of responsibilities/achievements"],
      "tags": ["relevant skill keywords"]
    }}
  ],
  "projects": [
    {{
      "projectName": "string",
      "description": "string",
      "role": "string",
      "tags": ["technologies used"],
      "startDate": "YYYY-MM",
      "endDate": "YYYY-MM",
      "link": "string (if available)"
    }}
  ],
  "skills": ["array of all skills, technologies, tools"],
  "languages": [
    {{
      "language": "string",
      "proficiency": "Native/Fluent/Professional/Basic"
    }}
  ],
  "certifications": [
    {{
      "certificationName": "string",
      "dateObtained": "YYYY-MM",
      "expirationDate": "YYYY-MM (if applicable)"
    }}
  ],
  "awards": [
    {{
      "awardName": "string",
      "issuingOrganization": "string",
      "dateReceived": "YYYY-MM",
      "description": "string"
    }}
  ],
  "volunteerExperience": [
    {{
      "role": "string",
      "organization": "string",
      "location": "string",
      "startDate": "YYYY-MM",
      "endDate": "YYYY-MM",
      "description": "string"
    }}
  ]
}}

INSTRUCTIONS:
1. Extract all information accurately from the resume
2. If a field is not found, use empty string "" or empty array []
3. For skills, extract ALL mentioned technologies, tools, frameworks, languages
4. Return ONLY the JSON object, no explanations or markdown
5. Ensure all dates are in YYYY-MM format or YYYY for year-only
6. For "tags" fields, extract relevant keywords/technologies

Return the structured JSON now:"""

    async def parse_cv_with_llm(self, raw_text: str) -> Dict[str, Any]:
        """
        Parse raw CV text into structured JSON using LLM with fallback parsing

        Args:
            raw_text: Raw extracted text from CV

        Returns:
            Structured resume dictionary
        """
        prompt = f"""You are a professional CV/Resume parser. Extract and structure the following resume into JSON format.

RESUME TEXT:
{raw_text[:4000]}

REQUIRED JSON STRUCTURE (return ONLY valid JSON, no markdown):
{{
  "personalInformation": {{
    "fullName": "string",
    "email": "string",
    "links": ["array of URLs/profiles"],
    "summary": "professional summary or objective"
  }},
  "education": [
    {{
      "degree": "string",
      "major": "string",
      "institution": "string",
      "location": "string",
      "startDate": "YYYY-MM or YYYY",
      "endDate": "YYYY-MM or YYYY or 'Present'",
      "gpa": "string (if available)"
    }}
  ],
  "workExperience": [
    {{
      "jobTitle": "string",
      "company": "string",
      "location": "string",
      "startDate": "YYYY-MM or YYYY",
      "endDate": "YYYY-MM or YYYY or 'Present'",
      "description": ["array of responsibilities/achievements"],
      "tags": ["relevant skill keywords"]
    }}
  ],
  "projects": [
    {{
      "projectName": "string",
      "description": "string",
      "role": "string",
      "tags": ["technologies used"],
      "startDate": "YYYY-MM",
      "endDate": "YYYY-MM",
      "link": "string (if available)"
    }}
  ],
  "skills": ["array of all skills, technologies, tools"],
  "languages": [
    {{
      "language": "string",
      "proficiency": "Native/Fluent/Professional/Basic"
    }}
  ],
  "certifications": [
    {{
      "certificationName": "string",
      "dateObtained": "YYYY-MM",
      "expirationDate": "YYYY-MM (if applicable)"
    }}
  ],
  "awards": [
    {{
      "awardName": "string",
      "issuingOrganization": "string",
      "dateReceived": "YYYY-MM",
      "description": "string"
    }}
  ],
  "volunteerExperience": [
    {{
      "role": "string",
      "organization": "string",
      "location": "string",
      "startDate": "YYYY-MM",
      "endDate": "YYYY-MM",
      "description": "string"
    }}
  ]
}}

INSTRUCTIONS:
1. Extract all information accurately from the resume
2. If a field is not found, use empty string "" or empty array []
3. For skills, extract ALL mentioned technologies, tools, frameworks, languages
4. Return ONLY the JSON object, no explanations or markdown
5. Ensure all dates are in YYYY-MM format or YYYY for year-only
6. For "tags" fields, extract relevant keywords/technologies

Return the structured JSON now:"""

        try:
            logger.info("🤖 Sending CV to LLM for structured parsing...")
            llm_response = await self.llm_client.generate(prompt, max_tokens=3000, temperature=0.1)

            # Check if we got a mock response (indicates LLM unavailable)
            if "Using mock response" in llm_response or "Sample Candidate" in llm_response:
                logger.warning("⚠️ LLM returned mock response, using regex-based fallback parsing")
                return self._parse_cv_with_regex(raw_text)

            # Clean response (remove markdown if present)
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "").strip()

            # Parse JSON
            structured_data = json.loads(cleaned_response)

            # Ensure all required keys exist
            template = self.get_empty_resume_template()
            for key in template:
                if key not in structured_data:
                    structured_data[key] = template[key]

            return structured_data

        except json.JSONDecodeError as e:
            logger.error(f"❌ LLM response is not valid JSON: {e}")
            logger.debug(f"LLM Response: {llm_response[:500]}")
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


# Standalone test function
async def test_cv_parser():
    """Test CV parser with sample CV"""
    from ai.qwen_vision_agent import QwenVisionAgent
    from ai.llm_client import LLMClient
    
    print("=" * 70)
    print("🧪 CV PARSER TEST")
    print("=" * 70)
    
    # Initialize agents
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


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_cv_parser())
