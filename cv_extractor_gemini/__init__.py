"""
Gemini CV Extractor Module
Extracts CV data into structured JSON format using Google Gemini
"""
import os
import json
from google import genai
from typing import Dict, Any

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass  # dotenv not available, use environment variables directly

class GeminiCVExtractor:
    """
    CV Extractor using Google Gemini API
    Converts resume files to structured JSON format
    """

    def __init__(self, api_key: str = None, cv_template_path: str = "./cv_template_camelCase.json"):
        """
        Initialize Gemini CV Extractor

        Args:
            api_key: Google Gemini API key (optional, will use env var if not provided)
            cv_template_path: Path to CV template JSON file
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)

        # Load CV template
        with open(cv_template_path, "r") as f:
            self.cv_template = f.read()

    def extract_cv_to_json(self, filepath: str) -> Dict[str, Any]:
        """
        Extract CV from file and convert to JSON

        Args:
            filepath: Path to the CV file (PDF, DOC, etc.)

        Returns:
            Dict containing structured CV data
        """
        # Upload file to Gemini
        uploaded_file = self.client.files.upload(file=filepath)

        # Generate content using Gemini
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                f"""You will receive a resume as the next input. Convert it into a single JSON object that exactly matches the template below:
                ```json
                {self.cv_template}
                ```
                Output requirements:
                - Return only one valid JSON object and nothing else (no explanations, no headings, no surrounding text or code fences).
                - Preserve the template's keys and structure exactly.
                - For any missing or unknown value, use the template's empty defaults: empty string "" for text fields and empty list [] for list fields.
                - For list fields (e.g. skills, languages, certifications) return arrays of short strings.
                - For workExperience and education entries, populate subfields (title, company, startDate, endDate, description); if a subfield is missing use an empty string.
                - Ensure the output is valid, parseable JSON.

                Do not include any additional text before or after the JSON object.
                """,
                uploaded_file,
            ],
        )

        text_output = response.text

        # Extract JSON from response
        json_start = text_output.find("{")
        json_end = text_output.rfind("}") + 1

        if json_start == -1 or json_end == 0:
            raise ValueError("No valid JSON found in Gemini response")

        json_str = text_output[json_start:json_end]
        return json.loads(json_str)

# Convenience function for backward compatibility
def resume_to_json(filepath: str) -> Dict[str, Any]:
    """
    Extract CV to JSON using Gemini (backward compatibility function)

    Args:
        filepath: Path to CV file

    Returns:
        Structured CV data as dict
    """
    extractor = GeminiCVExtractor()
    return extractor.extract_cv_to_json(filepath)