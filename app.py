import os
import json
import dotenv
import gradio as gr
from google import genai

dotenv.load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

with open("./cv_template_camelCase.json", "r") as f:
    CV_TEMPLATE = f.read()  # read as text and not json to pass to the model


experiences = """
[
    {
        "jobTitle": "Emergency Medical Technician (EMT)",
        "company": "Gospa Odv",
        "location": "",
        "startDate": "01/2025",
        "endDate": "Current",
        "description": ["Provided Basic Life Support (BLS) including CPR and AED.", "Stabilized patients during critical incidents for transport.", "Transported patients safely to medical facilities."],
        "tags": [""],
    },
    {
        "jobTitle": "Volunteer Firefighter",
        "company": "Gospa Odv",
        "location": "",
        "startDate": "01/2025",
        "endDate": "Current",
        "description": ["Responded to fire emergencies and assisted in fire suppression efforts.", "Conducted search and rescue operations in hazardous environments.", "Participated in community fire safety education programs."],
        "tags": ["firefighter", "emergency response"],
    },
]
"""


def resume_to_json(filepath):
    uploaded_file = client.files.upload(file=filepath)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            f"""You will receive a resume as the next input. Convert it into a single JSON object that exactly matches the template below:
            ```json
            {CV_TEMPLATE}
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
    # locate the first "{" and the last "}"
    json_start = text_output.find("{")
    json_end = text_output.rfind("}") + 1
    return json.loads(text_output[json_start:json_end])


def judge_experience(experience):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            f"""
You are an expert career advisor. Evaluate each description field of the following work experience for two criteria:
1. Does it follow the format "achieved X by Y", where X is a quantifiable result and Y is the action taken? If yes, respond with "Respected". If not, suggest a revised version that fits this format.
2. Does it use weak action verbs (e.g., "assisted", "helped", "participated")? If so, suggest stronger alternatives.

Return your output in the following JSON format (no extra text):
[
    {{
        "xyz_format": "Respected" | "Unrespected",
        "weak_verbs": "Weak" | "Strong",
        "suggested_modification": "..."
    }}
]
Example experience:
"Increased patient transport efficiency by 20% by optimizing ambulance routes."

Here is the experience to evaluate:
{experience}
"""
        ],
    )
    text_output = response.text
    json_start = text_output.find("[")
    json_end = text_output.rfind("]") + 1
    return json.loads(text_output[json_start:json_end])


resume_to_json_extractor = gr.Interface(
    fn=resume_to_json,
    inputs="file",
    outputs="json",
    title="Resume to JSON Extractor",
    api_name="resume_to_json_extractor",
)

Experience_judge = gr.Interface(
    fn=judge_experience,
    inputs=gr.Textbox(lines=10),
    outputs="json",
    title="Experience Judge",
    api_name="Experience_judge",
    examples=[experiences],
)

demo = gr.TabbedInterface(
    [resume_to_json_extractor, Experience_judge], [
        "Resume to JSON", "Experience Judge"]
)

demo.launch()
