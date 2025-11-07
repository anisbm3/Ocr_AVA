import os
import json
import dotenv
import gradio as gr
from google import genai
import chromadb
from uuid import uuid4

# load tokens
dotenv.load_dotenv()
# initialize clients
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# chroma db
chroma_client = chromadb.PersistentClient()
try:
    collection = chroma_client.create_collection(name="my_collection")
except:
    collection = chroma_client.get_collection(name="my_collection")


with open("./cv_template_camelCase.json", "r") as f:
    CV_TEMPLATE = f.read()  # read as text and not json to pass to the model

# jobs_descriptions = json.load(open("./job_descriptions.json", "r"))
with open("./job_descriptions.json", "r") as f:
    JOB_DESCRIPTIONS = json.load(f)
job_descriptions = [json.dumps(job, indent=2) for job in JOB_DESCRIPTIONS]
    



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


def add_new_job_to_db(job_data: str):
    # create unique job id
    job_id = str(uuid4())
    # Parse json
    job = json.loads(job_data)
    # Add job to database (simulated)
    collection.add(
        ids=[job_id],
        documents=[job_data],
        metadatas=[
            {
                "title": job.get("title", ""),
                "companyName": job.get("companyName", ""),
                "companyId": job.get("companyId", ""),
                "location": job.get("location", ""),
                "type": job.get("type", ""),
                "description": job.get("description", ""),
                "requirements": job.get("requirements", ""),
                "responsibilities": job.get("responsibilities", ""),
                "benefits": job.get("benefits", ""),
                "salaryMin": job.get("salaryMin", ""),
                "salaryMax": job.get("salaryMax", ""),
                "currency": job.get("currency", ""),
                "experienceLevel": job.get("experienceLevel", ""),
                "remote": job.get("remote", ""),
                "skills": ", ".join(job.get("skills", [])) if isinstance(job.get("skills", []), list) else job.get("skills", ""),
                "expiresAt": job.get("expiresAt", ""),
                "slug": job.get("slug", ""),
                "externalUrl": job.get("externalUrl", ""),
                "source": job.get("source", ""),
            }
        ],
    )
    return "Inserted job with ID: " + job_id


def match_cv_with_jobs(cv:str,n_results:int=5):
    results = collection.query(
        query_texts=[cv],
        n_results=n_results,
    )
    return results

def get_entire_collection():
    return collection.get()

def delete_job_from_db(job_id: str):
    collection.delete(ids=[job_id])
    return f"Deleted job with ID: {job_id}"

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


Add_To_Job_DB = gr.Interface(
    fn=add_new_job_to_db,
    inputs=gr.Textbox(lines=20),
    outputs="text",
    title="Add New Job to Database",
    api_name="add_to_job_db",
    examples=job_descriptions,
    cache_examples=False,
)

Match_CV_With_Jobs = gr.Interface(
    fn=match_cv_with_jobs,
    inputs=[gr.Textbox(lines=20), gr.Number(value=5, label="Number of Results")],
    outputs="json",
    title="Match CV with Jobs",
    description="the smaller the distance the closer it is to the original CV",
    api_name="match_cv_with_jobs",
    examples=[[CV_TEMPLATE,2]],
    cache_examples=False,
)

Get_Entire_Collection = gr.Interface(
    fn=get_entire_collection,
    inputs=[],
    outputs="json",
    title="Get Entire Job DB",
    api_name="get_entire_job_db",
)

Delete_Job_From_DB = gr.Interface(
    fn=delete_job_from_db,
    inputs=gr.Textbox(label="Job ID"),
    outputs="text",
    title="Delete Job from Database",
    api_name="delete_job_from_db",
)

demo = gr.TabbedInterface(
    [resume_to_json_extractor, Experience_judge, Add_To_Job_DB, Match_CV_With_Jobs, Get_Entire_Collection, Delete_Job_From_DB],
    ["Resume to JSON", "Experience Judge", "Add To Job DB", "Match CV With Jobs", "Get Entire Collection", "Delete Job from DB"],
)

demo.launch(debug=True,mcp_server=True)
