import json
import os
from collections import OrderedDict

from rapatriement_parser import SCHEMA_DEFAULTS


DEFAULT_MODEL = os.getenv("OCR_LLM_MODEL", "gpt-4.1-mini")
DEFAULT_MISTRAL_LLM_MODEL = os.getenv("MISTRAL_LLM_MODEL", "mistral-large-latest")

RAPATRIEMENT_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "dateDosRap": {
            "type": "string",
            "description": "Operation date in ISO format YYYY-MM-DD.",
        },
        "mntRap": {
            "type": "number",
            "description": "Droits a transfert cumules amount for the rapatriement operation.",
        },
        "typePieceBenef": {
            "type": "integer",
            "description": "Beneficiary document type. C maps to 1, S maps to 4.",
        },
        "noPieceBenef": {
            "type": "string",
            "description": "Beneficiary identification number.",
        },
    },
    "required": ["dateDosRap", "mntRap", "typePieceBenef", "noPieceBenef"],
}


def is_llm_configured():
    return bool(os.getenv("MISTRAL_API_KEY") or os.getenv("OPENAI_API_KEY"))


def extract_rapatriement_with_llm(text, filename="", next_step_context=""):
    if not is_llm_configured():
        raise RuntimeError("No LLM API key is configured. Set MISTRAL_API_KEY or OPENAI_API_KEY.")

    if os.getenv("MISTRAL_API_KEY"):
        return extract_rapatriement_with_mistral(text, filename, next_step_context)

    return extract_rapatriement_with_openai(text, filename, next_step_context)


def extract_rapatriement_with_mistral(text, filename="", next_step_context=""):
    try:
        from mistralai.client import Mistral
    except ImportError as error:
        raise RuntimeError("The mistralai package is not installed. Run: pip install mistralai") from error

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    prompt = build_prompt(text, filename, next_step_context)

    response = client.chat.complete(
        model=DEFAULT_MISTRAL_LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an intelligent document extraction engine. "
                    "You understand AVA rapatriement PDFs even when layout, labels, "
                    "table order, or OCR text order changes. Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw_json = response.choices[0].message.content
    return normalize_llm_data(json.loads(raw_json))


def extract_rapatriement_with_openai(text, filename="", next_step_context=""):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("The openai package is not installed. Run: pip install openai") from error

    client = OpenAI()
    prompt = build_prompt(text, filename, next_step_context)

    response = client.responses.create(
        model=DEFAULT_MODEL,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are an intelligent document extraction engine. "
                            "You understand AVA rapatriement PDFs even when layout, labels, "
                            "table order, or OCR text order changes. Return only the requested JSON."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "rapatriement_extraction",
                "schema": RAPATRIEMENT_JSON_SCHEMA,
                "strict": True,
            }
        },
    )

    raw_json = response.output_text
    return normalize_llm_data(json.loads(raw_json))


def build_prompt(text, filename, next_step_context):
    context = next_step_context.strip() if next_step_context else "No additional workflow context."
    return f"""
Extract the AVA rapatriement business JSON from the PDF text.

Do not rely on fixed positions. Understand the context and labels.
The PDF template can change, can be portrait or landscape, and OCR text order can be imperfect.

Required output fields:
- dateDosRap: date of the rapatriement operation row, not the document print date.
- mntRap: value corresponding to "DROITS A TRANSFERT CUMULES" or equivalent cumulative transfer rights.
- typePieceBenef: beneficiary document type. If beneficiary type is C, return 1. If it is S, return 4. If unknown, return 0.
- noPieceBenef: beneficiary identification number, usually near beneficiary code/type/numero labels.

Important:
- Dates must be returned as YYYY-MM-DD.
- Tunisian/French decimal comma numbers must be converted to JSON numbers.
- If the text contains several operations, choose the operation relevant to rapatriement/RAP unless the workflow context says otherwise.
- If a value is not present, return an empty string for string fields and 0 for numeric fields.

Filename: {filename}
Workflow/next-step context: {context}

PDF text:
---
{text}
---
""".strip()


def normalize_llm_data(data):
    normalized = OrderedDict(SCHEMA_DEFAULTS)
    normalized["dateDosRap"] = str(data.get("dateDosRap") or "")
    normalized["mntRap"] = float(data.get("mntRap") or 0)
    normalized["typePieceBenef"] = int(data.get("typePieceBenef") or 0)
    normalized["noPieceBenef"] = str(data.get("noPieceBenef") or "")
    return normalized
