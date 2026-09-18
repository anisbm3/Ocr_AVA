from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from llm_extractor import extract_rapatriement_with_llm, is_llm_configured
from mistral_ocr import extract_text_with_mistral_ocr, is_mistral_ocr_configured
from rapatriement_parser import extract_text_from_pdf, parse_rapatriement


app = FastAPI(
    title="Rapatriement OCR API",
    description="Extracts rapatriement data from AVA PDF documents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llmConfigured": is_llm_configured(),
        "mistralOcrConfigured": is_mistral_ocr_configured(),
    }


@app.post("/api/ocr/rapatriement")
async def extract_rapatriement(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    ocrProvider: str = Form("auto"),
    nextStepContext: str = Form(""),
):
    data, extraction_mode, extraction_method, warnings = await extract_from_upload(
        file,
        mode=mode,
        ocr_provider=ocrProvider,
        next_step_context=nextStepContext,
    )

    return {
        "success": True,
        "data": data,
        "extractionMode": extraction_mode,
        "extractionMethod": extraction_method,
        "errors": warnings,
    }


@app.post("/api/ocr/rapatriement/raw")
async def extract_rapatriement_raw(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    ocrProvider: str = Form("auto"),
    nextStepContext: str = Form(""),
):
    data, _, _, _ = await extract_from_upload(
        file,
        mode=mode,
        ocr_provider=ocrProvider,
        next_step_context=nextStepContext,
    )
    return data


async def extract_from_upload(file: UploadFile, mode="auto", ocr_provider="auto", next_step_context=""):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    mode = mode.lower().strip()
    if mode not in {"auto", "llm", "rules"}:
        raise HTTPException(status_code=400, detail="mode must be one of: auto, llm, rules.")

    ocr_provider = ocr_provider.lower().strip()
    if ocr_provider not in {"auto", "mistral", "local"}:
        raise HTTPException(status_code=400, detail="ocrProvider must be one of: auto, mistral, local.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")

    try:
        text, extraction_mode, ocr_warnings = extract_text_with_provider(pdf_bytes, ocr_provider)
        data, extraction_method, warnings = extract_with_requested_method(
            text=text,
            filename=file.filename,
            mode=mode,
            next_step_context=next_step_context,
        )
        warnings = ocr_warnings + warnings
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Could not extract PDF data: {error}") from error

    return data, extraction_mode, extraction_method, warnings


def extract_text_with_provider(pdf_bytes, ocr_provider):
    warnings = []

    if ocr_provider == "local":
        text, extraction_mode = extract_text_from_pdf(pdf_bytes)
        return text, extraction_mode, warnings

    if ocr_provider == "mistral":
        text = extract_text_with_mistral_ocr(pdf_bytes)
        return text, "mistral_ocr", warnings

    if is_mistral_ocr_configured():
        try:
            text = extract_text_with_mistral_ocr(pdf_bytes)
            return text, "mistral_ocr", warnings
        except Exception as error:
            warnings.append(f"Mistral OCR failed, fallback to local OCR/pdf text: {error}")

    text, extraction_mode = extract_text_from_pdf(pdf_bytes)
    return text, extraction_mode, warnings


def extract_with_requested_method(text, filename, mode, next_step_context):
    warnings = []

    if mode == "rules":
        return parse_rapatriement(text), "rules", warnings

    if mode == "llm":
        return (
            extract_rapatriement_with_llm(text, filename, next_step_context),
            "llm",
            warnings,
        )

    if is_llm_configured():
        try:
            return (
                extract_rapatriement_with_llm(text, filename, next_step_context),
                "llm",
                warnings,
            )
        except Exception as error:
            warnings.append(f"LLM extraction failed, fallback to rules: {error}")

    return parse_rapatriement(text), "rules", warnings
