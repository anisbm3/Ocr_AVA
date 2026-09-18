#!/usr/bin/env python3
"""REST API for rapatriement financial document OCR."""

import os
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from financial_extractor import FinancialDocumentParser
from llm_clients.llm_client import LLMClient
from llm_clients.qwen_vision_agent import QwenVisionAgent

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logging.getLogger("financial_extractor").setLevel(LOG_LEVEL)

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


class RapatriementExtraction(BaseModel):
    dateDosRap: str = Field(default="")
    mntRap: float = Field(default=0.0)
    typePieceBenef: int = Field(default=0)
    noPieceBenef: str = Field(default="")


class HealthResponse(BaseModel):
    status: str
    lmStudioHost: str
    qwenModel: str
    lmStudioConnected: bool


app = FastAPI(title="OCR Rapatriement API", version="1.0.0")

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parser_cache: Dict[tuple[str, str], FinancialDocumentParser] = {}


def get_max_upload_mb() -> int:
    return int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))


def get_lm_studio_host() -> str:
    return os.getenv("LM_STUDIO_HOST", "http://127.0.0.1:1234")


def get_qwen_model() -> str:
    return os.getenv("QWEN_MODEL", "qwen2.5-vl-7b")


def get_parser() -> FinancialDocumentParser:
    host = get_lm_studio_host()
    model = get_qwen_model()
    cache_key = (host, model)

    if cache_key not in parser_cache:
        qwen_agent = QwenVisionAgent(lm_studio_host=host, model_name=model)
        llm_client = LLMClient(lm_studio_host=host, preferred_model=model)
        parser_cache[cache_key] = FinancialDocumentParser(qwen_agent, llm_client)

    return parser_cache[cache_key]


def validate_filename(filename: Optional[str]) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Missing uploaded filename")

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    return suffix


async def save_upload_to_temp(file: UploadFile, suffix: str) -> str:
    max_bytes = get_max_upload_mb() * 1024 * 1024
    total_bytes = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name

        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break

            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                temp_file.close()
                os.remove(temp_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {get_max_upload_mb()} MB upload limit",
                )

            temp_file.write(chunk)

    return temp_path


def normalize_response(data: Dict[str, Any]) -> RapatriementExtraction:
    return RapatriementExtraction(
        dateDosRap=str(data.get("dateDosRap") or ""),
        mntRap=float(data.get("mntRap") or 0.0),
        typePieceBenef=int(data.get("typePieceBenef") or 0),
        noPieceBenef=str(data.get("noPieceBenef") or ""),
    )


def is_lm_studio_connected() -> bool:
    try:
        response = requests.get(f"{get_lm_studio_host()}/v1/models", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    connected = is_lm_studio_connected()
    return HealthResponse(
        status="ok" if connected else "degraded",
        lmStudioHost=get_lm_studio_host(),
        qwenModel=get_qwen_model(),
        lmStudioConnected=connected,
    )


@app.get("/api/ocr/parse/schema", response_model=RapatriementExtraction)
def ocr_parse_schema() -> RapatriementExtraction:
    return RapatriementExtraction()


@app.post("/api/ocr/parse", response_model=RapatriementExtraction)
async def parse_ocr_document(file: UploadFile = File(...)) -> RapatriementExtraction:
    suffix = validate_filename(file.filename)
    temp_path = await save_upload_to_temp(file, suffix)

    try:
        parser = get_parser()
        result = await parser.extract_and_parse_document(temp_path)
        return normalize_response(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to extract document: {exc}") from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn

    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api_ocr:app", host=api_host, port=api_port, reload=False)
