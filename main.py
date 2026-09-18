#!/usr/bin/env python3
"""Financial document extraction interface with a local Qwen parser."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logging.getLogger("financial_extractor").setLevel(LOG_LEVEL)

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from financial_extractor import FinancialDocumentParser
from llm_clients.llm_client import LLMClient
from llm_clients.qwen_vision_agent import QwenVisionAgent


qwen_agent: Optional[QwenVisionAgent] = None
llm_client: Optional[LLMClient] = None
financial_parser: Optional[FinancialDocumentParser] = None
qwen_parser_config: Optional[tuple[str, str]] = None


def get_max_upload_mb(max_upload_mb: Optional[int] = None) -> int:
    return int(max_upload_mb or os.getenv("MAX_UPLOAD_SIZE_MB", "10"))


def validate_upload(document_file: Any, max_upload_mb: Optional[int] = None) -> Optional[str]:
    if not document_file:
        return "Error: Please upload a financial document."

    file_name = getattr(document_file, "name", "")
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg"}
    if file_name and Path(file_name).suffix.lower() not in allowed_extensions:
        return "Error: Please upload a PDF, PNG, JPG, or JPEG file."

    limit_mb = get_max_upload_mb(max_upload_mb)
    if hasattr(document_file, "size") and document_file.size:
        file_size_mb = document_file.size / (1024 * 1024)
        if file_size_mb > limit_mb:
            return f"Error: File size ({file_size_mb:.1f} MB) exceeds the {limit_mb} MB limit."

    return None


def get_financial_parser(llm_host: Optional[str] = None, qwen_model: Optional[str] = None) -> FinancialDocumentParser:
    global qwen_agent, llm_client, financial_parser, qwen_parser_config
    host = llm_host or os.getenv("LM_STUDIO_HOST", "http://127.0.0.1:1234")
    model = qwen_model or os.getenv("QWEN_MODEL", "qwen2.5-vl-7b")
    config = (host, model)

    if financial_parser is None or qwen_parser_config != config:
        try:
            qwen_agent = QwenVisionAgent(lm_studio_host=host, model_name=model)
            llm_client = LLMClient(lm_studio_host=host, preferred_model=model)
            financial_parser = FinancialDocumentParser(qwen_agent, llm_client)
            qwen_parser_config = config
            print(f"Financial parser initialized with {model}")
        except Exception as exc:
            print(f"Could not initialize financial parser with LM Studio: {exc}")
            financial_parser = FinancialDocumentParser(None, None)
            qwen_parser_config = config

    return financial_parser


def format_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def summarize_financial_document(data: Dict[str, Any], method: str) -> str:
    lines = [
        f"## {method} extraction complete",
        "",
        f"**dateDosRap:** {data.get('dateDosRap') or 'Not found'}",
        f"**mntRap:** {data.get('mntRap')}",
        f"**typePieceBenef:** {data.get('typePieceBenef') or 'Not found'}",
        f"**noPieceBenef:** {data.get('noPieceBenef') or 'Not found'}",
    ]
    return "\n".join(lines)


def parse_financial_document(document_file, llm_host: str, qwen_model: str, max_upload_mb: int):
    error = validate_upload(document_file, max_upload_mb)
    if error:
        return error, None

    try:
        parser = get_financial_parser(llm_host, qwen_model)
        data = asyncio.run(parser.extract_and_parse_document(document_file.name))
        return summarize_financial_document(data, "Financial document"), format_json(data)
    except Exception as exc:
        return f"Error extracting financial document: {exc}", None


def create_interface() -> gr.Blocks:
    default_host = os.getenv("LM_STUDIO_HOST", "http://127.0.0.1:1234")
    default_model = os.getenv("QWEN_MODEL", "qwen2.5-vl-7b")
    default_max_upload = get_max_upload_mb()

    with gr.Blocks(title="Financial Document Extractor") as app:
        gr.Markdown("# Financial Document Extractor")

        with gr.Row():
            with gr.Column(scale=1):
                document_file = gr.File(
                    label="Financial document",
                )
                qwen_host = gr.Textbox(label="LM Studio host", value=default_host)
                qwen_model = gr.Textbox(label="Qwen model", value=default_model)
                max_upload = gr.Number(
                    label="Max upload size (MB)",
                    value=default_max_upload,
                    precision=0,
                )
                extract_button = gr.Button("Extract JSON", variant="primary")
            with gr.Column(scale=2):
                status = gr.Markdown("Waiting for a financial document.")
                extracted_json = gr.Code(label="Extracted JSON", language="json")

        extract_button.click(
            fn=parse_financial_document,
            inputs=[document_file, qwen_host, qwen_model, max_upload],
            outputs=[status, extracted_json],
        )

    return app


if __name__ == "__main__":
    host = os.getenv("APP_HOST", "localhost")
    port = int(os.getenv("APP_PORT", "7861"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    create_interface().launch(server_name=host, server_port=port, debug=debug, theme=gr.themes.Soft())
