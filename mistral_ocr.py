import base64
import os


DEFAULT_MISTRAL_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")


def is_mistral_ocr_configured():
    return bool(os.getenv("MISTRAL_API_KEY"))


def extract_text_with_mistral_ocr(pdf_bytes):
    if not is_mistral_ocr_configured():
        raise RuntimeError("MISTRAL_API_KEY is not configured.")

    try:
        from mistralai.client import Mistral
    except ImportError as error:
        raise RuntimeError("The mistralai package is not installed. Run: pip install mistralai") from error

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    response = client.ocr.process(
        model=DEFAULT_MISTRAL_OCR_MODEL,
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{encoded_pdf}",
        },
    )

    return ocr_response_to_text(response)


def ocr_response_to_text(response):
    pages = getattr(response, "pages", None)
    if pages is None and isinstance(response, dict):
        pages = response.get("pages", [])

    texts = []
    for page in pages or []:
        markdown = getattr(page, "markdown", None)
        if markdown is None and isinstance(page, dict):
            markdown = page.get("markdown")
        if markdown:
            texts.append(markdown)

    return "\n\n".join(texts)
