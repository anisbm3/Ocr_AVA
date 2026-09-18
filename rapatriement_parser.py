import json
import re
import shutil
from collections import OrderedDict
from datetime import datetime
from io import BytesIO, StringIO

import pdf2image
import pytesseract
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage


SCHEMA_DEFAULTS = OrderedDict(
    [
        ("dateDosRap", ""),
        ("mntRap", 0.0),
        ("typePieceBenef", 0),
        ("noPieceBenef", ""),
    ]
)

PIECE_TYPE_CODES = {
    "C": 1,
    "S": 4,
    "D": 3,
}

PRODUCT_SERVICE_CODES = {
    "RAP": 1,
    "RAV": 2,
    "RRV": 3,
    "MOC": 4,
}


def ensure_tesseract_cmd():
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if shutil.os.path.exists(candidate):
                tesseract_path = candidate
                break

    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        return True
    return False


def extract_text_from_pdf(pdf_bytes, ocr_language="fra+eng"):
    text = extract_embedded_text(pdf_bytes)
    if has_enough_text(text):
        return text, "embedded"

    if not ensure_tesseract_cmd():
        raise RuntimeError("Tesseract OCR was not found.")

    images = pdf2image.convert_from_bytes(pdf_bytes)
    pages = [pytesseract.image_to_string(image, lang=ocr_language) for image in images]
    return "\n\n".join(pages), "ocr"


def extract_embedded_text(pdf_bytes):
    output = StringIO()
    resource_manager = PDFResourceManager()
    device = TextConverter(resource_manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(resource_manager, device)

    try:
        for page in PDFPage.get_pages(BytesIO(pdf_bytes)):
            interpreter.process_page(page)
        return output.getvalue()
    finally:
        device.close()
        output.close()


def has_enough_text(text):
    compact = re.sub(r"\s+", "", text or "")
    return len(compact) >= 50 and "RAP" in text.upper()


def parse_rapatriement(text):
    normalized = normalize_text(text)
    flat = " ".join(normalized.split())

    data = OrderedDict(SCHEMA_DEFAULTS)
    data["dateDosRap"] = find_table_date(flat)

    row = find_operation_row(flat)
    if row:
        data["mntRap"] = find_transfer_amount(row)
        piece_type, piece_number = find_beneficiary_piece(row)
        data["typePieceBenef"] = piece_type
        data["noPieceBenef"] = piece_number

    return data


def normalize_text(text):
    replacements = {
        "\u00a0": " ",
        "Â°": "°",
        "Ã©": "e",
        "Ã¨": "e",
        "Ã ": "a",
        "Ã´": "o",
        "Ã®": "i",
        "Ã§": "c",
        "Ã‰": "E",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def find_num_dossier(text):
    patterns = [
        r"Titulaire\s+de\s+l['`]allocation\s*:?\s*([A-Z0-9]{5,})",
        r"Titulaire\s+de\s+l.?allocation\s*:?\s*([A-Z0-9]{5,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return coerce_int(match.group(1))
    return ""


def find_date(text):
    match = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*((?:\d\s*){4})", text)
    if not match:
        return ""

    day, month, year = match.group(1), match.group(2), re.sub(r"\s+", "", match.group(3))
    try:
        return datetime(int(year), int(month), int(day)).date().isoformat()
    except ValueError:
        return ""


def find_table_date(text):
    match = re.search(
        r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*((?:\d\s*){4})(?=.{0,120}\b(?:RAP|RAV|RRV|MOC)\b)",
        text,
        re.IGNORECASE,
    )
    if match:
        day, month, year = match.group(1), match.group(2), re.sub(r"\s+", "", match.group(3))
        try:
            return datetime(int(year), int(month), int(day)).date().isoformat()
        except ValueError:
            return ""

    return find_date(text)


def find_operation_row(text):
    match = re.search(r"\b(RAP|RAV|RRV|MOC)\b(.{0,350})", text, re.IGNORECASE)
    if not match:
        return ""
    return f"{match.group(1).upper()} {match.group(2)}"


def find_transfer_amount(row):
    amounts = re.findall(r"\b\d{1,9}(?:[,.]\d{1,3})\b", row)
    parsed = [parse_decimal(amount) for amount in amounts]
    parsed = [amount for amount in parsed if amount is not None]
    if not parsed:
        return 0.0
    return parsed[0]


def find_currency_or_country_code(row):
    match = re.search(r"\b(RAP|RAV|RRV|MOC)\b\s+\d+\s+(\d{3})\b", row, re.IGNORECASE)
    if match:
        return int(match.group(2))

    codes = [int(code) for code in re.findall(r"\b\d{3}\b", row)]
    return codes[0] if codes else 0


def find_product_service_code(row):
    match = re.search(r"\b(RAP|RAV|RRV|MOC)\b", row, re.IGNORECASE)
    if not match:
        return 0
    return PRODUCT_SERVICE_CODES.get(match.group(1).upper(), 0)


def find_beneficiary_piece(row):
    candidates = re.findall(r"\b([CSD])\s+([A-Z0-9]{4,})\b", row, re.IGNORECASE)
    if not candidates:
        return 0, ""

    piece_type, piece_number = candidates[-1]
    return PIECE_TYPE_CODES.get(piece_type.upper(), 0), piece_number


def find_account_number(text):
    patterns = [
        r"(?:numero|num[eé]ro|n[°o])\s+(?:de\s+)?compte\s*:?\s*([A-Z0-9]{8,30})",
        r"\bcompte\s*:?\s*([A-Z0-9]{8,30})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def parse_decimal(value):
    cleaned = value.strip().replace(" ", "")
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")

    try:
        return round(float(cleaned), 3)
    except ValueError:
        return None


def coerce_int(value):
    try:
        return int(value)
    except ValueError:
        return value


def to_pretty_json(data):
    return json.dumps(data, ensure_ascii=False, indent=2)
