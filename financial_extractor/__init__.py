"""Financial document extractor for rapatriement-style PDFs."""

import json
import logging
import os
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


FINANCIAL_TEMPLATE: Dict[str, Any] = {
    "dateDosRap": "",
    "mntRap": 0.0,
    "typePieceBenef": 0,
    "noPieceBenef": "",
}


class FinancialDocumentParser:
    """Extract the required rapatriement JSON from a PDF or image document."""

    def __init__(self, qwen_agent=None, llm_client=None):
        self.qwen_agent = qwen_agent
        self.llm_client = llm_client

    async def extract_and_parse_document(self, document_path: str) -> Dict[str, Any]:
        extraction_result = self._extract_text(document_path)
        raw_text = extraction_result.get("text", "")

        if not raw_text or len(raw_text.strip()) < 20:
            return FINANCIAL_TEMPLATE.copy()

        document_format = extraction_result.get("document_format") or self._detect_document_format_from_text(raw_text)
        if document_format == "letter":
            logger.info("Document format detected: letter")
            letter_data = await self._parse_letter_text(raw_text)
            return self._normalize_output(letter_data)

        logger.info("Document format detected: form")
        merged = await self._parse_extracted_text(raw_text)
        cell_data = self._extract_template_cell_data(document_path, extraction_result)
        for key in merged:
            merged[key] = cell_data.get(key) or merged.get(key) or FINANCIAL_TEMPLATE[key]

        normalized = self._normalize_output(merged)

        if self._needs_qwen_fallback(normalized):
            qwen_image_path = self._get_qwen_fallback_image_path(document_path, extraction_result)
            if qwen_image_path:
                logger.info("Trying Qwen Vision fallback because parsed fields are incomplete")
                qwen_result = self._extract_image_with_qwen(qwen_image_path)
                qwen_text = qwen_result.get("text", "")
                if qwen_text and len(qwen_text.strip()) >= 2:
                    qwen_data = await self._parse_extracted_text(qwen_text)
                    for key in merged:
                        merged[key] = cell_data.get(key) or qwen_data.get(key) or merged.get(key) or FINANCIAL_TEMPLATE[key]

        if cell_data:
            merged["dateDosRap"] = cell_data.get("dateDosRap") or FINANCIAL_TEMPLATE["dateDosRap"]
            merged["mntRap"] = cell_data.get("mntRap") or FINANCIAL_TEMPLATE["mntRap"]

        return self._normalize_output(merged)

    def _extract_text(self, document_path: str) -> Dict[str, Any]:
        suffix = Path(document_path).suffix.lower()
        if suffix == ".pdf" or not suffix:
            return self._extract_pdf_with_pymupdf(document_path)

        if self.qwen_agent and suffix in {".png", ".jpg", ".jpeg"}:
            letter_result = self._extract_letter_image_with_qwen(document_path)
            if letter_result.get("document_format") == "letter":
                letter_result["source_image_path"] = document_path
                return letter_result

        tesseract_result = self._extract_image_with_tesseract(document_path)
        if tesseract_result.get("text") and len(tesseract_result["text"].strip()) >= 20:
            tesseract_result["source_image_path"] = document_path
            return tesseract_result

        if self.qwen_agent:
            return self._extract_image_with_qwen(document_path)

        return {"text": "", "method": "unsupported_format"}

    def _extract_pdf_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        try:
            import fitz

            doc = fitz.open(pdf_path)
            text = "\n\n".join(page.get_text() for page in doc).strip()
            pages = len(doc)
            doc.close()

            if text and len(text) >= 20:
                return {"text": text, "method": "pymupdf", "pages": pages}

            return self._pdf_to_image_fallback(pdf_path)
        except Exception as exc:
            logger.warning("PyMuPDF extraction failed: %s", exc)
            return self._pdf_to_image_fallback(pdf_path)

    def _pdf_to_image_fallback(self, pdf_path: str) -> Dict[str, Any]:
        try:
            import fitz

            doc = fitz.open(pdf_path)
            temp_dir = Path("uploads/temp")
            temp_dir.mkdir(exist_ok=True, parents=True)

            image_paths = []
            for index, page in enumerate(doc):
                pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False)
                image_path = temp_dir / f"{Path(pdf_path).stem}_financial_{index + 1}.png"
                pix.save(str(image_path))
                image_paths.append(image_path)
            doc.close()

            if self.qwen_agent and image_paths:
                letter_result = self._extract_letter_image_with_qwen(str(image_paths[0]))
                if letter_result.get("document_format") == "letter":
                    letter_result["source_image_path"] = str(image_paths[0])
                    letter_result["image_paths"] = [str(path) for path in image_paths]
                    return letter_result

            tesseract_texts = []
            for image_path in image_paths:
                result = self._extract_image_with_tesseract(str(image_path))
                if result.get("text"):
                    tesseract_texts.append(result["text"])

            tesseract_text = "\n\n".join(tesseract_texts).strip()
            if tesseract_text and len(tesseract_text) >= 20:
                return {
                    "text": tesseract_text,
                    "method": "tesseract",
                    "pages": len(image_paths),
                    "source_image_path": str(image_paths[0]) if image_paths else "",
                    "image_paths": [str(path) for path in image_paths],
                }

            if self.qwen_agent and image_paths:
                return self._extract_image_with_qwen(str(image_paths[0]))

            return {"text": "", "method": "pdf_to_image_unavailable"}
        except Exception as exc:
            return {"text": "", "method": "pdf_to_image_failed", "error": str(exc)}

    def _extract_image_with_tesseract(self, image_path: str) -> Dict[str, Any]:
        try:
            from PIL import Image, ImageEnhance, ImageFilter, ImageOps
            import pytesseract

            configured_cmd = os.getenv("TESSERACT_CMD")
            discovered_cmd = configured_cmd or shutil.which("tesseract")
            windows_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if not discovered_cmd and Path(windows_cmd).exists():
                discovered_cmd = windows_cmd

            if discovered_cmd:
                pytesseract.pytesseract.tesseract_cmd = discovered_cmd

            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            gray = ImageOps.grayscale(image)
            gray = ImageOps.autocontrast(gray)
            enhanced = ImageEnhance.Contrast(gray).enhance(1.8).filter(ImageFilter.SHARPEN)
            threshold = enhanced.point(lambda pixel: 255 if pixel > 170 else 0)

            lang_candidates = [os.getenv("TESSERACT_LANG", "fra+eng"), "eng"]
            configs = [
                "--oem 3 --psm 6",
                "--oem 3 --psm 11",
                "--oem 3 --psm 12",
            ]
            images = [image, gray, enhanced, threshold]
            text_parts = []

            for ocr_image in images:
                for lang in lang_candidates:
                    for config in configs:
                        try:
                            text = pytesseract.image_to_string(ocr_image, lang=lang, config=config).strip()
                        except Exception:
                            continue
                        if text and text not in text_parts:
                            text_parts.append(text)

            return {"text": "\n\n".join(text_parts).strip(), "method": "tesseract"}
        except Exception as exc:
            logger.warning("Tesseract OCR failed: %s", exc)
            return {"text": "", "method": "tesseract_failed", "error": str(exc)}

    def _extract_image_with_qwen(self, image_path: str) -> Dict[str, Any]:
        if not self.qwen_agent:
            return {"text": "", "method": "qwen_unavailable"}

        prompt = """You are reading a fixed Tunisian AVA rapatriement form. Some filled values may be handwritten.

Extract only these fields from the table row, not from the explanatory notes:
- dateDosRap: operation date in the DATE column near the RAP row, formatted YYYY-MM-DD.
- mntRap: amount under "DROITS A TRANSFERT CUMULES (6)" for that row. Do not use the next cell "MONTANTS DES TRANSFERTS CUMULES".
- typePieceBenef: 1 if the beneficiary identification type is C, 4 if it is S.
- noPieceBenef: beneficiary identification number, usually 7 digits followed by one letter.

Read small table cells carefully. If a value is handwritten, infer it from the handwriting. Return exactly one JSON object and no extra text:
{"dateDosRap":"","mntRap":0.0,"typePieceBenef":0,"noPieceBenef":""}"""
        return self.qwen_agent.analyze_image(
            str(image_path),
            prompt=prompt,
            max_tokens=3000,
            temperature=0.0,
        )

    def _extract_letter_image_with_qwen(self, image_path: str) -> Dict[str, Any]:
        if not self.qwen_agent:
            return {"text": "", "method": "qwen_unavailable", "document_format": "other"}

        prompt = """Analyze this document image and classify its layout before extracting anything.

Classify as "letter" only if the page has a structured business-letter layout:
- sender/header and recipient/date area,
- an "Objet" line,
- greeting such as "Madame, Monsieur",
- paragraph blocks,
- closing/signature text such as "Veuillez agreer" or "salutations distinguees".

Classify as "other" if the page is a form, has boxes, fields to fill, rows/columns, a table, table headers, or a grid.

If it is not a letter, return exactly:
{"documentFormat":"other","dateDosRap":"","mntRap":0.0,"typePieceBenef":0,"noPieceBenef":""}

If it is a letter, extract the fields and return exactly:
{"documentFormat":"letter","dateDosRap":"YYYY-MM-DD","mntRap":0.0,"typePieceBenef":0,"noPieceBenef":""}

Letter rules:
- dateDosRap is the real operation/movement date, usually near "mouvement enregistre le" or "operation". Do not use the letter date, reference date, circular date, or period start/end dates.
- mntRap is the amount of "droits a transfert cumules" or an expression with the same meaning. Do not use "montants des transferts cumules".
- If the amount is written with comma separators, the comma is the decimal separator: 250,000 means 250.0 and 1,000,000 means 1000.0.
- typePieceBenef is 1 for piece type C and 4 for piece type S.
- noPieceBenef is the beneficiary identification number, exactly 7 digits followed by one letter.
- Return only JSON."""

        result = self.qwen_agent.analyze_image(
            str(image_path),
            prompt=prompt,
            max_tokens=1200,
            temperature=0.0,
        )
        data = self._parse_json_fragment(result.get("text", ""))
        if str(data.get("documentFormat", "")).strip().lower() != "letter":
            return {
                "text": result.get("text", ""),
                "method": result.get("method", "qwen_letter_probe"),
                "document_format": "other",
            }

        payload = {
            "dateDosRap": data.get("dateDosRap", ""),
            "mntRap": self._control_letter_amount(data.get("mntRap", 0.0)),
            "typePieceBenef": data.get("typePieceBenef", 0),
            "noPieceBenef": data.get("noPieceBenef", ""),
        }
        return {
            "text": json.dumps(payload, ensure_ascii=False),
            "method": result.get("method", "qwen_letter_direct"),
            "document_format": "letter",
        }

    async def _parse_extracted_text(self, raw_text: str) -> Dict[str, Any]:
        direct_json = self._parse_json_fragment(raw_text)
        rule_data = self._parse_with_rules(raw_text)
        llm_data = await self._parse_with_llm(raw_text)

        merged = FINANCIAL_TEMPLATE.copy()
        for key in merged:
            merged[key] = direct_json.get(key) or rule_data.get(key) or llm_data.get(key) or merged[key]
        return merged

    async def _parse_letter_text(self, raw_text: str) -> Dict[str, Any]:
        direct_json = self._parse_json_fragment(raw_text)
        llm_data = await self._parse_letter_with_llm(raw_text)
        rule_data = self._parse_letter_with_rules(raw_text)

        merged = FINANCIAL_TEMPLATE.copy()
        for key in merged:
            merged[key] = direct_json.get(key) or llm_data.get(key) or rule_data.get(key) or merged[key]
        if rule_data.get("mntRap", 0.0) > 0:
            merged["mntRap"] = rule_data["mntRap"]
        else:
            merged["mntRap"] = self._control_letter_amount(merged.get("mntRap"))
        merged["noPieceBenef"] = self._normalize_no_piece(merged.get("noPieceBenef")) or merged.get("noPieceBenef")
        return merged

    def _looks_like_ava_letter(self, raw_text: str) -> bool:
        return self._detect_document_format_from_text(raw_text) == "letter"

    def _detect_document_format_from_text(self, raw_text: str) -> str:
        text = self._fold_text(raw_text)

        form_patterns = [
            r"\bdate\s+devise\s+montant\b",
            r"\bdesignation\s+origine\s+pays\b",
            r"\bnature\s+de\s+l\s+operation\b",
            r"\bidentification\s+du\s+beneficiaire\b",
            r"\btype\s+piece\s+numero\b",
            r"\bdroits\s+a\s+transfert\s+cumules\s+6\b",
            r"\bmettre\s+une\s+croix\b",
            r"\bcase\s+reservee\b",
        ]
        form_score = sum(1 for pattern in form_patterns if re.search(pattern, text))
        if form_score >= 1:
            return "form"

        letter_checks = [
            bool(re.search(r"\bobjet\s+decompte\s+annuel\b", text))
            or bool(re.search(r"\bobjet\b.{0,80}\ballocation\s+pour\s+voyages\s+d\s+affaires\b", text)),
            bool(re.search(r"\bmadame\s+monsieur\b|\bmadame\b|\bmonsieur\b", text)),
            bool(re.search(r"\bnous\s+avons\s+l\s+honneur\s+de\s+vous\s+transmettre\b", text)),
            bool(re.search(r"\bmouvement\s+enregistre\s+le\b|\boperation\s+porte\s+la\s+designation\s+rap\b", text)),
            bool(re.search(r"\bnous\s+restons\s+a\s+votre\s+disposition\b|\bveuillez\s+agreer\b|\bsalutations\s+distinguees\b", text)),
        ]
        letter_score = sum(1 for matched in letter_checks if matched)

        paragraph_like_lines = [
            line.strip()
            for line in str(raw_text or "").splitlines()
            if len(line.strip()) >= 45 and not re.search(r"\s{4,}|\t", line)
        ]
        has_paragraph_structure = len(paragraph_like_lines) >= 3

        if letter_score >= 3 and has_paragraph_structure:
            return "letter"

        return "form"

    def _fold_text(self, value: Any) -> str:
        text = self._strip_accents(value)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _strip_accents(self, value: Any) -> str:
        text = str(value or "").lower()
        text = unicodedata.normalize("NFKD", text)
        return "".join(char for char in text if not unicodedata.combining(char))

    def _parse_json_fragment(self, raw_text: str) -> Dict[str, Any]:
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start == -1 or end == 0:
            return {}

        try:
            data = json.loads(raw_text[start:end])
        except json.JSONDecodeError:
            return {}

        return data if isinstance(data, dict) else {}

    def _needs_qwen_fallback(self, data: Dict[str, Any]) -> bool:
        normalized = self._normalize_output(data)
        critical_values = [
            bool(normalized.get("dateDosRap")),
            normalized.get("mntRap", 0.0) > 0,
            normalized.get("typePieceBenef") in {1, 4},
            bool(normalized.get("noPieceBenef")),
        ]
        return self.qwen_agent is not None and not all(critical_values)

    def _get_qwen_fallback_image_path(self, document_path: str, extraction_result: Dict[str, Any]) -> str:
        source_image_path = extraction_result.get("source_image_path")
        if source_image_path:
            return str(source_image_path)

        suffix = Path(document_path).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg"}:
            return document_path

        image_paths = extraction_result.get("image_paths") or []
        if image_paths:
            return str(image_paths[0])

        return ""

    def _extract_template_cell_data(self, document_path: str, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        image_path = self._get_qwen_fallback_image_path(document_path, extraction_result)
        if not image_path:
            return {}

        suffix = Path(image_path).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"} or not Path(image_path).exists():
            return {}

        try:
            grid = self._detect_table_grid(image_path)
            if not grid:
                return {}

            cells = self._ocr_template_cells(image_path, grid)
            parsed = {
                "dateDosRap": self._extract_operation_date(cells.get("dateDosRap", "")),
                "mntRap": self._extract_amount_from_cell_text(cells.get("mntRap", "")),
                "typePieceBenef": self._extract_piece_type_from_cell_text(cells.get("typePieceBenef", "")),
                "noPieceBenef": self._normalize_no_piece(cells.get("noPieceBenef", "")),
            }

            if not parsed["dateDosRap"] and self.qwen_agent:
                qwen_date = self._extract_template_cell_with_qwen(image_path, grid, "dateDosRap")
                parsed["dateDosRap"] = self._extract_operation_date(qwen_date)

            if not parsed["noPieceBenef"] and self.qwen_agent:
                qwen_no_piece = self._extract_template_cell_with_qwen(image_path, grid, "noPieceBenef")
                parsed["noPieceBenef"] = self._normalize_no_piece(qwen_no_piece)

            return self._normalize_output(parsed)
        except Exception as exc:
            logger.warning("Template cell extraction failed: %s", exc)
            return {}

    def _detect_table_grid(self, image_path: str) -> Optional[Dict[str, Any]]:
        try:
            import cv2
        except ImportError:
            return None

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        height, width = image.shape
        _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, width // 18), 1))
        horizontal = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        horizontal_lines = []
        for contour in horizontal_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= width * 0.35:
                horizontal_lines.append((x, y, w, h))

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 40)))
        vertical = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, vertical_kernel)
        vertical_contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vertical_lines = []
        for contour in vertical_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h >= height * 0.04:
                vertical_lines.append((x, y, w, h))

        y_lines = self._group_line_positions([y + h // 2 for _, y, _, h in horizontal_lines], tolerance=4)
        x_lines = self._group_line_positions([x + w // 2 for x, _, w, _ in vertical_lines], tolerance=4)
        if len(y_lines) < 3 or len(x_lines) < 13:
            return None

        x_lines = x_lines[:13]
        return {
            "x": x_lines,
            "y": y_lines,
            "width": width,
            "height": height,
        }

    def _group_line_positions(self, positions: list[int], tolerance: int = 4) -> list[int]:
        if not positions:
            return []

        grouped = []
        for position in sorted(positions):
            if not grouped or position > grouped[-1][-1] + tolerance:
                grouped.append([position])
            else:
                grouped[-1].append(position)

        return [round(sum(group) / len(group)) for group in grouped]

    def _ocr_template_cells(self, image_path: str, grid: Dict[str, Any]) -> Dict[str, str]:
        cells = {}
        cells["dateDosRap"] = self._ocr_table_cell(image_path, grid, 0, mode="date")
        cells["mntRap"] = self._ocr_table_cell(image_path, grid, 6, mode="amount")
        cells["typePieceBenef"] = self._ocr_table_cell(image_path, grid, 9, mode="piece_type")
        cells["noPieceBenef"] = self._ocr_table_cell(image_path, grid, 10, mode="piece_number")
        return cells

    def _ocr_table_cell(self, image_path: str, grid: Dict[str, Any], column_index: int, mode: str) -> str:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
        import pytesseract

        x_lines = grid["x"]
        y_lines = grid["y"]
        if column_index + 1 >= len(x_lines) or len(y_lines) < 2:
            return ""

        x1, x2 = x_lines[column_index], x_lines[column_index + 1]
        y1, y2 = y_lines[-2], y_lines[-1]
        pad_x = max(2, int((x2 - x1) * 0.04))
        pad_y = max(2, int((y2 - y1) * 0.08))

        image = Image.open(image_path).convert("RGB")
        crop = image.crop((max(0, x1 + pad_x), max(0, y1 + pad_y), min(image.width, x2 - pad_x), min(image.height, y2 - pad_y)))
        crop = crop.resize((crop.width * 8, crop.height * 8))
        gray = ImageOps.grayscale(crop)
        gray = ImageOps.autocontrast(gray)
        gray = ImageEnhance.Contrast(gray).enhance(2.5).filter(ImageFilter.SHARPEN)

        whitelist = {
            "date": "0123456789/-. ",
            "amount": "0123456789,.",
            "piece_type": "CS",
            "piece_number": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        }.get(mode, "")
        whitelist_config = f" -c tessedit_char_whitelist={whitelist}" if whitelist else ""

        results = []
        for psm in (7, 8, 13, 6):
            try:
                text = pytesseract.image_to_string(
                    gray,
                    lang="eng",
                    config=f"--oem 3 --psm {psm}{whitelist_config}",
                ).strip()
            except Exception:
                continue
            if text:
                results.append(text)

        return "\n".join(results)

    def _extract_template_cell_with_qwen(self, image_path: str, grid: Dict[str, Any], field_name: str) -> str:
        if not self.qwen_agent:
            return ""

        crop_path = self._save_template_cell_crop(image_path, grid, field_name)
        if not crop_path:
            return ""

        prompts = {
            "dateDosRap": "Read only this single DATE cell from an AVA table row. Return only the date as DD/MM/YYYY if visible. Do not use any other date.",
            "mntRap": "Read only this single amount cell under DROITS A TRANSFERT CUMULES (6). Return only the amount.",
            "noPieceBenef": "Read only this single NUMERO cell from the beneficiary identification area. The value must have exactly 7 digits followed by 1 letter, for example 0004545Y. If the last character looks handwritten, decide whether it is a letter, not a digit. Return only the 8-character value.",
        }
        result = self.qwen_agent.analyze_image(
            crop_path,
            prompt=prompts.get(field_name, "Read only the value in this single table cell. Return only the value."),
            max_tokens=100,
            temperature=0.0,
        )
        text = result.get("text", "")

        if field_name == "noPieceBenef" and not self._normalize_no_piece(text):
            retry_result = self.qwen_agent.analyze_image(
                crop_path,
                prompt=(
                    "Inspect this cropped NUMERO cell again. The required output format is exactly "
                    "7 digits followed by one uppercase letter: DDDDDDDL. The final character is a "
                    "letter, not part of the seven digits. If it resembles 9, g, y, or a handwritten "
                    "mark, choose the most likely uppercase letter. Return only the 8 characters."
                ),
                max_tokens=100,
                temperature=0.0,
            )
            retry_text = retry_result.get("text", "")
            if retry_text:
                text = f"{text}\n{retry_text}"

        return text

    def _save_template_cell_crop(self, image_path: str, grid: Dict[str, Any], field_name: str) -> str:
        from PIL import Image

        column_by_field = {
            "dateDosRap": 0,
            "mntRap": 6,
            "typePieceBenef": 9,
            "noPieceBenef": 10,
        }
        column_index = column_by_field.get(field_name)
        if column_index is None:
            return ""

        x_lines = grid["x"]
        y_lines = grid["y"]
        if column_index + 1 >= len(x_lines) or len(y_lines) < 2:
            return ""

        image = Image.open(image_path).convert("RGB")
        x1, x2 = x_lines[column_index], x_lines[column_index + 1]
        y1, y2 = y_lines[-2], y_lines[-1]
        crop = image.crop((max(0, x1 - 2), max(0, y1 - 2), min(image.width, x2 + 2), min(image.height, y2 + 2)))
        crop = crop.resize((crop.width * 4, crop.height * 4))

        temp_dir = Path("uploads/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        crop_path = temp_dir / f"{Path(image_path).stem}_{field_name}_cell.png"
        crop.save(crop_path)
        return str(crop_path)

    async def _parse_with_llm(self, raw_text: str) -> Dict[str, Any]:
        if not self.llm_client or not self.llm_client.provider:
            return {}

        prompt = f"""Extract data from this financial document.

Return exactly one valid JSON object with this structure and no extra text:
{{
  "dateDosRap": "YYYY-MM-DD",
  "mntRap": 0.0,
  "typePieceBenef": 0,
  "noPieceBenef": ""
}}

Rules:
- dateDosRap is the operation date in the document.
- mntRap is the amount in the table cell "DROITS A TRANSFERT CUMULES (6)" for the RAP row. Do not use the next cell "MONTANTS DES TRANSFERTS CUMULES".
- typePieceBenef is 1 when the beneficiary identification type is C.
- typePieceBenef is 4 when the beneficiary identification type is S.
- noPieceBenef is the beneficiary code made of exactly 7 digits with one letter.
- If a value is missing, keep the empty default value from the JSON shape.

DOCUMENT TEXT:
{raw_text[:6000]}
"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=800, temperature=0.0)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start == -1 or end == 0:
                return {}
            return json.loads(response[start:end])
        except Exception as exc:
            logger.warning("Financial LLM parsing failed: %s", exc)
            return {}

    async def _parse_letter_with_llm(self, raw_text: str) -> Dict[str, Any]:
        if not self.llm_client or not self.llm_client.provider:
            return {}

        prompt = f"""Extract rapatriement data from this AVA annual statement letter.

Return exactly one valid JSON object with this structure and no extra text:
{{
  "dateDosRap": "YYYY-MM-DD",
  "mntRap": 0.0,
  "typePieceBenef": 0,
  "noPieceBenef": ""
}}

Rules:
- dateDosRap is the real operation/movement date. Prefer the date near "mouvement enregistre le", "operation", or "operation porte la designation RAP".
- Do not use the letter date, circular/reference date, allocation year dates, period start date, or period end date.
- mntRap is the amount of this operation: "droits a transfert cumules" or wording with the same meaning.
- If the amount is written with comma separators, the comma is the decimal separator: 250,000 means 250.0 and 1,000,000 means 1000.0.
- Do not use "montants des transferts cumules" if it is present.
- typePieceBenef is 1 when the beneficiary is identified by piece type C.
- typePieceBenef is 4 when the beneficiary is identified by piece type S.
- noPieceBenef is exactly 7 digits followed by one letter.
- If a value is missing, keep the empty default value.

LETTER TEXT:
{raw_text[:6000]}
"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=800, temperature=0.0)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start == -1 or end == 0:
                return {}
            data = json.loads(response[start:end])
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("AVA letter LLM parsing failed: %s", exc)
            return {}

    def _parse_letter_with_rules(self, raw_text: str) -> Dict[str, Any]:
        no_piece = self._extract_no_piece(raw_text) or self._extract_letter_no_piece(raw_text)
        return {
            "dateDosRap": self._extract_letter_operation_date(raw_text),
            "mntRap": self._extract_letter_mnt_rap(raw_text),
            "typePieceBenef": self._extract_letter_type_piece(raw_text, no_piece),
            "noPieceBenef": no_piece,
        }

    def _parse_with_rules(self, raw_text: str) -> Dict[str, Any]:
        date_text = self._compact_numeric_whitespace(raw_text)
        no_piece = self._extract_no_piece(raw_text)

        return {
            "dateDosRap": self._extract_operation_date(date_text),
            "mntRap": self._extract_mnt_rap(raw_text),
            "typePieceBenef": self._extract_type_piece(raw_text, no_piece),
            "noPieceBenef": no_piece,
        }

    def _compact_numeric_whitespace(self, text: str) -> str:
        return re.sub(r"(?<=\d)\s+(?=\d)", "", text)

    def _extract_operation_date(self, text: str) -> str:
        date_matches = list(re.finditer(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", text))
        if not date_matches:
            return ""

        rap_index = text.upper().find("RAP")
        if rap_index != -1:
            date_matches.sort(key=lambda match: abs(match.start() - rap_index))

        for match in date_matches:
            normalized = self._normalize_date(match.group(0))
            if normalized:
                return normalized
        return ""

    def _extract_letter_operation_date(self, text: str) -> str:
        normalized_text = self._strip_accents(text)
        date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        priority_patterns = [
            rf"mouvement\s+enregistre\s+le\s+({date_pattern})",
            rf"operation.{0,140}?\b(?:le|du)\s+({date_pattern})",
            rf"designation\s+rap.{0,140}?\b(?:le|du)\s+({date_pattern})",
        ]

        for pattern in priority_patterns:
            match = re.search(pattern, normalized_text, re.IGNORECASE | re.DOTALL)
            if match:
                normalized = self._normalize_date(match.group(1))
                if normalized:
                    return normalized

        rap_index = normalized_text.find("rap")
        date_matches = list(re.finditer(date_pattern, normalized_text))
        if rap_index != -1 and date_matches:
            date_matches.sort(key=lambda match: abs(match.start() - rap_index))
            for match in date_matches:
                normalized = self._normalize_date(match.group(0))
                if normalized:
                    return normalized

        return ""


    def _normalize_date(self, value: Any) -> str:
        if not value:
            return ""

        text = str(value).strip()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
            try:
                return datetime.strptime(text, fmt).date().isoformat()
            except ValueError:
                pass
        return ""

    def _extract_mnt_rap(self, text: str) -> float:
        table_amount = self._extract_mnt_rap_from_table_row(text)
        if table_amount > 0:
            return table_amount

        return 0.0

    def _extract_letter_mnt_rap(self, text: str) -> float:
        normalized_text = self._strip_accents(text)
        amount_pattern = r"([0-9][0-9\s.,]*[,.]\d{1,3}|[0-9][0-9\s.]*)"
        patterns = [
            rf"droits\s+a\s+transfert\s+cumules?.{{0,120}}?s[' ]?elevent\s+a\s+{amount_pattern}",
            rf"droits\s+a\s+transfert\s+cumules?.{{0,120}}?{amount_pattern}",
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized_text, re.IGNORECASE | re.DOTALL)
            if match:
                amount = self._parse_amount(match.group(1))
                if amount > 0:
                    return amount

        return 0.0


    def _extract_mnt_rap_from_table_row(self, text: str) -> float:
        note_start = self._find_notes_start(text)
        table_text = text[:note_start] if note_start != -1 else text

        candidate_windows = []
        for match in re.finditer(r"\bRAP\b", table_text, re.IGNORECASE):
            candidate_windows.append(table_text[max(0, match.start() - 500):match.end() + 900])

        for match in re.finditer(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", table_text):
            candidate_windows.append(table_text[max(0, match.start() - 250):match.end() + 900])

        if not candidate_windows:
            cumules_index = table_text.upper().find("CUMULES")
            if cumules_index != -1:
                candidate_windows.append(table_text[cumules_index:cumules_index + 1200])

        for window in candidate_windows:
            amount = self._extract_first_table_amount_after_row_markers(window)
            if amount > 0:
                return amount

        return 0.0

    def _find_notes_start(self, text: str) -> int:
        markers = ["(1)-", "(1) -", "1)-METTRE", "(1)-METTRE"]
        indexes = [text.upper().find(marker) for marker in markers]
        valid_indexes = [index for index in indexes if index != -1]
        return min(valid_indexes) if valid_indexes else -1

    def _extract_first_table_amount_after_row_markers(self, text: str) -> float:
        amount_matches = list(self._iter_amount_matches(text))
        if not amount_matches:
            return 0.0

        row_marker = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
        if not row_marker:
            row_marker = re.search(r"\bRAP\b", text, re.IGNORECASE)

        row_amounts = amount_matches
        if row_marker:
            row_amounts = [match for match in amount_matches if match.start() > row_marker.end()]

        if not row_amounts:
            return 0.0

        return self._parse_amount(row_amounts[0].group(0))

    def _iter_amount_matches(self, text: str):
        amount_pattern = r"(?<![\w])\d{1,9}(?:[,.]\d{1,3})(?![\w])"
        return re.finditer(amount_pattern, text)

    def _extract_amount_from_cell_text(self, text: str) -> float:
        for match in self._iter_amount_matches(text):
            amount = self._parse_amount(match.group(0))
            if amount > 0:
                return amount
        return self._parse_amount(text)

    def _parse_amount(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return round(float(value), 3)

        text = str(value).strip().replace(" ", "").replace("\u00a0", "")
        if "," in text and "." in text:
            decimal_separator = "," if text.rfind(",") > text.rfind(".") else "."
            thousands_separator = "." if decimal_separator == "," else ","
            text = text.replace(thousands_separator, "").replace(decimal_separator, ".")
        elif text.count(",") > 1:
            last_comma = text.rfind(",")
            text = text[:last_comma].replace(",", "") + "." + text[last_comma + 1:]
        elif text.count(".") > 1:
            last_dot = text.rfind(".")
            text = text[:last_dot].replace(".", "") + "." + text[last_dot + 1:]
        elif "," in text:
            text = text.replace(",", ".")

        try:
            return round(float(text), 3)
        except ValueError:
            return 0.0

    def _control_letter_amount(self, value: Any) -> float:
        amount = self._parse_amount(value)
        if amount >= 10000 and float(amount).is_integer():
            corrected = round(amount / 1000, 3)
            logger.info("Letter amount decimal control: %s -> %s", amount, corrected)
            return corrected
        return amount

    def _extract_no_piece(self, text: str) -> str:
        match = re.search(r"\b\d{7}[A-Za-z]\b", text)
        return match.group(0).upper() if match else ""

    def _extract_letter_no_piece(self, text: str) -> str:
        match = re.search(r"\b(\d{7})\s*([A-Za-z])\b", text)
        if match:
            return f"{match.group(1)}{match.group(2)}".upper()
        return ""

    def _normalize_no_piece(self, value: Any) -> str:
        text = str(value or "").upper()
        text = text.replace("O", "0").replace("I", "1").replace("L", "1")
        text = re.sub(r"[^0-9A-Z]", "", text)
        match = re.search(r"\d{7}[A-Z]", text)
        if match:
            return match.group(0)
        return ""

    def _extract_piece_type_from_cell_text(self, text: str) -> int:
        match = re.search(r"\b[CS]\b", text, re.IGNORECASE)
        if match:
            return self._map_piece_type(match.group(0))
        return self._map_piece_type(text)

    def _extract_type_piece(self, text: str, no_piece: str) -> int:
        if no_piece:
            match = re.search(rf"\b([CS])\b\s*{re.escape(no_piece)}\b", text, re.IGNORECASE)
            if match:
                return self._map_piece_type(match.group(1))

            index = text.find(no_piece)
            if index != -1:
                before_no_piece = text[max(0, index - 80):index]
                nearby_codes = re.findall(r"\b[CS]\b", before_no_piece, re.IGNORECASE)
                if nearby_codes:
                    return self._map_piece_type(nearby_codes[-1])

        match = re.search(r"Code d[' ]?Identification\s*:?\s*([CS])\b", text, re.IGNORECASE)
        if match:
            return self._map_piece_type(match.group(1))

        return 0

    def _extract_letter_type_piece(self, text: str, no_piece: str) -> int:
        normalized_text = self._strip_accents(text)
        match = re.search(r"piece\s+de\s+type\s+([CS])\b", normalized_text, re.IGNORECASE)
        if match:
            return self._map_piece_type(match.group(1))

        return self._extract_type_piece(text, no_piece)

    def _map_piece_type(self, value: Any) -> int:
        code = str(value).strip().upper()
        if code == "C":
            return 1
        if code == "S":
            return 4
        return 0

    def _normalize_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            type_piece = int(data.get("typePieceBenef") or 0)
        except (TypeError, ValueError):
            type_piece = 0

        if type_piece not in {1, 4}:
            type_piece = 0

        return {
            "dateDosRap": self._normalize_date(data.get("dateDosRap")) or "",
            "mntRap": self._parse_amount(data.get("mntRap")),
            "typePieceBenef": self._map_piece_type(data.get("typePieceBenef"))
            if str(data.get("typePieceBenef")).upper() in {"C", "S"}
            else type_piece,
            "noPieceBenef": str(data.get("noPieceBenef") or "").strip().upper(),
        }
