# Rapatriement OCR API Integration

This service extracts AVA rapatriement data from a PDF and returns the JSON expected by the consuming application.

It supports two extraction methods:

- `rules`: deterministic extraction using OCR/text extraction and Python parsing rules.
- `llm`: intelligent extraction using OCR/text extraction, then an LLM that understands document context and returns strict JSON.

In production, use `mode=auto`: the API uses the LLM when `OPENAI_API_KEY` is configured and falls back to rules if needed.

It also supports multiple OCR/text providers:

- `local`: local PDF text extraction with `pdfminer`, then local OCR with Tesseract if needed.
- `mistral`: Mistral OCR for intelligent document reading.
- `auto`: use Mistral OCR when `MISTRAL_API_KEY` is configured, otherwise fallback to local extraction.

## Output JSON

The business payload contains exactly these fields:

```json
{
  "dateDosRap": "2026-06-10",
  "mntRap": 250.0,
  "typePieceBenef": 1,
  "noPieceBenef": "0000500R"
}
```

Extraction rules:

- `dateDosRap`: date from the first cell of the operation table row.
- `mntRap`: value from `DROITS A TRANSFERT CUMULES`.
- `typePieceBenef`: `C` returns `1`, `S` returns `4`.
- `noPieceBenef`: beneficiary identification number after `C` or `S`.

## Run Locally

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the API:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

To enable the LLM extraction mode, set your OpenAI API key before starting the API:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

To enable Mistral OCR, set your Mistral API key:

```powershell
$env:MISTRAL_API_KEY="your_mistral_api_key_here"
```

Optional model override:

```powershell
$env:OCR_LLM_MODEL="gpt-4.1-mini"
$env:MISTRAL_OCR_MODEL="mistral-ocr-latest"
```

Open the interactive API docs:

```text
http://localhost:8000/docs
```

## Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "llmConfigured": true,
  "mistralOcrConfigured": true
}
```

### Extract With Metadata

Use this endpoint for backend, workflow, and API gateway integration.

```http
POST /api/ocr/rapatriement
Content-Type: multipart/form-data
```

Form field:

```text
file: PDF file
mode: auto | llm | rules
ocrProvider: auto | mistral | local
nextStepContext: optional text describing the next workflow step
```

Response:

```json
{
  "success": true,
  "data": {
    "dateDosRap": "2026-06-10",
    "mntRap": 250.0,
    "typePieceBenef": 1,
    "noPieceBenef": "0000500R"
  },
  "extractionMode": "embedded",
  "extractionMethod": "llm",
  "errors": []
}
```

`extractionMode` can be:

- `embedded`: the PDF had readable text.
- `ocr`: the service used OCR because the PDF was scanned or had no readable text layer.
- `mistral_ocr`: the service used Mistral OCR.

`extractionMethod` can be:

- `llm`: the LLM understood the extracted text and produced the JSON.
- `rules`: the Python parser produced the JSON.

Recommended value for `mode`:

```text
auto
```

Recommended value for `ocrProvider` when using Mistral:

```text
auto
```

Use `ocrProvider=mistral` when you want to force Mistral OCR and fail if Mistral is unavailable.
Use `ocrProvider=local` when you want only local extraction.

Use `llm` when you want to force AI extraction and fail if the LLM is unavailable.
Use `rules` when you want deterministic extraction only.

`nextStepContext` lets the backend pass context from the workflow. Example:

```text
This extraction will be used to create a payment verification request.
Prioritize the rapatriement operation row and ignore footer examples.
```

### Extract Raw JSON

Use this endpoint only if the consuming system needs the business JSON directly.

```http
POST /api/ocr/rapatriement/raw
Content-Type: multipart/form-data
```

Response:

```json
{
  "dateDosRap": "2026-06-10",
  "mntRap": 250.0,
  "typePieceBenef": 1,
  "noPieceBenef": "0000500R"
}
```

## Backend Example

Python backend example:

```python
import requests


def extract_rapatriement(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        response = requests.post(
            "http://localhost:8000/api/ocr/rapatriement",
            files={"file": ("rapatriement.pdf", pdf_file, "application/pdf")},
            data={
                "mode": "auto",
                "ocrProvider": "auto",
                "nextStepContext": "Extract the values needed for the next workflow validation step.",
            },
            timeout=60,
        )

    response.raise_for_status()
    result = response.json()

    if not result["success"]:
        raise RuntimeError(result["errors"])

    return result["data"]
```

Node.js backend example:

```javascript
import fs from "node:fs";
import FormData from "form-data";
import fetch from "node-fetch";

export async function extractRapatriement(pdfPath) {
  const form = new FormData();
  form.append("file", fs.createReadStream(pdfPath));
  form.append("mode", "auto");
  form.append("ocrProvider", "auto");
  form.append("nextStepContext", "Extract values for the workflow validation step.");

  const response = await fetch("http://localhost:8000/api/ocr/rapatriement", {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  const result = await response.json();
  return result.data;
}
```

## Frontend Example

If the frontend sends the PDF directly through the API gateway:

```javascript
async function extractPdf(file) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("mode", "auto");
  formData.append("ocrProvider", "auto");

  const response = await fetch("/api/ocr/rapatriement", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  const result = await response.json();
  return result.data;
}
```

Recommended production flow:

```text
Frontend -> API Gateway -> Backend -> OCR API -> Backend -> Workflow
```

This keeps OCR processing behind the backend and avoids exposing the OCR service directly to users.

## API Gateway Notes

Route example:

```text
/api/ocr/rapatriement -> http://ocr-service:8000/api/ocr/rapatriement
```

Gateway requirements:

- Allow `multipart/form-data`.
- Set upload size limit high enough for PDFs.
- Set timeout to at least 60 seconds for scanned PDFs and LLM extraction because they can be slower.
- Forward authentication headers only if the OCR service needs them.
- Do not expose `OPENAI_API_KEY` to the frontend. Keep it only in the OCR service environment.
- Do not expose `MISTRAL_API_KEY` to the frontend. Keep it only in the OCR service environment.

## Workflow Integration

Typical workflow steps:

```text
1. Receive PDF
2. Call OCR API
3. Pass `nextStepContext` if the next workflow step affects which values should be extracted
4. Validate returned JSON
5. Save extracted data
6. Send data to the next workflow step
```

Validation suggestion:

- Reject the document if `dateDosRap` is empty.
- Reject or mark for manual review if `mntRap` is `0.0`.
- Reject or mark for manual review if `typePieceBenef` is `0`.
- Reject or mark for manual review if `noPieceBenef` is empty.

## Deployment Notes

The service needs:

- Python dependencies from `requirements.txt`.
- Poppler for `pdf2image`.
- Tesseract OCR for scanned PDFs.
- French and English OCR languages if scanned PDFs are expected.
- `OPENAI_API_KEY` when using `mode=llm` or AI extraction in `mode=auto`.
- `MISTRAL_API_KEY` when using `ocrProvider=mistral` or Mistral OCR in `ocrProvider=auto`.

For Windows local development, Tesseract is usually detected from:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

For Linux deployment, install system packages similar to:

```bash
apt-get update
apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng
```

## Error Responses

Invalid file:

```json
{
  "detail": "Only PDF files are accepted."
}
```

Empty file:

```json
{
  "detail": "The uploaded PDF is empty."
}
```

OCR dependency missing:

```json
{
  "detail": "Tesseract OCR was not found."
}
```
