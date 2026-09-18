# Document d'integration - OCR Qwen Rapatriement avec React

## Objectif

Integrer l'extracteur OCR/LLM local base sur Qwen dans une autre application React.

Le flux cible est le suivant:

1. L'utilisateur charge un PDF financier de rapatriement dans React.
2. React envoie le PDF vers une API backend.
3. Le backend appelle l'extracteur local `FinancialDocumentParser`.
4. L'API retourne un JSON normalise.
5. React utilise ce JSON pour pre-remplir le formulaire d'alimentation rapatriement.

## JSON de sortie obligatoire

L'API doit toujours retourner exactement cette structure:

```json
{
  "dateDosRap": "2026-06-08",
  "mntRap": 250.0,
  "typePieceBenef": 1,
  "noPieceBenef": "0004545Y"
}
```

### Definition des champs

| Champ | Type | Description |
| --- | --- | --- |
| `dateDosRap` | `string` | Date de l'operation trouvee dans le document, format `YYYY-MM-DD`. |
| `mntRap` | `number` | Montant du rapatriement. Priorite: montant sous/associe a `DROITS A TRANSFERT CUMULES (6)`, sinon le plus petit montant financier du document. |
| `typePieceBenef` | `number` | Type de piece du beneficiaire. `C` donne `1`, `S` donne `4`. |
| `noPieceBenef` | `string` | Numero/code beneficiaire compose de 7 chiffres et une lettre, exemple `0004545Y`. |

## Architecture recommandee

Ne pas appeler LM Studio directement depuis React.

React doit appeler uniquement votre backend applicatif, pour eviter:

- l'exposition de details internes comme `LM_STUDIO_HOST`;
- les problemes CORS;
- la dependance directe du navigateur a LM Studio;
- les erreurs de securite liees aux fichiers PDF.

Architecture:

```text
React App
  |
  | multipart/form-data PDF
  v
Backend API OCR
  |
  | Python FinancialDocumentParser
  v
LM Studio / Qwen local
  |
  v
JSON normalise
```

## Configuration backend

Variables `.env` cote backend Python:

```env
LM_STUDIO_HOST=http://127.0.0.1:1234
QWEN_MODEL=qwen/qwen2.5-vl-7b
MAX_UPLOAD_SIZE_MB=10
```

Si LM Studio expose le modele sans prefixe, utiliser:

```env
QWEN_MODEL=qwen2.5-vl-7b
```

## Endpoint API a exposer

Ces endpoints sont maintenant disponibles dans `api_ocr.py`.

### Healthcheck

```http
GET /api/health
```

Reponse exemple:

```json
{
  "status": "ok",
  "lmStudioHost": "http://127.0.0.1:1234",
  "qwenModel": "qwen2.5-vl-7b",
  "lmStudioConnected": true
}
```

### Schema de sortie

```http
GET /api/ocr/parse/schema
```

Reponse:

```json
{
  "dateDosRap": "",
  "mntRap": 0.0,
  "typePieceBenef": 0,
  "noPieceBenef": ""
}
```

### Extraction OCR

Endpoint:

```http
POST /api/ocr/parse
Content-Type: multipart/form-data
```

Champ fichier:

```text
file
```

Reponse succes:

```json
{
  "dateDosRap": "2026-06-08",
  "mntRap": 250.0,
  "typePieceBenef": 1,
  "noPieceBenef": "0004545Y"
}
```

Reponse erreur conseillee:

```json
{
  "message": "Impossible d'extraire le document",
  "details": "PDF invalide ou format non supporte"
}
```

## Exemple backend FastAPI

Le backend FastAPI existe dans le fichier `api_ocr.py`.

Installer les dependances si necessaire:

```powershell
pip install fastapi uvicorn python-multipart
```

Lancement:

```powershell
uvicorn api_ocr:app --host 127.0.0.1 --port 8000 --reload
```

ou:

```powershell
python api_ocr.py
```

Test rapide:

```powershell
curl -X POST http://localhost:8000/api/ocr/parse -F "file=@Rapatriment_AVA.pdf"
```

Le fichier contient la logique suivante:

```python
import asyncio
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from financial_extractor import FinancialDocumentParser
from llm_clients.llm_client import LLMClient
from llm_clients.qwen_vision_agent import QwenVisionAgent

load_dotenv()

app = FastAPI(title="OCR Rapatriement API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_parser() -> FinancialDocumentParser:
    host = os.getenv("LM_STUDIO_HOST", "http://127.0.0.1:1234")
    model = os.getenv("QWEN_MODEL", "qwen2.5-vl-7b")

    qwen_agent = QwenVisionAgent(lm_studio_host=host, model_name=model)
    llm_client = LLMClient(lm_studio_host=host, preferred_model=model)
    return FinancialDocumentParser(qwen_agent, llm_client)


@app.post("/api/ocr/parse")
async def parse_ocr_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont supportes")

    suffix = Path(file.filename).suffix or ".pdf"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        parser = create_parser()
        result = await parser.extract_and_parse_document(tmp_path)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
```

## Exemple React

### Service API

Fichier `src/services/rapatriementOcrApi.ts`:

```typescript
export type RapatriementOcrResult = {
  dateDosRap: string;
  mntRap: number;
  typePieceBenef: number;
  noPieceBenef: string;
};

export async function extractRapatriementPdf(file: File): Promise<RapatriementOcrResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://localhost:8000/api/ocr/parse", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail || "Erreur OCR rapatriement");
  }

  const payload = await response.json();
  return payload.data ?? payload;
}
```

### Composant upload + auto-remplissage

Exemple adapte a un formulaire React classique:

```tsx
import { useState } from "react";
import { extractRapatriementPdf, RapatriementOcrResult } from "./services/rapatriementOcrApi";

type FormState = {
  typePieceBenef: string;
  noPieceBenef: string;
  mntRap: string;
  dateDosRap: string;
};

const initialForm: FormState = {
  typePieceBenef: "",
  noPieceBenef: "",
  mntRap: "",
  dateDosRap: "",
};

export function RapatriementForm() {
  const [form, setForm] = useState<FormState>(initialForm);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function applyOcrResult(data: RapatriementOcrResult) {
    setForm((current) => ({
      ...current,
      typePieceBenef: String(data.typePieceBenef || ""),
      noPieceBenef: data.noPieceBenef || "",
      mntRap: data.mntRap ? String(data.mntRap) : "",
      dateDosRap: data.dateDosRap || "",
    }));
  }

  async function handleExtract() {
    if (!file) {
      setError("Veuillez selectionner un PDF.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const data = await extractRapatriementPdf(file);
      applyOcrResult(data);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Erreur OCR");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <section>
        <h3>Extraction OCR du rapatriement</h3>
        <p>Chargez le PDF du rapatriement pour pre-remplir les donnees de l'operation.</p>

        <input
          type="file"
          accept="application/pdf"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
        />

        <button type="button" onClick={handleExtract} disabled={loading}>
          {loading ? "Extraction..." : "Extraire"}
        </button>

        <button type="button" onClick={() => setForm(initialForm)}>
          Reinitialiser
        </button>

        {error && <p role="alert">{error}</p>}
      </section>

      <section>
        <h3>Formulaire d'Alimentation (Rapatriement)</h3>

        <label>
          Type de Piece *
          <select
            value={form.typePieceBenef}
            onChange={(event) => setForm({ ...form, typePieceBenef: event.target.value })}
          >
            <option value="">Selectionner le type</option>
            <option value="1">1 - Carte Nationale d'Identite</option>
            <option value="4">4 - Carte de Sejour</option>
          </select>
        </label>

        <label>
          Numero de piece *
          <input
            value={form.noPieceBenef}
            onChange={(event) => setForm({ ...form, noPieceBenef: event.target.value })}
            placeholder="Ex. 0004545Y"
          />
        </label>

        <label>
          Montant (Rapatriement) *
          <input
            type="number"
            step="0.001"
            value={form.mntRap}
            onChange={(event) => setForm({ ...form, mntRap: event.target.value })}
            placeholder="Montant du rapatriement"
          />
        </label>

        <label>
          Date de dossier *
          <input
            type="date"
            value={form.dateDosRap}
            onChange={(event) => setForm({ ...form, dateDosRap: event.target.value })}
          />
        </label>
      </section>
    </div>
  );
}
```

## Mapping JSON vers formulaire

| JSON OCR | Champ formulaire | Exemple |
| --- | --- | --- |
| `typePieceBenef` | Type de Piece | `1` |
| `noPieceBenef` | Numero de piece | `0004545Y` |
| `mntRap` | Montant (Rapatriement) | `250.0` |
| `dateDosRap` | Date de dossier | `2026-06-08` |

## Validation cote React

Avant d'autoriser `Rapatrier & Generer`, verifier:

```typescript
function validateForm(form: FormState) {
  if (!["1", "4"].includes(form.typePieceBenef)) {
    return "Type de piece invalide.";
  }

  if (!/^\d{7}[A-Za-z]$/.test(form.noPieceBenef)) {
    return "Numero de piece invalide. Format attendu: 7 chiffres + 1 lettre.";
  }

  if (!form.mntRap || Number(form.mntRap) <= 0) {
    return "Montant invalide.";
  }

  if (!form.dateDosRap) {
    return "Date de dossier obligatoire.";
  }

  return "";
}
```

## Points importants pour LM Studio

1. Ouvrir LM Studio.
2. Charger le modele Qwen Vision.
3. Demarrer le serveur local OpenAI-compatible.
4. Verifier que l'API repond:

```powershell
curl http://127.0.0.1:1234/v1/models
```

Le backend Python utilise:

```text
http://127.0.0.1:1234/v1/chat/completions
```

## Comportement sur plusieurs layouts

L'extracteur utilise deux niveaux:

1. Extraction texte PDF avec PyMuPDF quand le PDF contient du texte.
2. Qwen/LM Studio pour aider sur les layouts difficiles ou documents image/scannes.

Les regles metier restent fixes:

- date operation -> `dateDosRap`;
- `C` -> `typePieceBenef = 1`;
- `S` -> `typePieceBenef = 4`;
- code `7 chiffres + 1 lettre` -> `noPieceBenef`;
- montant `DROITS A TRANSFERT CUMULES (6)` ou plus petit montant -> `mntRap`.

## Checklist d'integration

- LM Studio lance sur `127.0.0.1:1234`.
- Modele Qwen charge.
- Backend FastAPI lance sur `localhost:8000`.
- React appelle `POST /api/ocr/parse`.
- Le champ fichier s'appelle `file`.
- Le JSON retourne exactement les 4 champs attendus.
- Le formulaire mappe `typePieceBenef`, `noPieceBenef`, `mntRap`, `dateDosRap`.
- Les champs restent modifiables manuellement apres l'auto-remplissage.
