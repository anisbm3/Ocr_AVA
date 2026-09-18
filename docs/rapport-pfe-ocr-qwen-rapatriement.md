# Documentation PFE - Module OCR intelligent pour documents de rapatriement

## 1. Presentation generale

Ce module a pour objectif d'automatiser l'extraction des donnees utiles depuis un document financier de rapatriement, puis de retourner un JSON normalise exploitable par une application web.

Le besoin principal est de remplacer la saisie manuelle de certaines informations par une extraction automatique a partir d'un PDF. L'utilisateur charge un document, le systeme analyse son contenu, identifie les champs importants, puis renvoie une structure JSON stable permettant le pre-remplissage d'un formulaire.

Le module extrait exactement les champs suivants:

```json
{
  "dateDosRap": "2026-06-08",
  "mntRap": 250.0,
  "typePieceBenef": 1,
  "noPieceBenef": "0004545Y"
}
```

## 2. Objectifs fonctionnels

- Charger un document PDF de rapatriement.
- Extraire la date de l'operation.
- Extraire le montant du rapatriement.
- Identifier le type de piece du beneficiaire.
- Extraire le numero de piece du beneficiaire.
- Retourner un JSON standardise.
- Permettre l'integration avec une application React.
- Pre-remplir automatiquement un formulaire metier a partir du JSON extrait.

## 3. Technologies utilisees

### Backend Python

Le backend est developpe en Python. Il contient la logique d'extraction, de validation, de parsing et d'exposition API.

Technologies principales:

- `Python 3.11`
- `FastAPI` pour exposer l'API REST.
- `Uvicorn` comme serveur ASGI.
- `PyMuPDF` pour extraire le texte des documents PDF.
- `requests` pour communiquer avec LM Studio.
- `python-dotenv` pour charger les variables d'environnement.
- `pydantic` via FastAPI pour typer et valider les reponses.

### Interface de test Gradio

Une interface Gradio est disponible pour tester rapidement le module sans developper de frontend externe.

Technologies:

- `Gradio`
- Upload de fichier PDF
- Affichage du JSON extrait

### LLM local avec Qwen

Le projet utilise un modele Qwen execute localement via LM Studio.

Modele utilise:

```text
qwen2.5-vl-7b
```

ou selon le nom expose par LM Studio:

```text
qwen/qwen2.5-vl-7b
```

LM Studio expose une API compatible OpenAI:

```text
GET  http://127.0.0.1:1234/v1/models
POST http://127.0.0.1:1234/v1/chat/completions
```

## 4. Role de l'intelligence artificielle

L'IA intervient comme moteur d'assistance a l'extraction et a la comprehension du document.

Le traitement utilise deux approches complementaires:

1. Extraction deterministe avec PyMuPDF et regles metier.
2. Assistance LLM locale avec Qwen pour les documents plus complexes ou les layouts variables.

Cette architecture permet d'avoir:

- une extraction rapide lorsque le PDF contient du texte lisible;
- une meilleure robustesse lorsque la mise en page varie;
- une execution locale sans envoyer les documents financiers vers un service cloud externe.

## 5. Gestion des API keys et confidentialite

Le module actuel ne necessite pas obligatoirement une cle API cloud pour l'extraction Qwen, car Qwen tourne localement avec LM Studio.

La configuration se fait dans le fichier `.env`:

```env
LM_STUDIO_HOST=http://127.0.0.1:1234
QWEN_MODEL=qwen2.5-vl-7b
LM_STUDIO_TIMEOUT=900
APP_HOST=localhost
APP_PORT=7861
API_HOST=127.0.0.1
API_PORT=8000
MAX_UPLOAD_SIZE_MB=10
```

Une ancienne integration Gemini peut exister dans le projet, mais elle n'est pas necessaire pour le module actuel de rapatriement. Si Gemini est utilise, une cle `GEMINI_API_KEY` est requise. Cette cle doit rester strictement privee et ne doit jamais etre exposee dans le frontend React.

Bonnes pratiques:

- ne jamais commiter le fichier `.env`;
- utiliser `.env.example` pour documenter les variables sans secrets;
- conserver les cles API uniquement cote backend;
- ne jamais appeler directement une API LLM depuis React si elle utilise une cle secrete.

## 6. Input du module

L'input principal est un fichier PDF.

Formats acceptes par l'API:

- `.pdf`
- `.png`
- `.jpg`
- `.jpeg`

Dans le cas du besoin metier actuel, le format recommande est:

```text
PDF de document financier de rapatriement
```

Exemple:

```text
Rapatriment_AVA.pdf
```

Le fichier est envoye via une requete HTTP multipart:

```text
multipart/form-data
champ fichier: file
```

## 7. Output du module

Le module retourne un JSON strict:

```json
{
  "dateDosRap": "2026-06-08",
  "mntRap": 250.0,
  "typePieceBenef": 1,
  "noPieceBenef": "0004545Y"
}
```

Description des champs:

| Champ | Type | Description |
| --- | --- | --- |
| `dateDosRap` | string | Date de l'operation au format `YYYY-MM-DD`. |
| `mntRap` | number | Montant du rapatriement extrait du document. |
| `typePieceBenef` | number | Type de piece beneficaire: `1` pour `C`, `4` pour `S`. |
| `noPieceBenef` | string | Code beneficiaire compose de 7 chiffres et une lettre. |

## 8. Regles metier d'extraction

### Date

Le champ `dateDosRap` correspond a la date de l'operation trouvee dans le document.

Exemple trouve dans le PDF:

```text
08/06/2026
```

Valeur retournee:

```json
"dateDosRap": "2026-06-08"
```

### Montant

Le champ `mntRap` correspond au montant du rapatriement.

La priorite est donnee au montant associe a:

```text
DROITS A TRANSFERT CUMULES (6)
```

Si plusieurs montants sont presents, le systeme peut choisir le plus petit montant financier pertinent.

Exemple:

```text
250,000
1000,000
```

Valeur retournee:

```json
"mntRap": 250.0
```

### Type de piece beneficiaire

Le document peut contenir un code d'identification:

```text
C
```

ou:

```text
S
```

Mapping:

| Code document | Valeur JSON |
| --- | --- |
| `C` | `1` |
| `S` | `4` |

Exemple:

```json
"typePieceBenef": 1
```

### Numero de piece beneficiaire

Le champ `noPieceBenef` correspond au code constitue de:

```text
7 chiffres + 1 lettre
```

Exemple:

```text
0004545Y
```

Valeur retournee:

```json
"noPieceBenef": "0004545Y"
```

## 9. Etapes de traitement

Le pipeline de traitement est le suivant:

1. L'utilisateur charge le PDF dans React ou dans l'interface Gradio.
2. Le fichier est envoye au backend via `multipart/form-data`.
3. Le backend verifie l'extension et la taille du fichier.
4. Le fichier est sauvegarde temporairement.
5. `PyMuPDF` tente d'extraire le texte du PDF.
6. Les regles metier recherchent les champs cibles.
7. Si necessaire, Qwen peut aider a interpreter le document.
8. Les donnees sont normalisees.
9. Le JSON final est retourne a l'application cliente.
10. React utilise le JSON pour pre-remplir le formulaire.

## 10. Architecture technique

Architecture globale:

```text
Utilisateur
  |
  v
Application React
  |
  | POST /api/ocr/parse
  | multipart/form-data
  v
Backend FastAPI
  |
  | Extraction texte PDF
  v
PyMuPDF + Regles metier
  |
  | si besoin
  v
LM Studio / Qwen local
  |
  v
JSON normalise
  |
  v
Formulaire pre-rempli
```

## 11. API REST exposee

### Healthcheck

Permet de verifier que l'API est active et que LM Studio est accessible.

```http
GET http://localhost:8000/api/health
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

### Schema JSON

Permet de connaitre la structure de sortie attendue.

```http
GET http://localhost:8000/api/ocr/parse/schema
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

Endpoint principal utilise par React.

```http
POST http://localhost:8000/api/ocr/parse
```

Body:

```text
multipart/form-data
file: fichier PDF
```

Exemple Postman:

```text
Method: POST
URL: http://localhost:8000/api/ocr/parse
Body: form-data
Key: file
Type: File
Value: Rapatriment_AVA.pdf
```

Reponse:

```json
{
  "dateDosRap": "2026-06-08",
  "mntRap": 250.0,
  "typePieceBenef": 1,
  "noPieceBenef": "0004545Y"
}
```

## 12. Integration React

Le frontend React doit consommer l'API backend, et non LM Studio directement.

Exemple de service React:

```typescript
export type OcrParseResult = {
  dateDosRap: string;
  mntRap: number;
  typePieceBenef: number;
  noPieceBenef: string;
};

export async function parseRapatriementPdf(file: File): Promise<OcrParseResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://localhost:8000/api/ocr/parse", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Erreur lors de l'extraction OCR");
  }

  return response.json();
}
```

Auto-remplissage du formulaire:

```typescript
const data = await parseRapatriementPdf(file);

setForm({
  typePieceBenef: String(data.typePieceBenef),
  noPieceBenef: data.noPieceBenef,
  mntRap: String(data.mntRap),
  dateDosRap: data.dateDosRap,
});
```

## 13. Securite

Mesures recommandees:

- limiter la taille maximale des fichiers;
- accepter uniquement les extensions autorisees;
- supprimer les fichiers temporaires apres traitement;
- ne pas exposer LM Studio directement a Internet;
- ne pas exposer de cles API dans React;
- journaliser les erreurs sans afficher de donnees sensibles;
- ajouter une authentification si l'API est exposee hors reseau local.

## 14. Avantages de l'approche locale

L'utilisation de Qwen via LM Studio presente plusieurs avantages:

- les documents restent sur la machine ou le serveur local;
- pas de dependance obligatoire a un fournisseur cloud;
- pas de cout direct par requete LLM cloud;
- possibilite de fonctionner meme avec des documents sensibles;
- controle total sur le modele et l'environnement d'execution.

## 15. Limites

Le systeme peut rencontrer des limites dans les cas suivants:

- PDF scanne de mauvaise qualite;
- document fortement incline ou flou;
- layouts tres differents du format attendu;
- champs absents ou ambigus;
- LM Studio non demarre;
- modele Qwen non charge;
- ressources machine insuffisantes pour executer le modele local.

## 16. Perspectives d'amelioration

Ameliorations possibles:

- ajouter un score de confiance par champ;
- stocker l'historique des extractions;
- ajouter une validation humaine avant soumission;
- enrichir les regles metier pour d'autres types de documents financiers;
- gerer plusieurs pages et plusieurs operations;
- ajouter une authentification JWT pour securiser l'API;
- ajouter des tests automatises sur plusieurs exemples PDF;
- conteneuriser l'API avec Docker pour faciliter le deploiement.

## 17. Conclusion

Ce module OCR intelligent permet d'automatiser l'extraction d'informations depuis un document financier de rapatriement. Il combine des techniques classiques d'extraction PDF avec l'apport d'un LLM local Qwen pour offrir une solution robuste, integrable et respectueuse de la confidentialite.

L'API `POST /api/ocr/parse` fournit une interface simple pour les applications externes, notamment une application React, qui peut ensuite pre-remplir automatiquement un formulaire metier a partir du JSON extrait.

