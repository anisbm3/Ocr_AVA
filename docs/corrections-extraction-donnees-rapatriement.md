# Corrections pour optimiser l'extraction des donnees AVA / rapatriement

## Objectif

Ameliorer l'extraction automatique des champs suivants depuis un formulaire AVA fixe, qu'il soit fourni en PDF ou en image PNG/JPG, avec des zones remplies par ecriture manuelle ou texte imprime :

```json
{
  "dateDosRap": "",
  "mntRap": 0.0,
  "typePieceBenef": 0,
  "noPieceBenef": ""
}
```

## Probleme observe

L'OCR Tesseract extrait beaucoup de texte depuis les images, mais il lit mal les petites cellules du tableau.

Exemples observes :

- `08/06/2026` peut devenir `08/06/202` ou `osvo6202`.
- `0004545Y` peut etre ignore ou lu de maniere incomplete.
- Les montants dans les colonnes du tableau peuvent disparaitre ou etre confondus entre deux cellules voisines.
- Le modele Qwen peut ensuite recevoir un texte OCR degrade, et retourner le JSON vide par prudence.

Le probleme n'est donc pas seulement "OCR fonctionne ou non". Le vrai probleme est la qualite des champs metier extraits.

## Correction deja appliquee

Fichier modifie :

```text
financial_extractor/__init__.py
```

### 1. Fallback Qwen Vision si les champs sont incomplets

Avant, le pipeline faisait :

```text
Image -> Tesseract -> si texte longueur >= 20, on accepte le resultat
```

Maintenant, le pipeline fait :

```text
Image -> Tesseract -> parsing des champs -> si champs critiques incomplets, Qwen Vision relit l'image
```

Les champs critiques controles sont :

- `dateDosRap`
- `mntRap`
- `typePieceBenef`
- `noPieceBenef`

Si un de ces champs manque, le code tente Qwen Vision sur l'image source.

### 2. Parsing centralise

Une nouvelle fonction regroupe les sources de parsing :

```python
_parse_extracted_text(...)
```

Elle combine :

- JSON direct si Qwen retourne deja un JSON.
- Regles regex locales.
- Parsing LLM depuis texte OCR.

Cela evite d'avoir deux pipelines separes pour Tesseract et Qwen.

### 3. Lecture JSON directe depuis Qwen

Une nouvelle fonction lit un JSON si le retour Qwen contient directement :

```json
{
  "dateDosRap": "2026-06-08",
  "mntRap": 250.0,
  "typePieceBenef": 1,
  "noPieceBenef": "0004545Y"
}
```

Fonction ajoutee :

```python
_parse_json_fragment(...)
```

### 4. Prompt Qwen adapte au formulaire fixe

Le prompt Qwen Vision demande maintenant explicitement les champs dans la ligne du tableau, pas dans les notes explicatives.

Il precise aussi que les valeurs peuvent etre manuscrites.

### 5. Extraction directe depuis les cellules du tableau

Une extraction par template fixe a ete ajoutee pour eviter que `dateDosRap` et `mntRap` soient pris depuis une mauvaise zone.

Le code detecte les lignes du tableau, puis lit directement :

- `dateDosRap` depuis la cellule `DATE` de la ligne de donnees ;
- `mntRap` depuis la cellule `DROITS A TRANSFERT CUMULES (6)` ;
- `typePieceBenef` depuis la cellule `TYPE` ;
- `noPieceBenef` depuis la cellule `NUMERO`.

Pour `dateDosRap` et `mntRap`, cette extraction cellule est prioritaire. Si la cellule DATE manuscrite n'est pas lisible par Tesseract, Qwen Vision est appele uniquement sur le crop de cette cellule, pas sur tout le formulaire.

Pour `noPieceBenef`, le format est strict :

```text
7 chiffres + 1 lettre
```

Exemple :

```text
0004545Y
```

Si Tesseract lit seulement les 7 chiffres, ou lit la derniere lettre comme un chiffre, Qwen Vision est relance uniquement sur la cellule `NUMERO` avec une consigne stricte : la derniere position doit etre une lettre.

Cela empeche les erreurs observees comme :

```json
{
  "dateDosRap": "2023-06-28",
  "mntRap": 1000000.0
}
```

alors que les cellules correctes donnent :

```json
{
  "dateDosRap": "2026-06-08",
  "mntRap": 250.0
}
```

## Corrections recommandees ensuite

### A. Donner la priorite a la zone du tableau

Comme le template est fixe, on peut cropper uniquement la zone du tableau avant OCR/Qwen.

Avantages :

- moins de bruit venant des notes en bas de page ;
- meilleure lecture des petites cellules ;
- moins de confusion entre les exemples des notes et les vraies donnees ;
- Qwen se concentre sur la ligne utile.

Pipeline recommande :

```text
image complete
  -> crop tableau
  -> Tesseract tableau
  -> Qwen Vision tableau si champs incomplets
```

### B. Augmenter la resolution avant OCR

Pour Tesseract, les petites cellules doivent etre agrandies.

Recommandation :

```text
image -> grayscale -> autocontrast -> resize x2 ou x3 -> sharpen -> OCR
```

Cela aidera surtout :

- la date ;
- le numero beneficiaire ;
- les montants ;
- les lettres manuscrites.

### C. Eviter de trop reduire l'image envoyee a Qwen

Dans `llm_clients/qwen_vision_agent.py`, la fonction :

```python
encode_image_to_base64(image_path, max_size=1024)
```

peut reduire l'image a 1024 pixels maximum.

Pour un formulaire avec petites cellules, c'est parfois trop faible.

Recommandation :

```text
max_size = 1600 ou 2048
```

Cela peut ameliorer la lecture des champs manuscrits et des tableaux.

### D. Rendre les regex plus tolerantes

Les regex actuelles attendent des valeurs propres.

Exemples :

```text
08/06/2026
250,000
0004545Y
```

Mais l'OCR peut produire :

```text
08/06/202 6
08/06/202
250.000
250,00
0004545 Y
O004545Y
```

Corrections recommandees :

- accepter les espaces dans les dates ;
- corriger `O` vers `0` dans les codes numeriques ;
- accepter un espace avant la lettre finale du numero ;
- privilegier la ligne contenant `RAP` ;
- retenir uniquement la cellule `DROITS A TRANSFERT CUMULES (6)` pour `mntRap` ;
- ne pas prendre la cellule suivante `MONTANTS DES TRANSFERTS CUMULES` ;
- convertir `250,000` et `250.000` en `250.000`, jamais en `250000`.

### E. Correction specifique du montant `mntRap`

Le montant a retenir est dans la cellule :

```text
DROITS A TRANSFERT CUMULES (6)
```

Il ne faut pas utiliser la cellule juste apres :

```text
MONTANTS DES TRANSFERTS CUMULES
```

Exemple de ligne :

```text
08/06/2026 | RAP | ... | PAYS 788 | DROITS A TRANSFERT CUMULES 250,000 | MONTANTS DES TRANSFERTS CUMULES 1000,000
```

Resultat attendu :

```json
{
  "mntRap": 250.0
}
```

Resultat incorrect :

```json
{
  "mntRap": 1000.0
}
```

La conversion des montants doit respecter le format decimal tunisien/francais :

```text
250,000 -> 250.000
250.000 -> 250.000
1000,000 -> 1000.000
```

Il ne faut pas supprimer le point decimal dans `250.000`.

### F. Ajouter un mode debug OCR

Pour comprendre les erreurs Postman, ajouter un endpoint ou une option qui retourne :

```json
{
  "method": "tesseract",
  "rawText": "...",
  "ruleData": {},
  "llmData": {},
  "qwenFallbackUsed": true
}
```

Cela permettra de savoir rapidement si l'erreur vient :

- de Tesseract ;
- de Qwen Vision ;
- des regex ;
- du format de sortie ;
- d'une image trop petite.

## Pipeline cible recommande

```text
Upload PDF/PNG/JPG
    |
    v
Validation extension + taille
    |
    v
Si PDF texte: PyMuPDF
    |
    v
Si PDF scanne: rendu PNG haute resolution
    |
    v
Pretraitement image complete
    |
    v
Crop zone tableau selon template fixe
    |
    v
Tesseract image complete + crop tableau
    |
    v
Parsing regex
    |
    v
Si champs incomplets: Qwen Vision sur crop tableau puis image complete
    |
    v
Fusion des resultats
    |
    v
Normalisation JSON final
```

## Priorite des prochaines taches

1. Augmenter la resolution des images avant Tesseract.
2. Envoyer a Qwen une image moins compressee ou moins reduite.
3. Rendre les regex plus robustes aux erreurs OCR.
4. Ajouter un endpoint debug pour voir le texte OCR et savoir quel fallback a ete utilise.

## Requete Postman de validation

Endpoint :

```text
POST http://127.0.0.1:8000/api/ocr/parse
```

Body :

```text
form-data
key: file
type: File
value: fichier PDF/PNG/JPG
```

Resultat attendu :

```json
{
  "dateDosRap": "2026-06-08",
  "mntRap": 250.0,
  "typePieceBenef": 1,
  "noPieceBenef": "0004545Y"
}
```

## Conclusion

La correction la plus importante est deja mise en place : ne plus faire confiance a Tesseract uniquement parce qu'il extrait beaucoup de texte.

Le critere correct est maintenant :

```text
Est-ce que les champs metier sont complets ?
```

Si non, Qwen Vision doit relire l'image, ce qui est beaucoup plus adapte pour un formulaire fixe rempli manuellement.
