# Rapatriement OCR Test Documents

This folder contains 20 synthetic variants generated from the same rapatriement document.

All documents keep the same expected target JSON:

```json
{
  "dateDosRap": "2026-06-08",
  "mntRap": 250.0,
  "typePieceBenef": 1,
  "noPieceBenef": "0004545Y"
}
```

The variants include:

- clean reference
- light and heavy rotation
- skewed capture
- grayscale
- black and white threshold
- low contrast
- high contrast
- blur
- light and heavy noise
- yellow and blue color casts
- dark scan
- overexposed scan
- slight crop
- extra unrelated data
- maximum degraded version

Ground truth file:

```text
ground_truth.json
```

Run the accuracy benchmark:

```powershell
python scripts/evaluate_rapatriement_accuracy.py
```

The script calls:

```text
POST http://localhost:8000/api/ocr/parse
```

Output:

```text
doc/test_documents/rapatriement_variants/accuracy_results.json
```

