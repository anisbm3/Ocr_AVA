"""Evaluate OCR accuracy on generated rapatriement test documents via the API."""

import argparse
import json
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "doc" / "test_documents" / "rapatriement_variants" / "ground_truth.json"
DEFAULT_API_URL = "http://localhost:8000/api/ocr/parse"
OUTPUT_PATH = ROOT / "doc" / "test_documents" / "rapatriement_variants" / "accuracy_results.json"


def normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, str):
        return value.strip()
    return value


def compare(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, bool]:
    return {
        field: normalize_value(actual.get(field)) == normalize_value(expected.get(field))
        for field in expected
    }


def post_document(api_url: str, pdf_path: Path) -> dict[str, Any]:
    with pdf_path.open("rb") as file_handle:
        response = requests.post(api_url, files={"file": (pdf_path.name, file_handle, "application/pdf")}, timeout=180)
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", payload)


def evaluate(dataset_path: Path, api_url: str) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    base_dir = dataset_path.parent
    results = []
    total_fields = 0
    correct_fields = 0
    exact_matches = 0

    for item in dataset:
        pdf_path = base_dir / item["file"]
        expected = item["expected"]

        try:
            actual = post_document(api_url, pdf_path)
            field_matches = compare(expected, actual)
            field_correct = sum(1 for value in field_matches.values() if value)
            exact_match = field_correct == len(field_matches)
            error = ""
        except Exception as exc:
            actual = {}
            field_matches = {field: False for field in expected}
            field_correct = 0
            exact_match = False
            error = str(exc)

        total_fields += len(expected)
        correct_fields += field_correct
        exact_matches += int(exact_match)

        results.append(
            {
                "file": item["file"],
                "issues": item.get("issues", []),
                "expected": expected,
                "actual": actual,
                "fieldMatches": field_matches,
                "correctFields": field_correct,
                "totalFields": len(expected),
                "exactMatch": exact_match,
                "error": error,
            }
        )

    summary = {
        "apiUrl": api_url,
        "documents": len(dataset),
        "correctFields": correct_fields,
        "totalFields": total_fields,
        "fieldLevelAccuracy": round((correct_fields / total_fields) * 100, 2) if total_fields else 0.0,
        "exactDocumentMatches": exact_matches,
        "documentExactMatchRate": round((exact_matches / len(dataset)) * 100, 2) if dataset else 0.0,
    }

    return {"summary": summary, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rapatriement OCR accuracy.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to ground_truth.json")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="OCR API URL")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSON path")
    args = parser.parse_args()

    report = evaluate(Path(args.dataset), args.api_url)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    print(f"Detailed results: {output_path}")


if __name__ == "__main__":
    main()
