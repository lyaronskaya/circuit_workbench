from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets"
BUILTIN_ROOT = DATASET_ROOT / "builtin"
CUSTOM_ROOT = DATASET_ROOT / "custom"


def ensure_dataset_dirs() -> None:
    """Create dataset directories if they do not exist."""
    BUILTIN_ROOT.mkdir(parents=True, exist_ok=True)
    CUSTOM_ROOT.mkdir(parents=True, exist_ok=True)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "dataset"


def _dataset_paths() -> List[Path]:
    ensure_dataset_dirs()
    return sorted(list(BUILTIN_ROOT.glob("*.json")) + list(CUSTOM_ROOT.glob("*.json")))


def _read_dataset(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["dataset_name"] = path.stem
    payload["storage"] = "custom" if path.parent == CUSTOM_ROOT else "builtin"
    return payload


def _supports_causal(examples: List[Dict[str, Any]]) -> bool:
    return bool(examples) and all(example.get("corrupted_text") for example in examples)


def validate_dataset_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the uploaded dataset payload and summarize issues."""
    errors: List[str] = []
    warnings: List[str] = []
    examples = payload.get("examples", [])

    if not payload.get("name"):
        errors.append("Dataset must include a non-empty 'name'.")

    if not isinstance(examples, list) or not examples:
        errors.append("Dataset must include a non-empty 'examples' list.")
        examples = []

    required_fields = ["id", "text", "target_token"]
    num_examples = len(examples)
    valid_examples = 0

    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            errors.append(f"Example {index} must be an object.")
            continue
        missing = [field for field in required_fields if not example.get(field)]
        if missing:
            errors.append(f"Example {index} is missing required fields: {', '.join(missing)}.")
            continue
        valid_examples += 1
        target_token = str(example.get("target_token", ""))
        if len(target_token.strip().split()) > 3:
            warnings.append(
                f"Example {index} has a long target token '{target_token}'. Consider verifying tokenization."
            )

    supports_causal = _supports_causal(examples)
    if not supports_causal:
        warnings.append("Dataset does not contain corrupted prompts for every example, so patching-style workflows are unavailable.")

    dataset_name = _slugify(str(payload.get("name", "")))
    return {
        "valid": len(errors) == 0,
        "dataset_name": dataset_name,
        "num_examples": num_examples,
        "valid_examples": valid_examples,
        "supports_causal": supports_causal,
        "errors": errors,
        "warnings": warnings,
    }


def list_datasets() -> List[Dict[str, Any]]:
    """List available built-in and custom datasets with summary metadata."""
    datasets: List[Dict[str, Any]] = []
    for path in _dataset_paths():
        payload = _read_dataset(path)
        examples = payload.get("examples", [])
        datasets.append(
            {
                "dataset_name": payload["dataset_name"],
                "name": payload.get("name", payload["dataset_name"]),
                "description": payload.get("description", ""),
                "metric": payload.get("metric", "target_probability"),
                "model": payload.get("model"),
                "storage": payload.get("storage"),
                "num_examples": len(examples),
                "supports_causal": _supports_causal(examples),
            }
        )
    return datasets


def load_dataset(dataset_name: str) -> Dict[str, Any]:
    """Load a dataset by file stem from builtin or custom storage."""
    for path in _dataset_paths():
        if path.stem == dataset_name:
            return _read_dataset(path)
    raise FileNotFoundError(f"Dataset '{dataset_name}' not found.")


def save_custom_dataset(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and save a custom dataset to datasets/custom."""
    ensure_dataset_dirs()
    validation = validate_dataset_payload(payload)
    if not validation["valid"]:
        raise ValueError("Dataset payload is invalid.")

    dataset_name = validation["dataset_name"]
    target_path = CUSTOM_ROOT / f"{dataset_name}.json"
    normalized_payload = dict(payload)
    normalized_payload["name"] = dataset_name if not payload.get("name") else payload["name"]
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(normalized_payload, handle, ensure_ascii=True, indent=2)

    saved_payload = load_dataset(dataset_name)
    return {
        "dataset_name": dataset_name,
        "path": str(target_path),
        "dataset": saved_payload,
        "validation": validation,
    }
