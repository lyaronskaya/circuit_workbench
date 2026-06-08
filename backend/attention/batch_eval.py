from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _mean(values: List[float]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def evaluate_dataset(
    model,
    dataset: Dict[str, Any],
    ablated_heads: List[Tuple[int, int]],
) -> Dict[str, Any]:
    """Evaluate a dataset example-by-example and aggregate causal metrics."""
    example_results: List[Dict[str, Any]] = []
    for example in dataset.get("examples", []):
        result = model.evaluate_text(
            text=example["text"],
            target_token=example.get("target_token"),
            ablated_heads=ablated_heads,
        )
        result["example_id"] = example["id"]
        result["text"] = example["text"]
        result["corrupted_text"] = example.get("corrupted_text")
        result["metadata"] = example.get("metadata", {})
        example_results.append(result)

    deltas = [result.get("delta") or {} for result in example_results]
    aggregate = {
        "mean_target_probability_delta": _mean([delta.get("target_probability_delta") for delta in deltas]),
        "mean_logit_diff_delta": _mean([delta.get("logit_diff_delta") for delta in deltas]),
        "mean_rank_delta": _mean([delta.get("target_rank_delta") for delta in deltas]),
        "mean_loss_delta": _mean([delta.get("loss_delta") for delta in deltas]),
        "top_prediction_changed_fraction": _mean(
            [1.0 if delta.get("top_prediction_changed") else 0.0 for delta in deltas]
        ),
    }

    return {
        "dataset_name": dataset.get("dataset_name"),
        "name": dataset.get("name"),
        "description": dataset.get("description"),
        "metric": dataset.get("metric"),
        "num_examples": len(example_results),
        "aggregate": aggregate,
        "examples": example_results,
    }
