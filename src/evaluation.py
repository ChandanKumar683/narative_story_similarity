"""
Evaluation metrics for SemEval-2026 Task 4 – Track A.
"""

import json
from pathlib import Path
from datetime import datetime, timezone


def evaluate_track_a(gold: list[dict], predictions: list[dict]) -> dict:
    """Compute accuracy and per-example correctness for Track-A.

    Args:
        gold: list of dicts with 'id' and 'text_a_is_closer' (ground truth).
        predictions: list of dicts with 'id' and 'text_a_is_closer' (predicted).

    Returns:
        dict with 'accuracy', 'correct', 'total', and 'per_example' breakdown.
    """
    gold_map = {g["id"]: g["text_a_is_closer"] for g in gold}
    pred_map = {p["id"]: p["text_a_is_closer"] for p in predictions}

    correct = 0
    total = 0
    per_example = []

    for eid in sorted(gold_map.keys()):
        if eid not in pred_map:
            per_example.append({"id": eid, "gold": gold_map[eid], "pred": None, "correct": False})
            total += 1
            continue
        is_correct = gold_map[eid] == pred_map[eid]
        correct += int(is_correct)
        total += 1
        per_example.append({
            "id": eid,
            "gold": gold_map[eid],
            "pred": pred_map[eid],
            "correct": is_correct,
        })

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "per_example": per_example,
    }


def save_results(
    metrics: dict,
    experiment_name: str,
    run_config: dict,
    results_dir: Path,
) -> Path:
    """Persist evaluation results and run config to a timestamped folder.

    Directory layout:
        results/<experiment_name>/<timestamp>/
            metrics.json
            config.json

    Returns the path to the run directory.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        # exclude per_example from the summary file (can be large)
        summary = {k: v for k, v in metrics.items() if k != "per_example"}
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(run_dir / "per_example.json", "w", encoding="utf-8") as f:
        json.dump(metrics.get("per_example", []), f, indent=2, ensure_ascii=False)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False, default=str)

    return run_dir
