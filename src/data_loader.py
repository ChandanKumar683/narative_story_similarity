"""
Data loading utilities for SemEval-2026 Task 4.
"""

import json
from pathlib import Path
from typing import Optional


def load_track_a(filepath: Path) -> list[dict]:
    """Load a Track-A JSONL file.

    Each record contains:
        - anchor_text (str)
        - text_a (str)
        - text_b (str)
        - text_a_is_closer (bool)   [absent in test split]

    Returns a list of dicts, each augmented with an 'id' field (0-indexed).
    """
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            record = json.loads(line.strip())
            record["id"] = idx
            records.append(record)
    return records


def save_predictions_track_a(predictions: list[dict], filepath: Path) -> None:
    """Save Track-A predictions as JSONL (one object per line).

    Each prediction dict should have at least:
        - id (int)
        - text_a_is_closer (bool)       — the predicted label
    Optionally:
        - raw_response (str)            — the raw LLM output
        - confidence (float)            — model confidence if available
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


def load_predictions_track_a(filepath: Path) -> list[dict]:
    """Load saved predictions from a JSONL file."""
    preds = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            preds.append(json.loads(line.strip()))
    return preds
