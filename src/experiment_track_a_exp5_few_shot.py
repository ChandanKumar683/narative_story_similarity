"""
Track-A  Experiment 5: Few-shot prompting.

Provides 3-5 labelled examples from the sample split in the prompt so the LLM
can learn the task pattern before answering.  Examples are selected to be
label-balanced (mix of A and B answers).

Usage:
    python src/experiment_track_a_exp5_few_shot.py \
        --model gpt-4o-mini \
        --split dev \
        [--num_shots 5] \
        [--temperature 0.0] \
        [--max_tokens 16] \
        [--limit 5]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    MODELS,
    SAMPLE_TRACK_A,
    DEV_TRACK_A,
    TEST_TRACK_A,
    RESULTS_DIR,
    PROMPTS_DIR,
)
from src.data_loader import load_track_a, save_predictions_track_a
from src.evaluation import evaluate_track_a, save_results
from src.llm_client import call_llm


SPLIT_MAP = {
    "sample": SAMPLE_TRACK_A,
    "dev": DEV_TRACK_A,
    "test": TEST_TRACK_A,
}

EXPERIMENT_NAME = "track_a_exp5_few_shot"

# Fixed seed for reproducible example selection
SEED = 42

# These sample indices are used as few-shot examples.
# Pre-selected for label balance and narrative diversity.
# Indices into the sample split (0-indexed).
DEFAULT_SHOT_INDICES = {
    3: [0, 1, 4],           # 2 × B, 1 × A
    5: [0, 1, 3, 4, 9],     # 2 × B, 3 × A
}


def select_shots(sample_data: list[dict], num_shots: int) -> list[dict]:
    """Select label-balanced few-shot examples from the sample split.

    Uses pre-selected indices for 3 and 5 shots, otherwise randomly samples
    with balanced labels.
    """
    if num_shots in DEFAULT_SHOT_INDICES:
        indices = DEFAULT_SHOT_INDICES[num_shots]
        return [sample_data[i] for i in indices]

    # Random balanced selection
    positives = [r for r in sample_data if r.get("text_a_is_closer") is True]
    negatives = [r for r in sample_data if r.get("text_a_is_closer") is False]

    rng = random.Random(SEED)
    n_pos = num_shots // 2
    n_neg = num_shots - n_pos

    rng.shuffle(positives)
    rng.shuffle(negatives)

    shots = positives[:n_pos] + negatives[:n_neg]
    rng.shuffle(shots)
    return shots


def format_example(record: dict, label: bool) -> str:
    """Format a single labelled example for the few-shot prompt."""
    answer = "A" if label else "B"
    return (
        f"### Anchor Story\n{record['anchor_text']}\n\n"
        f"### Text A\n{record['text_a']}\n\n"
        f"### Text B\n{record['text_b']}\n\n"
        f"Answer: {answer}"
    )


def build_user_prompt(record: dict, shots: list[dict]) -> str:
    """Build the full user prompt with few-shot examples followed by the query."""
    parts = []

    # Few-shot examples
    for idx, shot in enumerate(shots, 1):
        parts.append(f"--- Example {idx} ---")
        parts.append(format_example(shot, shot["text_a_is_closer"]))
        parts.append("")

    # Query
    parts.append(f"--- Now classify this ---")
    parts.append(
        f"### Anchor Story\n{record['anchor_text']}\n\n"
        f"### Text A\n{record['text_a']}\n\n"
        f"### Text B\n{record['text_b']}\n\n"
        f"Which text is more narratively similar to the anchor? Answer with ONLY \"A\" or \"B\"."
    )

    return "\n".join(parts)


def parse_response(raw: str) -> bool | None:
    """Convert LLM response to a boolean (text_a_is_closer)."""
    cleaned = raw.strip().upper()
    if cleaned.startswith("A"):
        return True
    elif cleaned.startswith("B"):
        return False
    return None


def run_experiment(
    model_key: str,
    split: str,
    num_shots: int = 5,
    temperature: float = 0.0,
    max_tokens: int = 16,
    limit: int | None = None,
) -> None:
    # ── Load sample data for few-shot examples ─────────────────────────────
    sample_data = load_track_a(SAMPLE_TRACK_A)
    shots = select_shots(sample_data, num_shots)
    shot_ids = {s["id"] for s in shots}

    print(f"Selected {len(shots)} few-shot examples (ids: {sorted(shot_ids)})")
    label_dist = sum(1 for s in shots if s["text_a_is_closer"])
    print(f"  Label balance: {label_dist} × A, {len(shots) - label_dist} × B")

    # ── Load evaluation data ───────────────────────────────────────────────
    data = load_track_a(SPLIT_MAP[split])

    # If evaluating on sample split, exclude the few-shot examples
    if split == "sample":
        data = [r for r in data if r["id"] not in shot_ids]
        print(f"  (Excluded {len(shot_ids)} shot examples from sample eval)")

    if limit:
        data = data[:limit]
    print(f"Loaded {len(data)} examples from '{split}' split.")

    # ── Load prompt ────────────────────────────────────────────────────────
    system_prompt = (PROMPTS_DIR / "few_shot.txt").read_text(encoding="utf-8")

    # ── Resolve model ─────────────────────────────────────────────────────
    model_cfg = MODELS[model_key]
    model_id = model_cfg["model_id"]

    print(f"Model: {model_key} (openrouter: {model_id})")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}, Shots: {num_shots}")
    print("-" * 60)

    # ── Run inference ─────────────────────────────────────────────────────
    predictions = []
    unparseable = 0

    for i, record in enumerate(data):
        user_prompt = build_user_prompt(record, shots)

        try:
            raw_response = call_llm(
                model_id=model_id,
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(data)}] ERROR on id={record['id']}: {e}")
            raw_response = ""

        predicted = parse_response(raw_response)
        if predicted is None:
            unparseable += 1
            print(f"  [{i+1}/{len(data)}] UNPARSEABLE: '{raw_response}'")
            predicted = True  # fallback

        predictions.append({
            "id": record["id"],
            "text_a_is_closer": predicted,
            "raw_response": raw_response,
        })

        # progress
        if (i + 1) % 10 == 0 or (i + 1) == len(data):
            print(f"  [{i+1}/{len(data)}] processed.")

    # ── Save predictions ──────────────────────────────────────────────────
    run_config = {
        "experiment": EXPERIMENT_NAME,
        "model_key": model_key,
        "provider": "openrouter",
        "model_id": model_id,
        "split": split,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "num_shots": num_shots,
        "shot_ids": sorted(shot_ids),
        "num_examples": len(data),
        "limit": limit,
        "unparseable_count": unparseable,
        "prompt_file": "prompts/few_shot.txt",
    }

    pred_dir = RESULTS_DIR / EXPERIMENT_NAME / f"{model_key}_{split}_{num_shots}shot"
    pred_dir.mkdir(parents=True, exist_ok=True)
    save_predictions_track_a(predictions, pred_dir / "predictions.jsonl")

    # ── Evaluate ──────────────────────────────────────────────────────────
    if split != "test":
        metrics = evaluate_track_a(data, predictions)
        run_dir = save_results(metrics, EXPERIMENT_NAME, run_config, RESULTS_DIR)
        print("=" * 60)
        print(f"Accuracy: {metrics['accuracy']:.4f}  ({metrics['correct']}/{metrics['total']})")
        print(f"Unparseable responses: {unparseable}")
        print(f"Results saved to: {run_dir}")
    else:
        submission_path = pred_dir / "track_a.jsonl"
        submission_preds = [{"text_a_is_closer": p["text_a_is_closer"]} for p in predictions]
        save_predictions_track_a(submission_preds, submission_path)
        print("=" * 60)
        print(f"Test predictions saved to: {submission_path}")
        print(f"Unparseable responses: {unparseable}")


def main():
    parser = argparse.ArgumentParser(
        description="Track-A Exp5: Few-shot prompting"
    )
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                        help="Model key from config.MODELS")
    parser.add_argument("--split", type=str, default="dev", choices=["sample", "dev", "test"],
                        help="Data split to run on")
    parser.add_argument("--num_shots", type=int, default=5, choices=[3, 4, 5],
                        help="Number of few-shot examples (3-5)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N examples (for quick tests)")

    args = parser.parse_args()
    run_experiment(
        model_key=args.model,
        split=args.split,
        num_shots=args.num_shots,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
