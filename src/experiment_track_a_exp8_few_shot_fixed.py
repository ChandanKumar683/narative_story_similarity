"""
Track-A  Experiment 8: Few-shot prompting (fixed 3 examples in prompt only).

Uses only the 3 labelled examples embedded in the prompt file (prompts/few_shot.txt).
No additional few-shot examples are injected from the sample split.

Usage:
    python src/experiment_track_a_exp8_few_shot_fixed.py \
        --model qwen-2.5-7b \
        --split dev \
        [--temperature 0.0] \
        [--max_tokens 16] \
        [--limit 5]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    MODELS,
    DEV_TRACK_A,
    TEST_TRACK_A,
    SAMPLE_TRACK_A,
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

EXPERIMENT_NAME = "track_a_exp8_few_shot_fixed"


def build_user_prompt(record: dict) -> str:
    """Build the user prompt for a single query (no extra shots added)."""
    return (
        f"### Anchor Story\n{record['anchor_text']}\n\n"
        f"### Text A\n{record['text_a']}\n\n"
        f"### Text B\n{record['text_b']}\n\n"
        f"Which text is more narratively similar to the anchor? Answer with ONLY \"A\" or \"B\"."
    )


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
    temperature: float = 0.0,
    max_tokens: int = 16,
    limit: int | None = None,
) -> None:
    # ── Load evaluation data ───────────────────────────────────────────────
    data = load_track_a(SPLIT_MAP[split])

    if limit:
        data = data[:limit]
    print(f"Loaded {len(data)} examples from '{split}' split.")

    # ── Load prompt (contains the 3 fixed few-shot examples) ──────────────
    system_prompt = (PROMPTS_DIR / "few_shot.txt").read_text(encoding="utf-8")

    # ── Resolve model ─────────────────────────────────────────────────────
    model_cfg = MODELS[model_key]
    model_id = model_cfg["model_id"]

    print(f"Model: {model_key} (openrouter: {model_id})")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")
    print(f"Few-shot examples: 3 (fixed in prompt file)")
    print("-" * 60)

    # ── Run inference ─────────────────────────────────────────────────────
    predictions = []
    unparseable = 0

    for i, record in enumerate(data):
        user_prompt = build_user_prompt(record)

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
        "num_shots": 3,
        "shot_source": "prompts/few_shot.txt (fixed)",
        "num_examples": len(data),
        "limit": limit,
        "unparseable_count": unparseable,
        "prompt_file": "prompts/few_shot.txt",
    }

    pred_dir = RESULTS_DIR / EXPERIMENT_NAME / f"{model_key}_{split}_3shot"
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
        description="Track-A Exp8: Few-shot prompting (3 fixed examples in prompt)"
    )
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                        help="Model key from config.MODELS")
    parser.add_argument("--split", type=str, default="dev", choices=["sample", "dev", "test"],
                        help="Data split to run on")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N examples (for quick tests)")

    args = parser.parse_args()
    run_experiment(
        model_key=args.model,
        split=args.split,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
