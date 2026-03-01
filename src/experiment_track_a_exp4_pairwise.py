"""
Track-A  Experiment 4: Pairwise scoring.

The LLM scores anchor-vs-text_a and anchor-vs-text_b narrative similarity
independently (1-10), then the higher score determines the answer.
This avoids positional bias from showing both candidates together.

Usage:
    python src/experiment_track_a_exp4_pairwise.py \
        --model gpt-4o-mini \
        --split dev \
        [--temperature 0.0] \
        [--max_tokens 512] \
        [--limit 5]
"""

import argparse
import json
import re
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

EXPERIMENT_NAME = "track_a_exp4_pairwise"


# ── Scoring ───────────────────────────────────────────────────────────────

def build_score_prompt(anchor_text: str, candidate_text: str) -> str:
    """Build the user prompt for scoring one anchor-candidate pair."""
    return (
        f"### Reference Story\n{anchor_text}\n\n"
        f"### Candidate Story\n{candidate_text}\n\n"
        f"Rate the narrative similarity between the Reference and the Candidate (1-10). "
        f"Give a brief justification, then end with SCORE: <number>"
    )


def parse_score(raw: str) -> float | None:
    """Extract a numeric score from the LLM response.

    Looks for the last SCORE: <number> pattern.
    Returns None if unparseable.
    """
    matches = re.findall(r"SCORE\s*:\s*(\d+(?:\.\d+)?)", raw, re.IGNORECASE)
    if matches:
        val = float(matches[-1])
        return max(1.0, min(10.0, val))  # clamp to [1, 10]

    # Fallback: look for a standalone number on the last line
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if lines:
        last_match = re.search(r"(\d+(?:\.\d+)?)", lines[-1])
        if last_match:
            val = float(last_match.group(1))
            if 1 <= val <= 10:
                return val

    return None


# ── Main experiment loop ─────────────────────────────────────────────────

def run_experiment(
    model_key: str,
    split: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    limit: int | None = None,
) -> None:
    # ── Load data ──────────────────────────────────────────────────────────
    data = load_track_a(SPLIT_MAP[split])
    if limit:
        data = data[:limit]
    print(f"Loaded {len(data)} examples from '{split}' split.")

    # ── Load prompt ────────────────────────────────────────────────────────
    system_prompt = (PROMPTS_DIR / "pairwise_score.txt").read_text(encoding="utf-8")

    # ── Resolve model ─────────────────────────────────────────────────────
    model_cfg = MODELS[model_key]
    model_id = model_cfg["model_id"]

    print(f"Model: {model_key} (openrouter: {model_id})")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")
    print("-" * 60)

    # ── Run inference ─────────────────────────────────────────────────────
    predictions = []
    unparseable = 0
    ties = 0

    for i, record in enumerate(data):
        score_a = None
        score_b = None
        raw_a = ""
        raw_b = ""

        # Score anchor vs text_a
        try:
            raw_a = call_llm(
                model_id=model_id,
                prompt=build_score_prompt(record["anchor_text"], record["text_a"]),
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            score_a = parse_score(raw_a)
        except Exception as e:
            print(f"  [{i+1}/{len(data)}] ERROR scoring text_a, id={record['id']}: {e}")

        # Score anchor vs text_b
        try:
            raw_b = call_llm(
                model_id=model_id,
                prompt=build_score_prompt(record["anchor_text"], record["text_b"]),
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            score_b = parse_score(raw_b)
        except Exception as e:
            print(f"  [{i+1}/{len(data)}] ERROR scoring text_b, id={record['id']}: {e}")

        # Handle parse failures
        if score_a is None or score_b is None:
            unparseable += 1
            if score_a is None:
                snippet = raw_a[-150:] if len(raw_a) > 150 else raw_a
                print(f"  [{i+1}/{len(data)}] UNPARSEABLE score_a: '{snippet}'")
                score_a = 5.0  # neutral fallback
            if score_b is None:
                snippet = raw_b[-150:] if len(raw_b) > 150 else raw_b
                print(f"  [{i+1}/{len(data)}] UNPARSEABLE score_b: '{snippet}'")
                score_b = 5.0  # neutral fallback

        # Decide: higher score = more similar to anchor
        if score_a == score_b:
            ties += 1
            predicted = True  # default to A on ties
        else:
            predicted = score_a > score_b

        predictions.append({
            "id": record["id"],
            "text_a_is_closer": predicted,
            "score_a": score_a,
            "score_b": score_b,
            "raw_response_a": raw_a,
            "raw_response_b": raw_b,
        })

        # progress
        if (i + 1) % 10 == 0 or (i + 1) == len(data):
            print(f"  [{i+1}/{len(data)}] processed.  (scores: A={score_a}, B={score_b})")

    # ── Save predictions ──────────────────────────────────────────────────
    run_config = {
        "experiment": EXPERIMENT_NAME,
        "model_key": model_key,
        "provider": "openrouter",
        "model_id": model_id,
        "split": split,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "num_examples": len(data),
        "limit": limit,
        "unparseable_count": unparseable,
        "tie_count": ties,
        "prompt_file": "prompts/pairwise_score.txt",
    }

    pred_dir = RESULTS_DIR / EXPERIMENT_NAME / f"{model_key}_{split}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    save_predictions_track_a(predictions, pred_dir / "predictions.jsonl")

    # ── Evaluate ──────────────────────────────────────────────────────────
    if split != "test":
        metrics = evaluate_track_a(data, predictions)
        run_dir = save_results(metrics, EXPERIMENT_NAME, run_config, RESULTS_DIR)
        print("=" * 60)
        print(f"Accuracy: {metrics['accuracy']:.4f}  ({metrics['correct']}/{metrics['total']})")
        print(f"Unparseable responses: {unparseable}")
        print(f"Ties (defaulted to A): {ties}")
        print(f"Results saved to: {run_dir}")
    else:
        submission_path = pred_dir / "track_a.jsonl"
        submission_preds = [{"text_a_is_closer": p["text_a_is_closer"]} for p in predictions]
        save_predictions_track_a(submission_preds, submission_path)
        print("=" * 60)
        print(f"Test predictions saved to: {submission_path}")
        print(f"Unparseable responses: {unparseable}")
        print(f"Ties (defaulted to A): {ties}")


def main():
    parser = argparse.ArgumentParser(
        description="Track-A Exp4: Pairwise scoring (1-10)"
    )
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                        help="Model key from config.MODELS")
    parser.add_argument("--split", type=str, default="dev", choices=["sample", "dev", "test"],
                        help="Data split to run on")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens per scoring call")
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
