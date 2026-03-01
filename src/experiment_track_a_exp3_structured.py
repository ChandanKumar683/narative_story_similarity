"""
Track-A  Experiment 3: Structured narrative feature extraction + comparison.

Two-stage pipeline:
  Stage 1 — Extract structured features (theme, events, outcome) for each of
             the three stories independently.
  Stage 2 — Compare the extracted features to decide which text is closer.

Usage:
    python src/experiment_track_a_exp3_structured.py \
        --model gpt-4o-mini \
        --split dev \
        [--temperature 0.0] \
        [--extract_max_tokens 512] \
        [--compare_max_tokens 1024] \
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

EXPERIMENT_NAME = "track_a_exp3_structured"


# ── Stage 1: Feature extraction ───────────────────────────────────────────

def build_extract_prompt(story_text: str) -> str:
    """Build the user prompt for extracting narrative features from one story."""
    return f"### Story\n{story_text}"


def extract_features(
    story_text: str,
    model_id: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call the LLM to extract structured features for a single story.

    Returns the raw extraction text (THEME / EVENTS / OUTCOME block).
    """
    return call_llm(
        model_id=model_id,
        prompt=build_extract_prompt(story_text),
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ── Stage 2: Comparison ──────────────────────────────────────────────────

def build_compare_prompt(
    anchor_features: str,
    text_a_features: str,
    text_b_features: str,
) -> str:
    """Build the user prompt for comparing extracted features."""
    return (
        f"### Anchor Story Features\n{anchor_features}\n\n"
        f"### Text A Features\n{text_a_features}\n\n"
        f"### Text B Features\n{text_b_features}\n\n"
        f"Which text's narrative features are more similar to the Anchor's? "
        f"Compare each component, then give your final answer as ANSWER: A or ANSWER: B"
    )


def parse_response(raw: str) -> bool | None:
    """Extract the final ANSWER: A/B from a comparison response."""
    matches = re.findall(r"ANSWER\s*:\s*([AB])", raw, re.IGNORECASE)
    if matches:
        return matches[-1].upper() == "A"

    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1].upper()
        if last in ("A", "B"):
            return last == "A"

    return None


# ── Main experiment loop ─────────────────────────────────────────────────

def run_experiment(
    model_key: str,
    split: str,
    temperature: float = 0.0,
    extract_max_tokens: int = 512,
    compare_max_tokens: int = 1024,
    limit: int | None = None,
) -> None:
    # ── Load data ──────────────────────────────────────────────────────────
    data = load_track_a(SPLIT_MAP[split])
    if limit:
        data = data[:limit]
    print(f"Loaded {len(data)} examples from '{split}' split.")

    # ── Load prompts ──────────────────────────────────────────────────────
    extract_system = (PROMPTS_DIR / "structured_extract.txt").read_text(encoding="utf-8")
    compare_system = (PROMPTS_DIR / "structured_compare.txt").read_text(encoding="utf-8")

    # ── Resolve model ─────────────────────────────────────────────────────
    model_cfg = MODELS[model_key]
    model_id = model_cfg["model_id"]

    print(f"Model: {model_key} (openrouter: {model_id})")
    print(f"Temperature: {temperature}")
    print(f"Extract max tokens: {extract_max_tokens}, Compare max tokens: {compare_max_tokens}")
    print("-" * 60)

    # ── Run inference ─────────────────────────────────────────────────────
    predictions = []
    unparseable = 0

    for i, record in enumerate(data):
        stage1_ok = True

        # Stage 1: extract features for all three stories
        try:
            anchor_feat = extract_features(
                record["anchor_text"], model_id, extract_system,
                temperature, extract_max_tokens,
            )
            text_a_feat = extract_features(
                record["text_a"], model_id, extract_system,
                temperature, extract_max_tokens,
            )
            text_b_feat = extract_features(
                record["text_b"], model_id, extract_system,
                temperature, extract_max_tokens,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(data)}] EXTRACTION ERROR on id={record['id']}: {e}")
            anchor_feat = text_a_feat = text_b_feat = ""
            stage1_ok = False

        # Stage 2: compare extracted features
        compare_prompt = build_compare_prompt(anchor_feat, text_a_feat, text_b_feat)
        try:
            raw_response = call_llm(
                model_id=model_id,
                prompt=compare_prompt,
                system_prompt=compare_system,
                temperature=temperature,
                max_tokens=compare_max_tokens,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(data)}] COMPARISON ERROR on id={record['id']}: {e}")
            raw_response = ""

        predicted = parse_response(raw_response)
        if predicted is None:
            unparseable += 1
            snippet = raw_response[-200:] if len(raw_response) > 200 else raw_response
            print(f"  [{i+1}/{len(data)}] UNPARSEABLE (tail): '{snippet}'")
            predicted = True  # fallback

        predictions.append({
            "id": record["id"],
            "text_a_is_closer": predicted,
            "anchor_features": anchor_feat,
            "text_a_features": text_a_feat,
            "text_b_features": text_b_feat,
            "raw_comparison": raw_response,
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
        "extract_max_tokens": extract_max_tokens,
        "compare_max_tokens": compare_max_tokens,
        "num_examples": len(data),
        "limit": limit,
        "unparseable_count": unparseable,
        "prompt_files": [
            "prompts/structured_extract.txt",
            "prompts/structured_compare.txt",
        ],
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
        description="Track-A Exp3: Structured feature extraction + comparison"
    )
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                        help="Model key from config.MODELS")
    parser.add_argument("--split", type=str, default="dev", choices=["sample", "dev", "test"],
                        help="Data split to run on")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--extract_max_tokens", type=int, default=512,
                        help="Max tokens for each feature extraction call")
    parser.add_argument("--compare_max_tokens", type=int, default=1024,
                        help="Max tokens for the comparison call")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N examples (for quick tests)")

    args = parser.parse_args()
    run_experiment(
        model_key=args.model,
        split=args.split,
        temperature=args.temperature,
        extract_max_tokens=args.extract_max_tokens,
        compare_max_tokens=args.compare_max_tokens,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
