"""
Track-A  Experiment 8 (OpenAI): Few-shot prompting (fixed 3 examples in prompt only).

Uses the OpenAI API directly (not OpenRouter) to test models like GPT-5.1.
Only the 3 labelled examples embedded in the prompt file are used.

Usage:
    python src/experiment_track_a_exp8_openai.py \
        --model gpt-5.1 \
        --split dev \
        [--temperature 0.0] \
        [--max_tokens 16] \
        [--limit 5]

Set your OpenAI API key via environment variable:
    export OPENAI_API_KEY="sk-..."       (Linux/Mac)
    set OPENAI_API_KEY=sk-...            (Windows)
"""

import argparse
import json
import sys
import time
from pathlib import Path

from openai import OpenAI, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DEV_TRACK_A,
    TEST_TRACK_A,
    SAMPLE_TRACK_A,
    RESULTS_DIR,
    PROMPTS_DIR,
)
from src.data_loader import load_track_a, save_predictions_track_a
from src.evaluation import evaluate_track_a, save_results


SPLIT_MAP = {
    "sample": SAMPLE_TRACK_A,
    "dev": DEV_TRACK_A,
    "test": TEST_TRACK_A,
}

EXPERIMENT_NAME = "track_a_exp8_openai"

MAX_RETRIES = 5
RETRY_BASE_DELAY = 5


def call_openai(
    client: OpenAI,
    model_id: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 16,
) -> str:
    """Send a prompt to the OpenAI API and return the text response."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"    Rate limited (attempt {attempt+1}/{MAX_RETRIES}), waiting {delay}s...")
            time.sleep(delay)
        except Exception:
            raise


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
    model_id: str,
    split: str,
    temperature: float = 0.0,
    max_tokens: int = 16,
    limit: int | None = None,
) -> None:
    # ── API key (hardcoded or from environment) ─────────────────────────
    api_key = "YOUR_OPENAI_API_KEY_HERE"  # <-- Replace with your actual OpenAI key
    client = OpenAI(api_key=api_key)

    # ── Load evaluation data ───────────────────────────────────────────────
    data = load_track_a(SPLIT_MAP[split])

    if limit:
        data = data[:limit]
    print(f"Loaded {len(data)} examples from '{split}' split.")

    # ── Load prompt (contains the 3 fixed few-shot examples) ──────────────
    system_prompt = (PROMPTS_DIR / "few_shot.txt").read_text(encoding="utf-8")

    print(f"Model: {model_id} (OpenAI API direct)")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")
    print(f"Few-shot examples: 3 (fixed in prompt file)")
    print("-" * 60)

    # ── Run inference ─────────────────────────────────────────────────────
    predictions = []
    unparseable = 0

    for i, record in enumerate(data):
        user_prompt = build_user_prompt(record)

        try:
            raw_response = call_openai(
                client=client,
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
    model_key = model_id.replace("/", "-")
    run_config = {
        "experiment": EXPERIMENT_NAME,
        "model_id": model_id,
        "provider": "openai",
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
        description="Track-A Exp8 (OpenAI): Few-shot prompting with GPT-5.1"
    )
    parser.add_argument("--model", type=str, default="gpt-5.1",
                        help="OpenAI model ID (default: gpt-5.1)")
    parser.add_argument("--split", type=str, default="dev", choices=["sample", "dev", "test"],
                        help="Data split to run on")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N examples (for quick tests)")

    args = parser.parse_args()
    run_experiment(
        model_id=args.model,
        split=args.split,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
