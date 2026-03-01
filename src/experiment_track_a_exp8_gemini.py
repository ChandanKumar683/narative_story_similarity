"""
Track-A  Experiment 8 (Gemini): Few-shot prompting (fixed 3 examples in prompt only).

Uses the Google Gemini API directly to test models like Gemini 3 Pro Preview.
Only the 3 labelled examples embedded in the prompt file are used.

Usage:
    python src/experiment_track_a_exp8_gemini.py \
        --model gemini-3-pro-preview \
        --split dev \
        [--temperature 0.0] \
        [--max_tokens 128] \
        [--limit 5]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

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


# ══════════════════════════════════════════════════════════════════════════
# PASTE YOUR GEMINI API KEY HERE
# ══════════════════════════════════════════════════════════════════════════
GEMINI_API_KEY = "AIzaSyA6OHAv8lyaS_GU32M400pq2T2s9zC982U"

SPLIT_MAP = {
    "sample": SAMPLE_TRACK_A,
    "dev": DEV_TRACK_A,
    "test": TEST_TRACK_A,
}

EXPERIMENT_NAME = "track_a_exp8_gemini"
MAX_RETRIES = 5
RETRY_BASE_DELAY = 5


def call_gemini(
    client: genai.Client,
    model_id: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
) -> str:
    """Send a prompt to the Gemini API and return the text response."""
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=1024),
    )
    if system_prompt:
        config.system_instruction = system_prompt

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=config,
            )
            # Debug: print raw response structure if text is empty
            try:
                if response.text:
                    return response.text.strip()
            except (ValueError, AttributeError):
                pass

            # Fallback: check candidates directly
            if response.candidates:
                candidate = response.candidates[0]
                # Check if blocked by safety filters
                if hasattr(candidate, "finish_reason"):
                    print(f"    DEBUG finish_reason: {candidate.finish_reason}")
                if hasattr(candidate, "content") and candidate.content:
                    parts = getattr(candidate.content, "parts", None) or []
                    for part in parts:
                        txt = getattr(part, "text", None)
                        if txt:
                            return txt.strip()

            # Check prompt feedback (safety block on the input)
            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                print(f"    DEBUG prompt_feedback: {response.prompt_feedback}")

            return ""
        except Exception as e:
            error_str = str(e).lower()
            if any(x in error_str for x in ["429", "resource exhausted", "rate"]):
                if attempt == MAX_RETRIES - 1:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    Rate limited (attempt {attempt+1}/{MAX_RETRIES}), waiting {delay}s...")
                time.sleep(delay)
            else:
                raise


def build_user_prompt(record: dict) -> str:
    """Build the user prompt for a single query."""
    return (
        f"### Anchor Story\n{record['anchor_text']}\n\n"
        f"### Text A\n{record['text_a']}\n\n"
        f"### Text B\n{record['text_b']}\n\n"
        f"Which text is more narratively similar to the anchor? Answer with ONLY \"A\" or \"B\"."
    )


def parse_response(raw: str) -> bool | None:
    """Convert LLM response to a boolean (True for A, False for B).

    Checks the last non-empty line first (model often puts reasoning before
    the final answer), then falls back to checking the start of the response.
    """
    if not raw:
        return None

    # Check the last non-empty line (where the answer usually is)
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1].strip().upper().rstrip(".,!? '\"")
        if last == "A" or last.startswith("A") and len(last) < 5:
            return True
        if last == "B" or last.startswith("B") and len(last) < 5:
            return False

    # Check the whole response for a standalone A or B
    cleaned = raw.strip().upper()

    # Look for "Answer: A" or "Answer: B" pattern anywhere
    for line in lines:
        upper = line.upper()
        if "ANSWER" in upper:
            if "A" in upper.split("ANSWER")[-1] and "B" not in upper.split("ANSWER")[-1]:
                return True
            if "B" in upper.split("ANSWER")[-1] and "A" not in upper.split("ANSWER")[-1]:
                return False

    # Last resort: check last character
    last_char = cleaned.rstrip(".,!? '\"")[-1] if cleaned else ""
    if last_char == "A":
        return True
    elif last_char == "B":
        return False

    # Absolute fallback: check start
    if cleaned.startswith("A"):
        return True
    elif cleaned.startswith("B"):
        return False

    return None


def run_experiment(
    model_id: str,
    split: str,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    limit: int | None = None,
) -> None:
    # ── Create Gemini client with timeout ─────────────────────────────────
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=120_000),
    )

    # ── Load evaluation data ───────────────────────────────────────────────
    data = load_track_a(SPLIT_MAP[split])
    if limit:
        data = data[:limit]

    # ── Load prompt (contains the 3 fixed few-shot examples) ──────────────
    prompt_path = PROMPTS_DIR / "few_shot.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt not found at {prompt_path}")
    system_prompt = prompt_path.read_text(encoding="utf-8")

    print(f"Loaded {len(data)} examples from '{split}' split.")
    print(f"Model: {model_id} (Gemini API direct)")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")
    print(f"Few-shot examples: 3 (fixed in prompt file)")
    print("-" * 60)

    # ── Run inference ─────────────────────────────────────────────────────
    predictions = []
    unparseable = 0

    for i, record in enumerate(data):
        user_prompt = build_user_prompt(record)

        try:
            raw_response = call_gemini(
                client=client,
                model_id=model_id,
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
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

        if (i + 1) % 10 == 0 or (i + 1) == len(data):
            print(f"  [{i+1}/{len(data)}] processed.")

    # ── Save predictions ──────────────────────────────────────────────────
    model_key = model_id.replace("/", "-")
    run_config = {
        "experiment": EXPERIMENT_NAME,
        "model_id": model_id,
        "provider": "google",
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
        description="Track-A Exp8 (Gemini): Few-shot prompting with Gemini 3 Pro"
    )
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview",
                        help="Gemini model ID (default: gemini-3-pro-preview)")
    parser.add_argument("--split", type=str, default="dev", choices=["sample", "dev", "test"],
                        help="Data split to run on")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=8192)
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
