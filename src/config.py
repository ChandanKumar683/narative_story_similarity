"""
Configuration for SemEval-2026 Task 4: Narrative Story Similarity experiments.
"""

from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT

SAMPLE_TRACK_A = DATA_DIR / "SemEval2026-Task_4-sample-v1" / "sample_track_a.jsonl"
DEV_TRACK_A    = DATA_DIR / "SemEval2026-Task_4-dev-v1 (1)" / "dev_track_a.jsonl"
TEST_TRACK_A   = DATA_DIR / "SemEval2026-Task_4-test-v1"   / "test_track_a.jsonl"

RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# ── OpenRouter configuration ──────────────────────────────────────────────
OPENROUTER_API_KEY= "API_KEY_HERE"  # Replace with your actual OpenRouter API key
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Model registry (all via OpenRouter) ───────────────────────────────────
MODELS = {
    # OpenAI
    "gpt-4o":           {"model_id": "openai/gpt-4o"},
    "gpt-4o-mini":      {"model_id": "openai/gpt-4o-mini"},
    # Anthropic
    "claude-sonnet":    {"model_id": "anthropic/claude-sonnet-4-20250514"},
    "claude-opus":      {"model_id": "anthropic/claude-opus-4-20250514"},
    # Google
    "gemini-1.5-pro":   {"model_id": "google/gemini-pro-1.5"},
    "gemini-2.0-flash":  {"model_id": "google/gemini-2.0-flash-001"},
    # Open-source
    "llama-3.3-70b":     {"model_id": "meta-llama/llama-3.3-70b-instruct"},
    "llama-3.1-70b":     {"model_id": "meta-llama/llama-3.1-70b-instruct"},
    "mistral-7b":       {"model_id": "mistralai/mistral-7b-instruct"},
    "qwen-2.5-7b":      {"model_id": "qwen/qwen-2.5-7b-instruct"},
    "qwen-2.5-72b":      {"model_id": "qwen/qwen-2.5-72b-instruct"},
    "qwen-3-4b":        {"model_id": "qwen/qwen3-4b:free"},
    "qwen-3-32b":        {"model_id": "qwen/qwen3-32b"},
    "qwen-3-80b":        {"model_id": "qwen/qwen3-next-80b-a3b-instruct"},
    "qwen-3-235b":        {"model_id": "qwen/qwen3-235b-a22b-2507"},
}
