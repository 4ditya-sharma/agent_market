"""All settings come from environment variables (or a .env file)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Provider endpoints. All are OpenAI-compatible, so one client covers them.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
    },
    "ollama": {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "key_env": None,  # Ollama needs no key
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
}

# Model specs are "provider:model". Check provider dashboards for current free model names.
MASTER_MODEL = os.getenv("MASTER_MODEL", "groq:llama-3.3-70b-versatile")
WORKER_MODEL = os.getenv("WORKER_MODEL", "groq:llama-3.1-8b-instant")
JUDGE_MODELS = [
    m.strip()
    for m in os.getenv(
        "JUDGE_MODELS",
        "groq:llama-3.3-70b-versatile,gemini:gemini-2.0-flash,ollama:qwen2.5:7b",
    ).split(",")
    if m.strip()
]

# "mock" forces offline mode. "auto" uses real models when keys exist, mock otherwise.
LLM_MODE = os.getenv("LLM_MODE", "auto").lower()

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Firewall: score at or above this is quarantined.
INJECTION_THRESHOLD = float(os.getenv("INJECTION_THRESHOLD", "0.6"))
# Judge: a subtask passes if the panel's average score is at or above this (0-10).
JUDGE_PASS_SCORE = float(os.getenv("JUDGE_PASS_SCORE", "6"))
