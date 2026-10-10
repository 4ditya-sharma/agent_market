"""Print the models your API keys can use, so you can set MASTER_MODEL / WORKER_MODEL / JUDGE_MODELS in .env.
    python -m scripts.list_models
"""
import os

from openai import OpenAI

from ai_layer import config

for provider, cfg in config.PROVIDERS.items():
    key = os.getenv(cfg["key_env"]) if cfg["key_env"] else None
    if not key:
        continue
    print(f"\n== {provider} ==")
    try:
        models = OpenAI(base_url=cfg["base_url"], api_key=key).models.list()
        names = sorted(m.id.replace("models/", "") for m in models.data)
        for n in names:
            if any(k in n.lower() for k in ("llama", "gemma", "qwen", "mixtral", "gemini", "deepseek", "gpt")):
                print(f"  {provider}:{n}")
    except Exception as e:
        print(f"  failed: {str(e)[:200]}")
