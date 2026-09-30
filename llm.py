"""
One place for the LLM. Switch provider in .env, no code changes needed.

.env examples:
  LLM_PROVIDER=groq        GROQ_API_KEY=...          (free, very fast, cloud)
  LLM_PROVIDER=ollama                                (free, runs on your laptop, no key)
  LLM_PROVIDER=openrouter  OPENROUTER_API_KEY=...    (free ":free" models)
  LLM_PROVIDER=mistral     MISTRAL_API_KEY=...       (European provider, free tier)
  LLM_MODEL=...            (optional: override the default model)

All four speak the same "OpenAI-compatible" API, so one client works for all.
"""
import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROVIDERS = {
    #  name        base url                              key variable          default model
    "groq":       ("https://api.groq.com/openai/v1",   "GROQ_API_KEY",       "llama-3.3-70b-versatile"),
    "ollama":     ("http://localhost:11434/v1",        None,                 "qwen2.5:7b"),
    "openrouter": ("https://openrouter.ai/api/v1",     "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    "mistral":    ("https://api.mistral.ai/v1",        "MISTRAL_API_KEY",    "mistral-small-latest"),
}

PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
BASE_URL, KEY_VAR, DEFAULT_MODEL = PROVIDERS[PROVIDER]
MODEL = os.getenv("LLM_MODEL") or DEFAULT_MODEL


def get_client():
    key = os.environ[KEY_VAR] if KEY_VAR else "ollama"   # raises KeyError if the key is missing
    return OpenAI(base_url=BASE_URL, api_key=key)


def _parse(text):
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)       # grab the JSON object if there's extra text
        if match:
            return json.loads(match.group(0))
        raise


def ask_json(client, prompt, temperature=0):
    """Send a prompt, get a JSON object (dict) back."""
    messages = [{"role": "user", "content": prompt}]
    try:
        resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature,
                                              response_format={"type": "json_object"})
    except Exception as e:
        if "response_format" not in str(e) and "json" not in str(e).lower():
            raise
        resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature)
    return _parse(resp.choices[0].message.content)


if __name__ == "__main__":
    # quick test: python llm.py
    print(f"Provider: {PROVIDER}, model: {MODEL}")
    print(ask_json(get_client(), 'Return JSON: {"status": "ok", "greeting": "<say hi>"}'))
