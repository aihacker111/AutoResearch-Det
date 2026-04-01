import json
import time
import urllib.request
import urllib.error
from config import Config

class LLMClient:
    def __init__(self, model: str):
        self.model = model
        self.api_key = Config.OPENROUTER_API_KEY

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = json.dumps({
            "model": self.model,
            "max_tokens": Config.LLM_MAX_TOKENS,
            "messages": messages,
        }).encode()
        
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        last_exc = None
        for attempt in range(1, Config.LLM_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    body = json.loads(r.read())
                    content = body["choices"][0]["message"].get("content")
                    if not content:
                        reason = body["choices"][0].get("finish_reason", "unknown")
                        raise ValueError(f"API returned empty content (finish_reason={reason!r})")
                    return content.strip()
            except (urllib.error.URLError, KeyError, json.JSONDecodeError, ValueError) as e:
                last_exc = e
                if attempt < Config.LLM_RETRIES:
                    print(f"  [LLM] attempt {attempt} failed ({e}), retrying in {Config.LLM_RETRY_WAIT}s...")
                    time.sleep(Config.LLM_RETRY_WAIT * attempt)
                    
        raise RuntimeError(f"LLM API failed after {Config.LLM_RETRIES} attempts: {last_exc}")