import os
import requests
from typing import Optional

def ollama_chat(prompt: str, model: Optional[str] = None, timeout: int = 30) -> str:
    """
    Call a local Ollama model and return its response text.
    Expects Ollama running at http://localhost:11434 and the model pulled.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = model or os.getenv("LLM_MODEL_NAME", "mistral")

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()
    except requests.exceptions.ConnectionError:
        return "[LLM error: Could not connect to Ollama at {}. Is it running?]".format(base_url)
    except requests.exceptions.Timeout:
        return "[LLM error: Request to Ollama timed out]"
    except requests.exceptions.RequestException as e:
        return f"[LLM error: {e}]"
    except Exception as e:
        return f"[LLM error: {e}]"
