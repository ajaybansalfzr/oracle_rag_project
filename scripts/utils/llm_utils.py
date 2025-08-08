# scripts/utils/llm_utils.py
import requests
from .. import config
from .utils import get_logger

logger = get_logger(__name__)

def llama3_call(prompt: str, system: str) -> str:
    """Performs a call to the Ollama API and returns the response."""
    try:
        res = requests.post(
            config.OLLAMA_URL,
            json={"model": config.LLM_MODEL, "prompt": prompt, "system": system, "stream": False},
            timeout=180 # Increased timeout for complex generation
        )
        res.raise_for_status()
        return res.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        return f"[LLM Error: Could not connect to the language model.]"