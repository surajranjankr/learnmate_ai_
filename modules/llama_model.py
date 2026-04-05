from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Optional

from learnmate_ai.config import get_config

try:
    from llama_cpp import Llama
except Exception:
    Llama = None


_llm_instance = None
_llm_load_error: Optional[str] = None
_llm_lock = Lock()


def get_model_path() -> Path:
    configured = Path(get_config().model_path)
    if configured.exists():
        return configured

    models_dir = configured.parent
    if models_dir.exists():
        candidates = sorted(models_dir.glob("*.gguf"))
        if candidates:
            return candidates[0]
    return configured


def load_llm():
    global _llm_instance, _llm_load_error

    if _llm_instance is not None:
        return _llm_instance

    if Llama is None:
        _llm_load_error = "llama-cpp-python is not installed in the active environment."
        return None

    model_path = get_model_path()
    if not model_path.exists():
        _llm_load_error = f"Model file not found: {model_path}"
        return None

    with _llm_lock:
        if _llm_instance is not None:
            return _llm_instance
        try:
            _llm_instance = Llama(
                model_path=str(model_path),
                n_ctx=4096,
                n_threads=8,
                n_gpu_layers=20,
                verbose=False,
            )
            _llm_load_error = None
            return _llm_instance
        except Exception as exc:
            _llm_load_error = str(exc)
            return None


def llm_is_available() -> bool:
    return load_llm() is not None


def get_llm_status() -> dict[str, str | bool]:
    model_path = get_model_path()
    return {
        "llama_cpp_available": Llama is not None,
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "ready": llm_is_available(),
        "error": _llm_load_error or "",
    }


def generate_llm_response(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    llm = load_llm()
    if llm is None:
        raise RuntimeError(_llm_load_error or "Local LLM is unavailable.")

    output = llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
    return output["choices"][0]["text"].strip()
