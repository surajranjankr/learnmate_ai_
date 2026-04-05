"""Public module exports for the LearnMate application with lazy imports."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "analytics",
    "chatbot_rag",
    "llama_model",
    "quiz_generator",
    "summarizer",
    "utils",
    "vectorstore",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"modules.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'modules' has no attribute {name!r}")
