from __future__ import annotations

import importlib
from typing import Any

from lotus.models.lm import LM
from lotus.models.reranker import Reranker
from lotus.models.rm import RM

_LAZY_IMPORTS = {
    "CrossEncoderReranker": ("lotus.models.cross_encoder_reranker", "CrossEncoderReranker"),
    "LiteLLMRM": ("lotus.models.litellm_rm", "LiteLLMRM"),
    "SentenceTransformersRM": ("lotus.models.sentence_transformers_rm", "SentenceTransformersRM"),
    "ColBERTv2RM": ("lotus.models.colbertv2_rm", "ColBERTv2RM"),
}

__all__ = [
    "CrossEncoderReranker",
    "LM",
    "RM",
    "Reranker",
    "LiteLLMRM",
    "SentenceTransformersRM",
    "ColBERTv2RM",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
