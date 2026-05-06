"""Tests for FALCONEYE_MAX_EMBEDDING_CHARS env var override.

The cap is read at class-body evaluation time (module import), so each
test reloads the adapter module after toggling the env var.
"""

from __future__ import annotations

import importlib
import sys

import pytest


_OLLAMA_MODULE = "falconeye.infrastructure.llm_providers.ollama_adapter"
_MLX_MODULE = "falconeye.infrastructure.llm_providers.mlx_adapter"


def _reload(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.unit
def test_ollama_default_cap_is_8000_when_env_unset(monkeypatch):
    monkeypatch.delenv("FALCONEYE_MAX_EMBEDDING_CHARS", raising=False)
    mod = _reload(_OLLAMA_MODULE)
    assert mod.OllamaLLMAdapter._MAX_EMBEDDING_CHARS == 8000


@pytest.mark.unit
def test_ollama_env_var_lowers_cap_for_small_context_models(monkeypatch):
    monkeypatch.setenv("FALCONEYE_MAX_EMBEDDING_CHARS", "6000")
    mod = _reload(_OLLAMA_MODULE)
    assert mod.OllamaLLMAdapter._MAX_EMBEDDING_CHARS == 6000


@pytest.mark.unit
def test_ollama_env_var_raises_cap_for_long_context_models(monkeypatch):
    monkeypatch.setenv("FALCONEYE_MAX_EMBEDDING_CHARS", "16000")
    mod = _reload(_OLLAMA_MODULE)
    assert mod.OllamaLLMAdapter._MAX_EMBEDDING_CHARS == 16000


@pytest.mark.unit
def test_mlx_default_cap_is_8000_when_env_unset(monkeypatch):
    monkeypatch.delenv("FALCONEYE_MAX_EMBEDDING_CHARS", raising=False)
    mod = _reload(_MLX_MODULE)
    assert mod.MLXLLMAdapter._MAX_EMBEDDING_CHARS == 8000


@pytest.mark.unit
def test_mlx_env_var_lowers_cap_for_small_context_models(monkeypatch):
    monkeypatch.setenv("FALCONEYE_MAX_EMBEDDING_CHARS", "6000")
    mod = _reload(_MLX_MODULE)
    assert mod.MLXLLMAdapter._MAX_EMBEDDING_CHARS == 6000


@pytest.mark.unit
def test_mlx_env_var_raises_cap_for_long_context_models(monkeypatch):
    monkeypatch.setenv("FALCONEYE_MAX_EMBEDDING_CHARS", "16000")
    mod = _reload(_MLX_MODULE)
    assert mod.MLXLLMAdapter._MAX_EMBEDDING_CHARS == 16000
