"""ASR layer.

``AsrClient`` is transport-neutral. The reference backend is a local llama.cpp
server running a Qwen3-ASR GGUF.
"""

from .llama_cpp import (
    AsrError,
    AsrResult,
    AsrSession,
    LlamaCppAsrClient,
    parse_qwen3_output,
)

__all__ = [
    "AsrError",
    "AsrResult",
    "AsrSession",
    "LlamaCppAsrClient",
    "parse_qwen3_output",
]
