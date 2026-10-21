"""AudioFlow Python package.

The native capture/VAD/chunker path lives in ``audioflow.dll`` and is loaded
with ctypes (not implemented yet). The ASR layer is transport-neutral: see
``audioflow.asr``.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
