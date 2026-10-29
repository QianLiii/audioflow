"""llama.cpp HTTP backend for AudioFlow.

Session contract:

* one session per chunker chunk;
* the client sends only new audio, tracked as ``submitted_samples``;
* every response is the cumulative tentative transcript for the session;
* ``finish`` returns the committed final text.

llama.cpp has no server-side session state, so the session object owns the
accumulated PCM and re-sends it on every decode.

Measured on an RX 9070 XT (Vulkan, Qwen3-ASR-1.7B Q8_0): ~0.02 s of model time
per second of audio, so a 4 s chunk re-fed four times costs ~0.4 s.

One instance handles one request in flight at a time.
"""

from __future__ import annotations

import base64
import json
import re
import struct
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

# Qwen3-ASR emits "language <Name><asr_text><transcript>".
_QWEN_LANGUAGE_RE = re.compile(r"^\s*language\s+([^<]*?)\s*<asr_text>", re.IGNORECASE)
_QWEN_TEXT_RE = re.compile(r"<asr_text>", re.IGNORECASE)


def parse_qwen3_output(raw: str) -> tuple[Optional[str], str]:
    """Split a Qwen3-ASR response into ``(language, text)``.

    Models that do not emit the marker are handled by the fallback: the whole
    string is the transcript and the language is unknown.
    """
    match = _QWEN_LANGUAGE_RE.match(raw)
    language = match.group(1).strip() if match else None
    marker = _QWEN_TEXT_RE.search(raw)
    text = raw[marker.end():] if marker else raw
    return language, text.strip()


@dataclass(frozen=True)
class AsrResult:
    """One ASR response. ``text`` is cumulative for the session."""

    language: Optional[str]
    text: str
    raw: str
    latency_s: float


class AsrError(RuntimeError):
    """A request failed. Only connection-refused failures are safe to repeat."""


def _wav_bytes(pcm_s16le: bytes) -> bytes:
    """Wrap raw PCM in a 44-byte WAV header.

    llama.cpp sniffs the container with miniaudio, so raw PCM cannot be posted
    directly. The wrapper is built in memory: AudioFlow never writes audio to
    disk.
    """
    size = len(pcm_s16le)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + size, b"WAVE",
        b"fmt ", 16, 1, 1,
        SAMPLE_RATE, SAMPLE_RATE * BYTES_PER_SAMPLE,
        BYTES_PER_SAMPLE, 16,
        b"data", size,
    )
    return header + pcm_s16le


class LlamaCppAsrClient:
    """Client for a ``llama-server`` instance serving a Qwen3-ASR GGUF."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        timeout_s: float = 120.0,
        max_tokens: int = 256,
        system_prompt: str = "",
        parser: Callable[[str], tuple[Optional[str], str]] = parse_qwen3_output,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._parser = parser

    # -- lifecycle ---------------------------------------------------------

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8")).get("status") == "ok"
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def open(self) -> "AsrSession":
        """Start a session for one chunk. Cheap: llama.cpp holds no state."""
        return AsrSession(self)

    def close(self) -> None:
        """Nothing to release."""

    # -- one-shot ----------------------------------------------------------

    def transcribe_pcm(self, pcm_s16le: bytes) -> AsrResult:
        return self._post(_wav_bytes(pcm_s16le))

    # -- HTTP --------------------------------------------------------------

    def _post(self, wav: bytes) -> AsrResult:
        body = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(wav).decode("ascii"),
                                "format": "wav",
                            },
                        }
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise AsrError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise AsrError(f"request failed: {exc}") from exc
        except ValueError as exc:
            raise AsrError(f"invalid JSON response: {exc}") from exc

        elapsed = time.perf_counter() - started
        try:
            raw = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AsrError(f"unexpected response shape: {payload!r}") from exc

        language, text = self._parser(raw)
        return AsrResult(language=language, text=text, raw=raw, latency_s=elapsed)


class AsrSession:
    """One chunk's ASR session.

    ``push`` accepts the s16le delta since the previous successful call and
    returns the cumulative transcript, or ``None`` when there is no new audio.
    ``finish`` commits the final text.
    """

    def __init__(self, client: LlamaCppAsrClient) -> None:
        self._client = client
        self._pcm = bytearray()
        self.submitted_samples = 0
        self.last_result: Optional[AsrResult] = None

    @property
    def total_samples(self) -> int:
        return len(self._pcm) // BYTES_PER_SAMPLE

    @property
    def last_text(self) -> str:
        return self.last_result.text if self.last_result else ""

    def push(self, pcm_s16le: bytes) -> Optional[AsrResult]:
        if not pcm_s16le:
            return None
        if len(pcm_s16le) % BYTES_PER_SAMPLE:
            raise ValueError("PCM delta must be a whole number of s16 samples")

        self._pcm += pcm_s16le
        result = self._client.transcribe_pcm(bytes(self._pcm))
        self.submitted_samples = self.total_samples
        self.last_result = result
        return result

    def finish(self) -> AsrResult:
        """Commit the final transcript.

        The previous decode already covered every submitted sample, so this
        repeats the last result. A session that received audio but never
        decoded is decoded once here; a session with no audio at all raises.
        """
        if self.last_result is not None:
            return self.last_result
        if not self._pcm:
            raise AsrError("finish() on a session with no audio")
        result = self._client.transcribe_pcm(bytes(self._pcm))
        self.submitted_samples = self.total_samples
        self.last_result = result
        return result

    def abort(self) -> None:
        """Drop the session. llama.cpp has no server-side state to clean up."""
        self._pcm = bytearray()
        self.last_result = None
