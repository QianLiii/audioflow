"""Deterministic fake ASR backend for scheduler tests.

It mirrors the ``AsrClient`` surface of :class:`audioflow.asr.LlamaCppAsrClient`
but costs no real time: latency is charged against an injected clock, so tests
can drive the scheduler by hand.

Failure injection covers the three cases the retry policy distinguishes. The
exception types are separate because the policy depends on them:

* :class:`ConnectionRefused` — the request never reached the service; safe to retry.
* :class:`AmbiguousTimeout` — the request may have been consumed; never resent.
* :class:`PermanentError` — 4xx; the session is broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .llama_cpp import AsrResult


class AsrFailure(RuntimeError):
    pass


class ConnectionRefused(AsrFailure):
    pass


class AmbiguousTimeout(AsrFailure):
    pass


class PermanentError(AsrFailure):
    pass


class VirtualClock:
    """Monotonic clock that only moves when someone advances it."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeStats:
    calls: int = 0
    max_concurrent: int = 0
    total_audio_s: float = 0.0
    failures: int = 0


@dataclass
class FakeAsrClient:
    """In-process stand-in for the HTTP client.

    ``latency = base_latency + per_second * audio_seconds`` is charged to the
    clock on every successful call, so admission decisions in the scheduler see
    a realistic cost model.
    """

    clock: VirtualClock
    base_latency: float = 0.05
    per_second: float = 0.02
    failures: list = field(default_factory=list)  # consumed one per call
    stats: FakeStats = field(default_factory=FakeStats)
    _concurrent: int = 0
    _sessions: int = 0

    # -- AsrClient surface -------------------------------------------------

    def health(self) -> bool:
        return True

    def open(self) -> "FakeAsrSession":
        self._sessions += 1
        return FakeAsrSession(self, self._sessions)

    def close(self) -> None:
        pass

    def transcribe_pcm(self, pcm_s16le: bytes) -> AsrResult:
        return self._run(pcm_s16le, tag="oneshot")

    # -- internals ---------------------------------------------------------

    def _run(self, pcm_s16le: bytes, tag: str) -> AsrResult:
        self._concurrent += 1
        self.stats.max_concurrent = max(self.stats.max_concurrent, self._concurrent)
        self.stats.calls += 1
        try:
            if self.failures:
                mode = self.failures.pop(0)
                if mode == "refused":
                    self.stats.failures += 1
                    raise ConnectionRefused("connection refused")
                if mode == "timeout":
                    self.stats.failures += 1
                    raise AmbiguousTimeout("request timed out")
                if mode == "http400":
                    self.stats.failures += 1
                    raise PermanentError("HTTP 400")
                raise ValueError(f"unknown failure mode {mode!r}")

            seconds = len(pcm_s16le) / 32000.0
            cost = self.base_latency + self.per_second * seconds
            self.clock.advance(cost)
            self.stats.total_audio_s += seconds
            text = f"{tag}:{seconds:.1f}s"
            return AsrResult(language="fake", text=text, raw=text, latency_s=cost)
        finally:
            self._concurrent -= 1


class FakeAsrSession:
    def __init__(self, client: FakeAsrClient, index: int) -> None:
        self._client = client
        self._index = index
        self._samples = 0
        self.submitted_samples = 0
        self.last_result = None

    def push(self, pcm_s16le: bytes) -> AsrResult | None:
        if not pcm_s16le:
            return None
        self._samples += len(pcm_s16le) // 2
        result = self._client._run(pcm_s16le, tag=f"chunk{self._index}")
        self.submitted_samples = self._samples
        self.last_result = result
        return result

    def finish(self) -> AsrResult:
        if self.last_result is None:
            raise AsrFailure("finish() on a session with no audio")
        return self.last_result

    def abort(self) -> None:
        self.last_result = None
