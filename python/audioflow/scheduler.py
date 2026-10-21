"""Single-lane scheduler, prototype.

Implements the parts of ARCHITECTURE.md section 6 that matter for the llama.cpp
backend and are testable without threads:

* one request in flight at a time (section 6.6, invariant 1);
* partial slots every ``P`` seconds with a maximum slip ``S``;
* finals are admitted only if they fit before ``next_slot + S``;
* a final that misses its deadline is lost: the last partial is committed as a
  degraded final and any uncovered tail becomes a gap (section 8.4);
* connection-refused failures retry with backoff, ambiguous failures never do.

Deliberate differences from the full design, all because llama.cpp cannot
cancel a running decode:

* **No preemption.** A request that has started runs to completion. The
  scheduler only decides what to start, which is the same guarantee section 16
  describes for a service that ignores cancellation.
* **Synchronous dispatch.** ``pump()`` runs one request inline. The prototype's
  caller is the chunker thread, not the audio thread, so blocking there is
  allowed. A threaded lane becomes worthwhile only once requests can outlive
  one pump.
* Partial slots are driven by the caller: it takes the chunker snapshot and
  calls :meth:`submit_partial`. The scheduler owns the timing decisions.

Time is passed in explicitly so tests can drive it with a virtual clock.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional

from .asr.fake import AmbiguousTimeout, AsrFailure, ConnectionRefused, PermanentError
from .asr.llama_cpp import AsrError, AsrResult

SAMPLE_RATE = 16000


@dataclass
class Event:
    kind: str                    # partial | final | gap
    chunk_id: int
    text: str = ""
    degraded: bool = False
    reason: str = ""
    t_start: float = 0.0
    t_end: float = 0.0
    latency: float = 0.0
    revision: int = 0


@dataclass
class Chunk:
    chunk_id: int
    start_sample: int
    open_time: float
    next_slot: float = 0.0
    end_sample: int = 0
    closed: bool = False
    close_time: float = 0.0
    last_partial_text: str = ""
    last_partial_end: int = 0
    revision: int = 0


@dataclass
class _FinalJob:
    chunk: Chunk
    pcm: bytes
    attempts: int = 0
    due: float = 0.0


@dataclass
class Stats:
    partials: int = 0
    finals: int = 0
    degraded: int = 0
    gaps: int = 0
    retries: int = 0
    lost: int = 0
    max_lane_depth: int = 0


class Scheduler:
    def __init__(
        self,
        client,
        emit: Callable[[Event], None],
        *,
        p: float = 0.5,
        s: float = 0.5,
        max_lag: float = 10.0,
        min_partial_age: float = 0.8,
        final_cost: float = 0.3,
        retry_backoff: tuple[float, ...] = (0.5, 1.5),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.emit = emit
        self.p = p
        self.s = s
        self.max_lag = max_lag
        self.min_partial_age = min_partial_age
        self.final_cost = final_cost
        self.retry_backoff = retry_backoff
        self.clock = clock
        self.stats = Stats()
        self._finals: Deque[_FinalJob] = deque()
        self._retries: list[_FinalJob] = []
        self._open: dict[int, Chunk] = {}

    # -- chunk lifecycle ---------------------------------------------------

    def open_chunk(self, chunk_id: int, start_sample: int, now: float) -> Chunk:
        chunk = Chunk(
            chunk_id=chunk_id,
            start_sample=start_sample,
            open_time=now,
            next_slot=now + self.min_partial_age,
        )
        self._open[chunk_id] = chunk
        return chunk

    def submit_partial(self, chunk: Chunk, pcm: bytes, now: float) -> Optional[AsrResult]:
        """Dispatch a partial if its slot is due. Returns the result, if any.

        Called by the chunker thread with a snapshot of the open chunk. The
        delta rule lives in the ASR session; the scheduler only decides whether
        this request may start now.
        """
        if chunk.closed or now + 1e-9 < chunk.next_slot:
            return None

        result = self._call(chunk, pcm, kind="partial")
        chunk.next_slot = now + self.p
        if result is None:
            return None

        chunk.revision += 1
        chunk.last_partial_text = result.text
        chunk.last_partial_end = chunk.start_sample + len(pcm) // 2
        self.stats.partials += 1
        self.emit(
            Event(
                kind="partial",
                chunk_id=chunk.chunk_id,
                text=result.text,
                t_start=chunk.start_sample / SAMPLE_RATE,
                t_end=chunk.last_partial_end / SAMPLE_RATE,
                latency=result.latency_s,
                revision=chunk.revision,
            )
        )
        return result

    def submit_final(self, chunk: Chunk, pcm: bytes, end_sample: int, now: float) -> None:
        chunk.closed = True
        chunk.close_time = now
        chunk.end_sample = end_sample
        job = _FinalJob(chunk=chunk, pcm=pcm, due=now)
        self._finals.append(job)
        self._open.pop(chunk.chunk_id, None)

    def abandon(self, chunk: Chunk) -> None:
        """Forget a chunk that will never produce a final (discarded or reset)."""
        chunk.closed = True
        self._open.pop(chunk.chunk_id, None)

    def pending(self) -> bool:
        return bool(self._finals or self._retries)

    # -- lane --------------------------------------------------------------

    def pump(self, now: float) -> bool:
        """Dispatch at most one job. Returns True when something ran."""
        self._expire(now)

        if self._retries:
            due = [job for job in self._retries if job.due <= now]
            if due:
                job = min(due, key=lambda j: j.due)
                self._retries.remove(job)
                self._dispatch_final(job, now)
                return True

        if self._finals:
            job = self._finals[0]
            if self._fits(job, now):
                self._finals.popleft()
                self._dispatch_final(job, now)
                return True
        return False

    def _fits(self, job: _FinalJob, now: float) -> bool:
        """A final may start only if it will not push a partial slot past S.

        Slots already in the past are ignored: the caller drives partials, so an
        overdue slot means no partial is pending, not that the lane is blocked.
        Once every chunk is closed there are no slots, so finals run freely.
        """
        upcoming = [c.next_slot for c in self._open.values()
                    if not c.closed and c.next_slot > now]
        if not upcoming:
            return True
        return now + self.final_cost <= min(upcoming) + self.s

    def _expire(self, now: float) -> None:
        keep: Deque[_FinalJob] = deque()
        for job in self._finals:
            if now > job.chunk.close_time + self.max_lag:
                self.stats.lost += 1
                self._lose(job.chunk, "deadline")
            else:
                keep.append(job)
        self._finals = keep

        still: list[_FinalJob] = []
        for job in self._retries:
            if now > job.chunk.close_time + self.max_lag:
                self.stats.lost += 1
                self._lose(job.chunk, "deadline")
            else:
                still.append(job)
        self._retries = still

    def _dispatch_final(self, job: _FinalJob, now: float) -> None:
        chunk = job.chunk
        job.attempts += 1
        try:
            result = self.client.transcribe_pcm(job.pcm)
        except ConnectionRefused:
            self._schedule_retry(job, now)
            return
        except (AmbiguousTimeout, PermanentError, AsrError):
            self._lose(chunk, "permanent_error")
            return

        self.stats.finals += 1
        self.emit(
            Event(
                kind="final",
                chunk_id=chunk.chunk_id,
                text=result.text,
                degraded=False,
                t_start=chunk.start_sample / SAMPLE_RATE,
                t_end=chunk.end_sample / SAMPLE_RATE,
                latency=result.latency_s,
            )
        )

    def _schedule_retry(self, job: _FinalJob, now: float) -> None:
        if job.attempts >= len(self.retry_backoff) + 1:
            self._lose(job.chunk, "permanent_error")
            return
        delay = self.retry_backoff[min(job.attempts - 1, len(self.retry_backoff) - 1)]
        job.due = now + delay
        self._retries.append(job)
        self.stats.retries += 1

    def _call(self, chunk: Chunk, pcm: bytes, kind: str) -> Optional[AsrResult]:
        """Run one request for ``kind`` and map failures to the loss policy."""
        try:
            return self.client.transcribe_pcm(pcm)
        except (AsrError, AsrFailure):
            self._lose(chunk, "permanent_error")
            return None

    # -- loss policy -------------------------------------------------------

    def _lose(self, chunk: Chunk, reason: str) -> None:
        """Commit a degraded final over the partial's range, then gap the rest."""
        if chunk.last_partial_text:
            self.stats.degraded += 1
            self.emit(
                Event(
                    kind="final",
                    chunk_id=chunk.chunk_id,
                    text=chunk.last_partial_text,
                    degraded=True,
                    reason=reason,
                    t_start=chunk.start_sample / SAMPLE_RATE,
                    t_end=chunk.last_partial_end / SAMPLE_RATE,
                )
            )
            if chunk.end_sample > chunk.last_partial_end:
                self.stats.gaps += 1
                self.emit(
                    Event(
                        kind="gap",
                        chunk_id=chunk.chunk_id,
                        reason="degraded_tail",
                        t_start=chunk.last_partial_end / SAMPLE_RATE,
                        t_end=chunk.end_sample / SAMPLE_RATE,
                    )
                )
        else:
            self.stats.gaps += 1
            self.emit(
                Event(
                    kind="gap",
                    chunk_id=chunk.chunk_id,
                    reason=reason,
                    t_start=chunk.start_sample / SAMPLE_RATE,
                    t_end=chunk.end_sample / SAMPLE_RATE,
                )
            )
