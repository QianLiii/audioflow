"""AudioFlow session: one capture run, end to end.

Owns the native capture/VAD/chunker handles, the scheduler, the assembler, and
the metrics for a single session (ARCHITECTURE.md section 3). It is the piece a
UI or a test drives; the tools are thin wrappers around it.

Threading model
---------------
* The capture callback runs on the DLL's delivery thread and only enqueues.
* One worker thread runs VAD, the chunker, the ASR requests, and the assembler.
  It never blocks the capture path.
* Public methods (``start_*``, ``stop``, ``render``) are meant to be called from
  the owner's thread; ``render`` is only safe after ``stop``.

Sources
-------
``start_capture()`` uses WASAPI loopback; ``start_file(path)`` replays a 16 kHz
mono WAV at a chosen speed. Both feed the same queue, so everything downstream
is source-agnostic (the ``AudioSource`` seam from section 12).
"""

from __future__ import annotations

import ctypes
import os
import time
import wave
from queue import Empty, SimpleQueue
from typing import Callable, Optional

from .assembler import Assembler, Segment
from .metrics import EventLog, Metrics
from .native import (
    AF_CHUNK_EVENT_CLOSED,
    AF_CHUNK_EVENT_DISCARDED,
    AF_CHUNK_EVENT_OPENED,
    AF_CHUNK_EVENT_SNAPSHOT,
    AF_EVENT_AUDIO,
    AF_EVENT_DISCONTINUITY,
    AF_EVENT_DROPPED,
    AudioFlow,
)
from .scheduler import Event, Scheduler

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE


class AudioFlowSession:
    def __init__(
        self,
        asr_client,
        *,
        vad_model: str,
        partial_ms: int = 500,
        min_partial_age: float = 0.8,
        max_chunk_ms: int = 4000,
        max_lag: float = 10.0,
        on_event: Optional[Callable[[Event], None]] = None,
        log_path: Optional[str] = None,
        flow: Optional[AudioFlow] = None,
    ) -> None:
        self.flow = flow or AudioFlow()
        self.flow.vad_open(vad_model)
        self.flow.chunker_open(max_chunk_ms=max_chunk_ms)
        self.assembler = Assembler()
        self.metrics = Metrics(partial_interval=partial_ms / 1000.0)
        self.on_event = on_event
        self.log = EventLog(log_path) if log_path else None
        self.scheduler = Scheduler(
            asr_client, self._emit,
            p=partial_ms / 1000.0, max_lag=max_lag, clock=time.monotonic,
            min_partial_age=min_partial_age,
        )
        self.stats = {"frames": 0, "dropped": 0, "discontinuities": 0}
        self._queue: SimpleQueue = SimpleQueue()
        self._worker: Optional[object] = None
        self._stop = False
        self._chunk = None
        self._next_partial = 0.0
        self._last_end = 0

    # -- lifecycle ---------------------------------------------------------

    def start_capture(self) -> None:
        self.flow.capture_start(self._on_capture)
        self._start_worker()

    def start_file(self, path: str, speed: float = 1.0) -> None:
        with wave.open(path, "rb") as handle:
            if (handle.getnchannels() != 1 or handle.getsampwidth() != 2
                    or handle.getframerate() != SAMPLE_RATE):
                raise ValueError("file source needs 16 kHz mono 16-bit PCM")
            pcm = handle.readframes(handle.getnframes())

        frames = [pcm[i:i + FRAME_SAMPLES * 2]
                  for i in range(0, len(pcm), FRAME_SAMPLES * 2)]
        self._start_worker()

        import threading

        def feeder() -> None:
            started = time.monotonic()
            step = FRAME_SECONDS / max(0.01, speed)
            for index, frame in enumerate(frames):
                if self._stop:
                    return
                delay = started + index * step - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                self._queue.put(("audio", frame, index * FRAME_SAMPLES))
            self._queue.put(("eof", b"", 0))

        threading.Thread(target=feeder, daemon=True).start()

    def stop(self) -> None:
        self._stop = True
        self.flow.capture_stop()
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=30)
            self._worker = None
        self.flow.close()
        if self.log is not None:
            self.log.close()

    # -- queries -----------------------------------------------------------

    def render(self) -> list[Segment]:
        return self.assembler.render()

    def transcript(self) -> str:
        return self.assembler.transcript()

    def snapshot(self) -> dict:
        return self.metrics.snapshot()

    # -- capture callback (delivery thread) --------------------------------

    def _on_capture(self, event_ptr, _user) -> None:
        event = event_ptr.contents
        if event.type == AF_EVENT_AUDIO:
            pcm = ctypes.string_at(ctypes.addressof(event.samples),
                                   event.sample_count * 2)
            self._queue.put(("audio", pcm, int(event.qpc_position)))
        elif event.type == AF_EVENT_DISCONTINUITY:
            self._queue.put(("disc", event.reason, 0))
        elif event.type == AF_EVENT_DROPPED:
            self._queue.put(("drop", event.dropped_frames, 0))

    # -- worker ------------------------------------------------------------

    def _start_worker(self) -> None:
        import threading

        self._worker = threading.Thread(target=self._run, name="audioflow-worker",
                                        daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            try:
                message = self._queue.get(timeout=0.02)
            except Empty:
                self._service(time.monotonic())
                continue
            if message is None:
                break

            kind, payload, qpc = message
            if kind == "audio":
                self._handle_audio(payload, qpc)
            elif kind == "disc":
                self._handle_discontinuity(payload)
            elif kind == "drop":
                self.stats["dropped"] += payload
            elif kind == "eof":
                break

            self._service(time.monotonic())

        self._drain()

    def _service(self, now: float) -> None:
        if self._chunk is not None and now >= self._next_partial:
            snap = self.flow.chunker_snapshot()
            if snap is not None:
                pcm = self.flow.event_pcm(snap)
                self.flow.release_event(snap)
                if pcm:
                    self.scheduler.submit_partial(self._chunk, pcm, now)
                    self._last_end = self._chunk.start_sample + len(pcm) // 2
            self._next_partial = now + self.scheduler.p

        while self.scheduler.pump(now):
            pass

    def _drain(self) -> None:
        """Flush the last chunk and finish every queued final."""
        events, produced = self.flow.chunker_flush()
        for index in range(produced):
            event = events[index]
            if event.type == AF_CHUNK_EVENT_CLOSED and self._chunk is not None:
                pcm = self.flow.event_pcm(event)
                self.scheduler.submit_final(self._chunk, pcm,
                                            end_sample=event.end_sample,
                                            now=time.monotonic())
                self._chunk = None
            self.flow.release_event(event)

        deadline = time.monotonic() + 30.0
        while self.scheduler.pump(time.monotonic()) or self.scheduler.pending():
            if time.monotonic() > deadline:
                break
            time.sleep(0.01)

    # -- message handlers --------------------------------------------------

    def _handle_audio(self, pcm: bytes, qpc: int) -> None:
        self.stats["frames"] += 1
        buf = (ctypes.c_int16 * (len(pcm) // 2)).from_buffer_copy(pcm)
        probability = self.flow.vad_process(buf, len(buf))
        events, produced = self.flow.chunker_push(buf, len(buf), probability, qpc)

        for index in range(produced):
            event = events[index]
            if event.type == AF_CHUNK_EVENT_OPENED:
                self._chunk = self.scheduler.open_chunk(
                    event.chunk_id, start_sample=event.start_sample,
                    now=time.monotonic())
                # The first slot is exactly min_partial_age after the open, not
                # one cadence P later: aligning to P makes the effective first
                # partial ceil(min_age / P) * P.
                self._next_partial = self._chunk.next_slot
                self._last_end = event.start_sample
            elif event.type == AF_CHUNK_EVENT_CLOSED:
                pcm_closed = self.flow.event_pcm(event)
                end_sample = event.end_sample
                self.flow.release_event(event)
                if self._chunk is not None:
                    self.scheduler.submit_final(self._chunk, pcm_closed,
                                                end_sample=end_sample,
                                                now=time.monotonic())
                    self._chunk = None
            elif event.type == AF_CHUNK_EVENT_DISCARDED:
                if self._chunk is not None:
                    self.scheduler.abandon(self._chunk)
                    self._chunk = None
                self.flow.release_event(event)
            elif event.type == AF_CHUNK_EVENT_SNAPSHOT:
                self.flow.release_event(event)

    def _handle_discontinuity(self, reason: int) -> None:
        self.stats["discontinuities"] += 1
        if self._chunk is not None:
            self._emit(Event("gap", self._chunk.chunk_id, reason="capture_loss",
                             t_start=self._chunk.start_sample / SAMPLE_RATE,
                             t_end=self._last_end / SAMPLE_RATE))
            self.scheduler.abandon(self._chunk)
            self._chunk = None

        self.flow.vad_reset()
        events, produced = self.flow.chunker_reset()
        for index in range(produced):
            self.flow.release_event(events[index])

    def _emit(self, event: Event) -> None:
        self.assembler.on_event(event)
        self.metrics.observe(event)
        if self.log is not None:
            self.log.write(event, now=time.monotonic())
        if self.on_event is not None:
            self.on_event(event)
