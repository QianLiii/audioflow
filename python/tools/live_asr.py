"""Live loop: capture -> Silero VAD -> chunker -> scheduler -> llama.cpp ASR.

The capture thread only enqueues audio. One worker thread runs VAD, the
chunker, and the scheduler; every partial slot takes a snapshot of the open
chunk and hands it to the scheduler, which owns admission and the loss policy.

Time is `time.monotonic()`. The scheduler is the prototype from
`audioflow.scheduler`: single lane, no preemption (llama.cpp cannot cancel a
decode), degraded finals and gaps on loss.

Usage:
  python tools/live_asr.py [--seconds 30] [--url http://127.0.0.1:8080]
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from queue import Empty, SimpleQueue

from audioflow.asr import LlamaCppAsrClient
from audioflow.native import (
    AF_CHUNK_EVENT_CLOSED,
    AF_CHUNK_EVENT_DISCARDED,
    AF_CHUNK_EVENT_OPENED,
    AF_CHUNK_EVENT_SNAPSHOT,
    AF_EVENT_AUDIO,
    AF_EVENT_DISCONTINUITY,
    AF_EVENT_DROPPED,
    AudioFlow,
)
from audioflow.scheduler import Event, Scheduler

SAMPLE_RATE = 16000


def fmt_time(seconds: float) -> str:
    return "%6.1fs" % seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="testdata/silero_vad.onnx")
    parser.add_argument("--partial-ms", type=int, default=500)
    parser.add_argument("--max-chunk-ms", type=int, default=4000)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    client = LlamaCppAsrClient(args.url)
    if not client.health():
        print(f"llama-server not reachable at {args.url}", file=sys.stderr)
        return 1

    clock = time.monotonic

    def emit(event: Event) -> None:
        if event.kind == "partial":
            print("  %s partial r%d  (+%.1fs, %.2fs)  %s"
                  % (fmt_time(event.t_end), event.revision, event.t_end - event.t_start,
                     event.latency, event.text), flush=True)
        elif event.kind == "final":
            tag = "DEGRADED " if event.degraded else ""
            print("%s final %s[%s] %.1f–%.1f  (%.2fs)  %s"
                  % (fmt_time(event.t_end), tag, event.reason or "ok",
                     event.t_start, event.t_end, event.latency, event.text), flush=True)
        else:
            print("%s gap [%s] %.1f–%.1f"
                  % (fmt_time(event.t_end), event.reason, event.t_start, event.t_end),
                  flush=True)

    sched = Scheduler(client, emit, p=args.partial_ms / 1000.0, clock=clock)
    flow = AudioFlow()
    flow.vad_open(args.model)
    flow.chunker_open(max_chunk_ms=args.max_chunk_ms)

    queue: SimpleQueue = SimpleQueue()
    stats = {"frames": 0, "dropped": 0, "disc": 0}

    def on_event(event_ptr, _user):
        event = event_ptr.contents
        if event.type == AF_EVENT_AUDIO:
            pcm = ctypes.string_at(ctypes.addressof(event.samples), event.sample_count * 2)
            queue.put(("audio", pcm, int(event.qpc_position)))
        elif event.type == AF_EVENT_DISCONTINUITY:
            queue.put(("disc", event.reason, 0))
        elif event.type == AF_EVENT_DROPPED:
            queue.put(("drop", event.dropped_frames, 0))

    flow.capture_start(on_event)
    print("listening for %.0f s ..." % args.seconds, flush=True)

    chunk = None                      # current scheduler chunk, if open
    next_partial = 0.0
    deadline = clock() + args.seconds

    try:
        while clock() < deadline:
            now = clock()

            if chunk is not None and now >= next_partial:
                snap = flow.chunker_snapshot()
                if snap is not None:
                    pcm = flow.event_pcm(snap)
                    flow.release_event(snap)
                    if pcm:
                        sched.submit_partial(chunk, pcm, now)
                next_partial = now + args.partial_ms / 1000.0

            try:
                kind, payload, qpc = queue.get(timeout=0.02)
            except Empty:
                continue

            if kind == "disc":
                stats["disc"] += 1
                print("  [discontinuity reason=%d] resetting VAD/chunker" % payload, flush=True)
                flow.vad_reset()
                events, produced = flow.chunker_reset()
                for i in range(produced):
                    flow.release_event(events[i])
                if chunk is not None:
                    sched.abandon(chunk)
                chunk = None
                continue

            if kind == "drop":
                stats["dropped"] += payload
                continue

            stats["frames"] += 1
            buf = (ctypes.c_int16 * (len(payload) // 2)).from_buffer_copy(payload)
            probability = flow.vad_process(buf, len(buf))
            events, produced = flow.chunker_push(buf, len(buf), probability, qpc)

            for i in range(produced):
                event = events[i]
                if event.type == AF_CHUNK_EVENT_OPENED:
                    chunk = sched.open_chunk(event.chunk_id,
                                             start_sample=event.start_sample,
                                             now=clock())
                    next_partial = clock() + args.partial_ms / 1000.0
                elif event.type == AF_CHUNK_EVENT_CLOSED:
                    pcm = flow.event_pcm(event)
                    flow.release_event(event)
                    if chunk is not None:
                        sched.submit_final(chunk, pcm, end_sample=event.end_sample, now=clock())
                        chunk = None
                elif event.type == AF_CHUNK_EVENT_DISCARDED:
                    print("%s discarded (reason=%d, speech=%.0f%%)"
                          % (fmt_time(event.end_sample / SAMPLE_RATE), event.reason,
                             event.speech_ratio * 100), flush=True)
                    if chunk is not None:
                        sched.abandon(chunk)
                        chunk = None
                    flow.release_event(event)
                elif event.type == AF_CHUNK_EVENT_SNAPSHOT:
                    flow.release_event(event)

            while sched.pump(clock()):
                pass
    finally:
        flow.capture_stop()
        events, produced = flow.chunker_flush()
        for i in range(produced):
            event = events[i]
            if event.type == AF_CHUNK_EVENT_CLOSED and chunk is not None:
                pcm = flow.event_pcm(event)
                sched.submit_final(chunk, pcm, end_sample=event.end_sample, now=clock())
                chunk = None
            flow.release_event(event)
        while sched.pump(clock()):
            pass
        flow.close()

        print("\nframes=%d dropped=%d disc=%d | partials=%d finals=%d degraded=%d gaps=%d retries=%d lost=%d"
              % (stats["frames"], stats["dropped"], stats["disc"],
                 sched.stats.partials, sched.stats.finals, sched.stats.degraded,
                 sched.stats.gaps, sched.stats.retries, sched.stats.lost), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
