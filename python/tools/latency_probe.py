"""Latency distribution probe.

Runs a session on the live capture and reports:

* end-to-end age: wall time when a partial/final is emitted minus the wall time
  at which the last audio it covers was captured (the number the viewer feels);
* ASR request time, split by how much audio the request carried;
* partial cadence and slot slip.

Audio time is mapped to wall time with the session start as the anchor; the
error is bounded by one capture frame (32 ms).

Usage:
  python tools/latency_probe.py --seconds 60
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

# Allow running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioflow.asr import LlamaCppAsrClient
from audioflow.metrics import percentile
from audioflow.session import AudioFlowSession


@dataclass
class Record:
    kind: str
    chunk_id: int
    t_start: float
    t_end: float
    latency: float
    wall: float
    audio_seconds: float = 0.0
    age: float = 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="native/model/silero_vad.onnx")
    parser.add_argument("--partial-ms", type=int, default=500)
    parser.add_argument("--min-partial-age", type=int, default=800)
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

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model = args.model if os.path.isabs(args.model) else os.path.join(root, args.model)

    records: list[Record] = []
    anchor = [0.0]

    def on_event(event) -> None:
        now = time.monotonic()
        records.append(Record(
            kind=event.kind,
            chunk_id=event.chunk_id,
            t_start=event.t_start,
            t_end=event.t_end,
            latency=event.latency,
            wall=now,
            audio_seconds=max(0.0, event.t_end - event.t_start),
            age=now - anchor[0] - event.t_end,
        ))

    session = AudioFlowSession(client, vad_model=model,
                               partial_ms=args.partial_ms,
                               min_partial_age=args.min_partial_age / 1000.0,
                               max_chunk_ms=args.max_chunk_ms,
                               on_event=on_event)
    anchor[0] = time.monotonic()
    session.start_capture()
    print("measuring %.0f s of live audio ..." % args.seconds, flush=True)

    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()

    partials = [r for r in records if r.kind == "partial"]
    finals = [r for r in records if r.kind == "final"]
    gaps = [r for r in records if r.kind == "gap"]

    def row(label: str, values: list[float], unit: str = "ms") -> str:
        if not values:
            return "  %-26s (none)" % label
        scale = 1000.0 if unit == "ms" else 1.0
        return ("  %-26s p50 %6.1f  p90 %6.1f  p99 %6.1f  max %6.1f %s"
                % (label, percentile(values, 0.5) * scale,
                   percentile(values, 0.9) * scale,
                   percentile(values, 0.99) * scale, max(values) * scale, unit))

    print("\n=== counts ===")
    print("  partials %d   finals %d   gaps %d" % (len(partials), len(finals), len(gaps)))

    ages = [r.age for r in partials + finals if r.age > 0]
    print("\n=== end-to-end age (audio end -> text shown) ===")
    print(row("partial", [r.age for r in partials]))
    print(row("final", [r.age for r in finals]))
    print(row("both", ages))

    print("\n=== ASR request time ===")
    print(row("all requests", [r.latency for r in partials + finals]))
    print(row("partials", [r.latency for r in partials]))
    print(row("finals", [r.latency for r in finals]))

    print("\n=== ASR time by audio carried ===")
    buckets = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 99.0)]
    for low, high in buckets:
        subset = [r.latency for r in partials if low <= r.audio_seconds < high]
        if subset:
            print("  %4.0f–%-4.0f s  n=%-4d p50 %5.0f ms  p95 %5.0f ms"
                  % (low, high, len(subset), percentile(subset, 0.5) * 1000,
                     percentile(subset, 0.95) * 1000))

    if partials:
        first: dict[int, Record] = {}
        for record in partials:
            first.setdefault(record.chunk_id, record)
        first_delays = [r.wall - (anchor[0] + r.t_start) for r in first.values()]
        print("\n=== first partial of a chunk (chunk open -> first text) ===")
        print(row("first partial", first_delays))

    if len(partials) > 1:
        # Cadence matters inside a chunk. Consecutive partials from different
        # chunks are separated by silence and say nothing about slot slip.
        intervals: list[float] = []
        for i in range(1, len(partials)):
            if partials[i].chunk_id == partials[i - 1].chunk_id:
                intervals.append(partials[i].wall - partials[i - 1].wall)
        slips = [value - args.partial_ms / 1000.0 for value in intervals]
        print("\n=== partial cadence within a chunk (target %.0f ms) ===" % args.partial_ms)
        print(row("wall interval", intervals))
        print(row("slip", slips))

        per_chunk: dict[int, int] = {}
        for record in partials:
            per_chunk[record.chunk_id] = per_chunk.get(record.chunk_id, 0) + 1
        print("  chunks with partials: %d, mean %.1f partials each"
              % (len(per_chunk), sum(per_chunk.values()) / len(per_chunk)))

    if finals:
        durations = [r.t_end - r.t_start for r in finals]
        print("\n=== final chunk length ===")
        print("  n=%d  mean %.2f s  max %.2f s"
              % (len(durations), sum(durations) / len(durations), max(durations)))

    covered = sum(r.t_end - r.t_start for r in finals)
    print("\n=== totals ===")
    print("  wall %.1f s   committed audio %.1f s   coverage %.0f%%"
          % (args.seconds, covered, 100.0 * covered / max(1e-9, args.seconds)))
    print("  session frames=%d dropped=%d"
          % (session.stats["frames"], session.stats["dropped"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
