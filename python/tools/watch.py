"""Live transcript view (ARCHITECTURE.md section 5.8: event sink).

Runs a session and repaints the transcript as it grows: committed text in full
colour, the tentative tail dimmed, degraded finals marked, gaps shown.

The sink is live-only and keeps no history on disk beyond the optional JSONL
event log, which never contains text.

Usage:
  python tools/watch.py --seconds 60
  python tools/watch.py --wav testdata/live_capture_10s.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from datetime import datetime

# Allow running this file directly, without installing the package or setting
# PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioflow.asr import LlamaCppAsrClient
from audioflow.session import AudioFlowSession

CLEAR = "\x1b[2J\x1b[H"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
RED = "\x1b[31m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"

WIDTH = 100


def wrap(text: str, width: int = WIDTH) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw(session: AudioFlowSession, elapsed: float, clear: bool) -> None:
    segments = session.render()
    out: list[str] = []
    if clear:
        out.append(CLEAR)

    snap = session.snapshot()
    out.append("%sAudioFlow%s  %5.1fs  chunk-partials %d  finals %d  degraded %d  "
               "gap %.1fs  p95 %.0f ms"
               % (BOLD, RESET, elapsed, snap["partials"], snap["finals"],
                  snap["degraded"], snap["gap_seconds"], snap["latency_p95_ms"]))
    out.append("─" * WIDTH)

    if not segments:
        out.append(f"{DIM}(waiting for speech){RESET}")
    for segment in segments:
        if segment.kind == "gap":
            out.append("%s[%5.1f–%5.1f]  gap: %s%s"
                       % (YELLOW, segment.t_start, segment.t_end,
                          segment.reason or "?", RESET))
            continue
        colour = DIM if segment.provisional else (RED if segment.degraded else "")
        tag = ""
        if segment.degraded:
            tag = " [degraded]"
        elif segment.provisional:
            tag = " [partial]"
        for index, line in enumerate(wrap(segment.text)):
            if index == 0:
                out.append("%s%6.1f–%-6.1f %s%s%s"
                           % (colour, segment.t_start, segment.t_end, line, tag, RESET))
            else:
                out.append("%s            %s%s" % (colour, line, RESET))

    out.append("─" * WIDTH)
    out.append(f"{DIM}ctrl+c to stop{RESET}")
    print("\n".join(out), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=None,
                        help="stop after N seconds; omit to run until Ctrl+C")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="native/model/silero_vad.onnx",
                        help="Silero VAD model, relative to the repo root by default")
    parser.add_argument("--partial-ms", type=int, default=500)
    parser.add_argument("--min-partial-age", type=int, default=800,
                        help="ms after a chunk opens before its first partial")
    parser.add_argument("--max-chunk-ms", type=int, default=4000)
    parser.add_argument("--wav", default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-clear", action="store_true")
    parser.add_argument("--interval-ms", type=int, default=500)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    clear = sys.stdout.isatty() and not args.no_clear

    client = LlamaCppAsrClient(args.url)
    if not client.health():
        print(f"llama-server not reachable at {args.url}", file=sys.stderr)
        return 1

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(root, "build", "logs")
    os.makedirs(log_dir, exist_ok=True)
    model = args.model if os.path.isabs(args.model) else os.path.join(root, args.model)

    session = AudioFlowSession(
        client,
        vad_model=model,
        partial_ms=args.partial_ms,
        min_partial_age=args.min_partial_age / 1000.0,
        max_chunk_ms=args.max_chunk_ms,
        on_event=None,
        log_path=os.path.join(log_dir, "watch-%s.jsonl"
                              % datetime.now().strftime("%Y%m%d-%H%M%S")),
    )

    runtime = args.seconds
    if args.wav:
        with wave.open(args.wav, "rb") as handle:
            duration = handle.getnframes() / handle.getframerate()
        session.start_file(args.wav, speed=args.speed)
        if runtime is None:
            runtime = duration / max(0.01, args.speed) + 2.0
    else:
        session.start_capture()
        # No --seconds means run until Ctrl+C.
        if runtime is not None and runtime <= 0:
            runtime = None

    started = time.monotonic()
    try:
        while runtime is None or time.monotonic() - started < runtime:
            draw(session, time.monotonic() - started, clear)
            time.sleep(args.interval_ms / 1000.0)
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()

    draw(session, time.monotonic() - started, clear)
    print("\ntranscript: %s" % (session.transcript() or "(empty)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
