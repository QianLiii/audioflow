"""Live ASR CLI over AudioFlowSession.

Capture (default) or replay a WAV, print partials and finals as they arrive,
then print the committed transcript and the session metrics.

Usage:
  python tools/live_asr.py --seconds 30
  python tools/live_asr.py --wav testdata/live_capture_10s.wav --speed 1
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

from audioflow.assembler import format_transcript
from audioflow.asr import LlamaCppAsrClient
from audioflow.session import SAMPLE_RATE, AudioFlowSession


def fmt(seconds: float) -> str:
    return "%6.1fs" % seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=None,
                        help="stop after N seconds; omit to run until Ctrl+C")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="native/model/silero_vad.onnx",
                        help="Silero VAD model, relative to the repo root by default")
    parser.add_argument("--partial-ms", type=int, default=500)
    parser.add_argument("--max-chunk-ms", type=int, default=4000)
    parser.add_argument("--wav", default=None, help="replay a 16 kHz mono WAV")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--log-dir", default=None)
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
    log_dir = args.log_dir or os.path.join(root, "build", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "live-%s.jsonl"
                            % datetime.now().strftime("%Y%m%d-%H%M%S"))

    def print_event(event) -> None:
        if event.kind == "partial":
            print("  %s partial r%d  (+%.1fs, %.2fs)  %s"
                  % (fmt(event.t_end), event.revision, event.t_end - event.t_start,
                     event.latency, event.text), flush=True)
        elif event.kind == "final":
            tag = "DEGRADED " if event.degraded else ""
            print("%s final %s[%s] %.1f–%.1f  (%.2fs)  %s"
                  % (fmt(event.t_end), tag, event.reason or "ok", event.t_start,
                     event.t_end, event.latency, event.text), flush=True)
        else:
            print("%s gap [%s] %.1f–%.1f"
                  % (fmt(event.t_end), event.reason, event.t_start, event.t_end),
                  flush=True)

    model = args.model if os.path.isabs(args.model) else os.path.join(root, args.model)

    session = AudioFlowSession(
        client,
        vad_model=model,
        partial_ms=args.partial_ms,
        max_chunk_ms=args.max_chunk_ms,
        on_event=print_event,
        log_path=log_path,
    )

    runtime = args.seconds
    try:
        if args.wav:
            with wave.open(args.wav, "rb") as handle:
                duration = handle.getnframes() / handle.getframerate()
            session.start_file(args.wav, speed=args.speed)
            print("replaying %s (%.1f s of audio, %.0fx) ..."
                  % (os.path.basename(args.wav), duration, args.speed), flush=True)
            if runtime is None:
                runtime = duration / max(0.01, args.speed) + 2.0
        else:
            session.start_capture()
            print("listening%s ... ctrl+c to stop"
                  % (" for %.0f s" % runtime if runtime else ""), flush=True)
            if runtime is not None and runtime <= 0:
                runtime = None

        if runtime is None:
            while True:
                time.sleep(0.5)
        else:
            time.sleep(runtime)
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    finally:
        session.stop()

    print("\nframes=%d dropped=%d discontinuities=%d"
          % (session.stats["frames"], session.stats["dropped"],
             session.stats["discontinuities"]), flush=True)
    print("scheduler: partials=%d finals=%d degraded=%d gaps=%d retries=%d lost=%d"
          % (session.scheduler.stats.partials, session.scheduler.stats.finals,
             session.scheduler.stats.degraded, session.scheduler.stats.gaps,
             session.scheduler.stats.retries, session.scheduler.stats.lost), flush=True)

    print("\n--- transcript ---")
    print(format_transcript(session.render()) or "(no committed segments)")
    print("\n--- metrics ---")
    for key, value in session.snapshot().items():
        print("  %-26s %s" % (key, value))
    print("\nlog: %s" % log_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
