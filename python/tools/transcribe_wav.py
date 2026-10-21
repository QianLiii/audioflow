"""Transcribe a 16 kHz mono s16 WAV through the ASR client.

Two modes:

  one-shot   transcribe the whole file in one request
  session    feed the file in 1 s deltas through an AsrSession, printing the
             cumulative transcript after each push (the AudioFlow partial path)

Usage:
  python tools/transcribe_wav.py [--session] [--url URL] file.wav ...
"""

from __future__ import annotations

import argparse
import sys
import wave

from audioflow.asr import AsrError, LlamaCppAsrClient


def read_wav(path: str) -> bytes:
    with wave.open(path, "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path}: need mono 16-bit PCM")
        if handle.getframerate() != 16000:
            raise SystemExit(f"{path}: need 16 kHz, got {handle.getframerate()}")
        return handle.readframes(handle.getnframes())


def run_one_shot(client: LlamaCppAsrClient, path: str) -> None:
    pcm = read_wav(path)
    result = client.transcribe_pcm(pcm)
    print("%-34s %6.2f s  [%s] %s"
          % (path.rsplit("\\", 1)[-1], result.latency_s,
             result.language or "?", result.text))


def run_session(client: LlamaCppAsrClient, path: str, delta_s: float) -> None:
    pcm = read_wav(path)
    step = int(delta_s * 16000) * 2
    session = client.open()
    print("-- %s, %d samples, delta %.1f s" % (path.rsplit("\\", 1)[-1],
                                               len(pcm) // 2, delta_s))
    elapsed = 0.0
    for offset in range(0, len(pcm), step):
        delta = pcm[offset:offset + step]
        result = session.push(delta)
        elapsed += result.latency_s if result else 0.0
        if result is not None:
            print("   %5.1f s  %6.2f s  %s"
                  % (session.total_samples / 16000, result.latency_s,
                     result.text[-60:]))
    final = session.finish()
    print("   final: %s  (decode time %.2f s for %.1f s audio)"
          % (final.text, elapsed, len(pcm) / 32000))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--session", action="store_true",
                        help="exercise the per-chunk AsrSession path")
    parser.add_argument("--delta", type=float, default=1.0,
                        help="delta length for --session (seconds)")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    client = LlamaCppAsrClient(args.url)
    if not client.health():
        print(f"llama-server not reachable at {args.url}", file=sys.stderr)
        return 1

    for path in args.files:
        try:
            if args.session:
                run_session(client, path, args.delta)
            else:
                run_one_shot(client, path)
        except (AsrError, SystemExit) as exc:
            print(f"FAIL {path}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
