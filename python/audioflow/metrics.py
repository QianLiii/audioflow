"""Metrics and the structured event log (ARCHITECTURE.md section 11).

The transcript text never enters the log: section 17 allows IDs, timings,
outcomes, and counters only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .scheduler import Event


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


@dataclass
class Metrics:
    """Running aggregate over scheduler events."""

    partial_interval: float = 0.5
    partials: int = 0
    finals: int = 0
    degraded: int = 0
    gaps: int = 0
    gap_seconds: float = 0.0
    text_seconds: float = 0.0
    retries: int = 0
    lost: int = 0
    decode_calls: int = 0
    decode_audio_seconds: float = 0.0
    latencies: list[float] = field(default_factory=list)
    intervals: list[float] = field(default_factory=list)
    _last_partial: dict[int, float] = field(default_factory=dict)

    def observe(self, event: Event, *, now: Optional[float] = None) -> None:
        if event.kind == "partial":
            self.partials += 1
            self.decode_calls += 1
            self.latencies.append(event.latency)
            previous = self._last_partial.get(event.chunk_id)
            if previous is not None:
                self.intervals.append(event.t_end - previous)
            self._last_partial[event.chunk_id] = event.t_end
            return

        if event.kind == "final":
            self.finals += 1
            if event.degraded:
                self.degraded += 1
            self.decode_calls += 1
            if event.latency:
                self.latencies.append(event.latency)
            self.text_seconds += max(0.0, event.t_end - event.t_start)
            self._last_partial.pop(event.chunk_id, None)
            return

        if event.kind == "gap":
            self.gaps += 1
            self.gap_seconds += max(0.0, event.t_end - event.t_start)
            return

    # -- derived -----------------------------------------------------------

    def slip(self) -> list[float]:
        """Delivery interval minus the target partial interval."""
        return [interval - self.partial_interval for interval in self.intervals]

    def coverage(self) -> float:
        total = self.text_seconds + self.gap_seconds
        return self.text_seconds / total if total else 0.0

    def snapshot(self) -> dict:
        slips = self.slip()
        return {
            "partials": self.partials,
            "finals": self.finals,
            "degraded": self.degraded,
            "gaps": self.gaps,
            "gap_seconds": round(self.gap_seconds, 2),
            "text_seconds": round(self.text_seconds, 2),
            "coverage": round(self.coverage(), 3),
            "retries": self.retries,
            "lost": self.lost,
            "latency_p50_ms": round(percentile(self.latencies, 0.50) * 1000, 1),
            "latency_p95_ms": round(percentile(self.latencies, 0.95) * 1000, 1),
            "partial_interval_mean_s": round(
                sum(self.intervals) / len(self.intervals), 3) if self.intervals else 0.0,
            "slip_p95_s": round(percentile(slips, 0.95), 3) if slips else 0.0,
        }


class EventLog:
    """JSONL log: one record per event, no transcript text."""

    def __init__(self, path: str) -> None:
        self._file = open(path, "w", encoding="utf-8")
        self.path = path

    def write(self, event: Event, *, now: Optional[float] = None) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "mono": round(now if now is not None else time.monotonic(), 3),
            "kind": event.kind,
            "chunk_id": event.chunk_id,
            "revision": event.revision,
            "t_start_ms": int(round(event.t_start * 1000)),
            "t_end_ms": int(round(event.t_end * 1000)),
            "latency_ms": int(round(event.latency * 1000)),
            "degraded": bool(event.degraded),
            "reason": event.reason or "",
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._file.close()
