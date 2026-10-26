"""Assembler: turns scheduler events into an ordered, committed transcript.

Event semantics:

* a partial sets the tentative text for its chunk and never commits;
* a final, authoritative or degraded, replaces every partial for that chunk;
* gaps are committed ranges with no text;
* adjacent forced-cut finals are trimmed textually, because the ASR returns no
  word timestamps;
* results are placed by ``t_start``, not arrival order, so a retried final that
  lands late still sits in the right place.

The assembler owns no threads and no I/O: it takes events and exposes state.
"""

from __future__ import annotations

import re
from bisect import insort
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .scheduler import Event

_TOKEN_RE = re.compile(r"\S+")
_NORMALISE_RE = re.compile(r"[^\w']+", re.UNICODE)


@dataclass
class Segment:
    kind: str                 # "text" | "gap"
    chunk_id: int
    t_start: float
    t_end: float
    text: str = ""
    degraded: bool = False
    provisional: bool = False
    reason: str = ""
    revision: int = 0


def _normalise(text: str) -> str:
    return _NORMALISE_RE.sub(" ", text.casefold()).strip()


def _tokens(text: str) -> list[str]:
    return _normalise(text).split()


def _token_spans(text: str) -> list[re.Match]:
    return list(_TOKEN_RE.finditer(text))


def trim_overlap(previous: str, current: str, max_tokens: int = 12) -> str:
    """Drop the part of ``current`` that repeats the tail of ``previous``.

    Forced cuts share up to O seconds of audio, so consecutive finals can
    transcribe the same words twice. There are no word timestamps, so the match
    is textual: take the longest suffix of the previous text that is also a
    prefix of the current one, bounded by ``max_tokens``, and cut it.

    A single shared token is not treated as overlap: "the" repeating at a
    boundary is far more likely to be two real words than a duplicate.
    """
    prev_tokens = _tokens(previous)[-max_tokens:]
    cur_tokens = _tokens(current)[:max_tokens]
    if not prev_tokens or not cur_tokens:
        return current

    shared = 0
    for size in range(min(len(prev_tokens), len(cur_tokens)), 0, -1):
        if prev_tokens[-size:] == cur_tokens[:size]:
            shared = size
            break
    if shared < 2:
        return current

    spans = _token_spans(current)
    if shared >= len(spans):
        return ""
    return current[spans[shared - 1].end():].lstrip()


class Assembler:
    """Collects events into committed segments plus a tentative overlay."""

    def __init__(self, max_trim_tokens: int = 12) -> None:
        self._committed: list[Segment] = []
        self._tentative: dict[int, Segment] = {}
        self._max_trim_tokens = max_trim_tokens

    # -- ingest ------------------------------------------------------------

    def on_event(self, event: Event) -> Optional[Segment]:
        if event.kind == "partial":
            segment = Segment(
                kind="text",
                chunk_id=event.chunk_id,
                t_start=event.t_start,
                t_end=event.t_end,
                text=event.text,
                provisional=True,
                revision=event.revision,
            )
            self._tentative[event.chunk_id] = segment
            return segment

        if event.kind == "final":
            self._tentative.pop(event.chunk_id, None)
            segment = Segment(
                kind="text",
                chunk_id=event.chunk_id,
                t_start=event.t_start,
                t_end=event.t_end,
                text=event.text,
                degraded=event.degraded,
                reason=event.reason,
            )
            self._trim_against_predecessor(segment)
            self._insert(segment)
            return segment

        if event.kind == "gap":
            self._tentative.pop(event.chunk_id, None)
            segment = Segment(
                kind="gap",
                chunk_id=event.chunk_id,
                t_start=event.t_start,
                t_end=event.t_end,
                reason=event.reason,
            )
            self._insert(segment)
            return segment

        return None

    # -- queries -----------------------------------------------------------

    def segments(self) -> list[Segment]:
        return list(self._committed)

    def tentative(self) -> list[Segment]:
        return sorted(self._tentative.values(), key=lambda s: s.t_start)

    def render(self) -> list[Segment]:
        """Committed segments plus the tentative overlay, ordered by time."""
        return sorted(self._committed + list(self._tentative.values()),
                      key=lambda s: s.t_start)

    def transcript(self) -> str:
        """Committed transcript; gaps are rendered as markers."""
        parts = []
        for segment in self._committed:
            if segment.kind == "gap":
                parts.append("[gap %.1fs]" % (segment.t_end - segment.t_start))
            elif segment.text:
                parts.append(segment.text)
        return " ".join(parts)

    # -- internals ---------------------------------------------------------

    def _insert(self, segment: Segment) -> None:
        insort(self._committed, segment, key=lambda s: s.t_start)

    def _predecessor(self, segment: Segment) -> Optional[Segment]:
        best = None
        for candidate in self._committed:
            if candidate.t_start < segment.t_start and candidate.kind == "text":
                best = candidate
        return best

    def _trim_against_predecessor(self, segment: Segment) -> None:
        previous = self._predecessor(segment)
        if previous is None or not previous.text or not segment.text:
            return
        if previous.t_end + 0.01 < segment.t_start:
            return                      # no shared audio, nothing to trim
        segment.text = trim_overlap(previous.text, segment.text,
                                    self._max_trim_tokens)


def format_transcript(segments: Iterable[Segment]) -> str:
    """Human-readable rendering used by the tools."""
    lines = []
    for segment in segments:
        if segment.kind == "gap":
            lines.append("[%6.1f–%6.1f]  <gap: %s>"
                         % (segment.t_start, segment.t_end, segment.reason or "?"))
            continue
        tag = "degraded" if segment.degraded else ("partial" if segment.provisional else "final")
        lines.append("[%6.1f–%6.1f]  %-8s  %s"
                     % (segment.t_start, segment.t_end, tag, segment.text))
    return "\n".join(lines)
