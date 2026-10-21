"""Assembler and metrics tests.

Run with:  python -m tests.test_assembler   (from the python/ directory)
"""

from __future__ import annotations

import sys

from audioflow.assembler import Assembler, format_transcript, trim_overlap
from audioflow.metrics import Metrics, percentile
from audioflow.scheduler import Event


def partial(chunk, t_start, t_end, text, revision=1):
    return Event("partial", chunk, text=text, t_start=t_start, t_end=t_end,
                 revision=revision)


def final(chunk, t_start, t_end, text, degraded=False, latency=0.1):
    return Event("final", chunk, text=text, t_start=t_start, t_end=t_end,
                 degraded=degraded, latency=latency)


def gap(chunk, t_start, t_end, reason="deadline"):
    return Event("gap", chunk, t_start=t_start, t_end=t_end, reason=reason)


def test_partial_replaces_tentative() -> None:
    asm = Assembler()
    asm.on_event(partial(1, 0.0, 1.0, "hello", revision=1))
    asm.on_event(partial(1, 0.0, 1.5, "hello world", revision=2))
    assert len(asm.tentative()) == 1
    assert asm.tentative()[0].text == "hello world"
    assert asm.tentative()[0].revision == 2
    assert not asm.segments()
    print("ok: partial replaces tentative")


def test_final_replaces_and_commits() -> None:
    asm = Assembler()
    asm.on_event(partial(1, 0.0, 1.0, "hello"))
    asm.on_event(final(1, 0.0, 2.0, "hello world."))
    assert not asm.tentative()
    assert [s.text for s in asm.segments()] == ["hello world."]
    print("ok: final replaces and commits")


def test_out_of_order_final_is_placed_by_time() -> None:
    asm = Assembler()
    asm.on_event(final(2, 5.0, 7.0, "second"))
    asm.on_event(final(1, 0.0, 2.0, "first"))
    assert [s.chunk_id for s in asm.segments()] == [1, 2]
    print("ok: out-of-order final placed by time")


def test_overlap_trimming() -> None:
    trimmed = trim_overlap("we are stuck in a story", "in a story that we have outgrown")
    assert trimmed == "that we have outgrown", trimmed
    print("ok: overlap trimmed")


def test_single_shared_token_is_not_overlap() -> None:
    assert trim_overlap("this is the", "the next thing") == "the next thing"
    print("ok: single token not trimmed")


def test_trim_applied_between_adjacent_finals() -> None:
    asm = Assembler()
    asm.on_event(final(1, 0.0, 4.0, "we are stuck in a story"))
    asm.on_event(final(2, 3.7, 7.0, "in a story that we have outgrown"))
    assert asm.segments()[1].text == "that we have outgrown"
    print("ok: adjacent finals trimmed in place")


def test_no_trim_when_ranges_do_not_touch() -> None:
    asm = Assembler()
    asm.on_event(final(1, 0.0, 2.0, "we are stuck in a story"))
    asm.on_event(final(2, 5.0, 7.0, "in a story that we have outgrown"))
    assert asm.segments()[1].text == "in a story that we have outgrown"
    print("ok: no trim without shared audio")


def test_gap_commits_and_renders() -> None:
    asm = Assembler()
    asm.on_event(final(1, 0.0, 2.0, "hello"))
    asm.on_event(gap(2, 2.0, 3.5))
    assert [s.kind for s in asm.segments()] == ["text", "gap"]
    assert "hello" in asm.transcript()
    assert "[gap 1.5s]" in asm.transcript()
    print("ok: gap committed and rendered")


def test_render_merges_tentative_in_time_order() -> None:
    asm = Assembler()
    asm.on_event(final(1, 0.0, 2.0, "committed"))
    asm.on_event(partial(2, 2.0, 2.8, "tentative"))
    rendered = asm.render()
    assert [s.text for s in rendered] == ["committed", "tentative"]
    assert rendered[1].provisional
    assert "partial" in format_transcript(rendered)
    print("ok: render includes tentative in order")


def test_metrics_aggregate() -> None:
    metrics = Metrics(partial_interval=0.5)
    metrics.observe(partial(1, 0.0, 0.5, "a", revision=1))
    metrics.observe(partial(1, 0.0, 1.0, "a b", revision=2))
    metrics.observe(final(1, 0.0, 2.0, "a b.", latency=0.12))
    metrics.observe(gap(2, 2.0, 3.0))

    snap = metrics.snapshot()
    assert snap["partials"] == 2 and snap["finals"] == 1 and snap["gaps"] == 1
    assert snap["gap_seconds"] == 1.0
    assert snap["text_seconds"] == 2.0
    assert abs(snap["coverage"] - round(2 / 3, 3)) < 1e-9
    assert snap["partial_interval_mean_s"] == 0.5
    assert snap["slip_p95_s"] == 0.0
    assert percentile([0.1, 0.2, 0.3], 0.5) == 0.2
    print("ok: metrics aggregate")


def test_metrics_counts_degraded_and_slip() -> None:
    metrics = Metrics(partial_interval=0.5)
    metrics.observe(partial(1, 0.0, 0.6, "a"))
    metrics.observe(partial(1, 0.0, 1.4, "a b"))
    metrics.observe(final(1, 0.0, 1.4, "a b", degraded=True))
    snap = metrics.snapshot()
    assert snap["degraded"] == 1
    assert snap["slip_p95_s"] == 0.3
    print("ok: degraded and slip counted")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"\n{len(tests)} assembler/metrics tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
