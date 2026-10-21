"""Scheduler tests driven by a virtual clock.

Run with:  python -m tests.test_scheduler   (from the python/ directory)
"""

from __future__ import annotations

import sys

from audioflow.asr.fake import FakeAsrClient, VirtualClock
from audioflow.scheduler import Scheduler


def pcm(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * 16000)


def make(**kwargs):
    clock = VirtualClock()
    events = []
    client = FakeAsrClient(clock=clock, **kwargs.pop("client", {}))
    sched = Scheduler(client, events.append, clock=clock, **kwargs)
    return clock, client, sched, events


def test_partial_before_slot_is_rejected() -> None:
    clock, _, sched, events = make()
    ch = sched.open_chunk(1, start_sample=0, now=clock())
    assert sched.submit_partial(ch, pcm(0.5), now=0.5) is None, "too early"
    assert not events
    result = sched.submit_partial(ch, pcm(0.9), now=0.9)
    assert result is not None
    assert events[0].kind == "partial" and events[0].revision == 1
    print("ok: partial slot gating")


def test_partial_text_is_cumulative_per_revision() -> None:
    clock, _, sched, events = make()
    ch = sched.open_chunk(1, start_sample=0, now=0.0)
    sched.submit_partial(ch, pcm(0.9), now=0.9)
    clock.now = 1.4
    sched.submit_partial(ch, pcm(1.4), now=1.4)
    assert [e.revision for e in events] == [1, 2]
    assert events[1].t_end > events[0].t_end
    print("ok: cumulative partial revisions")


def test_final_after_partial_commits_clean() -> None:
    clock, _, sched, events = make()
    ch = sched.open_chunk(1, start_sample=0, now=0.0)
    sched.submit_partial(ch, pcm(0.9), now=0.9)
    sched.submit_final(ch, pcm(2.0), end_sample=32000, now=2.0)
    assert sched.pump(2.0) is True
    final = events[-1]
    assert final.kind == "final" and not final.degraded
    assert sched.stats.finals == 1 and sched.stats.degraded == 0
    print("ok: clean final")


def test_deadline_loses_final_and_degrades() -> None:
    clock, _, sched, events = make(max_lag=2.0)
    ch = sched.open_chunk(1, start_sample=0, now=0.0)
    sched.submit_partial(ch, pcm(0.9), now=0.9)
    sched.submit_final(ch, pcm(2.0), end_sample=32000, now=2.0)

    assert sched.pump(2.5) is True, "should run while inside max_lag"
    assert events[-1].kind == "final" and not events[-1].degraded

    # Second chunk: let the deadline pass without pumping.
    ch2 = sched.open_chunk(2, start_sample=32000, now=3.0)
    sched.submit_partial(ch2, pcm(1.0), now=3.9)
    sched.submit_final(ch2, pcm(2.0), end_sample=64000, now=4.0)
    sched.pump(7.0)

    kinds = [e.kind for e in events if e.chunk_id == 2]
    assert kinds == ["partial", "final", "gap"], kinds
    degraded = [e for e in events if e.chunk_id == 2 and e.degraded]
    assert degraded and degraded[0].text == "oneshot:1.0s"
    assert sched.stats.degraded == 1 and sched.stats.gaps == 1
    print("ok: deadline -> degraded final + gap")


def test_no_partial_means_whole_chunk_is_a_gap() -> None:
    clock, _, sched, events = make(max_lag=1.0)
    ch = sched.open_chunk(1, start_sample=0, now=0.0)
    sched.submit_final(ch, pcm(2.0), end_sample=32000, now=1.0)
    sched.pump(5.0)
    assert [e.kind for e in events] == ["gap"]
    assert events[0].t_start == 0.0 and events[0].t_end == 2.0
    print("ok: gap without partial")


def test_connection_refused_retries_with_backoff() -> None:
    clock, client, sched, events = make(client={"failures": ["refused"]})
    ch = sched.open_chunk(1, start_sample=0, now=0.0)
    sched.submit_final(ch, pcm(2.0), end_sample=32000, now=2.0)

    assert sched.pump(2.0) is True
    assert not [e for e in events if e.kind == "final"], "first attempt failed"
    assert sched.stats.retries == 1

    assert sched.pump(2.2) is False, "backoff has not elapsed"
    assert sched.pump(2.6) is True
    assert events[-1].kind == "final" and not events[-1].degraded
    print("ok: refused -> retry -> final")


def test_ambiguous_timeout_is_not_retried() -> None:
    clock, client, sched, events = make()
    ch = sched.open_chunk(1, start_sample=0, now=0.0)
    sched.submit_partial(ch, pcm(0.9), now=0.9)
    client.failures.append("timeout")          # only the final's call fails
    sched.submit_final(ch, pcm(2.0), end_sample=32000, now=2.0)
    sched.pump(2.0)

    assert sched.stats.retries == 0
    degraded = [e for e in events if e.degraded]
    assert degraded and degraded[0].text == "oneshot:0.9s", events
    assert any(e.kind == "gap" for e in events)
    print("ok: ambiguous timeout not retried")


def test_final_waits_for_a_free_slot_window() -> None:
    clock, _, sched, events = make(s=0.5, final_cost=1.0)
    ch = sched.open_chunk(1, start_sample=0, now=0.0)

    other = sched.open_chunk(2, start_sample=0, now=0.0)
    other.next_slot = 1.0                      # another chunk's slot is imminent
    sched.submit_final(ch, pcm(2.0), end_sample=32000, now=0.9)
    assert sched.pump(0.9) is False, "1.0 s final would overrun the 1.0 s slot"

    other.next_slot = 5.0
    assert sched.pump(0.9) is True
    print("ok: final admission waits for the slot window")


def test_single_lane() -> None:
    clock, client, sched, events = make()
    ch = sched.open_chunk(1, start_sample=0, now=0.0)
    sched.submit_partial(ch, pcm(0.9), now=0.9)
    sched.submit_final(ch, pcm(2.0), end_sample=32000, now=2.0)
    sched.pump(2.0)
    assert client.stats.max_concurrent == 1
    assert client.stats.calls == 2
    print("ok: single lane")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"\n{len(tests)} scheduler tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
