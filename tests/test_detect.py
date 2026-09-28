"""Unit tests for segment merging / thresholding helpers."""

from __future__ import annotations

from volleyball_trim.detect import DetectOptions, Segment, _scores_to_segments, format_timestamp


def test_format_timestamp() -> None:
    assert format_timestamp(65) == "1:05"
    assert format_timestamp(3661) == "1:01:01"


def test_merge_and_pad_segments() -> None:
    times = [i * 0.5 for i in range(40)]  # 0..19.5
    # Two rallies close together, then a gap, then another.
    scores = []
    for t in times:
        if 2 <= t < 5 or 6 <= t < 8 or 14 <= t < 17:
            scores.append(10.0)
        else:
            scores.append(1.0)

    options = DetectOptions(
        min_rally_sec=1.0,
        merge_gap_sec=2.0,
        pad_before_sec=0.5,
        pad_after_sec=0.5,
    )
    segs = _scores_to_segments(times, scores, threshold=5.0, duration=20.0, options=options)
    # First two bursts should merge (gap ~1s < 2s) into one padded segment.
    assert len(segs) == 2
    assert segs[0].start <= 2.0
    assert segs[0].end >= 8.0
    assert isinstance(segs[0], Segment)
    assert segs[1].start >= 12.0


def test_drops_tiny_bursts() -> None:
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 5.5, 6.0]
    scores = [1, 10, 10, 1, 1, 1, 1, 1]
    options = DetectOptions(min_rally_sec=3.0, merge_gap_sec=0.5, pad_before_sec=0, pad_after_sec=0)
    segs = _scores_to_segments(times, scores, threshold=5.0, duration=7.0, options=options)
    assert segs == []
