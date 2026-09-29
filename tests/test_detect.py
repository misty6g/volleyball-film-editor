"""Unit tests for segment merging / thresholding helpers."""

from __future__ import annotations

import numpy as np

from volleyball_trim.detect import (
    CourtRoi,
    DetectOptions,
    Segment,
    _court_mask,
    _find_serve_onset,
    _scores_to_segments,
    _spatial_motion_score,
    format_timestamp,
    rows_to_segments,
    segments_to_rows,
)


def test_format_timestamp() -> None:
    assert format_timestamp(65) == "1:05"
    assert format_timestamp(3661) == "1:01:01"


def test_merge_and_pad_segments() -> None:
    times = [i * 0.5 for i in range(40)]  # 0..19.5
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
        serve_aware_pads=False,
    )
    segs, onsets = _scores_to_segments(
        times, scores, threshold=5.0, duration=20.0, options=options
    )
    assert len(segs) == 2
    assert segs[0].start <= 2.0
    assert segs[0].end >= 8.0
    assert isinstance(segs[0], Segment)
    assert segs[1].start >= 12.0
    assert onsets == []


def test_drops_tiny_bursts() -> None:
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 5.5, 6.0]
    scores = [1, 10, 10, 1, 1, 1, 1, 1]
    options = DetectOptions(
        min_rally_sec=3.0,
        merge_gap_sec=0.5,
        pad_before_sec=0,
        pad_after_sec=0,
        serve_aware_pads=False,
    )
    segs, _ = _scores_to_segments(
        times, scores, threshold=5.0, duration=7.0, options=options
    )
    assert segs == []


def test_court_mask_excludes_scorebug_bands() -> None:
    mask = _court_mask(100, 200, CourtRoi(top=0.10, bottom=0.10, left=0.0, right=0.0))
    assert mask[:10, :].sum() == 0
    assert mask[-10:, :].sum() == 0
    assert mask[50, 100] == 1.0


def test_spatial_score_prefers_hot_cells() -> None:
    diff = np.zeros((40, 60), dtype=np.float32)
    diff[5:15, 5:15] = 50  # localized burst
    mask = np.ones_like(diff)
    score = _spatial_motion_score(diff, mask, None, cols=6, rows=4)
    assert score > 10


def test_serve_onset_snaps_before_rally() -> None:
    times = [i * 0.25 for i in range(40)]
    scores = []
    for t in times:
        if 4.0 <= t < 4.5:
            scores.append(6.0)  # toss
        elif 5.0 <= t < 8.0:
            scores.append(12.0)  # rally
        else:
            scores.append(1.0)
    onset = _find_serve_onset(
        times, scores, rally_start=5.0, pad_before=1.5, quiet_level=1.5
    )
    assert onset is not None
    assert 3.5 <= onset <= 5.0

    options = DetectOptions(
        min_rally_sec=1.0,
        merge_gap_sec=0.5,
        pad_before_sec=1.5,
        pad_after_sec=0.25,
        serve_aware_pads=True,
    )
    segs, onsets = _scores_to_segments(
        times, scores, threshold=8.0, duration=10.0, options=options
    )
    assert len(segs) == 1
    assert onsets
    assert segs[0].start < 5.0
    assert segs[0].start >= onsets[0] - 0.5


def test_rows_roundtrip() -> None:
    segs = [Segment(1.0, 3.5), Segment(10.0, 12.25)]
    rows = segments_to_rows(segs)
    back = rows_to_segments(rows)
    assert len(back) == 2
    assert abs(back[0].start - 1.0) < 1e-6
    assert abs(back[1].end - 12.25) < 1e-6
