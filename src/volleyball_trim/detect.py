"""Detect active volleyball rallies via residual motion analysis."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Segment:
    """Inclusive start / exclusive end times in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class DetectOptions:
    """Tunables for rally detection on sideline / film video."""

    sample_fps: float = 4.0
    # Higher keeps more footage (lower motion threshold).
    sensitivity: float = 0.55
    min_rally_sec: float = 2.0
    merge_gap_sec: float = 2.5
    pad_before_sec: float = 1.5
    pad_after_sec: float = 1.0
    analysis_width: int = 320


@dataclass
class DetectResult:
    segments: list[Segment]
    duration: float
    fps: float
    sample_times: list[float]
    motion_scores: list[float]
    threshold: float


def _resize_gray(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        scale = width / w
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _residual_motion(prev: np.ndarray, curr: np.ndarray) -> float:
    """Motion score after removing bulk camera translation (pans/tilts)."""
    prev_f = np.float32(prev)
    curr_f = np.float32(curr)
    shift, _ = cv2.phaseCorrelate(prev_f, curr_f)
    dx, dy = shift
    # Cap shift so wild outliers don't blank the frame.
    dx = float(np.clip(dx, -40, 40))
    dy = float(np.clip(dy, -40, 40))
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = cv2.warpAffine(
        curr,
        m,
        (curr.shape[1], curr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    diff = cv2.absdiff(prev, aligned)
    # Ignore a thin border where warp introduces artifacts.
    y0, x0 = 4, 4
    y1, x1 = diff.shape[0] - 4, diff.shape[1] - 4
    if y1 <= y0 or x1 <= x0:
        return float(np.mean(diff))
    return float(np.mean(diff[y0:y1, x0:x1]))


def _adaptive_threshold(scores: list[float], sensitivity: float) -> float:
    """Pick a motion cutoff between quiet downtime and rally movement."""
    arr = np.asarray(scores, dtype=np.float64)
    sens = float(np.clip(sensitivity, 0.0, 1.0))
    quiet = float(np.percentile(arr, 20))
    play = float(np.percentile(arr, 80))
    separation = play - quiet

    if separation < 0.08:
        # Little contrast (constant motion or static film) — percentile fallback.
        pct = float(np.clip(75 - sens * 35, 40, 85))
        return float(np.percentile(arr, pct))

    # sens 0 → cut near play (strict); sens 1 → cut nearer quiet (keep more).
    alpha = 0.78 - sens * 0.55
    return float(quiet + alpha * separation)


def _scores_to_segments(
    times: list[float],
    scores: list[float],
    threshold: float,
    duration: float,
    options: DetectOptions,
) -> list[Segment]:
    if not times:
        return []

    # Hysteresis + short confirmation so one noisy sample doesn't stitch points.
    high = threshold
    low = threshold * 0.55
    active = False
    start: float | None = None
    pending_start: float | None = None
    enter_run = 0
    exit_run = 0
    need = 2
    raw: list[Segment] = []

    for t, score in zip(times, scores, strict=True):
        if not active:
            if score >= high:
                if pending_start is None:
                    pending_start = t
                enter_run += 1
                if enter_run >= need:
                    active = True
                    start = pending_start
                    pending_start = None
                    exit_run = 0
            else:
                enter_run = 0
                pending_start = None
        else:
            if score < low:
                exit_run += 1
                if exit_run >= need:
                    active = False
                    if start is not None:
                        raw.append(Segment(start, t))
                    start = None
                    enter_run = 0
                    pending_start = None
            else:
                exit_run = 0

    if active and start is not None:
        raw.append(Segment(start, duration))

    if not raw:
        return []

    # Merge rallies separated by only a short lull (set transitions, soft digs).
    merged: list[Segment] = [raw[0]]
    for seg in raw[1:]:
        prev = merged[-1]
        if seg.start - prev.end <= options.merge_gap_sec:
            merged[-1] = Segment(prev.start, seg.end)
        else:
            merged.append(seg)

    # Pad for serve toss / point end, then clamp and drop tiny clips.
    padded: list[Segment] = []
    for seg in merged:
        s = max(0.0, seg.start - options.pad_before_sec)
        e = min(duration, seg.end + options.pad_after_sec)
        if e - s >= options.min_rally_sec:
            padded.append(Segment(s, e))

    # Re-merge if padding caused overlaps.
    if not padded:
        return []
    final: list[Segment] = [padded[0]]
    for seg in padded[1:]:
        prev = final[-1]
        if seg.start <= prev.end:
            final[-1] = Segment(prev.start, max(prev.end, seg.end))
        else:
            final.append(seg)
    return final


def detect_rallies(video_path: str, options: DetectOptions | None = None) -> DetectResult:
    """Scan a game film and return kept rally segments."""
    options = options or DetectOptions()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count > 0 else 0.0

        step = max(1, int(round(fps / options.sample_fps)))
        times: list[float] = []
        scores: list[float] = []
        prev_gray: np.ndarray | None = None
        index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step != 0:
                index += 1
                continue

            gray = _resize_gray(frame, options.analysis_width)
            t = index / fps
            if prev_gray is not None:
                scores.append(_residual_motion(prev_gray, gray))
                times.append(t)
            prev_gray = gray
            index += 1

        if duration <= 0 and times:
            duration = times[-1] + (step / fps)

        if not scores:
            return DetectResult([], duration, fps, [], [], 0.0)

        threshold = _adaptive_threshold(scores, options.sensitivity)
        segments = _scores_to_segments(times, scores, threshold, duration, options)
        return DetectResult(segments, duration, fps, times, scores, threshold)
    finally:
        cap.release()


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def summarize_result(result: DetectResult) -> str:
    kept = sum(s.duration for s in result.segments)
    removed = max(0.0, result.duration - kept)
    lines = [
        f"Source length: {format_timestamp(result.duration)} ({result.duration:.1f}s)",
        f"Rallies kept: {len(result.segments)}",
        f"Output length: {format_timestamp(kept)} ({kept:.1f}s)",
        f"Downtime removed: {format_timestamp(removed)} ({removed:.1f}s)",
        f"Motion threshold: {result.threshold:.2f}",
        "",
        "Segments:",
    ]
    if not result.segments:
        lines.append("  (none — try raising sensitivity)")
    else:
        for i, seg in enumerate(result.segments, 1):
            lines.append(
                f"  {i:02d}. {format_timestamp(seg.start)} → {format_timestamp(seg.end)}"
                f"  ({seg.duration:.1f}s)"
            )
    return "\n".join(lines)
