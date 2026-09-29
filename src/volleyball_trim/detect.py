"""Detect active volleyball rallies via residual motion analysis.

Improvements over plain frame-mean motion:
- Court ROI that ignores top/bottom scorebug bands
- Spatial scoring (localized player motion, not whole-frame average)
- Optional person-prior weighting via OpenCV HOG (lightweight ML assist)
- Serve-aware lead-in pads that snap back to a toss/onset cue
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
class CourtRoi:
    """Normalized crop that keeps the court and drops scorebug/sideline chrome.

    Fractions are relative to frame height/width and are *excluded* from analysis.
    """

    top: float = 0.12
    bottom: float = 0.08
    left: float = 0.04
    right: float = 0.04

    def clamp(self) -> CourtRoi:
        return CourtRoi(
            top=float(np.clip(self.top, 0.0, 0.40)),
            bottom=float(np.clip(self.bottom, 0.0, 0.40)),
            left=float(np.clip(self.left, 0.0, 0.35)),
            right=float(np.clip(self.right, 0.0, 0.35)),
        )


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
    court_roi: CourtRoi = field(default_factory=CourtRoi)
    # Weight motion by HOG person detections when available.
    use_person_prior: bool = True
    # Snap pad-before back to a serve-toss / motion-onset cue.
    serve_aware_pads: bool = True
    # Spatial grid for localized motion (cols x rows inside the ROI).
    spatial_cols: int = 6
    spatial_rows: int = 4


@dataclass
class DetectResult:
    segments: list[Segment]
    duration: float
    fps: float
    sample_times: list[float]
    motion_scores: list[float]
    threshold: float
    serve_onsets: list[float] = field(default_factory=list)


def _resize_gray(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        scale = width / w
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _roi_slices(h: int, w: int, roi: CourtRoi) -> tuple[slice, slice]:
    roi = roi.clamp()
    y0 = int(round(h * roi.top))
    y1 = int(round(h * (1.0 - roi.bottom)))
    x0 = int(round(w * roi.left))
    x1 = int(round(w * (1.0 - roi.right)))
    y0 = max(0, min(y0, h - 2))
    y1 = max(y0 + 1, min(y1, h))
    x0 = max(0, min(x0, w - 2))
    x1 = max(x0 + 1, min(x1, w))
    return slice(y0, y1), slice(x0, x1)


def _court_mask(h: int, w: int, roi: CourtRoi) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    ys, xs = _roi_slices(h, w, roi)
    mask[ys, xs] = 1.0
    return mask


def _person_weight_map(gray: np.ndarray, roi: CourtRoi) -> np.ndarray | None:
    """Soft weight map from OpenCV HOG pedestrians (optional ML assist)."""
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    except Exception:  # pragma: no cover — OpenCV build without HOG
        return None

    h, w = gray.shape[:2]
    ys, xs = _roi_slices(h, w, roi)
    crop = gray[ys, xs]
    if crop.size == 0:
        return None

    # HOG prefers ~64px-wide upright people; upscale small analysis frames a bit.
    scale = 1.0
    if crop.shape[1] < 240:
        scale = 240.0 / crop.shape[1]
        crop_l = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        crop_l = crop

    try:
        boxes, weights = hog.detectMultiScale(
            crop_l,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
    except Exception:  # pragma: no cover
        return None

    weight = np.ones((h, w), dtype=np.float32) * 0.35
    weight[ys, xs] = 0.55
    if len(boxes) == 0:
        return weight

    y0, x0 = ys.start or 0, xs.start or 0
    for (bx, by, bw, bh), conf in zip(boxes, weights, strict=False):
        if scale != 1.0:
            bx, by, bw, bh = (int(v / scale) for v in (bx, by, bw, bh))
        strength = float(np.clip(0.8 + 0.6 * float(conf), 0.8, 2.2))
        x_a = max(0, x0 + bx)
        y_a = max(0, y0 + by)
        x_b = min(w, x0 + bx + bw)
        y_b = min(h, y0 + by + bh)
        if x_b > x_a and y_b > y_a:
            weight[y_a:y_b, x_a:x_b] = np.maximum(weight[y_a:y_b, x_a:x_b], strength)
    # Soft blur so weights don't create hard edges.
    weight = cv2.GaussianBlur(weight, (21, 21), 0)
    return weight


def _spatial_motion_score(
    diff: np.ndarray,
    mask: np.ndarray,
    person_weight: np.ndarray | None,
    *,
    cols: int,
    rows: int,
) -> float:
    """Score localized activity inside the court instead of a whole-frame mean."""
    weighted = diff.astype(np.float32) * mask
    if person_weight is not None:
        weighted *= person_weight

    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        return float(np.mean(diff))

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    region = weighted[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0

    rh, rw = region.shape
    cell_h = max(1, rh // max(1, rows))
    cell_w = max(1, rw // max(1, cols))
    cell_scores: list[float] = []
    for r in range(rows):
        for c in range(cols):
            cy0 = r * cell_h
            cx0 = c * cell_w
            cy1 = rh if r == rows - 1 else (r + 1) * cell_h
            cx1 = rw if c == cols - 1 else (c + 1) * cell_w
            cell = region[cy0:cy1, cx0:cx1]
            if cell.size == 0:
                continue
            # Mean of the top quartile inside the cell → local bursts matter.
            flat = cell.reshape(-1)
            if flat.size < 4:
                cell_scores.append(float(flat.mean()))
            else:
                thr = float(np.percentile(flat, 75))
                hot = flat[flat >= thr]
                cell_scores.append(float(hot.mean()) if hot.size else float(flat.mean()))

    if not cell_scores:
        return float(region.mean())

    arr = np.asarray(cell_scores, dtype=np.float64)
    # Emphasize the busiest court cells (players / ball), ignore quiet corners.
    top_k = max(1, len(arr) // 3)
    return float(np.mean(np.partition(arr, -top_k)[-top_k:]))


def _residual_motion(
    prev: np.ndarray,
    curr: np.ndarray,
    *,
    mask: np.ndarray,
    person_weight: np.ndarray | None,
    cols: int,
    rows: int,
) -> float:
    """Motion score after removing bulk camera translation (pans/tilts)."""
    prev_f = np.float32(prev)
    curr_f = np.float32(curr)
    # Estimate shift primarily from the court ROI so scorebugs don't dominate.
    ys, xs = np.where(mask > 0.5)
    if len(ys):
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        shift, _ = cv2.phaseCorrelate(prev_f[y0:y1, x0:x1], curr_f[y0:y1, x0:x1])
    else:
        shift, _ = cv2.phaseCorrelate(prev_f, curr_f)
    dx, dy = shift
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
    return _spatial_motion_score(diff, mask, person_weight, cols=cols, rows=rows)


def _adaptive_threshold(scores: list[float], sensitivity: float) -> float:
    """Pick a motion cutoff between quiet downtime and rally movement."""
    arr = np.asarray(scores, dtype=np.float64)
    sens = float(np.clip(sensitivity, 0.0, 1.0))
    quiet = float(np.percentile(arr, 20))
    play = float(np.percentile(arr, 80))
    separation = play - quiet

    if separation < 0.08:
        pct = float(np.clip(75 - sens * 35, 40, 85))
        return float(np.percentile(arr, pct))

    alpha = 0.78 - sens * 0.55
    return float(quiet + alpha * separation)


def _find_serve_onset(
    times: list[float],
    scores: list[float],
    rally_start: float,
    *,
    pad_before: float,
    quiet_level: float,
) -> float | None:
    """Search just before a rally for a serve-toss / motion-onset cue.

    Looks for the earliest sharp rise out of quiet within the pad window,
    which is typically the toss or approach rather than mid-rally chaos.
    """
    if pad_before <= 0 or not times:
        return None

    window_start = max(0.0, rally_start - max(pad_before * 2.5, pad_before + 1.0))
    idxs = [i for i, t in enumerate(times) if window_start <= t <= rally_start + 1e-6]
    if len(idxs) < 3:
        return None

    best_t: float | None = None
    # Require a rise from near-quiet into activity.
    rise_need = max(0.35, (float(np.median(scores)) - quiet_level) * 0.35)
    for k in range(1, len(idxs)):
        i_prev, i_cur = idxs[k - 1], idxs[k]
        prev_s, cur_s = scores[i_prev], scores[i_cur]
        t_cur = times[i_cur]
        if t_cur > rally_start:
            break
        if prev_s <= quiet_level * 1.15 and (cur_s - prev_s) >= rise_need and cur_s >= quiet_level + rise_need:
            # Prefer the earliest clear onset in the window.
            if best_t is None or t_cur < best_t:
                best_t = t_cur
                # Keep scanning for an even earlier one; don't break.
    return best_t


def _scores_to_segments(
    times: list[float],
    scores: list[float],
    threshold: float,
    duration: float,
    options: DetectOptions,
) -> tuple[list[Segment], list[float]]:
    if not times:
        return [], []

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
        return [], []

    merged: list[Segment] = [raw[0]]
    for seg in raw[1:]:
        prev = merged[-1]
        if seg.start - prev.end <= options.merge_gap_sec:
            merged[-1] = Segment(prev.start, seg.end)
        else:
            merged.append(seg)

    quiet_level = float(np.percentile(np.asarray(scores, dtype=np.float64), 25))
    serve_onsets: list[float] = []
    padded: list[Segment] = []
    for seg in merged:
        onset = None
        if options.serve_aware_pads:
            onset = _find_serve_onset(
                times,
                scores,
                seg.start,
                pad_before=options.pad_before_sec,
                quiet_level=quiet_level,
            )
        if onset is not None:
            # Keep a little pre-roll before the toss itself.
            s = max(0.0, onset - 0.35)
            serve_onsets.append(onset)
        else:
            s = max(0.0, seg.start - options.pad_before_sec)
        e = min(duration, seg.end + options.pad_after_sec)
        if e - s >= options.min_rally_sec:
            padded.append(Segment(s, e))

    if not padded:
        return [], serve_onsets

    final: list[Segment] = [padded[0]]
    for seg in padded[1:]:
        prev = final[-1]
        if seg.start <= prev.end:
            final[-1] = Segment(prev.start, max(prev.end, seg.end))
        else:
            final.append(seg)
    return final, serve_onsets


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
        mask: np.ndarray | None = None
        person_weight: np.ndarray | None = None
        person_refresh_every = max(1, int(round(options.sample_fps * 2)))  # ~2s
        sample_i = 0
        index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step != 0:
                index += 1
                continue

            gray = _resize_gray(frame, options.analysis_width)
            if mask is None:
                mask = _court_mask(gray.shape[0], gray.shape[1], options.court_roi)

            if options.use_person_prior and (
                person_weight is None or sample_i % person_refresh_every == 0
            ):
                person_weight = _person_weight_map(gray, options.court_roi)

            t = index / fps
            if prev_gray is not None and mask is not None:
                scores.append(
                    _residual_motion(
                        prev_gray,
                        gray,
                        mask=mask,
                        person_weight=person_weight if options.use_person_prior else None,
                        cols=options.spatial_cols,
                        rows=options.spatial_rows,
                    )
                )
                times.append(t)
            prev_gray = gray
            sample_i += 1
            index += 1

        if duration <= 0 and times:
            duration = times[-1] + (step / fps)

        if not scores:
            return DetectResult([], duration, fps, [], [], 0.0, [])

        threshold = _adaptive_threshold(scores, options.sensitivity)
        segments, serve_onsets = _scores_to_segments(
            times, scores, threshold, duration, options
        )
        return DetectResult(
            segments, duration, fps, times, scores, threshold, serve_onsets
        )
    finally:
        cap.release()


def render_motion_timeline(
    result: DetectResult,
    *,
    width: int = 900,
    height: int = 220,
) -> np.ndarray:
    """Render a simple motion / keep timeline as a BGR image for the review UI."""
    img = np.full((height, width, 3), 244, dtype=np.uint8)
    img[:] = (245, 247, 244)
    if result.duration <= 0 or not result.sample_times:
        cv2.putText(
            img,
            "No motion samples",
            (24, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 60, 50),
            1,
            cv2.LINE_AA,
        )
        return img

    pad_l, pad_r, pad_t, pad_b = 48, 16, 24, 36
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    scores = np.asarray(result.motion_scores, dtype=np.float64)
    times = np.asarray(result.sample_times, dtype=np.float64)
    smax = max(float(scores.max()), result.threshold * 1.5, 1e-3)

    # Kept rally bands.
    for seg in result.segments:
        x0 = pad_l + int(plot_w * (seg.start / result.duration))
        x1 = pad_l + int(plot_w * (seg.end / result.duration))
        cv2.rectangle(
            img,
            (x0, pad_t),
            (max(x0 + 1, x1), pad_t + plot_h),
            (210, 230, 200),
            -1,
        )

    # Threshold line.
    ty = pad_t + plot_h - int(plot_h * (result.threshold / smax))
    cv2.line(img, (pad_l, ty), (pad_l + plot_w, ty), (80, 120, 180), 1, cv2.LINE_AA)

    # Motion polyline.
    pts = []
    for t, s in zip(times, scores, strict=True):
        x = pad_l + int(plot_w * (float(t) / result.duration))
        y = pad_t + plot_h - int(plot_h * (float(s) / smax))
        pts.append((x, y))
    if len(pts) >= 2:
        cv2.polylines(
            img,
            [np.asarray(pts, dtype=np.int32)],
            False,
            (31, 107, 74),
            2,
            cv2.LINE_AA,
        )

    for onset in result.serve_onsets:
        x = pad_l + int(plot_w * (onset / result.duration))
        cv2.line(img, (x, pad_t), (x, pad_t + plot_h), (40, 90, 200), 1, cv2.LINE_AA)

    cv2.putText(
        img,
        "motion",
        (8, pad_t + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (60, 80, 70),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "0",
        (8, pad_t + plot_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (90, 100, 95),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        format_timestamp(result.duration),
        (width - 70, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (60, 80, 70),
        1,
        cv2.LINE_AA,
    )
    return img


def segments_to_rows(segments: list[Segment]) -> list[list[float | int]]:
    rows: list[list[float | int]] = []
    for i, seg in enumerate(segments, 1):
        rows.append(
            [
                i,
                round(seg.start, 3),
                round(seg.end, 3),
                round(seg.duration, 3),
            ]
        )
    return rows


def rows_to_segments(rows) -> list[Segment]:
    """Parse an editable review table back into segments."""
    if rows is None:
        return []
    # Gradio may hand back a pandas DataFrame or list of lists / dicts.
    try:
        import pandas as pd

        if isinstance(rows, pd.DataFrame):
            records = rows.to_dict(orient="records")
        else:
            records = rows
    except Exception:
        records = rows

    segs: list[Segment] = []
    if isinstance(records, list) and records and isinstance(records[0], dict):
        for row in records:
            try:
                start = float(row.get("start_sec", row.get("Start", 0)))
                end = float(row.get("end_sec", row.get("End", 0)))
            except (TypeError, ValueError):
                continue
            if end > start:
                segs.append(Segment(start, end))
    else:
        for row in records or []:
            try:
                if len(row) >= 3:
                    start = float(row[1])
                    end = float(row[2])
                else:
                    continue
            except (TypeError, ValueError, IndexError):
                continue
            if end > start:
                segs.append(Segment(start, end))

    segs.sort(key=lambda s: s.start)
    # Merge overlaps after manual edits.
    if not segs:
        return []
    out = [segs[0]]
    for seg in segs[1:]:
        prev = out[-1]
        if seg.start <= prev.end:
            out[-1] = Segment(prev.start, max(prev.end, seg.end))
        else:
            out.append(seg)
    return out


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
    ]
    if result.serve_onsets:
        lines.append(f"Serve onsets snapped: {len(result.serve_onsets)}")
    lines.extend(["", "Segments:"])
    if not result.segments:
        lines.append("  (none — try raising sensitivity)")
    else:
        for i, seg in enumerate(result.segments, 1):
            lines.append(
                f"  {i:02d}. {format_timestamp(seg.start)} → {format_timestamp(seg.end)}"
                f"  ({seg.duration:.1f}s)"
            )
    return "\n".join(lines)
