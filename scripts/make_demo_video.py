#!/usr/bin/env python3
"""Synthesize a short fake 'volleyball film' for smoke tests.

Pattern: quiet downtime (static court) → busy rally (moving players/ball) → repeat.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def draw_court(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    frame[:] = (52, 120, 72)  # BGR green court
    cv2.rectangle(frame, (40, 40), (w - 40, h - 40), (240, 240, 240), 2)
    cv2.line(frame, (w // 2, 40), (w // 2, h - 40), (230, 230, 230), 2)
    # Net
    cv2.line(frame, (w // 2, 80), (w // 2, h - 80), (30, 30, 30), 4)


def draw_players(frame: np.ndarray, t: float, active: bool, t0: float) -> None:
    h, w = frame.shape[:2]
    # Phase relative to rally start so motion is continuous (no RNG jumps).
    phase = t - t0
    base_positions = [
        (0.25, 0.35),
        (0.25, 0.55),
        (0.25, 0.75),
        (0.75, 0.35),
        (0.75, 0.55),
        (0.75, 0.75),
    ]
    for i, (bx, by) in enumerate(base_positions):
        if active:
            jx = 0.06 * np.sin(phase * 9 + i * 1.7)
            jy = 0.08 * np.cos(phase * 11 + i * 1.3)
        else:
            jx = 0.0
            jy = 0.0
        x = int((bx + jx) * w)
        y = int((by + jy) * h)
        color = (40, 40, 200) if bx < 0.5 else (200, 80, 40)
        cv2.circle(frame, (x, y), 16, color, -1)

    if active:
        bx = int(w * (0.18 + 0.64 * abs(np.sin(phase * 3.2))))
        by = int(h * (0.32 + 0.30 * abs(np.cos(phase * 4.1))))
        cv2.circle(frame, (bx, by), 10, (40, 220, 240), -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("demo_game.mp4"),
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # Timeline in seconds: downtime long enough that serve-padding won't merge points.
    timeline = [
        (4.0, False),
        (4.0, True),
        (8.0, False),
        (5.0, True),
        (7.0, False),
        (3.5, True),
        (3.0, False),
    ]
    w, h = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open writer for {args.output}")

    t = 0.0
    dt = 1.0 / args.fps
    for duration, active in timeline:
        end = t + duration
        t0 = t
        while t < end:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            draw_court(frame)
            draw_players(frame, t, active, t0)
            writer.write(frame)
            t += dt

    writer.release()
    print(f"Wrote {args.output} ({t:.1f}s)")


if __name__ == "__main__":
    main()
