"""End-to-end smoke test on a synthetic game film."""

from __future__ import annotations

from pathlib import Path

from volleyball_trim.detect import DetectOptions, detect_rallies
from volleyball_trim.pipeline import process_video


def test_detects_three_rallies_on_demo(tmp_path: Path) -> None:
    demo = tmp_path / "demo_game.mp4"
    # Import the generator inline so the test stays self-contained.
    import runpy
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "make_demo_video.py"
    sys.argv = ["make_demo_video.py", "-o", str(demo)]
    runpy.run_path(str(script), run_name="__main__")
    assert demo.is_file()

    result = detect_rallies(str(demo), DetectOptions())
    assert len(result.segments) == 3

    kept = sum(s.duration for s in result.segments)
    assert kept < result.duration * 0.85
    assert kept > result.duration * 0.35

    out = tmp_path / "rallies.mp4"
    _, written = process_video(demo, out, options=DetectOptions())
    assert written is not None and written.is_file()
    assert written.stat().st_size > 1000
