"""Volleyball film trimmer — cut downtime between serves."""

from volleyball_trim.detect import CourtRoi, DetectOptions, Segment, detect_rallies
from volleyball_trim.download import download_video, resolve_input
from volleyball_trim.pipeline import process_video

__all__ = [
    "CourtRoi",
    "DetectOptions",
    "Segment",
    "detect_rallies",
    "download_video",
    "main",
    "process_video",
    "resolve_input",
]


def __getattr__(name: str):
    # Lazy-load CLI entry so `python -m volleyball_trim.cli` does not warn
    # about cli already sitting in sys.modules via package import.
    if name == "main":
        from volleyball_trim.cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
