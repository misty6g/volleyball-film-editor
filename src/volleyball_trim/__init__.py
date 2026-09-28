"""Volleyball film trimmer — cut downtime between serves."""

from volleyball_trim.cli import main
from volleyball_trim.detect import DetectOptions, Segment, detect_rallies
from volleyball_trim.pipeline import process_video

__all__ = [
    "DetectOptions",
    "Segment",
    "detect_rallies",
    "main",
    "process_video",
]
