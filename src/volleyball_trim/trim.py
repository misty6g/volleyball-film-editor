"""Cut and concatenate rally segments with ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from volleyball_trim.detect import Segment


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg, then retry."
        )
    return path


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(f"ffmpeg failed:\n{err[-4000:]}")


def export_highlights(
    video_path: str | Path,
    segments: list[Segment],
    output_path: str | Path,
    *,
    reencode: bool = True,
) -> Path:
    """Write a single video containing only the given segments, in order."""
    ffmpeg = _require_ffmpeg()
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not segments:
        raise ValueError("No rally segments to export. Raise sensitivity and retry.")

    if len(segments) == 1:
        seg = segments[0]
        if reencode:
            cmd = [
                ffmpeg,
                "-y",
                "-ss",
                f"{seg.start:.3f}",
                "-to",
                f"{seg.end:.3f}",
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        else:
            cmd = [
                ffmpeg,
                "-y",
                "-ss",
                f"{seg.start:.3f}",
                "-to",
                f"{seg.end:.3f}",
                "-i",
                str(video_path),
                "-c",
                "copy",
                str(output_path),
            ]
        _run(cmd)
        return output_path

    # Multi-segment: filter concat for frame-accurate joins.
    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    for i, seg in enumerate(segments):
        filter_parts.append(
            f"[0:v]trim=start={seg.start:.3f}:end={seg.end:.3f},setpts=PTS-STARTPTS[v{i}];"
            f"[0:a]atrim=start={seg.start:.3f}:end={seg.end:.3f},asetpts=PTS-STARTPTS[a{i}]"
        )
        concat_inputs.append(f"[v{i}][a{i}]")

    # Some film has no audio — fall back to video-only concat.
    has_audio = _has_audio_stream(ffmpeg, video_path)
    if has_audio:
        filter_complex = (
            ";".join(filter_parts)
            + ";"
            + "".join(concat_inputs)
            + f"concat=n={len(segments)}:v=1:a=1[outv][outa]"
        )
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            _run(cmd)
            return output_path
        except RuntimeError:
            # Retry without audio if audio filters fail on odd containers.
            pass

    v_parts = []
    v_concat = []
    for i, seg in enumerate(segments):
        v_parts.append(
            f"[0:v]trim=start={seg.start:.3f}:end={seg.end:.3f},setpts=PTS-STARTPTS[v{i}]"
        )
        v_concat.append(f"[v{i}]")
    filter_complex = (
        ";".join(v_parts)
        + ";"
        + "".join(v_concat)
        + f"concat=n={len(segments)}:v=1:a=0[outv]"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-an",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run(cmd)
    return output_path


def _has_audio_stream(ffmpeg: str, video_path: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        # Probe via ffmpeg itself.
        proc = subprocess.run(
            [ffmpeg, "-i", str(video_path)],
            capture_output=True,
            text=True,
        )
        blob = (proc.stderr or "") + (proc.stdout or "")
        return "Audio:" in blob

    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    return bool(proc.stdout.strip())


def write_edl(segments: list[Segment], edl_path: str | Path) -> Path:
    """Write a simple CSV edit decision list for inspection / NLE import."""
    edl_path = Path(edl_path)
    edl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["index,start_sec,end_sec,duration_sec"]
    for i, seg in enumerate(segments, 1):
        lines.append(f"{i},{seg.start:.3f},{seg.end:.3f},{seg.duration:.3f}")
    edl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return edl_path


def export_with_temp_workdir(
    video_path: str | Path,
    segments: list[Segment],
    output_path: str | Path,
) -> Path:
    """Export highlights, using a temp dir for any intermediate artifacts."""
    with tempfile.TemporaryDirectory(prefix="vb-trim-") as tmp:
        # Currently export is single-pass; temp dir reserved for future chunking.
        _ = tmp
        return export_highlights(video_path, segments, output_path)
