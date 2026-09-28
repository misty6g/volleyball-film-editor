"""High-level detect → export pipeline."""

from __future__ import annotations

from pathlib import Path

from volleyball_trim.detect import DetectOptions, DetectResult, detect_rallies, summarize_result
from volleyball_trim.trim import export_highlights, write_edl


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    options: DetectOptions | None = None,
    edl_path: str | Path | None = None,
    dry_run: bool = False,
) -> tuple[DetectResult, Path | None]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Video not found: {input_path}")

    result = detect_rallies(str(input_path), options)
    if edl_path is not None:
        write_edl(result.segments, edl_path)

    if dry_run:
        return result, None

    out = export_highlights(input_path, result.segments, output_path)
    return result, out


def process_and_report(
    input_path: str | Path,
    output_path: str | Path,
    **kwargs,
) -> str:
    result, out = process_video(input_path, output_path, **kwargs)
    report = summarize_result(result)
    if out is not None:
        report += f"\n\nWrote: {out}"
    else:
        report += "\n\nDry run — no video written."
    return report
