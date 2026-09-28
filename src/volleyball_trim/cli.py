"""Command-line interface for volleyball film trimming."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from volleyball_trim.detect import DetectOptions
from volleyball_trim.download import looks_like_url, resolve_input
from volleyball_trim.pipeline import process_and_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volleyball-trim",
        description=(
            "Trim downtime between volleyball serves. "
            "Keeps rallies (high residual motion) and drops standing/reset time. "
            "Accepts a local video file or a YouTube URL."
        ),
    )
    p.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Path to game film, or a YouTube URL. Omit when using --ui.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input>_rallies.mp4)",
    )
    p.add_argument(
        "--sensitivity",
        type=float,
        default=0.55,
        help="0–1 keep more footage as this rises (default: 0.55)",
    )
    p.add_argument(
        "--pad-before",
        type=float,
        default=1.5,
        help="Seconds kept before each rally for the serve toss (default: 1.5)",
    )
    p.add_argument(
        "--pad-after",
        type=float,
        default=1.0,
        help="Seconds kept after each rally (default: 1.0)",
    )
    p.add_argument(
        "--min-rally",
        type=float,
        default=2.0,
        help="Drop detected bursts shorter than this many seconds (default: 2.0)",
    )
    p.add_argument(
        "--merge-gap",
        type=float,
        default=2.5,
        help="Merge rallies closer than this many seconds (default: 2.5)",
    )
    p.add_argument(
        "--sample-fps",
        type=float,
        default=4.0,
        help="Analysis sample rate (default: 4)",
    )
    p.add_argument(
        "--edl",
        type=Path,
        default=None,
        help="Optional CSV of kept segments",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and print segments without writing video",
    )
    p.add_argument(
        "--ui",
        action="store_true",
        help="Launch the web UI instead of processing a file",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Preferred port for --ui (default: 8765; auto-picks the next free port if busy)",
    )
    p.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        metavar="BROWSER",
        help="Browser to read YouTube cookies from (chrome, firefox, edge, …)",
    )
    p.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Netscape cookies.txt for YouTube downloads (if bot-checked)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.ui:
        from volleyball_trim.app import launch

        launch(port=args.port)
        return 0

    if args.input is None:
        print("error: input video or YouTube URL required (or pass --ui)", file=sys.stderr)
        return 1

    download_dir: Path | None = None
    try:
        if looks_like_url(args.input):
            download_dir = Path(tempfile.mkdtemp(prefix="vb-yt-cli-"))
            print(f"Downloading {args.input} …", file=sys.stderr)
            src = resolve_input(
                args.input,
                download_dir=download_dir,
                cookies_from_browser=args.cookies_from_browser,
                cookies_file=args.cookies,
            )
            print(f"Downloaded: {src}", file=sys.stderr)
        else:
            src = resolve_input(args.input)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output = args.output
    if output is None:
        output = src.with_name(f"{src.stem}_rallies.mp4")

    options = DetectOptions(
        sample_fps=args.sample_fps,
        sensitivity=float(max(0.0, min(1.0, args.sensitivity))),
        min_rally_sec=args.min_rally,
        merge_gap_sec=args.merge_gap,
        pad_before_sec=args.pad_before,
        pad_after_sec=args.pad_after,
    )

    try:
        report = process_and_report(
            src,
            output,
            options=options,
            edl_path=args.edl,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
