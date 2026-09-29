"""Gradio web UI for trimming volleyball film.

Flow: analyze → review/edit segments on a motion timeline → export.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from volleyball_trim.detect import (
    CourtRoi,
    DetectOptions,
    DetectResult,
    format_timestamp,
    render_motion_timeline,
    rows_to_segments,
    segments_to_rows,
    summarize_result,
)
from volleyball_trim.download import looks_like_url, resolve_input
from volleyball_trim.pipeline import detect_only, export_segments


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root {
  --vb-ink: #12231c;
  --vb-court: #e8f0ea;
  --vb-net: #1f6b4a;
  --vb-sand: #d4a574;
  --vb-line: #f4f7f5;
}
.gradio-container {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif !important;
  max-width: 1040px !important;
  background:
    radial-gradient(ellipse 80% 50% at 10% 0%, rgba(31, 107, 74, 0.12), transparent 55%),
    radial-gradient(ellipse 60% 40% at 90% 10%, rgba(212, 165, 116, 0.18), transparent 50%),
    linear-gradient(180deg, #f4f7f5 0%, #e8f0ea 100%) !important;
}
.vb-hero h1 {
  font-family: "Fraunces", "Palatino Linotype", Georgia, serif !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em;
  color: var(--vb-ink) !important;
  font-size: 2.4rem !important;
}
.vb-hero p {
  color: #3a5248 !important;
  font-size: 1.05rem !important;
}
"""

SEGMENT_HEADERS = ["#", "start_sec", "end_sec", "duration_sec"]


def _resolve_source(
    video,
    youtube_url: str,
    cookies_browser: str,
    cookies_upload,
    out_dir: Path,
    progress,
) -> tuple[Path, str]:
    url = (youtube_url or "").strip()
    browser = (cookies_browser or "").strip() or None
    cookies_file = None
    if cookies_upload is not None:
        cookies_file = Path(
            cookies_upload if isinstance(cookies_upload, str) else cookies_upload
        )
        if not cookies_file.is_file():
            raise gr.Error("Could not read the uploaded cookies.txt file.")

    if url:
        if not looks_like_url(url):
            raise gr.Error("That doesn't look like a valid URL. Paste a full YouTube link.")
        progress(0.02, desc="Resolving YouTube formats…")

        def _on_download(frac: float, message: str) -> None:
            overall = 0.02 + 0.22 * max(0.0, min(1.0, frac))
            progress(overall, desc=message)

        try:
            src = resolve_input(
                url,
                download_dir=out_dir / "download",
                cookies_from_browser=None if cookies_file else browser,
                cookies_file=cookies_file,
                on_progress=_on_download,
            )
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            raise gr.Error(str(exc)) from exc
        return src, url

    if video is not None:
        src = Path(video if isinstance(video, str) else video)
        if not src.is_file():
            raise gr.Error("Could not read the uploaded file.")
        return src, ""

    raise gr.Error("Upload a game video or paste a YouTube link.")


def _analyze(
    video,
    youtube_url: str,
    cookies_browser: str,
    cookies_upload,
    sensitivity: float,
    pad_before: float,
    pad_after: float,
    min_rally: float,
    merge_gap: float,
    roi_top: float,
    roi_bottom: float,
    use_person_prior: bool,
    serve_aware: bool,
    progress=gr.Progress(track_tqdm=False),
):
    out_dir = Path(tempfile.mkdtemp(prefix="vb-trim-ui-"))
    src, url = _resolve_source(
        video, youtube_url, cookies_browser, cookies_upload, out_dir, progress
    )
    progress(0.30, desc="Scanning court motion…")

    options = DetectOptions(
        sensitivity=float(sensitivity),
        pad_before_sec=float(pad_before),
        pad_after_sec=float(pad_after),
        min_rally_sec=float(min_rally),
        merge_gap_sec=float(merge_gap),
        court_roi=CourtRoi(top=float(roi_top), bottom=float(roi_bottom)),
        use_person_prior=bool(use_person_prior),
        serve_aware_pads=bool(serve_aware),
    )

    try:
        result = detect_only(src, options=options)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc

    progress(0.90, desc="Building review timeline…")
    timeline = render_motion_timeline(result)
    report = summarize_result(result)
    if url:
        report = f"Source: {url}\nDownloaded: {src.name}\n\n" + report
    report += (
        "\n\nReview the green keep-bands and edit start/end below, "
        "then click Export trimmed film."
    )
    rows = segments_to_rows(result.segments)
    state = {
        "video_path": str(src),
        "out_dir": str(out_dir),
        "duration": result.duration,
        "url": url,
    }
    return (
        timeline,
        rows,
        report,
        state,
        gr.update(interactive=True),
    )


def _export(
    segment_rows,
    state: dict | None,
    progress=gr.Progress(track_tqdm=False),
):
    if not state or not state.get("video_path"):
        raise gr.Error("Analyze a video first.")
    src = Path(state["video_path"])
    if not src.is_file():
        raise gr.Error("Source video is missing — analyze again.")

    segments = rows_to_segments(segment_rows)
    if not segments:
        raise gr.Error("No segments to export. Adjust the table or re-analyze.")

    # Clamp to source duration when known.
    duration = float(state.get("duration") or 0.0)
    if duration > 0:
        clamped = []
        for seg in segments:
            start = max(0.0, min(seg.start, duration))
            end = max(start, min(seg.end, duration))
            if end - start >= 0.2:
                clamped.append(type(seg)(start, end))
        segments = clamped
    if not segments:
        raise gr.Error("Segments fall outside the video — check start/end times.")

    out_dir = Path(state.get("out_dir") or tempfile.mkdtemp(prefix="vb-trim-ui-"))
    out_path = out_dir / f"{src.stem}_rallies.mp4"
    edl_path = out_dir / f"{src.stem}_segments.csv"

    progress(0.2, desc="Exporting trimmed film…")
    try:
        from volleyball_trim.trim import write_edl

        write_edl(segments, edl_path)
        written = export_segments(src, out_path, segments)
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc

    progress(0.95, desc="Finishing…")
    kept = sum(s.duration for s in segments)
    report = (
        f"Exported {len(segments)} segment(s)\n"
        f"Output length: {format_timestamp(kept)} ({kept:.1f}s)\n"
        f"Wrote: {written}"
    )
    return str(written), report, str(edl_path)


def _delete_selected(segment_rows, selected_index: float):
    segments = rows_to_segments(segment_rows)
    if not segments:
        return segments_to_rows([])
    idx = int(selected_index) - 1
    if 0 <= idx < len(segments):
        segments.pop(idx)
    return segments_to_rows(segments)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Rally Cut — Volleyball Film Trimmer") as demo:
        state = gr.State(None)

        with gr.Column(elem_classes=["vb-hero"]):
            gr.Markdown(
                """
# Rally Cut
Analyze game film, review the motion timeline, edit keep segments,
then export a rally-only cut.
                """
            )

        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="Game film (upload)", sources=["upload"])
                youtube_url = gr.Textbox(
                    label="Or paste a YouTube link",
                    placeholder="https://www.youtube.com/watch?v=…",
                    lines=1,
                )
                cookies_browser = gr.Dropdown(
                    label="Browser cookies for YouTube",
                    choices=["", "chrome", "firefox", "edge", "brave", "chromium", "safari"],
                    value="",
                    info=(
                        "Fully quit Chrome before selecting chrome — "
                        "otherwise the cookie DB stays locked."
                    ),
                )
                cookies_upload = gr.File(
                    label="Or upload cookies.txt (backup — use if Chrome is still open)",
                    file_types=[".txt"],
                    type="filepath",
                )
            with gr.Column():
                timeline = gr.Image(
                    label="Motion timeline (green = kept)",
                    interactive=False,
                )
                video_out = gr.Video(label="Rally-only output", interactive=False)

        with gr.Accordion("Detection", open=True):
            sensitivity = gr.Slider(
                0.05,
                0.95,
                value=0.55,
                step=0.05,
                label="Sensitivity",
                info="Raise if rallies are clipped. Lower to cut more downtime.",
            )
            with gr.Row():
                pad_before = gr.Slider(
                    0, 4, value=1.5, step=0.25, label="Max seconds before rally"
                )
                pad_after = gr.Slider(
                    0, 4, value=1.0, step=0.25, label="Seconds after rally"
                )
            with gr.Row():
                min_rally = gr.Slider(
                    0.5, 6, value=2.0, step=0.25, label="Minimum rally length (s)"
                )
                merge_gap = gr.Slider(
                    0.5, 6, value=2.5, step=0.25, label="Merge gap (s)"
                )
            with gr.Row():
                roi_top = gr.Slider(
                    0.0,
                    0.30,
                    value=0.12,
                    step=0.01,
                    label="Ignore top band (scorebug)",
                )
                roi_bottom = gr.Slider(
                    0.0,
                    0.30,
                    value=0.08,
                    step=0.01,
                    label="Ignore bottom band",
                )
            with gr.Row():
                use_person_prior = gr.Checkbox(
                    value=True,
                    label="Person-prior assist (HOG)",
                    info="Weights court motion toward detected people when possible.",
                )
                serve_aware = gr.Checkbox(
                    value=True,
                    label="Serve-aware lead-in",
                    info="Snap pad-before back to a toss / motion-onset cue.",
                )

        analyze_btn = gr.Button("1. Analyze rallies", variant="primary")
        report = gr.Textbox(label="Cut report", lines=12)

        gr.Markdown("### 2. Review & edit keep segments")
        segments_table = gr.Dataframe(
            headers=SEGMENT_HEADERS,
            datatype=["number", "number", "number", "number"],
            row_count=(0, "dynamic"),
            column_count=(4, "fixed"),
            label="Edit start_sec / end_sec, or add/remove rows",
            interactive=True,
        )
        with gr.Row():
            delete_idx = gr.Number(
                value=1,
                precision=0,
                label="Segment # to delete",
            )
            delete_btn = gr.Button("Delete segment #")
        export_btn = gr.Button(
            "3. Export trimmed film",
            variant="primary",
            interactive=False,
        )
        edl = gr.File(label="Segment CSV")

        gr.Markdown(
            """
Works best on **sideline / end-line film** with a mostly steady camera.
Green bands on the timeline are proposed keep regions; blue ticks mark
serve-onset snaps. Court ROI ignores scorebug bands; spatial scoring and
optional person-prior focus on player motion inside the court.
            """
        )

        analyze_btn.click(
            fn=_analyze,
            inputs=[
                video_in,
                youtube_url,
                cookies_browser,
                cookies_upload,
                sensitivity,
                pad_before,
                pad_after,
                min_rally,
                merge_gap,
                roi_top,
                roi_bottom,
                use_person_prior,
                serve_aware,
            ],
            outputs=[timeline, segments_table, report, state, export_btn],
        )
        delete_btn.click(
            fn=_delete_selected,
            inputs=[segments_table, delete_idx],
            outputs=[segments_table],
        )
        export_btn.click(
            fn=_export,
            inputs=[segments_table, state],
            outputs=[video_out, report, edl],
        )
    return demo


def _find_free_port(preferred: int, *, attempts: int = 40) -> int:
    """Return preferred if free, otherwise the next open TCP port."""
    import socket

    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            in_use = probe.connect_ex(("127.0.0.1", port)) == 0
        if in_use:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise OSError(
        f"No free port found in {preferred}–{preferred + attempts - 1}. "
        f"Stop the other Rally Cut / Gradio process, or pass --port PORT."
    )


def launch(port: int = 8765, share: bool = False) -> None:
    demo = build_ui()
    chosen = _find_free_port(port)
    if chosen != port:
        print(
            f"Port {port} is busy — launching on http://127.0.0.1:{chosen} instead.",
            flush=True,
        )
    else:
        print(f"Open http://127.0.0.1:{chosen}", flush=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=chosen,
        share=share,
        show_error=True,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    launch()
