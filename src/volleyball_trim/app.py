"""Gradio web UI for trimming volleyball film."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from volleyball_trim.detect import DetectOptions, summarize_result
from volleyball_trim.download import looks_like_url, resolve_input
from volleyball_trim.pipeline import process_video


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
  max-width: 960px !important;
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


def _process(
    video,
    youtube_url: str,
    cookies_browser: str,
    cookies_upload,
    sensitivity: float,
    pad_before: float,
    pad_after: float,
    min_rally: float,
    merge_gap: float,
    progress=gr.Progress(track_tqdm=False),
):
    url = (youtube_url or "").strip()
    out_dir = Path(tempfile.mkdtemp(prefix="vb-trim-ui-"))
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
        progress(0.05, desc="Downloading from YouTube…")
        try:
            src = resolve_input(
                url,
                download_dir=out_dir / "download",
                # Prefer an exported cookies.txt when both are provided — it
                # avoids Chrome's locked cookie DB while the browser is open.
                cookies_from_browser=None if cookies_file else browser,
                cookies_file=cookies_file,
            )
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            raise gr.Error(str(exc)) from exc
    elif video is not None:
        src = Path(video if isinstance(video, str) else video)
        if not src.is_file():
            raise gr.Error("Could not read the uploaded file.")
    else:
        raise gr.Error("Upload a game video or paste a YouTube link.")

    progress(0.25, desc="Scanning motion between plays…")
    options = DetectOptions(
        sensitivity=float(sensitivity),
        pad_before_sec=float(pad_before),
        pad_after_sec=float(pad_after),
        min_rally_sec=float(min_rally),
        merge_gap_sec=float(merge_gap),
    )

    out_path = out_dir / f"{src.stem}_rallies.mp4"
    edl_path = out_dir / f"{src.stem}_segments.csv"

    try:
        result, written = process_video(
            src,
            out_path,
            options=options,
            edl_path=edl_path,
            dry_run=False,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc

    progress(0.95, desc="Finishing…")
    report = summarize_result(result)
    if url:
        report = f"Source: {url}\nDownloaded: {src.name}\n\n" + report
    return str(written) if written else None, report, str(edl_path)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Rally Cut — Volleyball Film Trimmer") as demo:
        with gr.Column(elem_classes=["vb-hero"]):
            gr.Markdown(
                """
# Rally Cut
Drop a full volleyball game film — or paste a YouTube link. We keep the rallies
and cut the standing-around time between serves.
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
                        "Fully quit Chrome (not just close the tab) before selecting "
                        "chrome — otherwise the cookie DB stays locked and formats fail."
                    ),
                )
                cookies_upload = gr.File(
                    label="Or upload cookies.txt (backup)",
                    file_types=[".txt"],
                    type="filepath",
                    info=(
                        "Export with a cookies.txt extension, then upload here. "
                        "Use this if Chrome cookies still fail while Chrome is open."
                    ),
                )
            video_out = gr.Video(label="Rally-only output", interactive=False)

        with gr.Accordion("Tuning", open=False):
            sensitivity = gr.Slider(
                0.05,
                0.95,
                value=0.55,
                step=0.05,
                label="Sensitivity",
                info="Raise this if rallies are getting clipped. Lower to cut more downtime.",
            )
            with gr.Row():
                pad_before = gr.Slider(0, 4, value=1.5, step=0.25, label="Seconds before rally (serve toss)")
                pad_after = gr.Slider(0, 4, value=1.0, step=0.25, label="Seconds after rally")
            with gr.Row():
                min_rally = gr.Slider(0.5, 6, value=2.0, step=0.25, label="Minimum rally length (s)")
                merge_gap = gr.Slider(0.5, 6, value=2.5, step=0.25, label="Merge gap (s)")

        run_btn = gr.Button("Trim downtime", variant="primary")
        report = gr.Textbox(label="Cut report", lines=14)
        edl = gr.File(label="Segment CSV")

        gr.Markdown(
            """
Works best on **sideline / end-line film** with a mostly steady camera.
If both an upload and a YouTube link are provided, the link is used.
YouTube downloads may take a minute for longer match films.
            """
        )

        run_btn.click(
            fn=_process,
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
            ],
            outputs=[video_out, report, edl],
        )
    return demo


def launch(port: int = 8765, share: bool = False) -> None:
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    launch()
