"""Gradio web UI for trimming volleyball film."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from volleyball_trim.detect import DetectOptions, summarize_result
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
    sensitivity: float,
    pad_before: float,
    pad_after: float,
    min_rally: float,
    merge_gap: float,
    progress=gr.Progress(track_tqdm=False),
):
    if video is None:
        raise gr.Error("Upload a volleyball game video first.")

    src = Path(video if isinstance(video, str) else video)
    if not src.is_file():
        raise gr.Error("Could not read the uploaded file.")

    progress(0.1, desc="Scanning motion between plays…")
    options = DetectOptions(
        sensitivity=float(sensitivity),
        pad_before_sec=float(pad_before),
        pad_after_sec=float(pad_after),
        min_rally_sec=float(min_rally),
        merge_gap_sec=float(merge_gap),
    )

    out_dir = Path(tempfile.mkdtemp(prefix="vb-trim-ui-"))
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
    return str(written) if written else None, report, str(edl_path)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Rally Cut — Volleyball Film Trimmer") as demo:
        with gr.Column(elem_classes=["vb-hero"]):
            gr.Markdown(
                """
# Rally Cut
Drop a full volleyball game film. We keep the rallies and cut the standing-around
time between serves — so film study starts at the toss, not the huddle.
                """
            )

        with gr.Row():
            video_in = gr.Video(label="Game film", sources=["upload"])
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
It scores player/ball motion after removing camera pans, then stitches high-motion
stretches (with a little padding for the serve) into one watchable file.
            """
        )

        run_btn.click(
            fn=_process,
            inputs=[video_in, sensitivity, pad_before, pad_after, min_rally, merge_gap],
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
