# Rally Cut

Trim the downtime between volleyball serves. Point it at a game film — or paste a YouTube link — and it keeps the rallies while cutting standing-around time between points.

It scores **spatial residual motion inside a court ROI** (player/ball movement after removing camera pans, ignoring scorebug bands), optionally weights detections with a lightweight HOG person prior, snaps lead-ins to serve-toss onsets, and lets you **review/edit segments** before export.

## Requirements

- Python 3.12+
- [ffmpeg](https://ffmpeg.org/) on your `PATH`
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Quick start

```bash
git clone https://github.com/misty6g/volleyball-film-editor.git
cd volleyball-film-editor
uv sync
uv run volleyball-trim --ui
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) (or the next free port if 8765 is busy).

1. Upload a game video **or** paste a YouTube link  
2. Click **Analyze rallies** — review the motion timeline  
3. Edit keep segments (nudge start/end, delete rows)  
4. Click **Export trimmed film**

Or with pip:

```bash
pip install -r requirements.txt
python -m volleyball_trim.cli --ui
```

### CLI

```bash
# Local file (one-shot detect + export)
uv run volleyball-trim path/to/game.mp4 -o rallies.mp4

# YouTube URL
uv run volleyball-trim "https://www.youtube.com/watch?v=VIDEO_ID" -o rallies.mp4

# Cookies if YouTube bot-checks (fully quit Chrome first):
uv run volleyball-trim "https://youtu.be/VIDEO_ID" -o rallies.mp4 --cookies-from-browser chrome
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--sensitivity 0.7` | Keep more footage (raise if rallies get clipped) |
| `--pad-before 1.5` | Max seconds before rally (serve-aware may snap earlier) |
| `--pad-after 1.0` | Seconds kept after each rally |
| `--roi-top 0.12` | Ignore top scorebug band |
| `--roi-bottom 0.08` | Ignore bottom band |
| `--no-person-prior` | Disable HOG person weighting |
| `--no-serve-aware` | Use fixed pad-before instead of toss snapping |
| `--cookies-from-browser chrome` | Browser cookies for YouTube |
| `--dry-run` | Print segments without writing video |
| `--ui` | Launch the review UI |

## How it works

1. If the input is a URL, download it with yt-dlp.
2. Sample frames a few times per second inside a **court ROI** (scorebug bands ignored).
3. Estimate camera translation and score **spatial** residual motion (busiest court cells).
4. Optionally weight motion with an OpenCV **HOG person prior**.
5. Threshold → merge → **serve-aware** lead-in pads → segments.
6. In the UI: review timeline, edit segments, then ffmpeg-export.

Best results: **sideline or end-line film** with a mostly steady camera.

## Demo / tests

```bash
uv run python scripts/make_demo_video.py -o /tmp/demo_game.mp4
uv run volleyball-trim /tmp/demo_game.mp4 -o /tmp/demo_rallies.mp4
uv run pytest
```

## Project layout

```
src/volleyball_trim/
  download.py   # YouTube / URL download (yt-dlp)
  detect.py     # court ROI, spatial motion, serve pads, HOG prior
  trim.py       # ffmpeg export
  pipeline.py   # detect → export
  cli.py        # command line
  app.py        # Gradio review UI
```
