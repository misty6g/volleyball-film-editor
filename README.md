# Rally Cut

Trim the downtime between volleyball serves. Point it at a game film; it keeps the rallies and stitches them into a shorter watchable file.

It scores **residual motion** (player/ball movement after removing camera pans), finds high-activity stretches, pads a little for the serve toss, and exports one MP4 with ffmpeg.

## Requirements

- Python 3.12+
- [ffmpeg](https://ffmpeg.org/) on your `PATH`
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Quick start

```bash
uv sync
uv run volleyball-trim --ui
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765), upload a game video, click **Trim downtime**.

### CLI

```bash
uv run volleyball-trim path/to/game.mp4 -o rallies.mp4
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--sensitivity 0.7` | Keep more footage (raise if rallies get clipped) |
| `--pad-before 1.5` | Seconds kept before each rally for the serve toss |
| `--pad-after 1.0` | Seconds kept after each rally |
| `--dry-run` | Print detected segments without writing video |
| `--edl segments.csv` | Also write a CSV of kept ranges |
| `--ui` | Launch the web UI |

## How it works

1. Sample frames a few times per second.
2. Estimate camera translation and score the leftover motion.
3. Treat stretches above an adaptive threshold as rallies.
4. Merge nearby bursts, drop tiny blips, pad for serve/point end.
5. Concatenate kept clips with ffmpeg (`libx264`).

Best results: **sideline or end-line film** with a mostly steady camera. Heavy zooming, scoreboard overlays that animate constantly, or very shaky handheld footage may need a higher sensitivity.

## Demo / tests

```bash
uv run python scripts/make_demo_video.py -o /tmp/demo_game.mp4
uv run volleyball-trim /tmp/demo_game.mp4 -o /tmp/demo_rallies.mp4
uv run pytest
```

## Project layout

```
src/volleyball_trim/
  detect.py     # motion / rally detection
  trim.py       # ffmpeg export
  pipeline.py   # detect → export
  cli.py        # command line
  app.py        # Gradio UI
```
