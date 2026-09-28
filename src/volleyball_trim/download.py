"""Download game film from YouTube (and similar) URLs via yt-dlp."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

# Common hosts yt-dlp can pull volleyball film from; YouTube is the primary case.
_URL_HOST_HINTS = (
    "youtube.com",
    "youtu.be",
    "www.youtube.com",
    "m.youtube.com",
    "youtube-nocookie.com",
)

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def looks_like_url(value: str | Path) -> bool:
    text = str(value).strip()
    if not text or not _URL_RE.match(text):
        return False
    host = (urlparse(text).hostname or "").lower()
    if not host:
        return False
    if any(host == h or host.endswith("." + h) for h in _URL_HOST_HINTS):
        return True
    # Allow other http(s) URLs — yt-dlp supports many sites; fail later if unsupported.
    return True


def is_youtube_url(value: str | Path) -> bool:
    text = str(value).strip()
    if not _URL_RE.match(text):
        return False
    host = (urlparse(text).hostname or "").lower()
    return any(
        host == h or host.endswith("." + h)
        for h in ("youtube.com", "youtu.be", "youtube-nocookie.com")
    )


def download_video(
    url: str,
    dest_dir: str | Path | None = None,
    *,
    max_height: int = 1080,
    cookies_from_browser: str | None = None,
    cookies_file: str | Path | None = None,
) -> Path:
    """Download a video URL to dest_dir and return the local file path.

    Prefers an mp4 mux under max_height when available so OpenCV/ffmpeg
    can read it without exotic codecs.

    If YouTube asks to "Sign in to confirm you’re not a bot", pass browser
    cookies via cookies_from_browser (e.g. \"chrome\") or cookies_file.
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "yt-dlp is required to download from URLs. Install with: uv add yt-dlp"
        ) from exc

    url = url.strip()
    if not looks_like_url(url):
        raise ValueError(f"Not a downloadable URL: {url}")

    if dest_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="vb-yt-"))
    else:
        out_dir = Path(dest_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer mp4/h264 for broad compatibility; fall back to best available.
    fmt = (
        f"bv*[height<=?{max_height}][ext=mp4]+ba[ext=m4a]/"
        f"b[height<=?{max_height}][ext=mp4]/"
        f"bv*[height<=?{max_height}]+ba/b[height<=?{max_height}]/b"
    )
    outtmpl = str(out_dir / "%(title).80B [%(id)s].%(ext)s")

    opts: dict = {
        "format": fmt,
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "fragment_retries": 3,
        "merge_output_format": "mp4",
        "restrictfilenames": True,
        # Multiple clients improves odds against YouTube bot checks.
        "extractor_args": {
            "youtube": {"player_client": ["android", "ios", "web"]},
        },
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser.strip().lower(),)
    if cookies_file:
        cookie_path = Path(cookies_file)
        if not cookie_path.is_file():
            raise FileNotFoundError(f"Cookies file not found: {cookie_path}")
        opts["cookiefile"] = str(cookie_path)

    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise RuntimeError(f"Could not download: {url}")
            # Playlists should be blocked by noplaylist; still guard.
            if "entries" in info:
                entries = [e for e in info["entries"] if e]
                if not entries:
                    raise RuntimeError(f"No video found at URL: {url}")
                info = entries[0]
            prepared = ydl.prepare_filename(info)
            # merge_output_format may change extension to mp4 after remux.
            candidate = Path(prepared)
            if not candidate.is_file():
                mp4 = candidate.with_suffix(".mp4")
                if mp4.is_file():
                    candidate = mp4
                else:
                    # Fall back to newest file in dest dir.
                    files = sorted(
                        out_dir.iterdir(),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    files = [f for f in files if f.is_file()]
                    if not files:
                        raise RuntimeError(f"Download finished but no file found for: {url}")
                    candidate = files[0]
            return candidate.resolve()
    except Exception as exc:
        msg = str(exc).strip() or exc.__class__.__name__
        if "Sign in to confirm" in msg or "not a bot" in msg.lower():
            raise RuntimeError(
                "YouTube blocked the download (bot check). "
                "Pass browser cookies, e.g. "
                "`--cookies-from-browser chrome` "
                "or export cookies to a file and use `--cookies cookies.txt`. "
                f"Details: {msg}"
            ) from exc
        raise RuntimeError(f"Download failed: {msg}") from exc


def resolve_input(
    source: str | Path,
    *,
    download_dir: str | Path | None = None,
    cookies_from_browser: str | None = None,
    cookies_file: str | Path | None = None,
) -> Path:
    """Return a local video path, downloading first when source is a URL."""
    text = str(source).strip()
    if looks_like_url(text):
        return download_video(
            text,
            dest_dir=download_dir,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path.resolve()
