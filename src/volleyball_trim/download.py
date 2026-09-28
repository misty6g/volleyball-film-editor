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
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\[[0-9;]*m")


def _clean_error(message: str) -> str:
    """Strip terminal color codes yt-dlp sometimes leaves in exceptions."""
    return _ANSI_RE.sub("", message).strip()


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
    if not text or not _URL_RE.match(text):
        return False
    host = (urlparse(text).hostname or "").lower()
    return any(
        host == h or host.endswith("." + h)
        for h in ("youtube.com", "youtu.be", "youtube-nocookie.com")
    )


def _format_selector(max_height: int) -> str:
    """Prefer ≤max_height mp4, but always fall back to whatever yt-dlp can get."""
    return (
        f"bv*[height<=?{max_height}][ext=mp4]+ba[ext=m4a]/"
        f"b[height<=?{max_height}][ext=mp4]/"
        f"bv*[height<=?{max_height}]+ba/"
        f"b[height<=?{max_height}]/"
        f"bv*+ba/b"
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
    can read it without exotic codecs; falls back to best available.

    If YouTube asks to "Sign in to confirm you're not a bot", pass browser
    cookies via cookies_from_browser (e.g. "chrome") or cookies_file.
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

    outtmpl = str(out_dir / "%(title).80B [%(id)s].%(ext)s")
    has_cookies = bool(cookies_from_browser or cookies_file)

    opts: dict = {
        "format": _format_selector(max_height),
        # Prefer h264/mp4 when several formats match, without failing hard.
        "format_sort": [f"res:{max_height}", "vcodec:h264", "acodec:m4a", "ext:mp4:m4a"],
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "fragment_retries": 3,
        "merge_output_format": "mp4",
        "restrictfilenames": True,
    }
    # Cookies pair best with the web client; otherwise try mobile clients too.
    if has_cookies:
        opts["extractor_args"] = {"youtube": {"player_client": ["web", "android"]}}
    else:
        opts["extractor_args"] = {
            "youtube": {"player_client": ["android", "ios", "web"]},
        }

    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser.strip().lower(),)
    if cookies_file:
        cookie_path = Path(cookies_file)
        if not cookie_path.is_file():
            raise FileNotFoundError(f"Cookies file not found: {cookie_path}")
        opts["cookiefile"] = str(cookie_path)

    last_error: Exception | None = None
    # First try preferred formats; on "format not available" retry with absolute best.
    attempts = [
        opts,
        {**opts, "format": "bv*+ba/b", "format_sort": ["res", "br"]},
    ]

    for attempt_opts in attempts:
        try:
            with YoutubeDL(attempt_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise RuntimeError(f"Could not download: {url}")
                if "entries" in info:
                    entries = [e for e in info["entries"] if e]
                    if not entries:
                        raise RuntimeError(f"No video found at URL: {url}")
                    info = entries[0]
                prepared = ydl.prepare_filename(info)
                candidate = Path(prepared)
                if not candidate.is_file():
                    mp4 = candidate.with_suffix(".mp4")
                    if mp4.is_file():
                        candidate = mp4
                    else:
                        files = sorted(
                            (f for f in out_dir.iterdir() if f.is_file()),
                            key=lambda p: p.stat().st_mtime,
                            reverse=True,
                        )
                        if not files:
                            raise RuntimeError(
                                f"Download finished but no file found for: {url}"
                            )
                        candidate = files[0]
                return candidate.resolve()
        except Exception as exc:
            last_error = exc
            msg = _clean_error(str(exc))
            # Retry once when the preferred format set isn't offered for this video.
            if "Requested format is not available" in msg and attempt_opts is attempts[0]:
                continue
            break

    assert last_error is not None
    msg = _clean_error(str(last_error)) or last_error.__class__.__name__
    if "Sign in to confirm" in msg or "not a bot" in msg.lower():
        raise RuntimeError(
            "YouTube blocked the download (bot check). "
            "Pass browser cookies, e.g. "
            "`--cookies-from-browser chrome` "
            "or export cookies to a file and use `--cookies cookies.txt`. "
            f"Details: {msg}"
        ) from last_error
    if "Requested format is not available" in msg:
        raise RuntimeError(
            "YouTube did not offer a usable video format for this link. "
            "Try again with `--cookies-from-browser chrome`, or download "
            "the file manually and upload it instead. "
            f"Details: {msg}"
        ) from last_error
    raise RuntimeError(f"Download failed: {msg}") from last_error


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
