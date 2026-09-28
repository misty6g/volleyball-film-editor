"""Download game film from YouTube (and similar) URLs via yt-dlp."""

from __future__ import annotations

import re
import shutil
import tempfile
import time
from collections.abc import Callable
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

# on_progress(fraction 0..1, status message)
ProgressCallback = Callable[[float, str], None]


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


def _find_js_runtimes() -> dict[str, dict[str, str]]:
    """Enable deno/node so yt-dlp can resolve modern YouTube player JS."""
    runtimes: dict[str, dict[str, str]] = {}
    candidates = {
        "deno": [
            shutil.which("deno"),
            str(Path.home() / ".deno" / "bin" / "deno"),
        ],
        "node": [
            shutil.which("node"),
            "/exec-daemon/node",
            "/usr/local/bin/node",
            "/usr/bin/node",
        ],
    }
    for name, paths in candidates.items():
        for raw in paths:
            if not raw:
                continue
            path = Path(raw)
            if path.is_file():
                runtimes[name] = {"path": str(path)}
                break
        else:
            # Still declare the runtime so yt-dlp searches PATH itself.
            runtimes[name] = {}
    return runtimes


def _pick_downloaded_file(out_dir: Path, prepared: str) -> Path:
    candidate = Path(prepared)
    if candidate.is_file():
        return candidate.resolve()
    mp4 = candidate.with_suffix(".mp4")
    if mp4.is_file():
        return mp4.resolve()
    files = sorted(
        (f for f in out_dir.iterdir() if f.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise RuntimeError(f"Download finished but no file found in {out_dir}")
    return files[0].resolve()


def _fraction_from_hook(info: dict) -> float | None:
    """Extract 0..1 download fraction from a yt-dlp progress hook payload."""
    total = info.get("total_bytes") or info.get("total_bytes_estimate")
    downloaded = info.get("downloaded_bytes")
    if total and downloaded is not None and total > 0:
        return max(0.0, min(1.0, float(downloaded) / float(total)))
    raw = info.get("_percent_str")
    if isinstance(raw, str):
        cleaned = _ANSI_RE.sub("", raw).strip().rstrip("%")
        try:
            return max(0.0, min(1.0, float(cleaned) / 100.0))
        except ValueError:
            return None
    return None


def _make_yt_dlp_hook(
    on_progress: ProgressCallback | None,
    *,
    min_interval_sec: float = 0.25,
) -> Callable[[dict], None] | None:
    """Build a yt-dlp progress_hooks callback that throttles UI updates."""
    if on_progress is None:
        return None

    state = {"last_frac": -1.0, "last_t": 0.0}

    def hook(info: dict) -> None:
        status = info.get("status")
        if status == "downloading":
            frac = _fraction_from_hook(info)
            if frac is None:
                return
            now = time.monotonic()
            # Throttle: every ~0.25s or when percent ticks by ≥1.
            if (
                abs(frac - state["last_frac"]) < 0.01
                and now - state["last_t"] < min_interval_sec
            ):
                return
            state["last_frac"] = frac
            state["last_t"] = now
            pct = int(round(frac * 100))
            # Separate video/audio streams each report 0–100%; show which file.
            label = Path(str(info.get("filename") or "video")).name
            on_progress(frac, f"Downloading {label}… {pct}%")
        elif status == "finished":
            on_progress(1.0, "Download part finished — merging if needed…")

    return hook


def download_video(
    url: str,
    dest_dir: str | Path | None = None,
    *,
    max_height: int = 1080,
    cookies_from_browser: str | None = None,
    cookies_file: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Download a video URL to dest_dir and return the local file path.

    Prefers an mp4 under max_height when available; falls back aggressively
    so unusual YouTube format sets still download.

    If YouTube asks to \"Sign in to confirm you’re not a bot\", pass browser
    cookies via cookies_from_browser (e.g. \"chrome\") or cookies_file.
    Close Chrome fully before using cookies-from-browser on some systems.

    on_progress, when set, is called as on_progress(fraction, message) during
    the download (fraction is 0..1 for the current stream).
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
    browser = (cookies_from_browser or "").strip().lower() or None
    cookie_path: Path | None = None
    if cookies_file:
        cookie_path = Path(cookies_file)
        if not cookie_path.is_file():
            raise FileNotFoundError(f"Cookies file not found: {cookie_path}")
        # Prefer an explicit cookies file over a live browser DB (avoids lock issues).
        browser = None
    has_cookies = bool(browser or cookie_path)
    js_runtimes = _find_js_runtimes()
    progress_hook = _make_yt_dlp_hook(on_progress)

    base: dict = {
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "restrictfilenames": True,
        "js_runtimes": js_runtimes,
        # Allow yt-dlp to fetch the external JS challenge solver when needed.
        "remote_components": ["ejs:github"],
    }
    if progress_hook is not None:
        base["progress_hooks"] = [progress_hook]
    if browser:
        # chrome / chrome:Default / chrome:Profile 1 are all valid.
        base["cookiesfrombrowser"] = (browser,)
    if cookie_path is not None:
        base["cookiefile"] = str(cookie_path)

    if on_progress is not None:
        on_progress(0.0, "Resolving YouTube formats…")

    # Progressive attempts: simple formats first, then strip client restrictions.
    # Cookies work most reliably with the default / web clients — not android-only.
    # An empty format list usually means the wrong player client or missing JS runtime.
    attempts: list[dict] = [
        {
            **base,
            "format": f"bv*[height<=?{max_height}]+ba/b[height<=?{max_height}]/bv*+ba/b",
            "format_sort": [f"res:{max_height}", "vcodec:h264", "acodec:m4a"],
            "merge_output_format": "mp4",
            "extractor_args": (
                {"youtube": {"player_client": ["web", "web_safari"]}}
                if has_cookies
                else {"youtube": {"player_client": ["android", "ios", "web"]}}
            ),
        },
        {
            **base,
            "format": "bestvideo*+bestaudio/best",
            "merge_output_format": "mp4",
            "extractor_args": {"youtube": {"player_client": ["web"]}} if has_cookies else {},
        },
        {
            **base,
            "format": "best",
            "extractor_args": {
                "youtube": {"player_client": ["web", "mweb", "tv", "web_safari"]}
            },
        },
        {
            **base,
            "format": "best*",
            # Last resort: let yt-dlp pick its own client defaults.
        },
    ]

    last_error: Exception | None = None
    for attempt_opts in attempts:
        # Drop empty extractor_args so yt-dlp uses its defaults.
        if not attempt_opts.get("extractor_args"):
            attempt_opts = {k: v for k, v in attempt_opts.items() if k != "extractor_args"}
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
                if on_progress is not None:
                    on_progress(1.0, "Download complete")
                return _pick_downloaded_file(out_dir, prepared)
        except Exception as exc:
            last_error = exc
            msg = _clean_error(str(exc))
            retryable = (
                "Requested format is not available" in msg
                or "Only images are available" in msg
                or "No video formats" in msg
            )
            if retryable:
                if on_progress is not None:
                    on_progress(0.0, "Retrying with a different format…")
                continue
            break

    assert last_error is not None
    msg = _clean_error(str(last_error)) or last_error.__class__.__name__
    if "Sign in to confirm" in msg or "not a bot" in msg.lower():
        raise RuntimeError(
            "YouTube blocked the download (bot check). "
            "Select Chrome cookies in the UI (fully quit Chrome first), "
            "or export a cookies.txt and upload it. "
            f"Details: {msg}"
        ) from last_error
    if (
        "Requested format is not available" in msg
        or "No video formats" in msg
        or "Only images are available" in msg
    ):
        raise RuntimeError(
            "YouTube did not return any downloadable video streams for this link. "
            "Try: (1) fully quit Chrome, then select Chrome cookies again; "
            "(2) install Deno (https://deno.land) or Node.js so yt-dlp can solve "
            "YouTube's player JS; or (3) download the MP4 in your browser and upload it. "
            f"Details: {msg}"
        ) from last_error
    raise RuntimeError(f"Download failed: {msg}") from last_error


def resolve_input(
    source: str | Path,
    *,
    download_dir: str | Path | None = None,
    cookies_from_browser: str | None = None,
    cookies_file: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Return a local video path, downloading first when source is a URL."""
    text = str(source).strip()
    if looks_like_url(text):
        return download_video(
            text,
            dest_dir=download_dir,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
            on_progress=on_progress,
        )
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path.resolve()
