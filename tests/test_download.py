"""Unit tests for URL detection / input resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from volleyball_trim.download import is_youtube_url, looks_like_url, resolve_input


def test_looks_like_youtube_urls() -> None:
    assert looks_like_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert looks_like_url("https://youtu.be/dQw4w9WgXcQ")
    assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    assert not looks_like_url("/tmp/game.mp4")
    assert not looks_like_url("game.mp4")


def test_resolve_local_file(tmp_path: Path) -> None:
    video = tmp_path / "game.mp4"
    video.write_bytes(b"fake")
    assert resolve_input(video) == video.resolve()


def test_resolve_missing_local_raises() -> None:
    with pytest.raises(FileNotFoundError):
        resolve_input("/tmp/this-video-does-not-exist-vb-trim.mp4")


def test_resolve_bad_url_raises() -> None:
    with pytest.raises(FileNotFoundError):
        resolve_input("not-a-url")


def test_resolve_invalid_scheme_is_not_url() -> None:
    assert not looks_like_url("ftp://example.com/video.mp4")


def test_download_bot_check_message(monkeypatch, tmp_path: Path) -> None:
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=True):
            raise Exception(
                "ERROR: [youtube] abc: Sign in to confirm you’re not a bot."
            )

    import sys
    import types

    fake = types.ModuleType("yt_dlp")
    fake.YoutubeDL = FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)

    from volleyball_trim.download import download_video

    with pytest.raises(RuntimeError, match="bot check"):
        download_video("https://www.youtube.com/watch?v=abc123", dest_dir=tmp_path)


def test_download_retries_when_format_unavailable(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=True):
            calls.append(self.opts["format"])
            if len(calls) == 1:
                raise Exception(
                    "\x1b[0;31mERROR:\x1b[0m [youtube] W4rwOIwHLHI: "
                    "Requested format is not available. Use --list-formats"
                )
            # Second attempt succeeds.
            out = tmp_path / "match [W4rwOIwHLHI].mp4"
            out.write_bytes(b"fake-video")
            return {"id": "W4rwOIwHLHI", "title": "match", "ext": "mp4"}

        def prepare_filename(self, info):
            return str(tmp_path / "match [W4rwOIwHLHI].mp4")

    import sys
    import types

    fake = types.ModuleType("yt_dlp")
    fake.YoutubeDL = FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)

    from volleyball_trim.download import download_video

    path = download_video("https://youtu.be/W4rwOIwHLHI", dest_dir=tmp_path)
    assert path.is_file()
    assert len(calls) == 2
    assert calls[0].startswith("bv*")
    assert calls[1] == "bestvideo*+bestaudio/best"


def test_download_format_exhausted_message(monkeypatch, tmp_path: Path) -> None:
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=True):
            raise Exception(
                "ERROR: [youtube] W4rwOIwHLHI: Requested format is not available."
            )

    import sys
    import types

    fake = types.ModuleType("yt_dlp")
    fake.YoutubeDL = FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)

    from volleyball_trim.download import download_video

    with pytest.raises(RuntimeError, match="quit Chrome"):
        download_video(
            "https://youtu.be/W4rwOIwHLHI",
            dest_dir=tmp_path,
            cookies_from_browser="chrome",
        )


def test_download_uses_web_clients_with_cookies(monkeypatch, tmp_path: Path) -> None:
    seen: list[dict] = []

    class FakeYDL:
        def __init__(self, opts):
            seen.append(opts)
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=True):
            out = tmp_path / "match [abc].mp4"
            out.write_bytes(b"fake-video")
            return {"id": "abc", "title": "match", "ext": "mp4"}

        def prepare_filename(self, info):
            return str(tmp_path / "match [abc].mp4")

    import sys
    import types

    fake = types.ModuleType("yt_dlp")
    fake.YoutubeDL = FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)

    from volleyball_trim.download import download_video

    download_video(
        "https://youtu.be/abc",
        dest_dir=tmp_path,
        cookies_from_browser="chrome",
    )
    assert seen[0]["cookiesfrombrowser"] == ("chrome",)
    clients = seen[0]["extractor_args"]["youtube"]["player_client"]
    assert "web" in clients
    assert "android" not in clients


def test_clean_error_strips_ansi() -> None:
    from volleyball_trim.download import _clean_error

    raw = "\x1b[0;31mERROR:\x1b[0m [youtube] abc: Requested format is not available"
    assert _clean_error(raw) == "ERROR: [youtube] abc: Requested format is not available"
