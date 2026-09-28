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

    import volleyball_trim.download as download_mod

    monkeypatch.setattr(download_mod, "YoutubeDL", FakeYDL, raising=False)
    # Patch the import inside download_video
    import sys
    import types

    fake = types.ModuleType("yt_dlp")
    fake.YoutubeDL = FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)

    from volleyball_trim.download import download_video

    with pytest.raises(RuntimeError, match="browser cookies"):
        download_video("https://www.youtube.com/watch?v=abc123", dest_dir=tmp_path)
