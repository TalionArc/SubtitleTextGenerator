from __future__ import annotations

from pathlib import Path

import pytest

from subtitle_text_generator.constants import AUDIO_EXTENSIONS
from subtitle_text_generator.models import MediaKind
from subtitle_text_generator.scanner import (
    matching_outputs,
    matching_subtitles,
    media_entry,
    scan_media,
    scan_videos,
)


def test_media_entry_describes_one_file_and_rejects_others(tmp_path: Path) -> None:
    audio = tmp_path / "새 녹음" / "회의.M4A"
    audio.parent.mkdir()
    audio.write_bytes(b"audio")
    (audio.parent / "회의.txt").write_text("x", encoding="utf-8")
    (tmp_path / "메모.txt").write_text("x", encoding="utf-8")

    entry = media_entry(audio, tmp_path)

    assert entry is not None
    assert entry.path == audio.resolve()
    assert entry.kind is MediaKind.AUDIO
    assert entry.size == 5
    assert entry.relative_parent == "새 녹음"
    assert [path.name for path in entry.existing_outputs] == ["회의.txt"]
    assert entry == scan_media(tmp_path)[0]
    assert media_entry(tmp_path / "메모.txt", tmp_path) is None
    assert media_entry(tmp_path / "없는 파일.mp4", tmp_path) is None
    assert media_entry(audio.parent, tmp_path) is None


def test_matching_subtitles_exact_and_language_suffix(tmp_path: Path) -> None:
    video = tmp_path / "강의.1.mp4"
    video.write_bytes(b"video")
    exact = tmp_path / "강의.1.srt"
    korean = tmp_path / "강의.1.ko-KR.vtt"
    english = tmp_path / "강의.1_eng.ass"
    unrelated = tmp_path / "강의.1.notes.srt"
    backup = tmp_path / "강의.1.srt.bak"
    for path in (exact, korean, english, unrelated, backup):
        path.write_text("x", encoding="utf-8")

    assert matching_subtitles(video) == tuple(
        sorted((exact, korean, english), key=lambda item: item.name.casefold())
    )


def test_recursive_scan_handles_korean_brackets_and_duplicate_names(tmp_path: Path) -> None:
    first = tmp_path / "1주차 [기초]" / "DocZoomScreenCapture.mp4"
    second = tmp_path / "2주차 (심화)" / "DocZoomScreenCapture.mp4"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    (second.with_suffix(".srt")).write_text("subtitle", encoding="utf-8")

    entries = scan_videos(tmp_path)

    assert len(entries) == 2
    assert entries[0].path != entries[1].path
    assert sum(entry.has_subtitle for entry in entries) == 1
    assert {entry.path.name for entry in entries} == {"DocZoomScreenCapture.mp4"}


def test_scan_ignores_non_media_files(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "movie.MP4").write_bytes(b"x")
    assert [entry.path.name for entry in scan_videos(tmp_path)] == ["movie.MP4"]


def test_audio_txt_output_detection_is_separate_from_video_subtitles(tmp_path: Path) -> None:
    audio = tmp_path / "인터뷰.1.m4a"
    audio.write_bytes(b"audio")
    exact = tmp_path / "인터뷰.1.txt"
    korean = tmp_path / "인터뷰.1.ko-KR.txt"
    unrelated = tmp_path / "인터뷰.1.notes.txt"
    subtitle = tmp_path / "인터뷰.1.srt"
    backup = tmp_path / "인터뷰.1.txt.bak"
    for path in (exact, korean, unrelated, subtitle, backup):
        path.write_text("x", encoding="utf-8")

    assert matching_outputs(audio) == tuple(
        sorted((exact, korean), key=lambda item: item.name.casefold())
    )


@pytest.mark.parametrize("extension", sorted(AUDIO_EXTENSIONS))
def test_scan_supports_common_audio_extensions(tmp_path: Path, extension: str) -> None:
    audio = tmp_path / f"recording{extension.upper()}"
    audio.write_bytes(b"audio")

    entries = scan_media(tmp_path)

    assert len(entries) == 1
    assert entries[0].path.name == audio.name
    assert entries[0].kind is MediaKind.AUDIO
