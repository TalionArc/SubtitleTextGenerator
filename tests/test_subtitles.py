from __future__ import annotations

from pathlib import Path

import pytest

from lecture_subtitle_batcher.models import Segment, Transcript, Word
from lecture_subtitle_batcher.subtitles import (
    Cue,
    ExistingSubtitleError,
    commit_srt,
    cues_from_transcript,
    render_srt,
    validate_srt_text,
)


def _transcript() -> Transcript:
    words = [
        Word(0.0, 0.6, " 안녕하세요", 0.9),
        Word(0.7, 1.2, " 강의를", 0.9),
        Word(1.3, 2.0, " 시작합니다.", 0.9),
        Word(3.0, 3.5, " 이것은", 0.9),
        Word(3.6, 4.2, " 두번째", 0.9),
        Word(4.3, 5.0, " 문장입니다.", 0.9),
    ]
    return Transcript([Segment(0.0, 5.0, "", words)], duration=5.0)


def test_srt_has_two_lines_max_and_valid_timing() -> None:
    cues = cues_from_transcript(_transcript())
    text = render_srt(cues)
    validate_srt_text(text)
    assert all(len(cue.text.splitlines()) <= 2 for cue in cues)
    assert all(cue.end - cue.start <= 6.0 for cue in cues)


def test_commit_writes_utf8_bom_crlf_and_one_backup(tmp_path: Path) -> None:
    video = tmp_path / "강의 [1].mp4"
    video.write_bytes(b"original")
    first = [Cue(0, 1, "첫 자막")]
    second = [Cue(0, 1.2, "둘째 자막")]
    third = [Cue(0, 1.4, "셋째 자막")]
    output = commit_srt(video, first, allow_replace=False)
    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in output.read_bytes()
    commit_srt(video, second, allow_replace=True)
    backup = tmp_path / "강의 [1].srt.bak"
    assert "첫 자막" in backup.read_text(encoding="utf-8-sig")
    commit_srt(video, third, allow_replace=True)
    assert "둘째 자막" in backup.read_text(encoding="utf-8-sig")
    assert len(list(tmp_path.glob("*.bak"))) == 1
    assert video.read_bytes() == b"original"


def test_commit_refuses_unapproved_overwrite(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"x")
    commit_srt(video, [Cue(0, 1, "one")], allow_replace=False)
    with pytest.raises(ExistingSubtitleError):
        commit_srt(video, [Cue(0, 1, "two")], allow_replace=False)
