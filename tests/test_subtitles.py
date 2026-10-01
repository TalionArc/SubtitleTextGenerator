from __future__ import annotations

from pathlib import Path

import pytest

from subtitle_text_generator.models import Segment, Transcript, Word
from subtitle_text_generator.subtitles import (
    Cue,
    ExistingSubtitleError,
    ExistingTextOutputError,
    SubtitleError,
    commit_srt,
    commit_txt,
    cues_from_transcript,
    render_srt,
    render_txt,
    validate_srt_text,
    validate_txt_text,
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


def test_audio_txt_can_include_or_omit_timestamps() -> None:
    timestamped = render_txt(_transcript())
    validate_txt_text(timestamped, timestamps=True)
    assert timestamped.splitlines() == [
        "[00:00:00.000 - 00:00:02.000] 안녕하세요 강의를 시작합니다.",
        "[00:00:03.000 - 00:00:05.000] 이것은 두번째 문장입니다.",
    ]

    plain = render_txt(_transcript(), include_timestamps=False)
    validate_txt_text(plain, timestamps=False)
    assert "00:00:" not in plain
    assert plain.splitlines() == ["안녕하세요 강의를 시작합니다.", "이것은 두번째 문장입니다."]


def test_audio_txt_timestamp_validation_rejects_bad_order() -> None:
    with pytest.raises(SubtitleError, match="시간 순서"):
        validate_txt_text(
            "[00:00:02.000 - 00:00:01.000] 잘못된 구간\r\n",
            timestamps=True,
        )


def test_commit_txt_writes_bom_and_keeps_one_backup(tmp_path: Path) -> None:
    audio = tmp_path / "녹음 [1].m4a"
    audio.write_bytes(b"original audio")
    first = _transcript()
    second = Transcript(
        [Segment(0.0, 1.0, "", [Word(0.0, 1.0, " 두 번째 결과입니다.", 0.9)])],
        duration=1.0,
    )
    third = Transcript(
        [Segment(0.0, 1.0, "", [Word(0.0, 1.0, " 세 번째 결과입니다.", 0.9)])],
        duration=1.0,
    )

    output = commit_txt(audio, first, allow_replace=False)
    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in output.read_bytes()
    assert output.read_text(encoding="utf-8-sig").startswith("[00:00:00.000 - ")
    with pytest.raises(ExistingTextOutputError):
        commit_txt(audio, second, allow_replace=False)
    commit_txt(audio, second, allow_replace=True)
    backup = tmp_path / "녹음 [1].txt.bak"
    assert "안녕하세요" in backup.read_text(encoding="utf-8-sig")
    commit_txt(audio, third, allow_replace=True)
    assert "두 번째" in backup.read_text(encoding="utf-8-sig")
    assert len(list(tmp_path.glob("*.bak"))) == 1
    assert audio.read_bytes() == b"original audio"
