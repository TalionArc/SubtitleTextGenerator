from __future__ import annotations

from pathlib import Path

from subtitle_text_generator.constants import ENGINE_ARCHIVE_SIZE, MODELS, SETUP_DOWNLOAD_BYTES
from subtitle_text_generator.gui import (
    MESSAGE_PREVIEW_CHARS,
    entry_detail_text,
    format_gigabytes,
    setup_prompt_text,
    single_line_preview,
)
from subtitle_text_generator.models import JobStatus, MediaEntry, MediaKind


def test_preview_flattens_multiline_engine_errors() -> None:
    message = "CUDA 인식 실패 (종료 코드 1): line one\n  line two\r\n\tline three"
    assert single_line_preview(message) == "CUDA 인식 실패 (종료 코드 1): line one line two line three"


def test_preview_caps_length_so_the_column_width_stays_bounded() -> None:
    preview = single_line_preview("가" * 1_200)
    assert len(preview) == MESSAGE_PREVIEW_CHARS
    assert preview.endswith("…")
    assert single_line_preview("짧은 메시지") == "짧은 메시지"
    assert single_line_preview("") == ""


def test_format_gigabytes_drops_trailing_zero() -> None:
    assert format_gigabytes(10 * 1024**3) == "10GB"
    assert format_gigabytes(12 * 1024**3) == "12GB"
    assert format_gigabytes(int(5.71 * 1024**3)) == "5.7GB"


def test_setup_prompt_states_download_install_and_free_space(tmp_path: Path) -> None:
    expected = ENGINE_ARCHIVE_SIZE + sum(item.size for model in MODELS for item in model.files)
    assert expected == SETUP_DOWNLOAD_BYTES
    text = setup_prompt_text(tmp_path, 47 * 1024**3)
    assert f"다운로드: 약 {format_gigabytes(SETUP_DOWNLOAD_BYTES)}" in text
    assert "설치 후 사용 용량: 약 10GB" in text
    assert "필요한 여유 공간: 12GB 이상" in text
    assert str(tmp_path) in text
    assert "현재 여유 공간: 47GB" in text
    assert "현재 여유 공간" not in setup_prompt_text(tmp_path, None)


def test_detail_text_keeps_the_full_message(tmp_path: Path) -> None:
    message = "인식 엔진 실패 (종료 코드 1):\n" + "\n".join(f"line {index}" for index in range(40))
    entry = MediaEntry(
        path=tmp_path / "week 1" / "lecture.mp4",
        root=tmp_path,
        size=1,
        mtime_ns=1,
        kind=MediaKind.VIDEO,
        status=JobStatus.FAILED,
        stage="Turbo 초안",
        elapsed_seconds=75,
        message=message,
    )
    detail = entry_detail_text(entry)
    assert message in detail
    assert "상태: 실패" in detail
    assert "경과: 1:15" in detail
    assert "기존 결과: 없음" in detail
    entry.message = ""
    assert "(메시지 없음)" in entry_detail_text(entry)
