from __future__ import annotations

from pathlib import Path

from subtitle_text_generator.constants import ENGINE_ARCHIVE_SIZE, MODELS, SETUP_DOWNLOAD_BYTES
from subtitle_text_generator.gui import (
    COLUMN_HEADINGS,
    MESSAGE_PREVIEW_CHARS,
    entry_detail_text,
    format_gigabytes,
    heading_text,
    setup_prompt_text,
    single_line_preview,
    watch_state_text,
)
from subtitle_text_generator.models import JobStatus, MediaEntry, MediaKind
from subtitle_text_generator.sorting import SORT_COLUMNS
from subtitle_text_generator.watcher import WatchReport


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


def test_heading_shows_the_sort_direction_on_one_column_only() -> None:
    assert heading_text("name", "", False) == "파일명"
    assert heading_text("name", "name", False) == "파일명 ▲"
    assert heading_text("name", "name", True) == "파일명 ▼"
    assert heading_text("status", "name", True) == "상태"
    assert set(COLUMN_HEADINGS) == set(SORT_COLUMNS)


def test_watch_state_text_explains_what_the_watcher_is_doing() -> None:
    def text(**changes: object) -> str:
        options: dict[str, object] = {
            "enabled": True,
            "has_folder": True,
            "report": WatchReport(),
            "queued": 0,
            "engine_ready": True,
        }
        options.update(changes)
        return watch_state_text(**options)  # type: ignore[arg-type]

    assert text(enabled=False) == ""
    assert "폴더를 먼저" in text(has_folder=False)
    assert text(report=None) == "감시 준비 중"
    assert "접근할 수 없습니다" in text(report=WatchReport(available=False))
    assert text() == "감시 중 · 지금부터 추가되는 파일만 자동으로 처리합니다."
    assert text(report=WatchReport(settling=2)) == (
        "감시 중 · 새 파일 2개의 복사가 끝나기를 기다리는 중"
    )
    assert text(queued=3) == "감시 중 · 자동 처리 대기 3개"
    assert text(queued=3, engine_ready=False) == (
        "감시 중 · 자동 처리 대기 3개 (인식 환경이 준비되면 시작)"
    )


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
