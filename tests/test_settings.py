from __future__ import annotations

from pathlib import Path

from lecture_subtitle_batcher.app_paths import AppPaths
from lecture_subtitle_batcher.settings import (
    Settings,
    load_settings,
    normalize_glossary,
    save_settings,
)


def test_glossary_deduplicates_and_enforces_limits() -> None:
    lines = [" Whisper ", "whisper", "CTranslate2", *[f"용어{i}" for i in range(80)]]
    terms = normalize_glossary("\n".join(lines))
    assert terms[:2] == ("Whisper", "CTranslate2")
    assert len(terms) == 50
    assert len(",".join(terms)) <= 500


def test_settings_round_trip(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "app")
    expected = Settings(
        root_directory=str(tmp_path),
        glossary_text="용어",
        window_geometry="1000x700",
        audio_txt_timestamps=False,
    )
    save_settings(paths, expected)
    assert load_settings(paths) == expected


def test_old_settings_default_audio_timestamps_on(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "app")
    paths.ensure_base_dirs()
    paths.settings_file.write_text('{"root_directory": "C:/media"}', encoding="utf-8")
    assert load_settings(paths).audio_txt_timestamps is True


def test_corrupt_settings_fall_back_to_defaults(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "app")
    paths.ensure_base_dirs()
    paths.settings_file.write_text("{broken", encoding="utf-8")
    assert load_settings(paths).root_directory
