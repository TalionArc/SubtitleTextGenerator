from __future__ import annotations

from pathlib import Path

from subtitle_text_generator.app_paths import AppPaths
from subtitle_text_generator.settings import (
    Settings,
    load_settings,
    normalize_glossary,
    save_settings,
    valid_root_directory,
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
        skip_setup_prompt=True,
    )
    save_settings(paths, expected)
    assert load_settings(paths) == expected


def test_old_settings_default_audio_timestamps_on(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "app")
    paths.ensure_base_dirs()
    paths.settings_file.write_text('{"root_directory": "C:/media"}', encoding="utf-8")
    loaded = load_settings(paths)
    assert loaded.audio_txt_timestamps is True
    assert loaded.skip_setup_prompt is False
    assert loaded.root_directory == "C:/media"


def test_non_boolean_skip_setup_prompt_is_ignored(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "app")
    paths.ensure_base_dirs()
    paths.settings_file.write_text('{"skip_setup_prompt": "yes"}', encoding="utf-8")
    assert load_settings(paths).skip_setup_prompt is False


def test_corrupt_settings_fall_back_to_defaults(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "app")
    paths.ensure_base_dirs()
    paths.settings_file.write_text("{broken", encoding="utf-8")
    assert load_settings(paths) == Settings()


def test_default_media_folder_is_empty_and_never_scans_working_directory(tmp_path: Path) -> None:
    assert Settings().root_directory == ""
    assert valid_root_directory("") is None
    assert valid_root_directory("   ") is None
    assert valid_root_directory(str(tmp_path)) == tmp_path.resolve()
    assert valid_root_directory(str(tmp_path / "missing")) is None
