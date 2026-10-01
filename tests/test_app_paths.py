from __future__ import annotations

import sys
from pathlib import Path

import pytest

from subtitle_text_generator.app_paths import (
    HOME_OVERRIDE_ENV,
    AppPaths,
    legacy_portable_root,
    migrate_legacy_data,
    pending_legacy_root,
)


def _freeze(monkeypatch, executable: Path) -> None:
    monkeypatch.delenv(HOME_OVERRIDE_ENV, raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))


def test_frozen_exe_uses_adjacent_portable_data_folder(
    tmp_path: Path, monkeypatch
) -> None:
    executable = tmp_path / "D-drive-app" / "SubtitleTextGenerator.exe"
    _freeze(monkeypatch, executable)

    paths = AppPaths.discover()

    assert paths.root == executable.parent / "SubtitleTextGenerator-data"
    paths.ensure_base_dirs()
    assert paths.engine.parent == paths.root
    assert paths.models.parent == paths.root
    assert paths.downloads.parent == paths.root
    assert paths.temp.is_dir()


def test_explicit_home_override_wins_for_packaged_and_development_modes(
    tmp_path: Path, monkeypatch
) -> None:
    override = tmp_path / "custom-data"
    monkeypatch.setenv(HOME_OVERRIDE_ENV, str(override))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "app.exe"))

    assert AppPaths.discover().root == override.resolve()
    assert legacy_portable_root() is None


def test_legacy_data_folder_is_renamed_without_redownload(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "app" / "SubtitleTextGenerator.exe"
    legacy = executable.parent / "LectureSubtitleBatcher-data"
    (legacy / "models").mkdir(parents=True)
    (legacy / "models" / "model.bin").write_bytes(b"weights")
    (legacy / "settings.json").write_text('{"root_directory": "C:/media"}', encoding="utf-8")
    _freeze(monkeypatch, executable)
    paths = AppPaths.discover()

    assert pending_legacy_root(paths) == legacy
    assert migrate_legacy_data(paths) is True

    assert not legacy.exists()
    assert (paths.root / "models" / "model.bin").read_bytes() == b"weights"
    assert paths.settings_file.is_file()
    assert migrate_legacy_data(paths) is False


def test_existing_new_data_folder_leaves_legacy_untouched(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "app" / "SubtitleTextGenerator.exe"
    legacy = executable.parent / "LectureSubtitleBatcher-data"
    legacy.mkdir(parents=True)
    _freeze(monkeypatch, executable)
    paths = AppPaths.discover()
    paths.root.mkdir()

    assert pending_legacy_root(paths) is None
    assert migrate_legacy_data(paths) is False
    assert legacy.is_dir()


def test_failed_rename_keeps_legacy_and_creates_nothing(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "app" / "SubtitleTextGenerator.exe"
    legacy = executable.parent / "LectureSubtitleBatcher-data"
    legacy.mkdir(parents=True)
    _freeze(monkeypatch, executable)
    paths = AppPaths.discover()

    def locked(_source: Path, _target: Path) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr("subtitle_text_generator.app_paths.os.rename", locked)
    with pytest.raises(PermissionError):
        migrate_legacy_data(paths)

    assert legacy.is_dir()
    assert not paths.root.exists()


def test_development_mode_never_migrates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(HOME_OVERRIDE_ENV, raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    assert legacy_portable_root() is None
    assert migrate_legacy_data(AppPaths(tmp_path / "data")) is False
