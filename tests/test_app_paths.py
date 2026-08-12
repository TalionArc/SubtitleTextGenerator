from __future__ import annotations

import sys
from pathlib import Path

from lecture_subtitle_batcher.app_paths import AppPaths


def test_frozen_exe_uses_adjacent_portable_data_folder(
    tmp_path: Path, monkeypatch
) -> None:
    executable = tmp_path / "D-drive-app" / "LectureSubtitleBatcher.exe"
    monkeypatch.delenv("LECTURE_SUBTITLE_BATCHER_HOME", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))

    paths = AppPaths.discover()

    assert paths.root == executable.parent / "LectureSubtitleBatcher-data"
    paths.ensure_base_dirs()
    assert paths.engine.parent == paths.root
    assert paths.models.parent == paths.root
    assert paths.downloads.parent == paths.root
    assert paths.temp.is_dir()


def test_explicit_home_override_wins_for_packaged_and_development_modes(
    tmp_path: Path, monkeypatch
) -> None:
    override = tmp_path / "custom-data"
    monkeypatch.setenv("LECTURE_SUBTITLE_BATCHER_HOME", str(override))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "app.exe"))

    assert AppPaths.discover().root == override.resolve()
