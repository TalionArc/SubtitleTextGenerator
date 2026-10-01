from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .constants import APP_NAME, LEGACY_APP_NAME

HOME_OVERRIDE_ENV = "SUBTITLE_TEXT_GENERATOR_HOME"


@dataclass(frozen=True, slots=True)
class AppPaths:
    root: Path

    @classmethod
    def discover(cls) -> AppPaths:
        override = os.environ.get(HOME_OVERRIDE_ENV)
        if override:
            return cls(Path(override).expanduser().resolve())
        if getattr(sys, "frozen", False):
            executable_dir = Path(sys.executable).resolve().parent
            return cls(executable_dir / f"{APP_NAME}-data")
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return cls(base / APP_NAME)

    @property
    def settings_file(self) -> Path:
        return self.root / "settings.json"

    @property
    def downloads(self) -> Path:
        return self.root / "downloads"

    @property
    def engine(self) -> Path:
        return self.root / "engine"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def temp(self) -> Path:
        return self.root / "temp"

    def ensure_base_dirs(self) -> None:
        for path in (self.root, self.downloads, self.models, self.jobs, self.logs, self.temp):
            path.mkdir(parents=True, exist_ok=True)


def legacy_portable_root() -> Path | None:
    """Data folder that 1.2.x and older portable builds created next to the EXE."""
    if os.environ.get(HOME_OVERRIDE_ENV) or not getattr(sys, "frozen", False):
        return None
    return Path(sys.executable).resolve().parent / f"{LEGACY_APP_NAME}-data"


def pending_legacy_root(paths: AppPaths) -> Path | None:
    legacy = legacy_portable_root()
    if legacy is None or paths.root.exists() or not legacy.is_dir():
        return None
    return legacy


def migrate_legacy_data(paths: AppPaths) -> bool:
    """Rename the old data folder so the engine and models are not downloaded again."""
    legacy = pending_legacy_root(paths)
    if legacy is None:
        return False
    os.rename(legacy, paths.root)
    return True
