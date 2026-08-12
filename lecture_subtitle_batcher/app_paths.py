from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .constants import APP_NAME


@dataclass(frozen=True, slots=True)
class AppPaths:
    root: Path

    @classmethod
    def discover(cls) -> AppPaths:
        override = os.environ.get("LECTURE_SUBTITLE_BATCHER_HOME")
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
