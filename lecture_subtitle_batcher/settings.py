from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .app_paths import AppPaths
from .constants import DEFAULT_ROOT


@dataclass(slots=True)
class Settings:
    root_directory: str = DEFAULT_ROOT
    glossary_text: str = ""
    window_geometry: str = "1120x720"


def load_settings(paths: AppPaths) -> Settings:
    try:
        data = json.loads(paths.settings_file.read_text(encoding="utf-8"))
        return Settings(
            root_directory=str(data.get("root_directory") or DEFAULT_ROOT),
            glossary_text=str(data.get("glossary_text") or ""),
            window_geometry=str(data.get("window_geometry") or "1120x720"),
        )
    except (OSError, ValueError, TypeError):
        return Settings()


def save_settings(paths: AppPaths, settings: Settings) -> None:
    paths.ensure_base_dirs()
    payload = json.dumps(asdict(settings), ensure_ascii=False, indent=2)
    temporary = paths.settings_file.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, paths.settings_file)


def normalize_glossary(text: str) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    used_chars = 0
    for raw_line in text.splitlines():
        term = " ".join(raw_line.strip().split())
        key = term.casefold()
        if not term or key in seen:
            continue
        added = len(term) + (1 if result else 0)
        if len(result) >= 50 or used_chars + added > 500:
            break
        seen.add(key)
        result.append(term)
        used_chars += added
    return tuple(result)


def valid_root_directory(value: str) -> Path | None:
    try:
        path = Path(value).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    return path if path.is_dir() else None
