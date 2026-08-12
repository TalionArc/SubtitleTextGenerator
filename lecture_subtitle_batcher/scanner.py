from __future__ import annotations

import re
from pathlib import Path

from .constants import MEDIA_EXTENSIONS, SUBTITLE_EXTENSIONS
from .models import VideoEntry

_LANGUAGE_SUFFIX = re.compile(r"^[._-][a-z]{2,3}(?:[-_][a-z]{2,4})?$", re.IGNORECASE)


def matching_subtitles(video: Path) -> tuple[Path, ...]:
    matches: list[Path] = []
    try:
        candidates = video.parent.iterdir()
    except OSError:
        return ()
    stem_folded = video.stem.casefold()
    for candidate in candidates:
        if not candidate.is_file() or candidate.suffix.casefold() not in SUBTITLE_EXTENSIONS:
            continue
        if candidate.name.casefold().endswith(".srt.bak"):
            continue
        candidate_stem = candidate.stem.casefold()
        if candidate_stem == stem_folded:
            matches.append(candidate)
            continue
        if candidate_stem.startswith(stem_folded):
            remainder = candidate_stem[len(stem_folded) :]
            if _LANGUAGE_SUFFIX.fullmatch(remainder):
                matches.append(candidate)
    return tuple(sorted(matches, key=lambda item: item.name.casefold()))


def scan_videos(root: Path) -> list[VideoEntry]:
    root = root.resolve()
    entries: list[VideoEntry] = []
    try:
        candidates = root.rglob("*")
        for path in candidates:
            try:
                if not path.is_file() or path.suffix.casefold() not in MEDIA_EXTENSIONS:
                    continue
                stat = path.stat()
            except OSError:
                continue
            entries.append(
                VideoEntry(
                    path=path.resolve(),
                    root=root,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    existing_subtitles=matching_subtitles(path),
                )
            )
    except OSError:
        return []
    entries.sort(key=lambda item: str(item.path).casefold())
    return entries
