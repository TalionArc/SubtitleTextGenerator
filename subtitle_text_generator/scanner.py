from __future__ import annotations

import re
from pathlib import Path

from .constants import AUDIO_EXTENSIONS, MEDIA_EXTENSIONS, SUBTITLE_EXTENSIONS
from .models import MediaEntry, MediaKind

_LANGUAGE_SUFFIX = re.compile(r"^[._-][a-z]{2,3}(?:[-_][a-z]{2,4})?$", re.IGNORECASE)


def media_kind(path: Path) -> MediaKind:
    return MediaKind.AUDIO if path.suffix.casefold() in AUDIO_EXTENSIONS else MediaKind.VIDEO


def matching_outputs(source: Path) -> tuple[Path, ...]:
    matches: list[Path] = []
    try:
        candidates = source.parent.iterdir()
    except OSError:
        return ()
    stem_folded = source.stem.casefold()
    allowed_extensions = {".txt"} if media_kind(source) is MediaKind.AUDIO else SUBTITLE_EXTENSIONS
    for candidate in candidates:
        if not candidate.is_file() or candidate.suffix.casefold() not in allowed_extensions:
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


def matching_subtitles(video: Path) -> tuple[Path, ...]:
    """Compatibility wrapper for the 1.1 video-only scanner API."""
    return matching_outputs(video)


def scan_media(root: Path) -> list[MediaEntry]:
    root = root.resolve()
    entries: list[MediaEntry] = []
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
                MediaEntry(
                    path=path.resolve(),
                    root=root,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    kind=media_kind(path),
                    existing_outputs=matching_outputs(path),
                )
            )
    except OSError:
        return []
    entries.sort(key=lambda item: str(item.path).casefold())
    return entries


def scan_videos(root: Path) -> list[MediaEntry]:
    """Compatibility wrapper; since 1.2 this scans video and audio media."""
    return scan_media(root)
