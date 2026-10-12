from __future__ import annotations

import re
from collections.abc import Collection, Mapping

from .models import JobStatus, MediaEntry

SORT_COLUMNS = (
    "selected",
    "kind",
    "status",
    "folder",
    "name",
    "output",
    "stage",
    "elapsed",
    "message",
)

# 상태 열은 글자순이 아니라 손봐야 할 것이 먼저 오는 순서로 정렬합니다.
STATUS_ORDER = (
    JobStatus.FAILED,
    JobStatus.REVIEW,
    JobStatus.WARNING,
    JobStatus.CANCELLED,
    JobStatus.RUNNING,
    JobStatus.WAITING,
    JobStatus.SKIPPED,
    JobStatus.COMPLETE,
)

_DIGITS = re.compile(r"(\d+)")


def natural_key(text: str) -> tuple[tuple[int, int | str], ...]:
    """Order "2강" before "10강" the way Windows Explorer does."""
    parts: list[tuple[int, int | str]] = []
    for index, chunk in enumerate(_DIGITS.split(text.casefold())):
        if not chunk:
            continue
        parts.append((0, int(chunk)) if index % 2 else (1, chunk))
    return tuple(parts)


def entry_sort_key(column: str, entry: MediaEntry, *, selected: bool) -> tuple[object, ...]:
    if column == "selected":
        return (not selected,)
    if column == "kind":
        return (entry.kind.value,)
    if column == "status":
        return (STATUS_ORDER.index(entry.status),)
    if column == "folder":
        return natural_key(entry.relative_parent)
    if column == "name":
        return natural_key(entry.path.name)
    if column == "output":
        extensions = sorted(path.suffix.casefold() for path in entry.existing_outputs)
        return (entry.has_output, *extensions)
    if column == "stage":
        return natural_key(entry.stage)
    if column == "elapsed":
        return (entry.elapsed_seconds,)
    if column == "message":
        return natural_key(" ".join(entry.message.split()))
    raise ValueError(f"unknown sort column: {column}")


def sorted_ids(
    entries: Mapping[str, MediaEntry],
    selected: Collection[str],
    column: str = "",
    descending: bool = False,
) -> list[str]:
    """Row order for the media list; rows that compare equal stay in path order."""
    ordered = sorted(entries, key=lambda iid: natural_key(str(entries[iid].path)))
    if column not in SORT_COLUMNS:
        return ordered
    return sorted(
        ordered,
        key=lambda iid: entry_sort_key(column, entries[iid], selected=iid in selected),
        reverse=descending,
    )
