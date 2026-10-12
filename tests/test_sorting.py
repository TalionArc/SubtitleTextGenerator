from __future__ import annotations

from pathlib import Path

from subtitle_text_generator.models import JobStatus, MediaEntry, MediaKind
from subtitle_text_generator.sorting import SORT_COLUMNS, natural_key, sorted_ids


def make_entry(root: Path, relative: str, **changes: object) -> MediaEntry:
    path = root / relative
    kind = MediaKind.AUDIO if path.suffix == ".m4a" else MediaKind.VIDEO
    entry = MediaEntry(path=path, root=root, size=1, mtime_ns=1, kind=kind)
    for name, value in changes.items():
        setattr(entry, name, value)
    return entry


def names(entries: dict[str, MediaEntry], order: list[str]) -> list[str]:
    return [entries[iid].path.name for iid in order]


def test_natural_key_orders_numbers_by_value_and_ignores_case() -> None:
    files = ["강의 10강.mp4", "강의 2강.mp4", "강의 1강.mp4", "B.mp4", "a.mp4", "1²3.mp4"]
    assert sorted(files, key=natural_key) == [
        "1²3.mp4",
        "a.mp4",
        "B.mp4",
        "강의 1강.mp4",
        "강의 2강.mp4",
        "강의 10강.mp4",
    ]
    assert natural_key("") == ()


def test_without_a_sort_column_rows_follow_natural_path_order(tmp_path: Path) -> None:
    entries = {
        "c": make_entry(tmp_path, "10주차/1.mp4"),
        "b": make_entry(tmp_path, "2주차/10.mp4"),
        "a": make_entry(tmp_path, "2주차/2.mp4"),
    }
    assert sorted_ids(entries, set()) == ["a", "b", "c"]
    assert sorted_ids(entries, set(), "no-such-column") == ["a", "b", "c"]


def test_name_sort_is_natural_and_reversible(tmp_path: Path) -> None:
    entries = {
        "ten": make_entry(tmp_path, "week/강의 10강.mp4"),
        "two": make_entry(tmp_path, "week/강의 2강.mp4"),
        "one": make_entry(tmp_path, "other/강의 1강.mp4"),
    }
    assert sorted_ids(entries, set(), "name") == ["one", "two", "ten"]
    assert sorted_ids(entries, set(), "name", descending=True) == ["ten", "two", "one"]


def test_rows_that_compare_equal_keep_path_order_in_both_directions(tmp_path: Path) -> None:
    entries = {
        "v2": make_entry(tmp_path, "10강.mp4"),
        "audio": make_entry(tmp_path, "1강.m4a"),
        "v1": make_entry(tmp_path, "2강.mp4"),
    }
    assert sorted_ids(entries, set(), "kind") == ["v1", "v2", "audio"]
    assert sorted_ids(entries, set(), "kind", descending=True) == ["audio", "v1", "v2"]


def test_status_sort_puts_rows_needing_attention_first(tmp_path: Path) -> None:
    statuses = [
        JobStatus.COMPLETE,
        JobStatus.WAITING,
        JobStatus.FAILED,
        JobStatus.SKIPPED,
        JobStatus.REVIEW,
        JobStatus.RUNNING,
        JobStatus.WARNING,
        JobStatus.CANCELLED,
    ]
    entries = {
        status.name: make_entry(tmp_path, f"{index}.mp4", status=status)
        for index, status in enumerate(statuses)
    }
    assert sorted_ids(entries, set(), "status") == [
        "FAILED",
        "REVIEW",
        "WARNING",
        "CANCELLED",
        "RUNNING",
        "WAITING",
        "SKIPPED",
        "COMPLETE",
    ]


def test_selected_output_and_elapsed_sorts(tmp_path: Path) -> None:
    entries = {
        "a": make_entry(tmp_path, "a.mp4", elapsed_seconds=90.0),
        "b": make_entry(tmp_path, "b.mp4", existing_outputs=(tmp_path / "b.srt",)),
        "c": make_entry(tmp_path, "c.mp4", elapsed_seconds=600.0),
    }
    assert sorted_ids(entries, {"c"}, "selected") == ["c", "a", "b"]
    assert sorted_ids(entries, set(), "output") == ["a", "c", "b"]
    assert sorted_ids(entries, set(), "elapsed") == ["b", "a", "c"]
    assert sorted_ids(entries, set(), "elapsed", descending=True) == ["c", "a", "b"]


def test_every_column_can_be_sorted(tmp_path: Path) -> None:
    entries = {
        "a": make_entry(tmp_path, "sub/a.mp4", stage="Turbo 초안", message="둘째\n줄"),
        "b": make_entry(tmp_path, "b.m4a", message="첫째"),
    }
    for column in SORT_COLUMNS:
        assert sorted(sorted_ids(entries, {"a"}, column)) == ["a", "b"]
    assert names(entries, sorted_ids(entries, set(), "folder")) == ["b.m4a", "a.mp4"]
    assert names(entries, sorted_ids(entries, set(), "message")) == ["a.mp4", "b.m4a"]
