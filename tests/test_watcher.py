from __future__ import annotations

import os
import queue
from pathlib import Path

import pytest

from subtitle_text_generator.watcher import (
    WATCH_MAX_INTERVAL_SECONDS,
    FolderWatcher,
    NewFileTracker,
    WatchReport,
    is_file_free,
    list_media,
    next_delay,
)

SETTLE = 9.0


class FakeFiles:
    def __init__(self) -> None:
        self.stamps: dict[Path, tuple[int, int]] = {}
        self.locked: set[Path] = set()

    def tracker(self, baseline: list[Path]) -> NewFileTracker:
        return NewFileTracker(
            baseline,
            settle_seconds=SETTLE,
            stamp=self.stamps.get,
            is_free=lambda path: path not in self.locked,
        )


def test_list_media_finds_nested_media_only(tmp_path: Path) -> None:
    (tmp_path / "1주차 [기초]").mkdir()
    video = tmp_path / "1주차 [기초]" / "강의.MP4"
    audio = tmp_path / "녹음.m4a"
    for path in (video, audio, tmp_path / "메모.txt", tmp_path / "강의.srt"):
        path.write_bytes(b"x")
    (tmp_path / "폴더.mp4").mkdir()

    assert list_media(tmp_path) == {video, audio}


def test_list_media_reports_an_unreadable_root_as_none(tmp_path: Path) -> None:
    assert list_media(tmp_path / "missing") is None
    assert list_media(tmp_path) == frozenset()


def test_files_present_at_start_are_never_reported(tmp_path: Path) -> None:
    files = FakeFiles()
    old = tmp_path / "old.mp4"
    files.stamps[old] = (10, 1)
    tracker = files.tracker([old])

    assert tracker.update({old}, 0.0) == ((), 0)
    assert tracker.update({old}, 100.0) == ((), 0)


def test_new_file_is_reported_once_after_it_stops_changing(tmp_path: Path) -> None:
    files = FakeFiles()
    new = tmp_path / "new.m4a"
    tracker = files.tracker([])

    files.stamps[new] = (100, 1)
    assert tracker.update({new}, 0.0) == ((), 1)
    assert tracker.update({new}, SETTLE - 0.1) == ((), 1)
    assert tracker.update({new}, SETTLE) == ((new,), 0)
    assert tracker.update({new}, SETTLE + 100) == ((), 0)


def test_growing_file_restarts_the_wait(tmp_path: Path) -> None:
    files = FakeFiles()
    new = tmp_path / "recording.mkv"
    tracker = files.tracker([])

    files.stamps[new] = (100, 1)
    tracker.update({new}, 0.0)
    files.stamps[new] = (200, 2)
    assert tracker.update({new}, 8.0) == ((), 1)
    assert tracker.update({new}, 16.0) == ((), 1)
    assert tracker.update({new}, 17.0) == ((new,), 0)


def test_file_still_open_for_writing_waits_until_released(tmp_path: Path) -> None:
    files = FakeFiles()
    new = tmp_path / "copying.mp4"
    tracker = files.tracker([])
    files.stamps[new] = (4_000_000_000, 1)
    files.locked.add(new)

    tracker.update({new}, 0.0)
    assert tracker.update({new}, 60.0) == ((), 1)
    files.locked.clear()
    assert tracker.update({new}, 63.0) == ((new,), 0)


def test_empty_and_vanished_files_are_not_reported(tmp_path: Path) -> None:
    files = FakeFiles()
    empty = tmp_path / "empty.wav"
    temporary = tmp_path / "temporary.mp4"
    tracker = files.tracker([])
    files.stamps[empty] = (0, 1)
    files.stamps[temporary] = (50, 1)

    assert tracker.update({empty, temporary}, 0.0) == ((), 2)
    del files.stamps[temporary]
    assert tracker.update({empty}, 60.0) == ((), 1)
    # 사라졌다가 다시 생긴 파일은 처음부터 다시 기다립니다.
    files.stamps[temporary] = (50, 1)
    assert tracker.update({empty, temporary}, 61.0) == ((), 2)
    assert tracker.update({empty, temporary}, 70.0) == ((temporary,), 1)


def test_known_files_stay_known_when_the_folder_drops_out(tmp_path: Path) -> None:
    files = FakeFiles()
    old = tmp_path / "old.mp4"
    done = tmp_path / "done.mp4"
    files.stamps.update({old: (10, 1), done: (10, 1)})
    tracker = files.tracker([old])
    tracker.update({old, done}, 0.0)
    assert tracker.update({old, done}, SETTLE) == ((done,), 0)

    # 네트워크 폴더가 잠깐 비어 보였다가 돌아와도 기존 파일을 새 파일로 보지 않습니다.
    assert tracker.update(set(), 20.0) == ((), 0)
    assert tracker.update({old, done}, 40.0) == ((), 0)
    assert tracker.update({old, done}, 80.0) == ((), 0)


def test_ready_files_come_in_path_order(tmp_path: Path) -> None:
    files = FakeFiles()
    second = tmp_path / "b.mp4"
    first = tmp_path / "A.mp4"
    files.stamps.update({second: (1, 1), first: (1, 1)})
    tracker = files.tracker([])
    tracker.update({second, first}, 0.0)

    assert tracker.update({second, first}, SETTLE) == ((first, second), 0)


def test_slow_listing_stretches_the_wait_only_up_to_the_cap() -> None:
    assert next_delay(3.0, 0.01) == 3.0
    assert next_delay(3.0, 2.0) == 10.0
    assert next_delay(3.0, 25.0) == WATCH_MAX_INTERVAL_SECONDS
    assert next_delay(3.0, 600.0) == WATCH_MAX_INTERVAL_SECONDS
    # 기본 간격이 상한보다 길게 설정돼 있으면 그 간격을 그대로 씁니다.
    assert next_delay(60.0, 1.0) == 60.0
    assert next_delay(0.05, 0.001) == 0.05


@pytest.mark.skipif(os.name != "nt", reason="write-lock detection uses the Windows API")
def test_is_file_free_detects_a_writer(tmp_path: Path) -> None:
    target = tmp_path / "녹화 중.mp4"
    with target.open("wb") as handle:
        handle.write(b"partial")
        handle.flush()
        assert is_file_free(target) is False
    assert is_file_free(target) is True
    with target.open("rb"):
        assert is_file_free(target) is True
    assert is_file_free(tmp_path / "missing.mp4") is True


def test_watcher_thread_reports_only_files_added_after_start(tmp_path: Path) -> None:
    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"old")
    reports: queue.Queue[tuple[FolderWatcher, WatchReport]] = queue.Queue()
    watcher = FolderWatcher(
        tmp_path,
        lambda source, report: reports.put((source, report)),
        interval_seconds=0.05,
        settle_seconds=0.2,
    )
    watcher.start()
    try:
        source, report = reports.get(timeout=10)
        assert source is watcher
        assert report == WatchReport()

        added = tmp_path / "sub" / "added.m4a"
        added.parent.mkdir()
        added.write_bytes(b"new")
        ready: tuple[Path, ...] = ()
        while not ready:
            ready = reports.get(timeout=10)[1].ready
        assert ready == (added,)
    finally:
        watcher.stop()
    watcher._thread.join(timeout=10)
    assert not watcher._thread.is_alive()
    assert reports.empty()


def test_watcher_thread_reports_a_missing_folder_and_recovers(tmp_path: Path) -> None:
    root = tmp_path / "share"
    reports: queue.Queue[tuple[FolderWatcher, WatchReport]] = queue.Queue()
    watcher = FolderWatcher(
        root,
        lambda source, report: reports.put((source, report)),
        interval_seconds=0.05,
        settle_seconds=0.2,
    )
    watcher.start()
    try:
        assert reports.get(timeout=10)[1] == WatchReport(available=False)
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "first.mp4").write_bytes(b"x")
        staging.rename(root)
        # 폴더가 처음 보인 순간에 있던 파일이 기준이 되므로 새 파일로 보고되지 않습니다.
        assert reports.get(timeout=10)[1] == WatchReport()
        with pytest.raises(queue.Empty):
            reports.get(timeout=0.6)
    finally:
        watcher.stop()
