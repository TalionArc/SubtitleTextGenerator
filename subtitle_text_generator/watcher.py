from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from pathlib import Path

from .constants import MEDIA_EXTENSIONS

WATCH_INTERVAL_SECONDS = 3.0
# 복사·다운로드·녹화 중인 파일을 건드리지 않도록, 크기와 수정 시각이 이만큼 그대로일 때만 처리합니다.
WATCH_SETTLE_SECONDS = 9.0
WATCH_MAX_INTERVAL_SECONDS = 30.0

_GENERIC_READ = 0x80000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_DELETE = 0x00000004
_OPEN_EXISTING = 3
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_SHARING_ERRORS = frozenset({32, 33})  # ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION

Stamp = tuple[int, int]


@dataclass(frozen=True, slots=True)
class WatchReport:
    ready: tuple[Path, ...] = ()
    settling: int = 0
    available: bool = True


def list_media(root: Path) -> frozenset[Path] | None:
    """Media files under root, or None when root itself cannot be read.

    Follows directory junctions like the list scan does, but enters each junction target once.
    """
    found: set[Path] = set()
    visited = {_real_key(str(root))}
    stack = [str(root)]
    at_root = True
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                for item in iterator:
                    try:
                        if item.is_dir(follow_symlinks=False):
                            if _is_junction(item):
                                target = _real_key(item.path)
                                if target in visited:
                                    continue
                                visited.add(target)
                            stack.append(item.path)
                        elif (
                            os.path.splitext(item.name)[1].casefold() in MEDIA_EXTENSIONS
                            and item.is_file()
                        ):
                            found.add(Path(item.path))
                    except OSError:
                        continue
        except OSError:
            if at_root:
                return None
        at_root = False
    return frozenset(found)


def file_stamp(path: Path) -> Stamp | None:
    # 폴더 목록에 적힌 크기는 쓰는 중인 파일에서 늦게 바뀌므로 파일을 직접 조회합니다.
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def is_file_free(path: Path) -> bool:
    """False while another program still has the file open for writing."""
    if os.name != "nt":
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    # 쓰기 공유를 허용하지 않고 열면, 다른 프로그램이 쓰기용으로 열어 둔 동안에는 실패합니다.
    handle = kernel32.CreateFileW(
        str(path),
        _GENERIC_READ,
        _FILE_SHARE_READ | _FILE_SHARE_DELETE,
        None,
        _OPEN_EXISTING,
        0,
        None,
    )
    if handle is None or handle == _INVALID_HANDLE:
        return ctypes.get_last_error() not in _SHARING_ERRORS
    kernel32.CloseHandle(handle)
    return True


class NewFileTracker:
    """Reports each media file once, after it appeared and stopped changing.

    Files that vanish are never forgotten, so a folder that drops out for a moment
    (network share, removable drive) cannot make existing files look new.
    """

    def __init__(
        self,
        baseline: Iterable[Path],
        *,
        settle_seconds: float = WATCH_SETTLE_SECONDS,
        stamp: Callable[[Path], Stamp | None] = file_stamp,
        is_free: Callable[[Path], bool] = is_file_free,
    ) -> None:
        self._known = set(baseline)
        self._settle_seconds = settle_seconds
        self._stamp = stamp
        self._is_free = is_free
        self._pending: dict[Path, tuple[Stamp, float]] = {}

    def update(self, present: Collection[Path], now: float) -> tuple[tuple[Path, ...], int]:
        """Return (files ready to process, files still being written)."""
        for path in [path for path in self._pending if path not in present]:
            del self._pending[path]
        ready: list[Path] = []
        for path in present:
            if path in self._known:
                continue
            stamp = self._stamp(path)
            if stamp is None:
                self._pending.pop(path, None)
                continue
            seen = self._pending.get(path)
            if seen is None or seen[0] != stamp:
                self._pending[path] = (stamp, now)
                continue
            if stamp[0] <= 0 or now - seen[1] < self._settle_seconds:
                continue
            if self._is_free(path):
                ready.append(path)
        for path in ready:
            del self._pending[path]
            self._known.add(path)
        ready.sort(key=lambda path: str(path).casefold())
        return tuple(ready), len(self._pending)


class FolderWatcher:
    """Polls a folder on a background thread and reports newly added media files."""

    def __init__(
        self,
        root: Path,
        callback: Callable[[FolderWatcher, WatchReport], None],
        *,
        interval_seconds: float = WATCH_INTERVAL_SECONDS,
        settle_seconds: float = WATCH_SETTLE_SECONDS,
    ) -> None:
        self.root = root
        self._callback = callback
        self._interval_seconds = interval_seconds
        self._settle_seconds = settle_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        tracker: NewFileTracker | None = None
        last: WatchReport | None = None
        delay = 0.0
        while not self._stop_event.wait(delay):
            started = time.monotonic()
            listing_seconds = 0.0
            try:
                present = list_media(self.root)
                listing_seconds = time.monotonic() - started
                if present is None:
                    report = WatchReport(available=False)
                elif tracker is None:
                    # 감시를 시작할 때 이미 있던 파일은 새 파일로 보지 않습니다.
                    tracker = NewFileTracker(present, settle_seconds=self._settle_seconds)
                    report = WatchReport()
                else:
                    ready, settling = tracker.update(present, time.monotonic())
                    report = WatchReport(ready=ready, settling=settling)
            except Exception:
                logging.getLogger(__name__).exception("Folder watch failed for %s", self.root)
                report = WatchReport(available=False)
            if self._stop_event.is_set():
                return
            if report.ready or report != last:
                self._callback(self, report)
            last = WatchReport(settling=report.settling, available=report.available)
            delay = next_delay(self._interval_seconds, listing_seconds)


def next_delay(interval_seconds: float, listing_seconds: float) -> float:
    """Seconds to wait before looking at the folder again.

    A slow listing (huge folder, slow drive) stretches the wait so the watch does not keep the
    disk busy, but only up to WATCH_MAX_INTERVAL_SECONDS: one stalled look must not turn into
    minutes before a new file is picked up.
    """
    longest = max(interval_seconds, WATCH_MAX_INTERVAL_SECONDS)
    return min(max(interval_seconds, listing_seconds * 5), longest)


def _is_junction(item: os.DirEntry[str]) -> bool:
    check = getattr(item, "is_junction", None)
    return bool(check()) if check is not None else False


def _real_key(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))
