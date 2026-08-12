from __future__ import annotations

import ctypes
import logging
import os
import tkinter as tk
from contextlib import suppress
from pathlib import Path
from tkinter import messagebox

from .app_paths import AppPaths
from .constants import APP_DISPLAY_NAME
from .gui import SubtitleBatcherApp
from .settings import load_settings

_MUTEX_HANDLE: int | None = None


def main() -> None:
    if os.name == "nt":
        _enable_dpi_awareness()
    paths = AppPaths.discover()
    try:
        paths.ensure_base_dirs()
        _configure_logging(paths.logs / "application.log")
    except OSError as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            APP_DISPLAY_NAME,
            "EXE가 있는 폴더에 데이터 폴더를 만들 수 없습니다.\n"
            "쓰기 가능한 폴더(예: D 드라이브)로 EXE를 옮긴 뒤 다시 실행하세요.\n\n"
            f"데이터 경로: {paths.root}\n오류: {exc}",
        )
        root.destroy()
        return
    if os.name == "nt" and not _claim_single_instance():
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(APP_DISPLAY_NAME, "프로그램이 이미 실행 중입니다.")
        root.destroy()
        return
    root = tk.Tk()

    def report_exception(exc_type: type[BaseException], exc: BaseException, trace: object) -> None:
        logging.getLogger(__name__).exception("Unhandled Tk exception", exc_info=(exc_type, exc, trace))
        messagebox.showerror(APP_DISPLAY_NAME, f"예상하지 못한 오류가 발생했습니다.\n\n{exc}")

    root.report_callback_exception = report_exception  # type: ignore[method-assign]
    SubtitleBatcherApp(root, paths, load_settings(paths))
    root.mainloop()


def _configure_logging(log_file: Path) -> None:
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        encoding="utf-8",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _enable_dpi_awareness() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        with suppress(AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()


def _claim_single_instance() -> bool:
    global _MUTEX_HANDLE
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, "Local\\LectureSubtitleBatcher-1.0")
    if not handle:
        return True
    _MUTEX_HANDLE = int(handle)
    return kernel32.GetLastError() != 183


if __name__ == "__main__":
    main()
