from __future__ import annotations

import ctypes
import logging
import os
import tkinter as tk
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from tkinter import messagebox

from .app_paths import AppPaths, migrate_legacy_data, pending_legacy_root
from .constants import APP_DISPLAY_NAME, APP_NAME, LEGACY_APP_NAME
from .gui import SubtitleBatcherApp
from .settings import load_settings

_MUTEX_HANDLE: int | None = None
_SYNCHRONIZE = 0x00100000


def main() -> None:
    if os.name == "nt":
        _enable_dpi_awareness()
    paths = AppPaths.discover()
    if os.name == "nt" and not _claim_single_instance():
        _show_message(messagebox.showinfo, "프로그램이 이미 실행 중입니다.")
        return
    legacy = pending_legacy_root(paths)
    if legacy is not None:
        if os.name == "nt" and _legacy_instance_running():
            _show_message(
                messagebox.showwarning,
                f"이전 버전({LEGACY_APP_NAME})이 실행 중입니다.\n"
                "이전 버전을 종료한 뒤 다시 실행하면 기존 엔진과 모델을 그대로 이어서 사용합니다.\n\n"
                f"기존 데이터 폴더: {legacy}",
            )
            return
        try:
            migrate_legacy_data(paths)
        except OSError as exc:
            _show_message(
                messagebox.showerror,
                "이전 버전의 데이터 폴더 이름을 바꾸지 못했습니다.\n"
                "엔진과 모델은 새로 내려받지 않았습니다. 폴더를 사용 중인 프로그램을 닫고 "
                "다시 실행하거나, 폴더 이름을 직접 바꿔 주세요.\n\n"
                f"기존 폴더: {legacy}\n새 폴더: {paths.root}\n오류: {exc}",
            )
            return
    try:
        paths.ensure_base_dirs()
        _configure_logging(paths.logs / "application.log")
    except OSError as exc:
        _show_message(
            messagebox.showerror,
            "EXE가 있는 폴더에 데이터 폴더를 만들 수 없습니다.\n"
            "쓰기 가능한 폴더(예: D 드라이브)로 EXE를 옮긴 뒤 다시 실행하세요.\n\n"
            f"데이터 경로: {paths.root}\n오류: {exc}",
        )
        return
    if legacy is not None:
        logging.getLogger(__name__).info("Renamed legacy data folder %s to %s", legacy, paths.root)
    root = tk.Tk()

    def report_exception(exc_type: type[BaseException], exc: BaseException, trace: object) -> None:
        logging.getLogger(__name__).exception("Unhandled Tk exception", exc_info=(exc_type, exc, trace))
        messagebox.showerror(APP_DISPLAY_NAME, f"예상하지 못한 오류가 발생했습니다.\n\n{exc}")

    root.report_callback_exception = report_exception  # type: ignore[method-assign]
    SubtitleBatcherApp(root, paths, load_settings(paths))
    root.mainloop()


def _show_message(show: Callable[[str, str], object], message: str) -> None:
    root = tk.Tk()
    root.withdraw()
    show(APP_DISPLAY_NAME, message)
    root.destroy()


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
    handle = kernel32.CreateMutexW(None, False, f"Local\\{APP_NAME}-1.0")
    if not handle:
        return True
    _MUTEX_HANDLE = int(handle)
    return kernel32.GetLastError() != 183


def _legacy_instance_running() -> bool:
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenMutexW.restype = ctypes.c_void_p
    handle = kernel32.OpenMutexW(_SYNCHRONIZE, False, f"Local\\{LEGACY_APP_NAME}-1.0")
    if not handle:
        return False
    kernel32.CloseHandle(ctypes.c_void_p(handle))
    return True


if __name__ == "__main__":
    main()
