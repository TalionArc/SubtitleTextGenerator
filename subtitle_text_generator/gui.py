from __future__ import annotations

import hashlib
import os
import queue
import shutil
import time
import tkinter as tk
from contextlib import suppress
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

from .app_paths import AppPaths
from .constants import (
    APP_DISPLAY_NAME,
    APP_VERSION,
    ENGINE_VERSION,
    MIN_SETUP_FREE_BYTES,
    SETUP_DOWNLOAD_BYTES,
    SETUP_INSTALLED_BYTES,
)
from .controller import BatchController, ControllerEvent
from .engine import EngineRunner
from .models import JobMode, JobRequest, JobStatus, MediaEntry, MediaKind
from .processor import JobProcessor
from .runtime import RuntimeManager
from .scanner import matching_outputs, media_entry, scan_media
from .settings import Settings, normalize_glossary, save_settings, valid_root_directory
from .sorting import SORT_COLUMNS, sorted_ids
from .watcher import FolderWatcher, WatchReport

MESSAGE_PREVIEW_CHARS = 120
STATUS_PREVIEW_CHARS = 200
TOOLTIP_MAX_CHARS = 1_500
MESSAGE_MIN_WIDTH = 120
CELL_PADDING = 24
TOOLTIP_DELAY_MS = 600
SETUP_PROMPT_DELAY_MS = 150
SETUP_PENDING_MESSAGE = (
    "인식 환경이 설치되지 않았습니다. '용어집 / 설정' 탭의 '환경 검사 / 복구'로 설치할 수 있습니다."
)
COLUMN_HEADINGS = {
    "selected": "선택",
    "kind": "종류",
    "status": "상태",
    "folder": "상대 폴더",
    "name": "파일명",
    "output": "기존 결과",
    "stage": "처리 단계",
    "elapsed": "경과",
    "message": "메시지",
}
QUEUED_STAGE = "대기열"
WATCH_STAGE = "감시 대기"
WATCH_NAME_CHARS = 48


def single_line_preview(text: str, limit: int = MESSAGE_PREVIEW_CHARS) -> str:
    flattened = " ".join(text.split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 1].rstrip() + "…"


def format_gigabytes(size: int) -> str:
    return f"{size / 1024**3:.1f}".rstrip("0").rstrip(".") + "GB"


def setup_prompt_text(install_root: Path, free_bytes: int | None) -> str:
    lines = [
        f"· 다운로드: 약 {format_gigabytes(SETUP_DOWNLOAD_BYTES)} (인터넷 필요, 중단하면 다음에 이어받습니다)",
        f"· 설치 후 사용 용량: 약 {format_gigabytes(SETUP_INSTALLED_BYTES)}",
        f"· 필요한 여유 공간: {format_gigabytes(MIN_SETUP_FREE_BYTES)} 이상",
        f"· 설치 위치: {install_root}",
    ]
    if free_bytes is not None:
        lines.append(f"· 현재 여유 공간: {format_gigabytes(free_bytes)}")
    lines.extend(
        (
            "",
            "NVIDIA GPU(CUDA)가 있어야 인식할 수 있습니다.",
            "'아니오'를 고르면 설치하지 않으며, '용어집 / 설정' 탭의 '환경 검사 / 복구'로 "
            "나중에 설치할 수 있습니다.",
        )
    )
    return "\n".join(lines)


def heading_text(column: str, sort_column: str, descending: bool) -> str:
    title = COLUMN_HEADINGS[column]
    if column != sort_column:
        return title
    return f"{title} {'▼' if descending else '▲'}"


def watch_state_text(
    *,
    enabled: bool,
    has_folder: bool,
    report: WatchReport | None,
    queued: int,
    engine_ready: bool,
    running: bool = False,
    latest: str = "",
) -> str:
    if not enabled:
        return ""
    if not has_folder:
        return "감시할 폴더를 먼저 고르세요."
    if report is None:
        return "감시 준비 중"
    if not report.available:
        return "폴더에 접근할 수 없습니다. 다시 연결되면 이어서 감시합니다."
    parts = ["감시 중"]
    if report.settling:
        parts.append(f"새 파일 {report.settling}개의 복사가 끝나기를 기다리는 중")
    if queued:
        waiting = f"자동 처리 대기 {queued}개"
        parts.append(waiting if engine_ready else f"{waiting} (인식 환경이 준비되면 시작)")
    if running:
        parts.append("자동 처리 중")
    if latest:
        parts.append(f"최근 자동 처리: {latest}")
    if len(parts) == 1:
        parts.append("지금부터 추가되는 파일만 자동으로 처리합니다.")
    return " · ".join(parts)


def watch_result_text(name: str, status: JobStatus, clock: str) -> str:
    return f"{single_line_preview(name, WATCH_NAME_CHARS)} — {status.value} ({clock})"


def entry_detail_text(entry: MediaEntry) -> str:
    outputs = ", ".join(path.name for path in entry.existing_outputs) or "없음"
    lines = [
        f"파일: {entry.path}",
        f"종류: {entry.kind.value}",
        f"상태: {entry.status.value}",
        f"처리 단계: {entry.stage or '-'}",
        f"경과: {_format_elapsed(entry.elapsed_seconds) or '-'}",
        f"기존 결과: {outputs}",
        "",
        "메시지:",
        entry.message.strip() or "(메시지 없음)",
    ]
    return "\n".join(lines)


class SetupPrompt:
    def __init__(self, parent: tk.Misc, install_root: Path) -> None:
        self.accepted = False
        self.remember_decline = False
        try:
            free_bytes: int | None = shutil.disk_usage(install_root).free
        except OSError:
            free_bytes = None
        self.window = tk.Toplevel(parent)
        self.window.title(APP_DISPLAY_NAME)
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        body = ttk.Frame(self.window, padding=16)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            body,
            text="인식 엔진과 모델을 설치할까요?",
            font=("Malgun Gothic", 11, "bold"),
        ).pack(anchor=tk.W)
        ttk.Label(
            body,
            text=setup_prompt_text(install_root, free_bytes),
            justify=tk.LEFT,
            wraplength="15c",
        ).pack(anchor=tk.W, pady=(8, 12))
        self.skip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            body,
            text="다시 표시하지 않기 ('아니오'를 고를 때만 적용)",
            variable=self.skip_var,
        ).pack(anchor=tk.W)
        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(buttons, text="아니오", command=self._decline).pack(side=tk.RIGHT)
        accept = ttk.Button(buttons, text="예, 설치", style="Primary.TButton", command=self._accept)
        accept.pack(side=tk.RIGHT, padx=(0, 6))
        self.window.bind("<Escape>", lambda _event: self._close())
        self.window.update_idletasks()
        x = parent.winfo_rootx() + max((parent.winfo_width() - self.window.winfo_width()) // 2, 0)
        y = parent.winfo_rooty() + max((parent.winfo_height() - self.window.winfo_height()) // 3, 0)
        self.window.geometry(f"+{x}+{y}")
        accept.focus_set()

    def show(self) -> tuple[bool, bool]:
        self.window.grab_set()
        self.window.wait_window()
        return self.accepted, self.remember_decline

    def _accept(self) -> None:
        self.accepted = True
        self.window.destroy()

    def _decline(self) -> None:
        self.remember_decline = bool(self.skip_var.get())
        self.window.destroy()

    def _close(self) -> None:
        self.window.destroy()


class SubtitleBatcherApp:
    POLL_MS = 100

    def __init__(self, root: tk.Tk, paths: AppPaths, settings: Settings) -> None:
        self.root = root
        self.paths = paths
        self.settings = settings
        self.events: queue.Queue[ControllerEvent] = queue.Queue()
        self.entries: dict[str, MediaEntry] = {}
        self.selected: set[str] = set()
        self.ready = False
        self._sort_column = settings.sort_column if settings.sort_column in SORT_COLUMNS else ""
        self._sort_descending = settings.sort_descending
        self._scanned_root: Path | None = None
        self._watcher: FolderWatcher | None = None
        self._watch_report: WatchReport | None = None
        self._watch_queue: list[str] = []
        self._watch_batch = 0
        self._watch_latest = ""
        self._batch_iids: set[str] = set()
        self._tooltip: tk.Toplevel | None = None
        self._tooltip_after: str | None = None
        self._tooltip_iid = ""

        self.runtime = RuntimeManager(paths)
        self.engine = EngineRunner(self.runtime)
        self.processor = JobProcessor(paths, self.runtime, self.engine)
        self.controller = BatchController(
            self.runtime, self.engine, self.processor, self.events.put
        )

        self.root.title(f"{APP_DISPLAY_NAME} {APP_VERSION}")
        self.root.geometry(settings.window_geometry)
        self.root.minsize(940, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._configure_style()
        self._build_ui()
        self._scan()
        self.root.after(self.POLL_MS, self._drain_events)
        self.root.after(SETUP_PROMPT_DELAY_MS, self._begin_setup)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Treeview", rowheight=25)
        style.configure("Primary.TButton", padding=(12, 6))
        style.configure("Danger.TButton", padding=(10, 6))
        try:
            self._tree_font = tkfont.nametofont(style.lookup("Treeview", "font") or "TkDefaultFont")
        except tk.TclError:
            self._tree_font = tkfont.nametofont("TkDefaultFont")

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(container)
        notebook.pack(fill=tk.BOTH, expand=True)
        work_tab = ttk.Frame(notebook, padding=8)
        settings_tab = ttk.Frame(notebook, padding=12)
        notebook.add(work_tab, text="미디어 작업")
        notebook.add(settings_tab, text="용어집 / 설정")

        folder_row = ttk.Frame(work_tab)
        folder_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(folder_row, text="미디어 폴더").pack(side=tk.LEFT, padx=(0, 8))
        self.root_var = tk.StringVar(value=self.settings.root_directory)
        self.root_entry = ttk.Entry(folder_row, textvariable=self.root_var)
        self.root_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(folder_row, text="폴더 선택", command=self._browse).pack(side=tk.LEFT, padx=(8, 4))
        ttk.Button(folder_row, text="다시 검색", command=self._scan).pack(side=tk.LEFT)

        selection_row = ttk.Frame(work_tab)
        selection_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(
            selection_row,
            text="결과 없는 파일 모두 선택",
            command=self._select_without_outputs,
        ).pack(side=tk.LEFT)
        ttk.Button(selection_row, text="선택 해제", command=self._clear_selection).pack(
            side=tk.LEFT, padx=5
        )
        self.count_var = tk.StringVar(value="전체 0개 / 선택 0개")
        ttk.Label(selection_row, textvariable=self.count_var).pack(side=tk.RIGHT)

        output_option_row = ttk.Frame(work_tab)
        output_option_row.pack(fill=tk.X, pady=(0, 8))
        self.audio_timestamp_var = tk.BooleanVar(value=self.settings.audio_txt_timestamps)
        self.audio_timestamp_check = ttk.Checkbutton(
            output_option_row,
            text="오디오 TXT에 타임스탬프 포함",
            variable=self.audio_timestamp_var,
        )
        self.audio_timestamp_check.pack(side=tk.LEFT)
        ttk.Label(
            output_option_row,
            text="예: [00:01:23.456 - 00:01:28.900] (영상 SRT에는 영향 없음)",
        ).pack(side=tk.LEFT, padx=(8, 0))

        watch_row = ttk.Frame(work_tab)
        watch_row.pack(fill=tk.X, pady=(0, 8))
        self.watch_var = tk.BooleanVar(value=self.settings.watch_folder)
        ttk.Checkbutton(
            watch_row,
            text="폴더 감시 (새로 추가되는 파일 자동 생성)",
            variable=self.watch_var,
            command=self._sync_watcher,
        ).pack(side=tk.LEFT)
        self.watch_state_var = tk.StringVar()
        ttk.Label(watch_row, textvariable=self.watch_state_var).pack(side=tk.LEFT, padx=(8, 0))

        # 아래쪽 행을 먼저 배치해야 상태 문구가 여러 줄이 되어도 목록에 가려지지 않습니다.
        progress_row = ttk.Frame(work_tab)
        progress_row.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        self.progress = ttk.Progressbar(progress_row, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="시작 중")
        self.status_label = ttk.Label(
            progress_row, textvariable=self.status_var, anchor=tk.W, justify=tk.LEFT
        )
        self.status_label.pack(fill=tk.X, pady=(4, 0))
        self.status_label.bind(
            "<Configure>",
            lambda event: self.status_label.configure(wraplength=max(event.width - 4, 1)),
        )

        action_row = ttk.Frame(work_tab)
        action_row.pack(side=tk.BOTTOM, fill=tk.X, pady=(9, 0))
        self.generate_button = ttk.Button(
            action_row,
            text="선택 생성 (Turbo → 보정)",
            style="Primary.TButton",
            command=lambda: self._start_selected(JobMode.ADAPTIVE),
        )
        self.generate_button.pack(side=tk.LEFT)
        self.reprocess_button = ttk.Button(
            action_row,
            text="선택 재생성",
            command=lambda: self._start_selected(JobMode.REPROCESS),
        )
        self.reprocess_button.pack(side=tk.LEFT, padx=5)
        self.full_large_button = ttk.Button(
            action_row,
            text="선택 전체 large-v3",
            command=lambda: self._start_selected(JobMode.FULL_LARGE),
        )
        self.full_large_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(
            action_row, text="중지", style="Danger.TButton", command=self._stop
        )
        self.stop_button.pack(side=tk.RIGHT)
        ttk.Label(action_row, text="행을 더블클릭하면 전체 메시지를 볼 수 있습니다.").pack(
            side=tk.RIGHT, padx=(0, 12)
        )

        tree_frame = ttk.Frame(work_tab)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(
            tree_frame, columns=SORT_COLUMNS, show="headings", selectmode="browse"
        )
        widths = {
            "selected": 56,
            "kind": 66,
            "status": 105,
            "folder": 145,
            "name": 230,
            "output": 96,
            "stage": 135,
            "elapsed": 66,
            "message": MESSAGE_MIN_WIDTH,
        }
        # stretch 열이 있으면 Tk가 열 폭 합계를 목록 폭에 맞춰 줄여서 가로 스크롤이 메시지 끝까지
        # 가지 못합니다. 폭을 고정하고 메시지 열만 내용과 남는 공간에 맞춰 직접 조절합니다.
        for column in SORT_COLUMNS:
            self.tree.heading(column, command=lambda column=column: self._sort_by(column))
            self.tree.column(column, width=widths[column], minwidth=40, stretch=False)
        self._refresh_headings()
        vertical = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        horizontal = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.tree.bind("<Button-1>", self._tree_click)
        self.tree.bind("<Double-Button-1>", self._tree_double_click)
        self.tree.bind("<space>", self._tree_space)
        self.tree.bind("<Configure>", lambda _event: self._fit_message_column())
        self.tree.bind("<Motion>", self._tree_motion)
        for sequence in ("<Leave>", "<ButtonPress>", "<MouseWheel>", "<Key>"):
            self.tree.bind(sequence, lambda _event: self._hide_tooltip(), add="+")

        ttk.Label(
            settings_tab,
            text="전역 용어집",
            font=("Malgun Gothic", 11, "bold"),
        ).pack(anchor=tk.W)
        ttk.Label(
            settings_tab,
            text="한 줄에 한 용어를 입력하세요. 중복을 제거해 최대 50개·500자까지 모든 작업에 적용합니다.",
        ).pack(anchor=tk.W, pady=(3, 7))
        glossary_frame = ttk.Frame(settings_tab)
        glossary_frame.pack(fill=tk.BOTH, expand=True)
        self.glossary_text = tk.Text(glossary_frame, height=14, wrap=tk.WORD, undo=True)
        self.glossary_text.insert("1.0", self.settings.glossary_text)
        glossary_scroll = ttk.Scrollbar(
            glossary_frame, orient=tk.VERTICAL, command=self.glossary_text.yview
        )
        self.glossary_text.configure(yscrollcommand=glossary_scroll.set)
        self.glossary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        glossary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.glossary_text.bind("<KeyRelease>", lambda _event: self._update_glossary_count())
        self.glossary_count_var = tk.StringVar()
        ttk.Label(settings_tab, textvariable=self.glossary_count_var).pack(anchor=tk.E, pady=(4, 12))
        self._update_glossary_count()

        environment = ttk.LabelFrame(settings_tab, text="로컬 인식 환경", padding=10)
        environment.pack(fill=tk.X)
        ttk.Label(
            environment,
            text=f"Faster-Whisper-XXL {ENGINE_VERSION} / large-v3-turbo + large-v3 (한국어 인식 전용)",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(environment, text=str(self.paths.root)).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(environment, text="앱 데이터 폴더 열기", command=self._open_data_folder).grid(
            row=0, column=1, rowspan=2, padx=(15, 5)
        )
        self.repair_button = ttk.Button(environment, text="환경 검사 / 복구", command=self._repair)
        self.repair_button.grid(row=0, column=2, rowspan=2)
        environment.columnconfigure(0, weight=1)

        self._set_action_state(False)
        self.repair_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.DISABLED)

    def _begin_setup(self) -> None:
        if self.runtime.ready():
            self.controller.start_setup()
            return
        if not self.settings.skip_setup_prompt:
            accepted, remember_decline = SetupPrompt(self.root, self.paths.root).show()
            if accepted:
                self.controller.start_setup()
                return
            if remember_decline:
                self.settings.skip_setup_prompt = True
                with suppress(OSError):
                    save_settings(self.paths, self.settings)
        self._set_status(SETUP_PENDING_MESSAGE)
        self.repair_button.configure(state=tk.NORMAL)

    def _set_status(self, text: str) -> None:
        self.status_var.set(single_line_preview(text, STATUS_PREVIEW_CHARS))

    def _scan(self) -> None:
        root_path = valid_root_directory(self.root_var.get())
        if root_path is None:
            if self.root_var.get().strip():
                self._set_status("미디어 폴더를 찾을 수 없습니다.")
            else:
                self._set_status("'폴더 선택'으로 영상·녹음 파일이 있는 폴더를 고르세요.")
            self._refresh_watch_state()
            return
        self.settings.root_directory = str(root_path)
        self._scanned_root = root_path
        self._hide_tooltip()
        self.entries.clear()
        self.selected.clear()
        self.tree.delete(*self.tree.get_children())
        for entry in scan_media(root_path):
            iid = self._iid(entry.path)
            self.entries[iid] = entry
            self.tree.insert("", tk.END, iid=iid, values=self._row_values(iid, entry))
        # 같은 폴더를 다시 검색한 경우, 감시로 대기 중이던 파일은 그대로 대기시킵니다.
        self._watch_queue = [iid for iid in self._watch_queue if iid in self.entries]
        for iid in self._watch_queue:
            self._mark_watch_queued(self.entries[iid])
            self._refresh_row(iid)
        self._apply_order()
        self._set_status(f"미디어 {len(self.entries)}개 검색 완료")
        self._update_count()
        self._fit_message_column()
        self._sync_watcher()

    def _browse(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.root_var.get() or None)
        if selected:
            self.root_var.set(selected)
            self._scan()

    def _sort_by(self, column: str) -> None:
        self._hide_tooltip()
        if column == self._sort_column:
            self._sort_descending = not self._sort_descending
        else:
            self._sort_column = column
            self._sort_descending = False
        self._refresh_headings()
        self._apply_order()

    def _refresh_headings(self) -> None:
        for column in SORT_COLUMNS:
            self.tree.heading(
                column, text=heading_text(column, self._sort_column, self._sort_descending)
            )

    def _apply_order(self) -> None:
        ordered = sorted_ids(self.entries, self.selected, self._sort_column, self._sort_descending)
        if ordered != list(self.tree.get_children()):
            for index, iid in enumerate(ordered):
                self.tree.move(iid, "", index)

    def _sync_watcher(self) -> None:
        root_path = self._scanned_root if self.watch_var.get() else None
        current = self._watcher
        if current is not None and current.root != root_path:
            current.stop()
            self._watcher = current = None
            self._watch_report = None
        if root_path is None:
            self._drop_watch_queue("폴더 감시를 꺼서 자동으로 처리하지 않았습니다.")
        elif current is None:
            self._watcher = FolderWatcher(
                root_path,
                lambda source, report: self.events.put(
                    ControllerEvent(kind="watch", payload=(source, report))
                ),
            )
            self._watcher.start()
        self._refresh_watch_state()

    def _refresh_watch_state(self) -> None:
        self.watch_state_var.set(
            watch_state_text(
                enabled=bool(self.watch_var.get()),
                has_folder=self._scanned_root is not None,
                report=self._watch_report,
                queued=len(self._watch_queue),
                engine_ready=self.ready,
                running=bool(self._watch_batch),
                latest=self._watch_latest,
            )
        )

    def _handle_watch(self, source: FolderWatcher, report: WatchReport) -> None:
        # 폴더를 바꾸거나 감시를 끈 뒤에 도착한 이전 감시의 보고는 버립니다.
        if source is not self._watcher:
            return
        self._watch_report = report
        found = 0
        for path in report.ready:
            iid = self._iid(path)
            entry = self.entries.get(iid)
            if entry is None:
                entry = media_entry(path, source.root)
                if entry is None:
                    continue
                self.entries[iid] = entry
                self.tree.insert("", tk.END, iid=iid, values=self._row_values(iid, entry))
            else:
                # '다시 검색'으로 먼저 목록에 올라온 파일입니다. 이미 처리 중이거나 대기 중이거나
                # 그사이 결과가 생겼으면 그대로 둡니다.
                if iid in self._batch_iids or iid in self._watch_queue:
                    continue
                entry.existing_outputs = matching_outputs(entry.path)
                if entry.has_output:
                    self._refresh_row(iid)
                    continue
            found += 1
            if entry.has_output:
                entry.message = "새로 추가됐지만 결과 파일이 이미 있어 자동으로 처리하지 않았습니다."
            else:
                self._mark_watch_queued(entry)
                self._watch_queue.append(iid)
            self._refresh_row(iid)
        if found:
            self._apply_order()
            self._update_count()
            self._fit_message_column()
            self._set_status(f"폴더 감시: 새 파일 {found}개를 찾았습니다.")
        self._refresh_watch_state()

    @staticmethod
    def _mark_watch_queued(entry: MediaEntry) -> None:
        entry.status = JobStatus.WAITING
        entry.stage = WATCH_STAGE
        entry.message = "폴더 감시로 찾은 새 파일입니다. 차례가 되면 자동으로 생성합니다."

    def _drop_watch_queue(self, message: str) -> None:
        for iid in self._watch_queue:
            entry = self.entries.get(iid)
            if entry is not None:
                entry.stage = ""
                entry.message = message
                self._refresh_row(iid)
        if self._watch_queue:
            self._watch_queue.clear()
            self._fit_message_column()

    def _start_watch_jobs(self) -> None:
        if not self.ready or self.controller.busy:
            return
        ordered: list[str] = []
        for iid in self._watch_queue:
            entry = self.entries.get(iid)
            if entry is None:
                continue
            if not entry.path.is_file():
                # 처리하기 전에 사라진 임시 파일은 목록에서도 뺍니다.
                del self.entries[iid]
                self.selected.discard(iid)
                self.tree.delete(iid)
                continue
            entry.existing_outputs = matching_outputs(entry.path)
            if entry.has_output:
                entry.stage = ""
                entry.message = "결과 파일이 이미 있어 자동으로 처리하지 않았습니다."
                self._refresh_row(iid)
                continue
            ordered.append(iid)
        self._watch_queue = ordered
        if ordered and self._launch(ordered, JobMode.ADAPTIVE):
            self._watch_batch = len(ordered)
        self._update_count()
        self._fit_message_column()
        self._refresh_watch_state()

    def _stop(self) -> None:
        self._drop_watch_queue("중지해서 자동으로 처리하지 않았습니다.")
        self._watch_batch = 0
        self._refresh_watch_state()
        self.controller.cancel()

    def _select_without_outputs(self) -> None:
        self.selected = {iid for iid, entry in self.entries.items() if not entry.has_output}
        self._refresh_checks()

    def _clear_selection(self) -> None:
        self.selected.clear()
        self._refresh_checks()

    def _tree_click(self, event: tk.Event[tk.Misc]) -> None:
        self._hide_tooltip()
        if self.tree.identify_region(event.x, event.y) != "cell":
            return
        iid = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if iid and column == "#1":
            self._toggle(iid)

    def _tree_double_click(self, event: tk.Event[tk.Misc]) -> str | None:
        if self.tree.identify_region(event.x, event.y) != "cell":
            return None
        iid = self.tree.identify_row(event.y)
        if not iid:
            return None
        if self.tree.identify_column(event.x) == "#1":
            self._toggle(iid)
        else:
            self._show_details(iid)
        return "break"

    def _tree_space(self, _event: tk.Event[tk.Misc]) -> str:
        selection = self.tree.selection()
        if selection:
            self._toggle(selection[0])
        return "break"

    def _tree_motion(self, event: tk.Event[tk.Misc]) -> None:
        iid = ""
        if self.tree.identify_region(event.x, event.y) == "cell":
            iid = self.tree.identify_row(event.y)
        if iid == self._tooltip_iid:
            return
        self._hide_tooltip()
        self._tooltip_iid = iid
        entry = self.entries.get(iid)
        if entry is not None and entry.message.strip():
            x, y = event.x_root, event.y_root
            self._tooltip_after = self.root.after(
                TOOLTIP_DELAY_MS, lambda: self._show_tooltip(iid, x, y)
            )

    def _show_tooltip(self, iid: str, x: int, y: int) -> None:
        self._tooltip_after = None
        entry = self.entries.get(iid)
        if entry is None or not entry.message.strip():
            return
        text = entry.message.strip()
        if len(text) > TOOLTIP_MAX_CHARS:
            text = text[:TOOLTIP_MAX_CHARS].rstrip() + "\n… (더블클릭하면 전체를 볼 수 있습니다)"
        tip = tk.Toplevel(self.root)
        tip.wm_overrideredirect(True)
        tk.Label(
            tip,
            text=text,
            justify=tk.LEFT,
            wraplength="16c",
            background="#ffffe1",
            relief=tk.SOLID,
            borderwidth=1,
            padx=6,
            pady=4,
        ).pack()
        tip.update_idletasks()
        left = self.root.winfo_rootx()
        right = left + self.root.winfo_width()
        bottom = self.root.winfo_rooty() + self.root.winfo_height()
        tip_x = max(min(x + 16, right - tip.winfo_width()), left)
        tip_y = y + 20
        if tip_y + tip.winfo_height() > bottom:
            tip_y = y - tip.winfo_height() - 12
        tip.wm_geometry(f"+{tip_x}+{tip_y}")
        self._tooltip = tip

    def _hide_tooltip(self) -> None:
        if self._tooltip_after is not None:
            self.root.after_cancel(self._tooltip_after)
            self._tooltip_after = None
        if self._tooltip is not None:
            self._tooltip.destroy()
            self._tooltip = None
        self._tooltip_iid = ""

    def _show_details(self, iid: str) -> None:
        entry = self.entries.get(iid)
        if entry is None:
            return
        self._hide_tooltip()
        detail = entry_detail_text(entry)
        window = tk.Toplevel(self.root)
        window.title(f"{entry.path.name} — 상세")
        window.geometry("760x380")
        window.minsize(420, 240)
        window.transient(self.root)
        frame = ttk.Frame(window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        buttons = ttk.Frame(frame)
        buttons.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        ttk.Button(buttons, text="닫기", command=window.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="복사", command=lambda: self._copy_text(detail)).pack(
            side=tk.RIGHT, padx=(0, 6)
        )
        text = tk.Text(frame, wrap=tk.WORD, height=12, font="TkDefaultFont")
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text.insert("1.0", detail)
        text.configure(state=tk.DISABLED)
        window.bind("<Escape>", lambda _event: window.destroy())
        text.focus_set()

    def _copy_text(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _fit_message_column(self) -> None:
        other = sum(
            int(self.tree.column(column, "width"))
            for column in self.tree["columns"]
            if column != "message"
        )
        content = max(
            (
                self._tree_font.measure(self.tree.set(iid, "message"))
                for iid in self.tree.get_children()
            ),
            default=0,
        )
        available = self.tree.winfo_width() - other - 4
        width = max(MESSAGE_MIN_WIDTH, content + CELL_PADDING, available)
        if width != int(self.tree.column("message", "width")):
            self.tree.column("message", width=width)

    def _toggle(self, iid: str) -> None:
        if iid in self.selected:
            self.selected.remove(iid)
        else:
            self.selected.add(iid)
        self._refresh_row(iid)
        self._update_count()

    def _start_selected(self, mode: JobMode) -> None:
        if not self.ready:
            messagebox.showwarning(APP_DISPLAY_NAME, "엔진과 모델 설치가 아직 완료되지 않았습니다.")
            return
        ordered = [iid for iid in self.tree.get_children() if iid in self.selected]
        if not ordered:
            messagebox.showinfo(APP_DISPLAY_NAME, "처리할 미디어 파일을 먼저 선택하세요.")
            return
        if mode in {JobMode.REPROCESS, JobMode.FULL_LARGE}:
            label = "전체 large-v3로 재생성" if mode is JobMode.FULL_LARGE else "재생성"
            if not messagebox.askokcancel(
                APP_DISPLAY_NAME,
                f"선택한 {len(ordered)}개 파일을 {label}합니다.\n"
                "기존 SRT/TXT는 각각 .bak 하나로 교체됩니다.",
            ):
                return
        if not self._launch(ordered, mode):
            messagebox.showwarning(APP_DISPLAY_NAME, "다른 작업이 이미 진행 중입니다.")

    def _launch(self, ordered: list[str], mode: JobMode) -> bool:
        glossary = normalize_glossary(self.glossary_text.get("1.0", "end-1c"))
        audio_txt_timestamps = bool(self.audio_timestamp_var.get())
        jobs = [
            JobRequest(
                video=self.entries[iid].path,
                root=self.entries[iid].root,
                mode=mode,
                glossary=glossary,
                audio_txt_timestamps=audio_txt_timestamps,
            )
            for iid in ordered
        ]
        if not self.controller.start_jobs(jobs):
            return False
        self._batch_iids = set(ordered)
        # 방금 시작한 파일은 감시 대기에서 빼서 두 번 처리하지 않습니다.
        self._watch_queue = [iid for iid in self._watch_queue if iid not in self._batch_iids]
        for iid in ordered:
            entry = self.entries[iid]
            entry.status = JobStatus.WAITING
            entry.stage = QUEUED_STAGE
            entry.message = ""
            self._refresh_row(iid)
        self._fit_message_column()
        return True

    def _repair(self) -> None:
        # 미설치 상태에서는 빠진 구성만 받고, 설치가 끝난 상태에서만 전체를 다시 설치합니다.
        if not self.controller.start_setup(force_repair=self.runtime.ready()):
            messagebox.showwarning(APP_DISPLAY_NAME, "다른 작업이 이미 진행 중입니다.")

    def _open_data_folder(self) -> None:
        self.paths.ensure_base_dirs()
        try:
            os.startfile(self.paths.root)  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror(APP_DISPLAY_NAME, str(exc))

    def _drain_events(self) -> None:
        try:
            while True:
                self._handle_event(self.events.get_nowait())
        except queue.Empty:
            pass
        if self._watch_queue:
            self._start_watch_jobs()
        self.root.after(self.POLL_MS, self._drain_events)

    def _handle_event(self, event: ControllerEvent) -> None:
        if event.kind == "watch":
            self._handle_watch(*event.payload)
        elif event.kind in {"setup_started", "batch_started"}:
            self._set_action_state(False)
            self.repair_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
        elif event.kind == "setup_ready":
            self.ready = True
            self.settings.skip_setup_prompt = False
            self._set_status(event.message)
        elif event.kind == "setup_failed":
            self.ready = False
            self._set_status(f"환경 준비 실패: {event.message}")
            messagebox.showerror(APP_DISPLAY_NAME, f"인식 환경 준비 실패\n\n{event.message}")
        elif event.kind == "setup_cancelled":
            self._set_status(event.message)
        elif event.kind in {"setup_progress", "job_progress"}:
            self._set_status(f"{event.stage}: {event.message}")
            self._set_progress(event.fraction)
        elif event.kind == "job_started" and event.video:
            self._update_video_event(event.video, event)
        elif event.kind == "job_finished" and event.video:
            self._update_video_event(event.video, event)
            if self._watch_batch and event.status is not None:
                # 새 행이 정렬 순서상 화면 밖에 있어도 결과를 알 수 있게 감시 문구에 남깁니다.
                self._watch_latest = watch_result_text(
                    event.video.name, event.status, time.strftime("%H:%M")
                )
                self._refresh_watch_state()
            if event.status is JobStatus.FAILED:
                self._set_status(f"실패: {event.video.name} — {event.message}")
        elif event.kind == "cancelling":
            self._set_status(event.message)
        elif event.kind == "batch_finished":
            if self._watch_batch:
                self._set_status(
                    f"폴더 감시: 새 파일 {self._watch_batch}개의 자동 처리가 끝났습니다."
                )
            else:
                self._set_status(event.message)
            self._set_progress(1.0)
        elif event.kind == "idle":
            self.ready = self.runtime.ready()
            self._batch_iids.clear()
            self._watch_batch = 0
            self._set_action_state(self.ready)
            self.repair_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self._refresh_watch_state()

    def _update_video_event(self, video: Path, event: ControllerEvent) -> None:
        iid = self._iid(video)
        entry = self.entries.get(iid)
        if entry is None:
            return
        if event.status:
            entry.status = event.status
        if event.stage:
            entry.stage = event.stage
        if event.message:
            entry.message = event.message
        if event.result:
            entry.elapsed_seconds = event.result.elapsed_seconds
            entry.existing_outputs = matching_outputs(entry.path)
            if event.result.status in {
                JobStatus.COMPLETE,
                JobStatus.REVIEW,
                JobStatus.WARNING,
                JobStatus.SKIPPED,
            }:
                entry.stage = "완료"
            self._update_count()
        self._refresh_row(iid)
        self._fit_message_column()

    def _set_progress(self, fraction: float | None) -> None:
        if fraction is None:
            if str(self.progress["mode"]) != "indeterminate":
                self.progress.configure(mode="indeterminate")
                self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate", value=max(0.0, min(fraction, 1.0)) * 100)

    def _set_action_state(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in (
            self.generate_button,
            self.reprocess_button,
            self.full_large_button,
        ):
            button.configure(state=state)
        self.audio_timestamp_check.configure(state=state)

    def _refresh_checks(self) -> None:
        for iid in self.entries:
            self._refresh_row(iid)
        self._update_count()

    def _refresh_row(self, iid: str) -> None:
        entry = self.entries.get(iid)
        if entry and self.tree.exists(iid):
            self.tree.item(iid, values=self._row_values(iid, entry))

    def _row_values(self, iid: str, entry: MediaEntry) -> tuple[str, ...]:
        output = "없음"
        if entry.existing_outputs:
            extensions = ", ".join(
                sorted({path.suffix.lstrip(".").upper() for path in entry.existing_outputs})
            )
            output = f"있음 ({extensions})"
        return (
            "☑" if iid in self.selected else "☐",
            entry.kind.value,
            entry.status.value,
            entry.relative_parent,
            entry.path.name,
            output,
            entry.stage,
            _format_elapsed(entry.elapsed_seconds),
            single_line_preview(entry.message),
        )

    def _update_count(self) -> None:
        missing = sum(not entry.has_output for entry in self.entries.values())
        video_count = sum(entry.kind is MediaKind.VIDEO for entry in self.entries.values())
        audio_count = sum(entry.kind is MediaKind.AUDIO for entry in self.entries.values())
        self.count_var.set(
            f"전체 {len(self.entries)}개 (영상 {video_count} / 오디오 {audio_count}) / "
            f"결과 없음 {missing}개 / 선택 {len(self.selected)}개"
        )

    def _update_glossary_count(self) -> None:
        terms = normalize_glossary(self.glossary_text.get("1.0", "end-1c"))
        self.glossary_count_var.set(f"적용: {len(terms)}/50개, {len(', '.join(terms))}/500자")

    def _on_close(self) -> None:
        if self.controller.busy and not messagebox.askyesno(
            APP_DISPLAY_NAME, "진행 중인 작업을 중지하고 종료할까요?"
        ):
            return
        if self.controller.busy:
            self.controller.cancel()
        if self._watcher is not None:
            self._watcher.stop()
        self._hide_tooltip()
        self.settings.watch_folder = bool(self.watch_var.get())
        self.settings.sort_column = self._sort_column
        self.settings.sort_descending = self._sort_descending
        self.settings.root_directory = self.root_var.get().strip()
        self.settings.glossary_text = self.glossary_text.get("1.0", "end-1c")
        self.settings.window_geometry = self.root.geometry()
        self.settings.audio_txt_timestamps = bool(self.audio_timestamp_var.get())
        with suppress(OSError):
            save_settings(self.paths, self.settings)
        self.root.destroy()

    @staticmethod
    def _iid(path: Path) -> str:
        return hashlib.sha1(os.path.normcase(str(path.resolve())).encode("utf-8")).hexdigest()


def _format_elapsed(seconds: float) -> str:
    if seconds <= 0:
        return ""
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"
