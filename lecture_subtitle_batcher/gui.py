from __future__ import annotations

import hashlib
import os
import queue
import tkinter as tk
from contextlib import suppress
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .app_paths import AppPaths
from .constants import APP_DISPLAY_NAME, APP_VERSION, ENGINE_VERSION
from .controller import BatchController, ControllerEvent
from .engine import EngineRunner
from .models import JobMode, JobRequest, JobStatus, MediaEntry, MediaKind
from .processor import JobProcessor
from .runtime import RuntimeManager
from .scanner import matching_outputs, scan_media
from .settings import Settings, normalize_glossary, save_settings, valid_root_directory


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
        self.controller.start_setup()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Treeview", rowheight=25)
        style.configure("Primary.TButton", padding=(12, 6))
        style.configure("Danger.TButton", padding=(10, 6))

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

        tree_frame = ttk.Frame(work_tab)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        columns = (
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
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")
        headings = {
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
        widths = {
            "selected": 45,
            "kind": 60,
            "status": 105,
            "folder": 145,
            "name": 230,
            "output": 90,
            "stage": 135,
            "elapsed": 65,
            "message": 260,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], minwidth=40, stretch=column in {"folder", "name", "message"})
        vertical = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        horizontal = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.tree.bind("<Button-1>", self._tree_click)
        self.tree.bind("<space>", self._tree_space)

        action_row = ttk.Frame(work_tab)
        action_row.pack(fill=tk.X, pady=(9, 0))
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
            action_row, text="중지", style="Danger.TButton", command=self.controller.cancel
        )
        self.stop_button.pack(side=tk.RIGHT)

        progress_row = ttk.Frame(work_tab)
        progress_row.pack(fill=tk.X, pady=(8, 0))
        self.progress = ttk.Progressbar(progress_row, mode="determinate", maximum=100)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.status_var = tk.StringVar(value="시작 중")
        ttk.Label(progress_row, textvariable=self.status_var, width=48, anchor=tk.W).pack(
            side=tk.LEFT, padx=(8, 0)
        )

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
            text=f"Faster-Whisper-XXL {ENGINE_VERSION} / large-v3-turbo + large-v3",
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

    def _scan(self) -> None:
        root_path = valid_root_directory(self.root_var.get())
        if root_path is None:
            self.status_var.set("미디어 폴더를 찾을 수 없습니다.")
            return
        self.settings.root_directory = str(root_path)
        self.entries.clear()
        self.selected.clear()
        self.tree.delete(*self.tree.get_children())
        for entry in scan_media(root_path):
            iid = self._iid(entry.path)
            self.entries[iid] = entry
            self.tree.insert("", tk.END, iid=iid, values=self._row_values(iid, entry))
        self.status_var.set(f"미디어 {len(self.entries)}개 검색 완료")
        self._update_count()

    def _browse(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.root_var.get() or None)
        if selected:
            self.root_var.set(selected)
            self._scan()

    def _select_without_outputs(self) -> None:
        self.selected = {iid for iid, entry in self.entries.items() if not entry.has_output}
        self._refresh_checks()

    def _clear_selection(self) -> None:
        self.selected.clear()
        self._refresh_checks()

    def _tree_click(self, event: tk.Event[tk.Misc]) -> None:
        if self.tree.identify_region(event.x, event.y) != "cell":
            return
        iid = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if iid and column == "#1":
            self._toggle(iid)

    def _tree_space(self, _event: tk.Event[tk.Misc]) -> str:
        selection = self.tree.selection()
        if selection:
            self._toggle(selection[0])
        return "break"

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
        for iid in ordered:
            entry = self.entries[iid]
            entry.status = JobStatus.WAITING
            entry.stage = "대기열"
            entry.message = ""
            self._refresh_row(iid)
        if not self.controller.start_jobs(jobs):
            messagebox.showwarning(APP_DISPLAY_NAME, "다른 작업이 이미 진행 중입니다.")

    def _repair(self) -> None:
        if not self.controller.start_setup(force_repair=True):
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
        self.root.after(self.POLL_MS, self._drain_events)

    def _handle_event(self, event: ControllerEvent) -> None:
        if event.kind in {"setup_started", "batch_started"}:
            self._set_action_state(False)
            self.repair_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
        elif event.kind == "setup_ready":
            self.ready = True
            self.status_var.set(event.message)
        elif event.kind == "setup_failed":
            self.ready = False
            self.status_var.set(f"환경 준비 실패: {event.message}")
            messagebox.showerror(APP_DISPLAY_NAME, f"인식 환경 준비 실패\n\n{event.message}")
        elif event.kind == "setup_cancelled":
            self.status_var.set(event.message)
        elif event.kind in {"setup_progress", "job_progress"}:
            self.status_var.set(f"{event.stage}: {event.message}")
            self._set_progress(event.fraction)
        elif event.kind == "job_started" and event.video:
            self._update_video_event(event.video, event)
        elif event.kind == "job_finished" and event.video:
            self._update_video_event(event.video, event)
            if event.status is JobStatus.FAILED:
                self.status_var.set(f"실패: {event.video.name} — {event.message}")
        elif event.kind == "cancelling":
            self.status_var.set(event.message)
        elif event.kind == "batch_finished":
            self.status_var.set(event.message)
            self._set_progress(1.0)
        elif event.kind == "idle":
            self.ready = self.runtime.ready()
            self._set_action_state(self.ready)
            self.repair_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)

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
        self._refresh_row(iid)

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
            entry.message,
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
