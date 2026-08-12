from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .runtime import RuntimeManager

LineCallback = Callable[[str], None]
ProgressCallback = Callable[[float | None, str], None]


class EngineExecutionError(RuntimeError):
    def __init__(self, message: str, *, output: str = "", returncode: int | None = None) -> None:
        super().__init__(message)
        self.output = output
        self.returncode = returncode

    @property
    def is_out_of_memory(self) -> bool:
        folded = self.output.casefold()
        patterns = (
            "out of memory",
            "cublas_status_alloc_failed",
            "cuda_error_out_of_memory",
            "failed to allocate",
        )
        return any(pattern in folded for pattern in patterns)


class EngineCancelled(EngineExecutionError):
    pass


@dataclass(frozen=True, slots=True)
class EngineRun:
    video: Path
    model_key: str
    output_dir: Path
    batch_size: int | None = None
    clip_timestamps: tuple[float, ...] = ()
    glossary: tuple[str, ...] = ()


class EngineRunner:
    _PROGRESS_RE = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%")

    def __init__(self, runtime: RuntimeManager) -> None:
        self.runtime = runtime
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def build_command(self, run: EngineRun) -> list[str]:
        command = [
            str(self.runtime.engine_executable()),
            str(run.video),
            "--model",
            run.model_key,
            "--model_dir",
            str(self.runtime.model_path(run.model_key).parent),
            "--device",
            "cuda",
            "--compute_type",
            "float16",
            "--language",
            "ko",
            "--task",
            "transcribe",
            "--beam_size",
            "5",
            "--word_timestamps",
            "True",
            "--vad_filter",
            "True",
            "--vad_method",
            "silero_v5_fw",
            "--output_format",
            "json",
            "--output_dir",
            str(run.output_dir),
            "--print_progress",
            "--beep_off",
        ]
        if run.batch_size is not None:
            command.extend(("--batched", "--batch_size", str(run.batch_size)))
        if run.clip_timestamps:
            clips = ",".join(self._format_timestamp(value) for value in run.clip_timestamps)
            command.extend(("--clip_timestamps", clips))
        if run.glossary:
            command.extend(("--hotwords", ", ".join(run.glossary)))
        return command

    def run(
        self,
        spec: EngineRun,
        *,
        cancelled: Callable[[], bool],
        progress: ProgressCallback,
        line_callback: LineCallback | None = None,
    ) -> Path:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_command(spec)
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        self.runtime.paths.temp.mkdir(parents=True, exist_ok=True)
        environment["TEMP"] = str(self.runtime.paths.temp)
        environment["TMP"] = str(self.runtime.paths.temp)
        environment["HF_HOME"] = str(self.runtime.paths.root / "huggingface-cache")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(self.runtime.engine_executable().parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=flags,
                env=environment,
            )
        except OSError as exc:
            raise EngineExecutionError(f"인식 엔진을 시작하지 못했습니다: {exc}") from exc
        with self._lock:
            self._process = process
        lines: list[str] = []
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    output_queue.put(line.rstrip())
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        stream_finished = False
        try:
            while process.poll() is None or not stream_finished:
                if cancelled():
                    self.stop()
                    raise EngineCancelled("사용자가 작업을 중지했습니다.")
                try:
                    line = output_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    stream_finished = True
                    continue
                lines.append(line)
                if line_callback:
                    line_callback(line)
                match = self._PROGRESS_RE.search(line)
                if match:
                    progress(min(float(match.group(1)) / 100.0, 1.0), line)
            returncode = process.wait()
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None
        combined = "\n".join(lines)
        json_files = sorted(
            spec.output_dir.glob("*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True
        )
        if returncode != 0:
            lowered = combined.casefold()
            if _has_complete_output_after_nonzero_exit(lowered, json_files):
                warning = f"엔진이 결과 저장 후 비정상 종료 코드 {returncode}를 반환했지만 JSON을 회수했습니다."
                if line_callback:
                    line_callback(warning)
                progress(1.0, warning)
                return json_files[0]
            cuda_markers = ("cuda error", "cublas", "out of memory", "failed to allocate")
            prefix = "CUDA 인식 실패" if any(marker in lowered for marker in cuda_markers) else "인식 엔진 실패"
            tail = "\n".join(lines[-12:]).strip()[-1_200:]
            raise EngineExecutionError(
                f"{prefix} (종료 코드 {returncode}): {tail or '출력 없음'}",
                output=combined,
                returncode=returncode,
            )
        if not json_files:
            raise EngineExecutionError(
                "인식 엔진이 JSON 결과를 만들지 않았습니다.", output=combined, returncode=returncode
            )
        progress(1.0, "인식 완료")
        return json_files[0]

    def stop(self) -> None:
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW
            subprocess.run(
                ["taskkill", "/PID", str(int(process.pid)), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=flags,
            )
        else:
            process.kill()
        deadline = time.monotonic() + 3
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            process.kill()

    @staticmethod
    def _format_timestamp(value: float) -> str:
        formatted = f"{max(value, 0.0):.3f}".rstrip("0").rstrip(".")
        return formatted or "0"


def clip_argument(windows: Sequence[tuple[float, float]]) -> tuple[float, ...]:
    values: list[float] = []
    for start, end in windows:
        if end > start:
            values.extend((start, end))
    return tuple(values)


def _has_complete_output_after_nonzero_exit(output: str, json_files: Sequence[Path]) -> bool:
    if not json_files:
        return False
    try:
        payload = json.loads(json_files[0].read_text(encoding="utf-8-sig"))
        has_nonempty_json = (
            isinstance(payload, dict)
            and isinstance(payload.get("segments"), list)
            and bool(payload["segments"])
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    completion_markers = (
        "subtitles are written to",
        "operation finished in:",
        "100%",
    )
    return has_nonempty_json and all(marker in output for marker in completion_markers)
