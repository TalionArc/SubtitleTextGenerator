from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine import EngineCancelled, EngineRunner
from .models import JobRequest, JobResult, JobStatus
from .processor import JobProcessor
from .runtime import RuntimeManager, SetupCancelled


@dataclass(frozen=True, slots=True)
class ControllerEvent:
    kind: str
    video: Path | None = None
    status: JobStatus | None = None
    stage: str = ""
    message: str = ""
    fraction: float | None = None
    result: JobResult | None = None
    payload: Any = None


EventCallback = Callable[[ControllerEvent], None]


class BatchController:
    def __init__(
        self,
        runtime: RuntimeManager,
        engine: EngineRunner,
        processor: JobProcessor,
        callback: EventCallback,
    ) -> None:
        self.runtime = runtime
        self.engine = engine
        self.processor = processor
        self.callback = callback
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start_setup(self, *, force_repair: bool = False) -> bool:
        return self._start_thread(self._setup_worker, force_repair)

    def start_jobs(self, jobs: Sequence[JobRequest]) -> bool:
        if not jobs:
            return False
        return self._start_thread(self._job_worker, list(jobs))

    def cancel(self) -> None:
        self._cancel_event.set()
        self.engine.stop()
        self.callback(ControllerEvent(kind="cancelling", message="작업 중지 요청을 보냈습니다."))

    def _start_thread(self, target: Callable[..., None], *args: object) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._cancel_event.clear()
            self._thread = threading.Thread(target=target, args=args, daemon=True)
            self._thread.start()
        return True

    def _setup_worker(self, force_repair: bool) -> None:
        self.callback(ControllerEvent(kind="setup_started", message="첫 실행 환경을 확인합니다."))
        try:
            self.runtime.ensure_ready(
                lambda stage, message, fraction: self.callback(
                    ControllerEvent(
                        kind="setup_progress",
                        stage=stage,
                        message=message,
                        fraction=fraction,
                    )
                ),
                self._cancel_event.is_set,
                force_repair=force_repair,
            )
        except SetupCancelled:
            self.callback(ControllerEvent(kind="setup_cancelled", message="설치가 취소되었습니다."))
        except BaseException as exc:
            self.callback(ControllerEvent(kind="setup_failed", message=str(exc), payload=exc))
        else:
            self.callback(ControllerEvent(kind="setup_ready", message="인식 환경 준비 완료"))
        finally:
            self.callback(ControllerEvent(kind="idle"))

    def _job_worker(self, jobs: list[JobRequest]) -> None:
        self.callback(ControllerEvent(kind="batch_started", message=f"{len(jobs)}개 작업 시작"))
        if not self.runtime.ready():
            try:
                self.runtime.ensure_ready(
                    lambda stage, message, fraction: self.callback(
                        ControllerEvent(
                            kind="setup_progress",
                            stage=stage,
                            message=message,
                            fraction=fraction,
                        )
                    ),
                    self._cancel_event.is_set,
                )
            except BaseException as exc:
                self.callback(ControllerEvent(kind="setup_failed", message=str(exc), payload=exc))
                self.callback(ControllerEvent(kind="idle"))
                return
        for index, request in enumerate(jobs):
            if self._cancel_event.is_set():
                self._cancel_remaining(jobs[index:])
                break
            self.callback(
                ControllerEvent(
                    kind="job_started",
                    video=request.video,
                    status=JobStatus.RUNNING,
                    message=f"{index + 1}/{len(jobs)}",
                )
            )
            try:
                result = self.processor.process(
                    request,
                    cancelled=self._cancel_event.is_set,
                    progress=lambda stage, message, fraction, video=request.video: self.callback(
                        ControllerEvent(
                            kind="job_progress",
                            video=video,
                            status=JobStatus.RUNNING,
                            stage=stage,
                            message=message,
                            fraction=fraction,
                        )
                    ),
                )
            except (EngineCancelled, SetupCancelled):
                cancelled_result = JobResult(
                    video=request.video,
                    status=JobStatus.CANCELLED,
                    message="작업이 중지되었습니다.",
                )
                self.callback(
                    ControllerEvent(
                        kind="job_finished",
                        video=request.video,
                        status=JobStatus.CANCELLED,
                        result=cancelled_result,
                        message=cancelled_result.message,
                    )
                )
                self._cancel_remaining(jobs[index + 1 :])
                break
            except BaseException as exc:
                failed = JobResult(
                    video=request.video,
                    status=JobStatus.FAILED,
                    message=str(exc),
                )
                self.callback(
                    ControllerEvent(
                        kind="job_finished",
                        video=request.video,
                        status=JobStatus.FAILED,
                        result=failed,
                        message=failed.message,
                        payload=exc,
                    )
                )
                continue
            self.callback(
                ControllerEvent(
                    kind="job_finished",
                    video=request.video,
                    status=result.status,
                    result=result,
                    message=result.message,
                )
            )
        self.callback(ControllerEvent(kind="batch_finished", message="대기열 처리가 끝났습니다."))
        self.callback(ControllerEvent(kind="idle"))

    def _cancel_remaining(self, jobs: Sequence[JobRequest]) -> None:
        for request in jobs:
            result = JobResult(
                video=request.video,
                status=JobStatus.CANCELLED,
                message="대기열 중지로 취소되었습니다.",
            )
            self.callback(
                ControllerEvent(
                    kind="job_finished",
                    video=request.video,
                    status=JobStatus.CANCELLED,
                    result=result,
                    message=result.message,
                )
            )
