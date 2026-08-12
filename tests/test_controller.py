from __future__ import annotations

import time
from pathlib import Path

from lecture_subtitle_batcher.controller import BatchController
from lecture_subtitle_batcher.models import JobMode, JobRequest, JobResult, JobStatus


class ReadyRuntime:
    def ready(self) -> bool:
        return True


class NoopEngine:
    def stop(self) -> None:
        return None


class RecordingProcessor:
    def __init__(self) -> None:
        self.order: list[str] = []

    def process(self, request, *, cancelled, progress):
        self.order.append(request.video.name)
        progress("stage", "message", 0.5)
        if request.video.name.startswith("fail"):
            raise RuntimeError("expected failure")
        return JobResult(request.video, JobStatus.COMPLETE, message="ok", elapsed_seconds=0.01)


def test_controller_runs_sequentially_and_continues_after_failure(tmp_path: Path) -> None:
    events = []
    processor = RecordingProcessor()
    controller = BatchController(
        ReadyRuntime(),  # type: ignore[arg-type]
        NoopEngine(),  # type: ignore[arg-type]
        processor,  # type: ignore[arg-type]
        events.append,
    )
    jobs = [
        JobRequest(tmp_path / "fail.mp4", tmp_path, JobMode.ADAPTIVE),
        JobRequest(tmp_path / "success.mp4", tmp_path, JobMode.ADAPTIVE),
    ]
    assert controller.start_jobs(jobs)
    deadline = time.monotonic() + 3
    while controller.busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert processor.order == ["fail.mp4", "success.mp4"]
    finished = [event for event in events if event.kind == "job_finished"]
    assert [event.status for event in finished] == [JobStatus.FAILED, JobStatus.COMPLETE]
