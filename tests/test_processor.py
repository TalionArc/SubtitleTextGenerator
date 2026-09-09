from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from lecture_subtitle_batcher.app_paths import AppPaths
from lecture_subtitle_batcher.engine import EngineExecutionError
from lecture_subtitle_batcher.models import JobMode, JobRequest, JobStatus
from lecture_subtitle_batcher.processor import JobProcessor, SourceChangedError, job_identity


class FakeRuntime:
    def ready(self) -> bool:
        return True


class FakeEngine:
    def __init__(self, turbo: dict[str, object], refined: dict[str, object] | None = None) -> None:
        self.turbo = turbo
        self.refined = refined or turbo
        self.calls: list[object] = []
        self.fail_refine = False
        self.oom_batches: set[int] = set()
        self.modify_source = False

    def run(self, spec, *, cancelled, progress, line_callback=None) -> Path:
        self.calls.append(spec)
        if cancelled():
            raise RuntimeError("cancelled")
        if spec.batch_size in self.oom_batches:
            raise EngineExecutionError("OOM", output="CUDA out of memory")
        if spec.model_key == "large-v3" and spec.clip_timestamps and self.fail_refine:
            raise EngineExecutionError("refine failed", output="decoder error")
        payload = self.refined if spec.model_key == "large-v3" else self.turbo
        if self.modify_source:
            spec.video.write_bytes(spec.video.read_bytes() + b"changed")
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        result = spec.output_dir / "result.json"
        result.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        progress(1.0, "100%")
        if line_callback:
            line_callback("done")
        return result

    def stop(self) -> None:
        return None


def _request(
    video: Path,
    mode: JobMode = JobMode.ADAPTIVE,
    *,
    audio_txt_timestamps: bool = True,
) -> JobRequest:
    return JobRequest(
        video=video,
        root=video.parent,
        mode=mode,
        glossary=("Whisper",),
        audio_txt_timestamps=audio_txt_timestamps,
    )


def _processor(tmp_path: Path, engine: FakeEngine) -> JobProcessor:
    paths = AppPaths(tmp_path / "appdata")
    paths.ensure_base_dirs()
    return JobProcessor(paths, FakeRuntime(), engine)  # type: ignore[arg-type]


def test_adaptive_success_and_cache_resume(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    video = tmp_path / "강의.mp4"
    video.write_bytes(b"video")
    engine = FakeEngine(clean_transcript_payload)
    processor = _processor(tmp_path, engine)
    result = processor.process(_request(video), cancelled=lambda: False, progress=lambda *_: None)
    assert result.status is JobStatus.COMPLETE
    assert video.with_suffix(".srt").is_file()
    assert len(engine.calls) == 1

    source_stat = video.stat()
    meta_path = processor.paths.jobs / job_identity(
        video, source_stat.st_size, source_stat.st_mtime_ns
    ) / "meta.json"
    old_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    old_metadata["app_version"] = "1.2.0"
    meta_path.write_text(json.dumps(old_metadata), encoding="utf-8")

    result2 = processor.process(
        _request(video, JobMode.REPROCESS), cancelled=lambda: False, progress=lambda *_: None
    )
    assert result2.status is JobStatus.COMPLETE
    assert len(engine.calls) == 1
    assert video.with_suffix(".srt.bak").is_file()


def test_default_mode_skips_any_existing_sidecar(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"v")
    video.with_suffix(".ko.vtt").write_text("WEBVTT", encoding="utf-8")
    engine = FakeEngine(clean_transcript_payload)
    result = _processor(tmp_path, engine).process(
        _request(video), cancelled=lambda: False, progress=lambda *_: None
    )
    assert result.status is JobStatus.SKIPPED
    assert not engine.calls


def test_refine_failure_saves_turbo_with_warning(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    uncertain = deepcopy(clean_transcript_payload)
    uncertain["segments"][0]["avg_logprob"] = -1.2  # type: ignore[index]
    video = tmp_path / "uncertain.mp4"
    video.write_bytes(b"v")
    engine = FakeEngine(uncertain)
    engine.fail_refine = True
    result = _processor(tmp_path, engine).process(
        _request(video), cancelled=lambda: False, progress=lambda *_: None
    )
    assert result.status is JobStatus.WARNING
    assert video.with_suffix(".srt").is_file()
    assert "Turbo" in result.message


def test_oom_retries_batch_four_then_two(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    video = tmp_path / "oom.mp4"
    video.write_bytes(b"v")
    engine = FakeEngine(clean_transcript_payload)
    engine.oom_batches = {4}
    result = _processor(tmp_path, engine).process(
        _request(video), cancelled=lambda: False, progress=lambda *_: None
    )
    assert result.status is JobStatus.COMPLETE
    assert [call.batch_size for call in engine.calls] == [4, 2]


def test_successful_refinement_builds_hybrid_srt(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    uncertain = deepcopy(clean_transcript_payload)
    uncertain["segments"][0]["avg_logprob"] = -1.2  # type: ignore[index]
    refined = deepcopy(clean_transcript_payload)
    refined["segments"][0]["text"] = " 정확한 새 문장입니다."  # type: ignore[index]
    refined["segments"][0]["words"] = [  # type: ignore[index]
        {"start": 0.2, "end": 0.8, "word": " 정확한", "probability": 0.99},
        {"start": 0.9, "end": 1.3, "word": " 새", "probability": 0.99},
        {"start": 1.4, "end": 2.2, "word": " 문장입니다.", "probability": 0.99},
    ]
    video = tmp_path / "hybrid.mp4"
    video.write_bytes(b"v")
    engine = FakeEngine(uncertain, refined)
    result = _processor(tmp_path, engine).process(
        _request(video), cancelled=lambda: False, progress=lambda *_: None
    )
    assert result.status in {JobStatus.COMPLETE, JobStatus.REVIEW}
    assert "정확한" in video.with_suffix(".srt").read_text(encoding="utf-8-sig")
    assert len(engine.calls) == 2
    assert engine.calls[1].clip_timestamps


def test_source_change_aborts_before_srt_write(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    video = tmp_path / "changing.mp4"
    video.write_bytes(b"v")
    engine = FakeEngine(clean_transcript_payload)
    engine.modify_source = True
    with pytest.raises(SourceChangedError):
        _processor(tmp_path, engine).process(
            _request(video), cancelled=lambda: False, progress=lambda *_: None
        )
    assert not video.with_suffix(".srt").exists()


def test_turbo_failure_does_not_move_existing_srt(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    video = tmp_path / "failed.mp4"
    video.write_bytes(b"v")
    existing = video.with_suffix(".srt")
    existing.write_text("existing", encoding="utf-8")
    engine = FakeEngine(clean_transcript_payload)

    def fail(*_args, **_kwargs):
        raise EngineExecutionError("decoder failed", output="decoder failed")

    engine.run = fail  # type: ignore[method-assign]
    with pytest.raises(EngineExecutionError):
        _processor(tmp_path, engine).process(
            _request(video, JobMode.REPROCESS),
            cancelled=lambda: False,
            progress=lambda *_: None,
        )
    assert existing.read_text(encoding="utf-8") == "existing"
    assert not video.with_suffix(".srt.bak").exists()


def test_job_identity_uses_full_path_size_and_mtime(tmp_path: Path) -> None:
    left = tmp_path / "a" / "same.mp4"
    right = tmp_path / "b" / "same.mp4"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_bytes(b"x")
    right.write_bytes(b"x")
    left_stat = left.stat()
    right_stat = right.stat()
    assert job_identity(left, left_stat.st_size, left_stat.st_mtime_ns) != job_identity(
        right, right_stat.st_size, right_stat.st_mtime_ns
    )


def test_audio_adaptive_writes_timestamped_txt_not_srt(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    audio = tmp_path / "회의 녹음.m4a"
    audio.write_bytes(b"audio")
    result = _processor(tmp_path, FakeEngine(clean_transcript_payload)).process(
        _request(audio), cancelled=lambda: False, progress=lambda *_: None
    )

    output = audio.with_suffix(".txt")
    text = output.read_text(encoding="utf-8-sig")
    assert result.status is JobStatus.COMPLETE
    assert result.output_path == output
    assert output.is_file()
    assert not audio.with_suffix(".srt").exists()
    assert "안녕하세요" in text
    assert text.startswith("[00:00:00.200 - 00:00:02.200] ")


def test_audio_txt_can_disable_timestamps(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    audio = tmp_path / "타임스탬프 제외.m4a"
    audio.write_bytes(b"audio")
    result = _processor(tmp_path, FakeEngine(clean_transcript_payload)).process(
        _request(audio, audio_txt_timestamps=False),
        cancelled=lambda: False,
        progress=lambda *_: None,
    )

    text = audio.with_suffix(".txt").read_text(encoding="utf-8-sig")
    assert result.status is JobStatus.COMPLETE
    assert text.startswith("안녕하세요 강의를 시작합니다.")
    assert "[00:00:" not in text


def test_audio_existing_txt_skips_and_reprocess_creates_one_backup(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    audio = tmp_path / "recording.m4a"
    audio.write_bytes(b"audio")
    output = audio.with_suffix(".txt")
    output.write_text("사용자의 기존 텍스트", encoding="utf-8")
    engine = FakeEngine(clean_transcript_payload)
    processor = _processor(tmp_path, engine)

    skipped = processor.process(
        _request(audio), cancelled=lambda: False, progress=lambda *_: None
    )
    assert skipped.status is JobStatus.SKIPPED
    assert not engine.calls

    completed = processor.process(
        _request(audio, JobMode.REPROCESS),
        cancelled=lambda: False,
        progress=lambda *_: None,
    )
    assert completed.status is JobStatus.COMPLETE
    assert "사용자의 기존 텍스트" in audio.with_suffix(".txt.bak").read_text(
        encoding="utf-8"
    )
