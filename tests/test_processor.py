from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from subtitle_text_generator.app_paths import AppPaths
from subtitle_text_generator.engine import EngineCancelled, EngineExecutionError
from subtitle_text_generator.models import JobMode, JobRequest, JobStatus
from subtitle_text_generator.processor import JobProcessor, SourceChangedError, job_identity


class FakeRuntime:
    def ready(self) -> bool:
        return True


class FakeEngine:
    def __init__(self, turbo: dict[str, object], refined: dict[str, object] | None = None) -> None:
        self.turbo = turbo
        self.refined = refined or turbo
        self.calls: list[object] = []
        self.fail_refine = False
        self.fail_full = False
        self.oom_batches: set[int] = set()
        self.modify_source = False
        self.mean_volume: float | None = -20.0
        self.measured: list[Path] = []
        self.normalized: list[tuple[Path, Path, str]] = []
        self.cancel_normalize = False
        self.fail_normalize = False

    def measure_mean_volume(self, source: Path, *, cancelled) -> float | None:
        self.measured.append(source)
        return self.mean_volume

    def normalize_audio(self, source: Path, target: Path, audio_filter: str, *, cancelled) -> None:
        if self.cancel_normalize:
            target.write_bytes(b"partial")
            raise EngineCancelled("cancelled")
        if self.fail_normalize:
            raise EngineExecutionError("음량 보정 실패", output="Invalid data")
        self.normalized.append((source, target, audio_filter))
        target.write_bytes(b"wav")

    def run(self, spec, *, cancelled, progress, line_callback=None) -> Path:
        self.calls.append(spec)
        if cancelled():
            raise RuntimeError("cancelled")
        if spec.batch_size in self.oom_batches:
            raise EngineExecutionError("OOM", output="CUDA out of memory")
        if spec.model_key == "large-v3" and spec.clip_timestamps and self.fail_refine:
            raise EngineExecutionError("refine failed", output="decoder error")
        if spec.model_key == "large-v3" and not spec.clip_timestamps and self.fail_full:
            raise EngineExecutionError("full failed", output="decoder error")
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
    auto_full_large: bool = True,
) -> JobRequest:
    return JobRequest(
        video=video,
        root=video.parent,
        mode=mode,
        glossary=("Whisper",),
        audio_txt_timestamps=audio_txt_timestamps,
        auto_full_large=auto_full_large,
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
        _request(video, auto_full_large=False), cancelled=lambda: False, progress=lambda *_: None
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
        _request(video, auto_full_large=False), cancelled=lambda: False, progress=lambda *_: None
    )
    assert result.status in {JobStatus.COMPLETE, JobStatus.REVIEW}
    assert "정확한" in video.with_suffix(".srt").read_text(encoding="utf-8-sig")
    assert len(engine.calls) == 2
    assert engine.calls[1].clip_timestamps
    assert "전체 large-v3를 권장" in result.message
    assert engine.measured == []


def _uncertain_and_refined(clean: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    uncertain = deepcopy(clean)
    uncertain["segments"][0]["avg_logprob"] = -1.2  # type: ignore[index]
    refined = deepcopy(clean)
    refined["segments"][0]["text"] = " 정확한 새 문장입니다."  # type: ignore[index]
    refined["segments"][0]["words"] = [  # type: ignore[index]
        {"start": 0.2, "end": 0.8, "word": " 정확한", "probability": 0.99},
        {"start": 0.9, "end": 1.3, "word": " 새", "probability": 0.99},
        {"start": 1.4, "end": 2.2, "word": " 문장입니다.", "probability": 0.99},
    ]
    return uncertain, refined


def test_high_uncertainty_rerecognizes_whole_file_with_large(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    uncertain, refined = _uncertain_and_refined(clean_transcript_payload)
    video = tmp_path / "auto.mp4"
    video.write_bytes(b"v")
    engine = FakeEngine(uncertain, refined)
    result = _processor(tmp_path, engine).process(
        _request(video), cancelled=lambda: False, progress=lambda *_: None
    )
    assert result.status is JobStatus.COMPLETE
    assert "전체 large-v3로 다시 인식했습니다" in result.message
    assert "음량 보정" not in result.message
    assert [(call.model_key, call.clip_timestamps) for call in engine.calls] == [
        ("large-v3-turbo", ()),
        ("large-v3", ()),
    ]
    assert engine.calls[1].video == video.resolve()
    assert engine.normalized == []
    assert "정확한" in video.with_suffix(".srt").read_text(encoding="utf-8-sig")


def test_quiet_audio_is_normalized_before_full_large_and_cleaned_up(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    uncertain, refined = _uncertain_and_refined(clean_transcript_payload)
    audio = tmp_path / "녹음.m4a"
    audio.write_bytes(b"a")
    engine = FakeEngine(uncertain, refined)
    engine.mean_volume = -35.9
    processor = _processor(tmp_path, engine)
    result = processor.process(_request(audio), cancelled=lambda: False, progress=lambda *_: None)
    job_dir = processor.paths.jobs / job_identity(
        audio.resolve(), audio.stat().st_size, audio.stat().st_mtime_ns
    )
    assert result.status is JobStatus.COMPLETE
    assert "음량 보정(평균 -35.9dB) 후 전체 large-v3로 다시 인식했습니다" in result.message
    assert len(engine.normalized) == 1
    source, target, audio_filter = engine.normalized[0]
    assert source == audio.resolve()
    assert "dynaudnorm" in audio_filter
    assert engine.calls[1].video == target
    assert not target.exists()
    assert (job_dir / "full-large-normalized.json").is_file()
    assert json.loads((job_dir / "loudness.json").read_text(encoding="utf-8")) == {
        "mean_volume_db": -35.9
    }

    # 같은 파일을 다시 처리하면 측정값과 large-v3 결과를 모두 캐시에서 씁니다.
    engine.calls.clear()
    engine.measured.clear()
    processor.process(
        _request(audio, JobMode.REPROCESS), cancelled=lambda: False, progress=lambda *_: None
    )
    assert engine.measured == []
    assert engine.calls == []


def test_rerecognized_result_that_stays_uncertain_needs_review(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    uncertain, _ = _uncertain_and_refined(clean_transcript_payload)
    video = tmp_path / "still.mp4"
    video.write_bytes(b"v")
    engine = FakeEngine(uncertain, uncertain)
    result = _processor(tmp_path, engine).process(
        _request(video), cancelled=lambda: False, progress=lambda *_: None
    )
    assert result.status is JobStatus.REVIEW
    assert "다시 인식한 결과도 불확실 비율" in result.message


def test_auto_full_large_failure_falls_back_to_refinement_with_warning(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    uncertain, refined = _uncertain_and_refined(clean_transcript_payload)
    video = tmp_path / "fallback.mp4"
    video.write_bytes(b"v")
    engine = FakeEngine(uncertain, refined)
    engine.fail_full = True
    result = _processor(tmp_path, engine).process(
        _request(video), cancelled=lambda: False, progress=lambda *_: None
    )
    assert result.status is JobStatus.WARNING
    assert result.message.startswith("자동 전체 large-v3 실패")
    assert "large-v3로 교체했습니다" in result.message
    assert engine.calls[-1].clip_timestamps
    assert "정확한" in video.with_suffix(".srt").read_text(encoding="utf-8-sig")


def test_low_uncertainty_never_measures_or_rerecognizes(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    video = tmp_path / "clean.mp4"
    video.write_bytes(b"v")
    engine = FakeEngine(clean_transcript_payload)
    engine.mean_volume = -40.0
    result = _processor(tmp_path, engine).process(
        _request(video), cancelled=lambda: False, progress=lambda *_: None
    )
    assert result.status is JobStatus.COMPLETE
    assert engine.measured == []
    assert [call.model_key for call in engine.calls] == ["large-v3-turbo"]


def test_full_large_mode_normalizes_quiet_audio_only(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    quiet = tmp_path / "quiet.mp4"
    quiet.write_bytes(b"q")
    loud = tmp_path / "loud.mp4"
    loud.write_bytes(b"l")
    engine = FakeEngine(clean_transcript_payload)
    processor = _processor(tmp_path, engine)
    engine.mean_volume = -30.0
    quiet_result = processor.process(
        _request(quiet, JobMode.FULL_LARGE), cancelled=lambda: False, progress=lambda *_: None
    )
    engine.mean_volume = -18.0
    loud_result = processor.process(
        _request(loud, JobMode.FULL_LARGE), cancelled=lambda: False, progress=lambda *_: None
    )
    assert quiet_result.message == "음량 보정(평균 -30.0dB) 후 전체 large-v3 인식 완료"
    assert loud_result.message == "전체 large-v3 인식 완료"
    assert [source for source, _, _ in engine.normalized] == [quiet.resolve()]
    assert engine.calls[-1].video == loud.resolve()


def test_unmeasurable_volume_runs_full_large_without_normalizing(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    video = tmp_path / "unknown.mp4"
    video.write_bytes(b"v")
    engine = FakeEngine(clean_transcript_payload)
    engine.mean_volume = None
    result = _processor(tmp_path, engine).process(
        _request(video, JobMode.FULL_LARGE), cancelled=lambda: False, progress=lambda *_: None
    )
    assert result.message == "전체 large-v3 인식 완료"
    assert engine.normalized == []


def test_failed_normalization_falls_back_to_original_audio(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    video = tmp_path / "broken-audio.mp4"
    video.write_bytes(b"v")
    engine = FakeEngine(clean_transcript_payload)
    engine.mean_volume = -40.0
    engine.fail_normalize = True
    processor = _processor(tmp_path, engine)
    result = processor.process(
        _request(video, JobMode.FULL_LARGE), cancelled=lambda: False, progress=lambda *_: None
    )
    stat = video.stat()
    job_dir = processor.paths.jobs / job_identity(video.resolve(), stat.st_size, stat.st_mtime_ns)
    assert result.status is JobStatus.COMPLETE
    assert result.message == "전체 large-v3 인식 완료"
    assert engine.calls[-1].video == video.resolve()
    assert (job_dir / "full-large.json").is_file()
    assert not (job_dir / "full-large-normalized.json").exists()


def test_cancel_reported_even_if_cleanup_is_blocked(
    tmp_path: Path, clean_transcript_payload: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    uncertain, refined = _uncertain_and_refined(clean_transcript_payload)
    audio = tmp_path / "locked.m4a"
    audio.write_bytes(b"a")
    engine = FakeEngine(uncertain, refined)
    engine.mean_volume = -35.0
    stop = {"now": False}

    def run_then_cancel(spec, *, cancelled, progress, line_callback=None):
        if spec.model_key == "large-v3":
            stop["now"] = True
            raise EngineCancelled("cancelled")
        return FakeEngine.run(engine, spec, cancelled=cancelled, progress=progress)

    engine.run = run_then_cancel  # type: ignore[method-assign]
    processor = _processor(tmp_path, engine)
    original_remove = JobProcessor._safe_remove_run_dir

    def locked_remove(path: Path, job_dir: Path) -> None:
        if path.name == "normalized.wav" and path.exists():
            raise PermissionError("file in use")
        original_remove(path, job_dir)

    monkeypatch.setattr(JobProcessor, "_safe_remove_run_dir", staticmethod(locked_remove))
    with pytest.raises(EngineCancelled):
        processor.process(
            _request(audio), cancelled=lambda: stop["now"], progress=lambda *_: None
        )
    assert not audio.with_suffix(".txt").exists()


def test_changed_glossary_discards_old_caches_before_running(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    video = tmp_path / "glossary.mp4"
    video.write_bytes(b"v")
    engine = FakeEngine(clean_transcript_payload)
    processor = _processor(tmp_path, engine)
    processor.process(
        _request(video, JobMode.FULL_LARGE), cancelled=lambda: False, progress=lambda *_: None
    )
    stat = video.stat()
    job_dir = processor.paths.jobs / job_identity(video.resolve(), stat.st_size, stat.st_mtime_ns)
    assert (job_dir / "full-large.json").is_file()
    changed = JobRequest(
        video=video, root=video.parent, mode=JobMode.FULL_LARGE, glossary=("새 용어",)
    )
    with pytest.raises(RuntimeError):
        processor.process(changed, cancelled=lambda: True, progress=lambda *_: None)
    assert not (job_dir / "full-large.json").exists()


def test_cancel_during_normalization_removes_partial_audio(
    tmp_path: Path, clean_transcript_payload: dict[str, object]
) -> None:
    uncertain, refined = _uncertain_and_refined(clean_transcript_payload)
    audio = tmp_path / "cancel.m4a"
    audio.write_bytes(b"a")
    engine = FakeEngine(uncertain, refined)
    engine.mean_volume = -35.0
    engine.cancel_normalize = True
    processor = _processor(tmp_path, engine)
    with pytest.raises(EngineCancelled):
        processor.process(_request(audio), cancelled=lambda: False, progress=lambda *_: None)
    assert not list(processor.paths.jobs.rglob("normalized.wav"))
    assert not audio.with_suffix(".txt").exists()


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
