from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from .app_paths import AppPaths
from .constants import APP_VERSION, LARGE_MODEL, TURBO_MODEL
from .engine import EngineCancelled, EngineExecutionError, EngineRun, EngineRunner, clip_argument
from .models import JobMode, JobRequest, JobResult, JobStatus, RefineWindow, Transcript
from .quality import (
    REVIEW_RATIO_THRESHOLD,
    build_refine_windows,
    merge_refined_transcript,
    uncertain_ratio,
)
from .runtime import RuntimeManager
from .scanner import matching_subtitles
from .subtitles import commit_srt, cues_from_transcript
from .transcript import TranscriptError, load_transcript, transcript_is_valid

StageCallback = Callable[[str, str, float | None], None]
CancelCheck = Callable[[], bool]


class SourceChangedError(RuntimeError):
    pass


class JobProcessor:
    PIPELINE_VERSION = 1

    def __init__(self, paths: AppPaths, runtime: RuntimeManager, engine: EngineRunner) -> None:
        self.paths = paths
        self.runtime = runtime
        self.engine = engine

    def process(
        self,
        request: JobRequest,
        *,
        cancelled: CancelCheck,
        progress: StageCallback,
    ) -> JobResult:
        started = time.monotonic()
        video = request.video.resolve()
        source_stat = video.stat()
        if request.mode is JobMode.ADAPTIVE and matching_subtitles(video):
            return JobResult(
                video=video,
                status=JobStatus.SKIPPED,
                message="기존 자막이 있어 건너뛰었습니다.",
                elapsed_seconds=time.monotonic() - started,
            )
        if not self.runtime.ready():
            raise RuntimeError("인식 엔진 설치가 완료되지 않았습니다.")

        job_id = job_identity(video, source_stat.st_size, source_stat.st_mtime_ns)
        job_dir = self.paths.jobs / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        metadata = self._metadata(request, source_stat.st_size, source_stat.st_mtime_ns)
        cache_matches = self._metadata_matches(job_dir / "meta.json", metadata)
        self._write_json_atomic(job_dir / "meta.json", metadata)
        log_path = self.paths.logs / f"{job_id}.log"

        warning = ""
        ratio = 0.0
        if request.mode is JobMode.FULL_LARGE:
            progress("전체 large-v3", "전체 영상을 large-v3로 인식합니다.", 0.0)
            final_transcript = self._run_cached_full_large(
                video,
                request,
                job_dir,
                cache_matches,
                log_path,
                cancelled,
                progress,
            )
            status = JobStatus.COMPLETE
            message = "전체 large-v3 자막 생성 완료"
        else:
            progress("Turbo 초안", "Turbo 초안을 준비합니다.", 0.0)
            turbo = self._run_cached_turbo(
                video,
                request,
                job_dir,
                cache_matches,
                log_path,
                cancelled,
                progress,
            )
            windows = build_refine_windows(turbo)
            ratio = uncertain_ratio(turbo, windows)
            final_transcript = turbo
            status = JobStatus.COMPLETE
            message = "Turbo 결과에 불확실 구간이 없습니다."
            if windows:
                progress(
                    "불확실 구간 분석",
                    f"large-v3 재처리 구간 {len(windows)}개 ({ratio * 100:.1f}%)",
                    None,
                )
                try:
                    refined = self._run_refinement(
                        video,
                        request,
                        job_dir,
                        log_path,
                        windows,
                        cancelled,
                        progress,
                    )
                    merged, replaced, unresolved = merge_refined_transcript(turbo, refined, windows)
                    if not transcript_is_valid(merged) or replaced == 0:
                        warning = "large-v3 결과가 유효하지 않아 Turbo 초안을 사용했습니다."
                    else:
                        final_transcript = merged
                        message = f"불확실 구간 {replaced}/{len(windows)}개를 large-v3로 교체했습니다."
                        if unresolved or replaced < len(windows):
                            status = JobStatus.REVIEW
                            message += " 일부 구간은 검토가 필요합니다."
                except EngineCancelled:
                    raise
                except (EngineExecutionError, TranscriptError, OSError, ValueError) as exc:
                    warning = f"large-v3 구간 재처리 실패. Turbo 초안을 저장합니다: {exc}"
            if warning:
                status = JobStatus.WARNING
                message = warning
            elif ratio >= REVIEW_RATIO_THRESHOLD:
                status = JobStatus.REVIEW
                message += " 불확실 비율이 30% 이상이므로 전체 large-v3를 권장합니다."

        if cancelled():
            raise EngineCancelled("사용자가 작업을 중지했습니다.")
        self._verify_source_unchanged(video, source_stat.st_size, source_stat.st_mtime_ns)
        progress("SRT 생성", "단어 타임스탬프를 자막 큐로 정리합니다.", None)
        cues = cues_from_transcript(final_transcript)
        if not cues:
            raise RuntimeError("음성이 인식되지 않아 SRT를 만들 수 없습니다.")
        srt_path = commit_srt(
            video,
            cues,
            allow_replace=request.mode in {JobMode.REPROCESS, JobMode.FULL_LARGE},
        )
        self._verify_source_unchanged(video, source_stat.st_size, source_stat.st_mtime_ns)
        elapsed = time.monotonic() - started
        progress("완료", message, 1.0)
        return JobResult(
            video=video,
            status=status,
            srt_path=srt_path,
            message=message,
            uncertain_ratio=ratio,
            elapsed_seconds=elapsed,
        )

    def _run_cached_turbo(
        self,
        video: Path,
        request: JobRequest,
        job_dir: Path,
        cache_matches: bool,
        log_path: Path,
        cancelled: CancelCheck,
        progress: StageCallback,
    ) -> Transcript:
        cache = job_dir / "turbo.json"
        if cache_matches and cache.is_file():
            try:
                transcript = load_transcript(cache, source="turbo-cache")
                if transcript_is_valid(transcript):
                    progress("Turbo 초안", "중단 전 Turbo JSON을 재사용합니다.", 1.0)
                    return transcript
            except TranscriptError:
                pass
        last_oom: EngineExecutionError | None = None
        for batch_size in (4, 2, 1):
            run_dir = job_dir / f"turbo-run-b{batch_size}"
            self._safe_remove_run_dir(run_dir, job_dir)
            progress("Turbo 초안", f"Turbo 인식 시작 (batch {batch_size})", 0.0)
            try:
                result = self.engine.run(
                    EngineRun(
                        video=video,
                        model_key=TURBO_MODEL.key,
                        output_dir=run_dir,
                        batch_size=batch_size,
                        glossary=request.glossary,
                    ),
                    cancelled=cancelled,
                    progress=lambda fraction, line: progress("Turbo 초안", line, fraction),
                    line_callback=lambda line: self._append_log(log_path, "turbo", line),
                )
                transcript = load_transcript(result, source="turbo")
                if not transcript_is_valid(transcript):
                    raise TranscriptError("Turbo JSON에 유효한 발화가 없습니다.")
                self._copy_atomic(result, cache)
                self._safe_remove_run_dir(run_dir, job_dir)
                return transcript
            except EngineExecutionError as exc:
                if exc.is_out_of_memory:
                    last_oom = exc
                    progress(
                        "Turbo 초안",
                        f"GPU 메모리 부족: batch {batch_size} 실패",
                        None,
                    )
                    continue
                raise
        raise last_oom or RuntimeError("Turbo 인식에 실패했습니다.")

    def _run_refinement(
        self,
        video: Path,
        request: JobRequest,
        job_dir: Path,
        log_path: Path,
        windows: list[RefineWindow],
        cancelled: CancelCheck,
        progress: StageCallback,
    ) -> Transcript:
        run_dir = job_dir / "refine-run"
        self._safe_remove_run_dir(run_dir, job_dir)
        ranges = [(window.decode_start, window.decode_end) for window in windows]
        result = self.engine.run(
            EngineRun(
                video=video,
                model_key=LARGE_MODEL.key,
                output_dir=run_dir,
                clip_timestamps=clip_argument(ranges),
                glossary=request.glossary,
            ),
            cancelled=cancelled,
            progress=lambda fraction, line: progress("large-v3 구간 재처리", line, fraction),
            line_callback=lambda line: self._append_log(log_path, "refine", line),
        )
        transcript = load_transcript(result, source="large-v3-refine")
        if not transcript_is_valid(transcript):
            raise TranscriptError("large-v3 구간 JSON에 유효한 발화가 없습니다.")
        return transcript

    def _run_cached_full_large(
        self,
        video: Path,
        request: JobRequest,
        job_dir: Path,
        cache_matches: bool,
        log_path: Path,
        cancelled: CancelCheck,
        progress: StageCallback,
    ) -> Transcript:
        cache = job_dir / "full-large.json"
        if cache_matches and cache.is_file():
            try:
                transcript = load_transcript(cache, source="large-v3-cache")
                if transcript_is_valid(transcript):
                    progress("전체 large-v3", "중단 전 large-v3 JSON을 재사용합니다.", 1.0)
                    return transcript
            except TranscriptError:
                pass
        run_dir = job_dir / "full-large-run"
        self._safe_remove_run_dir(run_dir, job_dir)
        result = self.engine.run(
            EngineRun(
                video=video,
                model_key=LARGE_MODEL.key,
                output_dir=run_dir,
                glossary=request.glossary,
            ),
            cancelled=cancelled,
            progress=lambda fraction, line: progress("전체 large-v3", line, fraction),
            line_callback=lambda line: self._append_log(log_path, "full-large", line),
        )
        transcript = load_transcript(result, source="large-v3-full")
        if not transcript_is_valid(transcript):
            raise TranscriptError("large-v3 JSON에 유효한 발화가 없습니다.")
        self._copy_atomic(result, cache)
        self._safe_remove_run_dir(run_dir, job_dir)
        return transcript

    def _metadata(self, request: JobRequest, size: int, mtime_ns: int) -> dict[str, object]:
        glossary_hash = hashlib.sha256("\n".join(request.glossary).encode("utf-8")).hexdigest()
        return {
            "pipeline_version": self.PIPELINE_VERSION,
            "app_version": APP_VERSION,
            "source": os.path.normcase(str(request.video.resolve())),
            "size": size,
            "mtime_ns": mtime_ns,
            "glossary_sha256": glossary_hash,
        }

    @staticmethod
    def _metadata_matches(path: Path, expected: dict[str, object]) -> bool:
        try:
            return json.loads(path.read_text(encoding="utf-8")) == expected
        except (OSError, ValueError, TypeError):
            return False

    @staticmethod
    def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _copy_atomic(source: Path, target: Path) -> None:
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)

    @staticmethod
    def _append_log(log_path: Path, stage: str, line: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a", encoding="utf-8", errors="replace") as stream:
            stream.write(f"[{timestamp}] [{stage}] {line}\n")

    @staticmethod
    def _verify_source_unchanged(video: Path, size: int, mtime_ns: int) -> None:
        stat = video.stat()
        if stat.st_size != size or stat.st_mtime_ns != mtime_ns:
            raise SourceChangedError("처리 중 원본 영상이 변경되어 자막 저장을 중단했습니다.")

    @staticmethod
    def _safe_remove_run_dir(path: Path, job_dir: Path) -> None:
        candidate = path.resolve()
        root = job_dir.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"안전하지 않은 캐시 경로: {candidate}") from exc
        if candidate == root:
            raise RuntimeError("작업 캐시 루트는 삭제할 수 없습니다.")
        if candidate.is_dir():
            shutil.rmtree(candidate)
        elif candidate.exists():
            candidate.unlink()


def job_identity(video: Path, size: int, mtime_ns: int) -> str:
    canonical = os.path.normcase(str(video.resolve()))
    payload = f"{canonical}\0{size}\0{mtime_ns}".encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()
