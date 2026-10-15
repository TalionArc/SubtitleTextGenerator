from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from .app_paths import AppPaths
from .constants import APP_VERSION, LARGE_MODEL, TURBO_MODEL
from .engine import EngineCancelled, EngineExecutionError, EngineRun, EngineRunner, clip_argument
from .hallucination import filter_hallucinations
from .models import JobMode, JobRequest, JobResult, JobStatus, MediaKind, RefineWindow, Transcript
from .quality import (
    LOUDNESS_FILTER,
    QUIET_MEAN_VOLUME_DB,
    REVIEW_RATIO_THRESHOLD,
    build_refine_windows,
    merge_refined_transcript,
    uncertain_ratio,
)
from .runtime import RuntimeManager, RuntimeSetupError
from .scanner import matching_outputs, media_kind
from .subtitles import commit_srt, commit_txt, cues_from_transcript, format_txt_timestamp
from .transcript import TranscriptError, load_transcript, transcript_is_valid

StageCallback = Callable[[str, str, float | None], None]
CancelCheck = Callable[[], bool]


_CACHE_FILES = ("turbo.json", "full-large.json", "full-large-normalized.json", "loudness.json")


class SourceChangedError(RuntimeError):
    pass


class JobProcessor:
    PIPELINE_VERSION = 2

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
        source = request.video.resolve()
        source_kind = media_kind(source)
        source_stat = source.stat()
        if request.mode is JobMode.ADAPTIVE and matching_outputs(source):
            output_label = "TXT" if source_kind is MediaKind.AUDIO else "자막"
            return JobResult(
                video=source,
                status=JobStatus.SKIPPED,
                message=f"기존 {output_label} 결과가 있어 건너뛰었습니다.",
                elapsed_seconds=time.monotonic() - started,
            )
        if not self.runtime.ready():
            raise RuntimeError("인식 엔진 설치가 완료되지 않았습니다.")

        job_id = job_identity(source, source_stat.st_size, source_stat.st_mtime_ns)
        job_dir = self.paths.jobs / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        metadata = self._metadata(request, source_stat.st_size, source_stat.st_mtime_ns)
        cache_matches = self._metadata_matches(job_dir / "meta.json", metadata)
        log_path = self.paths.logs / f"{job_id}.log"
        if not cache_matches:
            # 용어집 등이 바뀐 실행이 도중에 멈춰도 다음 실행이 옛 결과를 다시 쓰지 않게 합니다.
            for name in _CACHE_FILES:
                self._remove_quietly(job_dir / name, job_dir, log_path)
        self._write_json_atomic(job_dir / "meta.json", metadata)

        warning = ""
        ratio = 0.0
        if request.mode is JobMode.FULL_LARGE:
            stage = "전체 large-v3"
            progress(stage, "전체 파일을 large-v3로 인식합니다.", 0.0)
            final_transcript, volume_note = self._run_full_large(
                source,
                request,
                job_dir,
                cache_matches,
                log_path,
                cancelled,
                progress,
                stage,
            )
            status = JobStatus.COMPLETE
            message = f"{volume_note}전체 large-v3 인식 완료"
        else:
            progress("Turbo 초안", "Turbo 초안을 준비합니다.", 0.0)
            turbo = self._run_cached_turbo(
                source,
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
            auto_failure = ""
            rerecognized = False
            if request.auto_full_large and ratio >= REVIEW_RATIO_THRESHOLD:
                # 불확실 구간이 많으면 구간 재처리보다 전체를 다시 인식하는 편이 낫습니다.
                # 조용한 녹음에서는 구간 재처리가 Turbo 초안보다 내용을 더 잃었습니다.
                stage = "자동 전체 large-v3"
                progress(
                    stage,
                    f"불확실 비율 {ratio * 100:.1f}%: 전체를 large-v3로 다시 인식합니다.",
                    0.0,
                )
                try:
                    large, volume_note = self._run_full_large(
                        source,
                        request,
                        job_dir,
                        cache_matches,
                        log_path,
                        cancelled,
                        progress,
                        stage,
                    )
                except EngineCancelled:
                    raise
                except (
                    EngineExecutionError,
                    TranscriptError,
                    RuntimeSetupError,
                    OSError,
                    ValueError,
                ) as exc:
                    if cancelled():
                        raise EngineCancelled("사용자가 작업을 중지했습니다.") from exc
                    auto_failure = f"자동 전체 large-v3 실패: {exc}"
                    with suppress(OSError):
                        self._append_log(log_path, "auto-full-large", auto_failure)
                else:
                    rerecognized = True
                    final_transcript = large
                    message = (
                        f"불확실 비율 {ratio * 100:.0f}%라 {volume_note}"
                        "전체 large-v3로 다시 인식했습니다."
                    )
                    large_ratio = uncertain_ratio(large, build_refine_windows(large))
                    if large_ratio >= REVIEW_RATIO_THRESHOLD:
                        status = JobStatus.REVIEW
                        message += (
                            f" 다시 인식한 결과도 불확실 비율이 {large_ratio * 100:.0f}%라 "
                            "검토가 필요합니다."
                        )
            if not rerecognized:
                if windows:
                    progress(
                        "불확실 구간 분석",
                        f"large-v3 재처리 구간 {len(windows)}개 ({ratio * 100:.1f}%)",
                        None,
                    )
                    try:
                        refined = self._run_refinement(
                            source,
                            request,
                            job_dir,
                            log_path,
                            windows,
                            cancelled,
                            progress,
                        )
                        merged, replaced, unresolved = merge_refined_transcript(
                            turbo, refined, windows
                        )
                        if not transcript_is_valid(merged) or replaced == 0:
                            warning = "large-v3 결과가 유효하지 않아 Turbo 초안을 사용했습니다."
                        else:
                            final_transcript = merged
                            message = (
                                f"불확실 구간 {replaced}/{len(windows)}개를 large-v3로 교체했습니다."
                            )
                            if unresolved or replaced < len(windows):
                                status = JobStatus.REVIEW
                                message += " 일부 구간은 검토가 필요합니다."
                    except EngineCancelled:
                        raise
                    except (EngineExecutionError, TranscriptError, OSError, ValueError) as exc:
                        warning = f"large-v3 구간 재처리 실패. Turbo 초안을 저장합니다: {exc}"
                if auto_failure:
                    status = JobStatus.WARNING
                    message = f"{auto_failure} → {warning or message}"
                elif warning:
                    status = JobStatus.WARNING
                    message = warning
                elif ratio >= REVIEW_RATIO_THRESHOLD:
                    status = JobStatus.REVIEW
                    message += " 불확실 비율이 30% 이상이므로 전체 large-v3를 권장합니다."

        if cancelled():
            raise EngineCancelled("사용자가 작업을 중지했습니다.")
        self._verify_source_unchanged(source, source_stat.st_size, source_stat.st_mtime_ns)
        if request.filter_hallucinations:
            final_transcript, filter_note = self._filter_hallucinations(
                final_transcript, source_kind, log_path
            )
            message += filter_note
        allow_replace = request.mode in {JobMode.REPROCESS, JobMode.FULL_LARGE}
        if source_kind is MediaKind.AUDIO:
            timestamp_label = "타임스탬프 포함" if request.audio_txt_timestamps else "타임스탬프 제외"
            progress("TXT 생성", f"인식 결과를 {timestamp_label} 텍스트로 정리합니다.", None)
            output_path = commit_txt(
                source,
                final_transcript,
                allow_replace=allow_replace,
                include_timestamps=request.audio_txt_timestamps,
            )
        else:
            progress("SRT 생성", "단어 타임스탬프를 자막 큐로 정리합니다.", None)
            cues = cues_from_transcript(final_transcript)
            if not cues:
                raise RuntimeError("음성이 인식되지 않아 SRT를 만들 수 없습니다.")
            output_path = commit_srt(source, cues, allow_replace=allow_replace)
        self._verify_source_unchanged(source, source_stat.st_size, source_stat.st_mtime_ns)
        elapsed = time.monotonic() - started
        progress("완료", message, 1.0)
        return JobResult(
            video=source,
            status=status,
            output_path=output_path,
            message=message,
            uncertain_ratio=ratio,
            elapsed_seconds=elapsed,
        )

    def _filter_hallucinations(
        self, transcript: Transcript, source_kind: MediaKind, log_path: Path
    ) -> tuple[Transcript, str]:
        filtered, removed = filter_hallucinations(transcript, video=source_kind is MediaKind.VIDEO)
        if not removed:
            return transcript, ""
        if not transcript_is_valid(filtered):
            # 전부 헛소리로 보이면 빈 결과 대신 원래 결과를 저장합니다.
            with suppress(OSError):
                self._append_log(log_path, "filter", "남는 말이 없어 헛소리를 거르지 않았습니다.")
            return transcript, ""
        with suppress(OSError):
            for item in removed:
                self._append_log(
                    log_path,
                    "filter",
                    f"{format_txt_timestamp(item.start)} {item.reason}: {item.text[:80]}",
                )
        return filtered, f" 헛소리 {len(removed)}곳을 걸렀습니다."

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

    def _run_full_large(
        self,
        source: Path,
        request: JobRequest,
        job_dir: Path,
        cache_matches: bool,
        log_path: Path,
        cancelled: CancelCheck,
        progress: StageCallback,
        stage: str,
    ) -> tuple[Transcript, str]:
        """Recognise the whole file with large-v3, boosting quiet audio first.

        Returns the transcript and a short note for the result message.
        """
        normalized_audio = job_dir / "normalized.wav"
        # 강제 종료로 남은 보정 파일은 먼저 치웁니다.
        self._remove_quietly(normalized_audio, job_dir, log_path)
        self._remove_quietly(job_dir / "normalized.partial.wav", job_dir, log_path)
        mean_volume = self._mean_volume(source, job_dir, cache_matches, log_path, cancelled)
        quiet = mean_volume is not None and mean_volume < QUIET_MEAN_VOLUME_DB
        note = f"음량 보정(평균 {mean_volume:.1f}dB) 후 " if quiet else ""
        cache = job_dir / ("full-large-normalized.json" if quiet else "full-large.json")
        if cache_matches and cache.is_file():
            try:
                transcript = load_transcript(cache, source="large-v3-cache")
                if transcript_is_valid(transcript):
                    progress(stage, "중단 전 large-v3 JSON을 재사용합니다.", 1.0)
                    return transcript, note
            except TranscriptError:
                pass
        run_dir = job_dir / "full-large-run"
        self._safe_remove_run_dir(run_dir, job_dir)
        recognition_input = source
        try:
            if quiet:
                progress(stage, f"평균 음량 {mean_volume:.1f}dB: 음량을 보정합니다.", None)
                try:
                    self.engine.normalize_audio(
                        source, normalized_audio, LOUDNESS_FILTER, cancelled=cancelled
                    )
                except EngineCancelled:
                    raise
                except (EngineExecutionError, RuntimeSetupError, OSError) as exc:
                    # 보정이 안 되면 원본으로라도 인식합니다. 1.5.0까지와 같은 동작입니다.
                    self._append_log(log_path, "loudness", f"normalization failed: {exc}")
                    note = ""
                    cache = job_dir / "full-large.json"
                else:
                    self._append_log(log_path, "loudness", f"normalized with {LOUDNESS_FILTER}")
                    recognition_input = normalized_audio
            result = self.engine.run(
                EngineRun(
                    video=recognition_input,
                    model_key=LARGE_MODEL.key,
                    output_dir=run_dir,
                    glossary=request.glossary,
                ),
                cancelled=cancelled,
                progress=lambda fraction, line: progress(stage, line, fraction),
                line_callback=lambda line: self._append_log(log_path, "full-large", line),
            )
            transcript = load_transcript(result, source="large-v3-full")
            if not transcript_is_valid(transcript):
                raise TranscriptError("large-v3 JSON에 유효한 발화가 없습니다.")
            self._copy_atomic(result, cache)
            self._safe_remove_run_dir(run_dir, job_dir)
        finally:
            # 지우기가 막혀도(백신 검사 등) 중지·실패 원인을 가리지 않게 합니다.
            self._remove_quietly(normalized_audio, job_dir, log_path)
        return transcript, note

    def _remove_quietly(self, path: Path, job_dir: Path, log_path: Path) -> None:
        try:
            self._safe_remove_run_dir(path, job_dir)
        except OSError as exc:
            with suppress(OSError):
                self._append_log(log_path, "cleanup", f"could not remove {path.name}: {exc}")

    def _mean_volume(
        self,
        source: Path,
        job_dir: Path,
        cache_matches: bool,
        log_path: Path,
        cancelled: CancelCheck,
    ) -> float | None:
        cache = job_dir / "loudness.json"
        if cache_matches and cache.is_file():
            try:
                value = json.loads(cache.read_text(encoding="utf-8")).get("mean_volume_db")
                if isinstance(value, (int, float)):
                    return float(value)
            except (OSError, ValueError, AttributeError):
                pass
        try:
            value = self.engine.measure_mean_volume(source, cancelled=cancelled)
        except EngineCancelled:
            raise
        except (EngineExecutionError, RuntimeSetupError, OSError) as exc:
            self._append_log(log_path, "loudness", f"measurement failed: {exc}")
            return None
        self._append_log(log_path, "loudness", f"mean volume {value} dB")
        if value is not None:
            with suppress(OSError):
                self._write_json_atomic(cache, {"mean_volume_db": value})
        return value

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
            actual = json.loads(path.read_text(encoding="utf-8"))
            recognition_keys = (
                "pipeline_version",
                "source",
                "size",
                "mtime_ns",
                "glossary_sha256",
            )
            return isinstance(actual, dict) and all(
                actual.get(key) == expected.get(key) for key in recognition_keys
            )
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
            raise SourceChangedError("처리 중 원본 미디어가 변경되어 결과 저장을 중단했습니다.")

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
