from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .app_paths import AppPaths
from .constants import (
    APP_NAME,
    APP_VERSION,
    ENGINE_ARCHIVE_NAME,
    ENGINE_ARCHIVE_SHA256,
    ENGINE_ARCHIVE_SIZE,
    ENGINE_ARCHIVE_URL,
    ENGINE_VERSION,
    MIN_SETUP_FREE_BYTES,
    MODELS,
    ModelManifest,
)

ProgressCallback = Callable[[str, str, float | None], None]
CancelCheck = Callable[[], bool]


class RuntimeSetupError(RuntimeError):
    pass


class SetupCancelled(RuntimeSetupError):
    pass


def sha256_file(path: Path, cancel: CancelCheck | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            if cancel and cancel():
                raise SetupCancelled("설치가 취소되었습니다.")
            digest.update(chunk)
    return digest.hexdigest()


def gpu_summary() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            creationflags=flags,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeSetupError(
            "NVIDIA GPU를 확인할 수 없습니다. NVIDIA 드라이버와 nvidia-smi를 확인하세요."
        ) from exc
    if completed.returncode != 0 or not completed.stdout.strip():
        detail = completed.stderr.strip() or "nvidia-smi 응답 없음"
        raise RuntimeSetupError(f"CUDA용 NVIDIA GPU 확인 실패: {detail}")
    return completed.stdout.strip().splitlines()[0]


class ResumableDownloader:
    def __init__(self, progress: ProgressCallback, cancelled: CancelCheck) -> None:
        self.progress = progress
        self.cancelled = cancelled

    def fetch(
        self,
        url: str,
        target: Path,
        *,
        expected_size: int,
        expected_sha256: str,
        label: str,
    ) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        if self._valid_existing(target, expected_size, expected_sha256, label):
            return target

        partial = target.with_name(target.name + ".part")
        if partial.exists() and partial.stat().st_size > expected_size:
            partial.unlink()

        last_error: BaseException | None = None
        for attempt in range(1, 6):
            if self.cancelled():
                raise SetupCancelled("설치가 취소되었습니다.")
            try:
                self._download_once(url, partial, expected_size, label)
                if partial.stat().st_size != expected_size:
                    raise RuntimeSetupError(
                        f"{label} 크기 불일치: {partial.stat().st_size:,} / {expected_size:,} bytes"
                    )
                self.progress("설치", f"{label} 무결성 검사 중", None)
                actual = sha256_file(partial, self.cancelled)
                if actual.casefold() != expected_sha256.casefold():
                    partial.unlink(missing_ok=True)
                    raise RuntimeSetupError(
                        f"{label} SHA-256 불일치. 다운로드 파일을 폐기했습니다."
                    )
                os.replace(partial, target)
                return target
            except SetupCancelled:
                raise
            except (OSError, urllib.error.URLError, RuntimeSetupError) as exc:
                last_error = exc
                if attempt == 5:
                    break
                self.progress("설치", f"{label} 다운로드 재시도 {attempt}/4", None)
                self._cancelable_sleep(min(2**attempt, 8))
        raise RuntimeSetupError(f"{label} 다운로드 실패: {last_error}") from last_error

    def _valid_existing(
        self, target: Path, expected_size: int, expected_sha256: str, label: str
    ) -> bool:
        try:
            if target.stat().st_size != expected_size:
                return False
        except OSError:
            return False
        self.progress("설치", f"{label} 기존 파일 검사 중", None)
        try:
            return sha256_file(target, self.cancelled).casefold() == expected_sha256.casefold()
        except OSError:
            return False

    def _download_once(self, url: str, partial: Path, expected_size: int, label: str) -> None:
        current = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": f"{APP_NAME}/{APP_VERSION}"}
        if current:
            headers["Range"] = f"bytes={current}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            status = getattr(response, "status", response.getcode())
            if current and status != 206:
                current = 0
                partial.unlink(missing_ok=True)
            mode = "ab" if current else "wb"
            received = current
            last_report = 0.0
            with partial.open(mode) as stream:
                while chunk := response.read(1024 * 1024):
                    if self.cancelled():
                        raise SetupCancelled("설치가 취소되었습니다.")
                    stream.write(chunk)
                    received += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 0.2 or received >= expected_size:
                        fraction = min(received / expected_size, 1.0) if expected_size else None
                        self.progress(
                            "설치",
                            f"{label} 다운로드 {received / 1024**2:,.0f} / {expected_size / 1024**2:,.0f} MB",
                            fraction,
                        )
                        last_report = now

    def _cancelable_sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.cancelled():
                raise SetupCancelled("설치가 취소되었습니다.")
            time.sleep(min(0.1, max(deadline - time.monotonic(), 0.0)))


class RuntimeManager:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths

    def engine_executable(self) -> Path:
        candidates = sorted(self.paths.engine.rglob("faster-whisper-xxl.exe"))
        if not candidates:
            raise RuntimeSetupError("Faster-Whisper-XXL 실행 파일을 찾을 수 없습니다.")
        return candidates[0]

    def model_path(self, key: str) -> Path:
        return self.paths.models / f"faster-whisper-{key}"

    def ready(self) -> bool:
        return self._engine_ready() and all(self._model_ready(model) for model in MODELS)

    def ensure_ready(
        self,
        progress: ProgressCallback,
        cancelled: CancelCheck,
        *,
        force_repair: bool = False,
    ) -> None:
        self.paths.ensure_base_dirs()
        gpu = gpu_summary()
        progress("환경 확인", f"GPU: {gpu}", None)
        if not self.ready() or force_repair:
            free = shutil.disk_usage(self.paths.root).free
            if free < MIN_SETUP_FREE_BYTES:
                raise RuntimeSetupError(
                    f"설치 공간이 부족합니다. 최소 12 GB 필요, 현재 {free / 1024**3:.1f} GB 여유."
                )
        downloader = ResumableDownloader(progress, cancelled)
        if force_repair or not self._engine_ready():
            self._install_engine(downloader, progress, cancelled)
        for model in MODELS:
            if force_repair or not self._model_ready(model):
                self._install_model(model, downloader, progress, cancelled)
        progress("준비 완료", "엔진과 두 모델이 준비되었습니다.", 1.0)

    def _engine_ready(self) -> bool:
        manifest_path = self.paths.engine / "install.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != ENGINE_VERSION:
                return False
            executable = self.paths.engine / str(manifest["executable"])
            return executable.is_file() and executable.stat().st_size > 0
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _model_ready(self, model: ModelManifest) -> bool:
        target = self.model_path(model.key)
        try:
            manifest = json.loads((target / "install.json").read_text(encoding="utf-8"))
            if manifest.get("revision") != model.revision:
                return False
            return all((target / item.name).stat().st_size == item.size for item in model.files)
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _install_engine(
        self,
        downloader: ResumableDownloader,
        progress: ProgressCallback,
        cancelled: CancelCheck,
    ) -> None:
        archive = downloader.fetch(
            ENGINE_ARCHIVE_URL,
            self.paths.downloads / ENGINE_ARCHIVE_NAME,
            expected_size=ENGINE_ARCHIVE_SIZE,
            expected_sha256=ENGINE_ARCHIVE_SHA256,
            label="Faster-Whisper-XXL",
        )
        if cancelled():
            raise SetupCancelled("설치가 취소되었습니다.")
        tar = shutil.which("tar")
        if not tar:
            raise RuntimeSetupError("Windows 기본 tar.exe를 찾을 수 없어 엔진 압축을 풀 수 없습니다.")
        staging = self.paths.root / "engine.staging"
        self._safe_remove(staging)
        staging.mkdir(parents=True)
        progress("설치", "Faster-Whisper-XXL 압축 해제 중", None)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            [tar, "-xf", str(archive), "-C", str(staging)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=flags,
        )
        while process.poll() is None:
            if cancelled():
                process.kill()
                process.wait(timeout=5)
                self._safe_remove(staging)
                raise SetupCancelled("설치가 취소되었습니다.")
            time.sleep(0.1)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            self._safe_remove(staging)
            detail = stderr.strip() or stdout.strip()
            raise RuntimeSetupError(f"엔진 압축 해제 실패: {detail}")
        executables = sorted(staging.rglob("faster-whisper-xxl.exe"))
        ffmpeg = sorted(staging.rglob("ffmpeg.exe"))
        if not executables or not ffmpeg or executables[0].stat().st_size == 0:
            self._safe_remove(staging)
            raise RuntimeSetupError("압축을 푼 엔진 구성이 불완전합니다.")
        relative_executable = executables[0].relative_to(staging)
        install_data = {
            "version": ENGINE_VERSION,
            "archive_sha256": ENGINE_ARCHIVE_SHA256,
            "executable": str(relative_executable),
            "executable_sha256": sha256_file(executables[0], cancelled),
        }
        (staging / "install.json").write_text(
            json.dumps(install_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._replace_directory(staging, self.paths.engine)

    def _install_model(
        self,
        model: ModelManifest,
        downloader: ResumableDownloader,
        progress: ProgressCallback,
        cancelled: CancelCheck,
    ) -> None:
        target = self.model_path(model.key)
        staging = self.paths.models / f"faster-whisper-{model.key}.staging"
        staging.mkdir(parents=True, exist_ok=True)
        for item in model.files:
            downloader.fetch(
                model.url_for(item.name),
                staging / item.name,
                expected_size=item.size,
                expected_sha256=item.sha256,
                label=f"{model.display_name} / {item.name}",
            )
        if cancelled():
            raise SetupCancelled("설치가 취소되었습니다.")
        (staging / "install.json").write_text(
            json.dumps(
                {
                    "repository": model.repository,
                    "revision": model.revision,
                    "files": {item.name: item.sha256 for item in model.files},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        progress("설치", f"{model.display_name} 설치 마무리 중", None)
        self._replace_directory(staging, target)

    def _replace_directory(self, staging: Path, target: Path) -> None:
        backup = target.with_name(target.name + ".old")
        self._safe_remove(backup)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except BaseException:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        self._safe_remove(backup)

    def _safe_remove(self, path: Path) -> None:
        root = self.paths.root.resolve()
        candidate = path.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeSetupError(f"안전하지 않은 삭제 경로: {candidate}") from exc
        if candidate == root:
            raise RuntimeSetupError("앱 데이터 루트는 삭제할 수 없습니다.")
        if candidate.is_dir():
            shutil.rmtree(candidate)
        elif candidate.exists():
            candidate.unlink()
