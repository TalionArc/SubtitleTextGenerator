from __future__ import annotations

from dataclasses import dataclass

APP_NAME = "SubtitleTextGenerator"
LEGACY_APP_NAME = "LectureSubtitleBatcher"
APP_DISPLAY_NAME = "영상·녹음 자막 텍스트 생성기"
APP_VERSION = "1.3.0"

DEFAULT_ROOT = ""
VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".ts"}
)
AUDIO_EXTENSIONS = frozenset(
    {".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".wma", ".aif", ".aiff", ".mka"}
)
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
SUBTITLE_EXTENSIONS = frozenset({".srt", ".vtt", ".ass", ".ssa"})

ENGINE_VERSION = "r245.4"
ENGINE_ARCHIVE_NAME = "Faster-Whisper-XXL_r245.4_windows.7z"
ENGINE_ARCHIVE_URL = (
    "https://github.com/Purfview/whisper-standalone-win/releases/download/"
    "Faster-Whisper-XXL/Faster-Whisper-XXL_r245.4_windows.7z"
)
ENGINE_ARCHIVE_SIZE = 1_424_256_246
ENGINE_ARCHIVE_SHA256 = "237dee23939cdabfc96ef859fc5e584b842c3a5557e0d2ca744e1f87c14c5844"
MIN_SETUP_FREE_BYTES = 12 * 1024**3


@dataclass(frozen=True, slots=True)
class DownloadFile:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelManifest:
    key: str
    display_name: str
    repository: str
    revision: str
    files: tuple[DownloadFile, ...]

    def url_for(self, filename: str) -> str:
        return (
            f"https://huggingface.co/{self.repository}/resolve/"
            f"{self.revision}/{filename}?download=true"
        )


COMMON_MODEL_FILES = (
    DownloadFile(
        "preprocessor_config.json",
        340,
        "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711",
    ),
    DownloadFile(
        "tokenizer.json",
        2_480_617,
        "6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca",
    ),
    DownloadFile(
        "vocabulary.json",
        1_068_114,
        "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1",
    ),
)

TURBO_MODEL = ModelManifest(
    key="large-v3-turbo",
    display_name="large-v3-turbo",
    repository="Purfview/faster-whisper-large-v3-turbo",
    revision="09b34be54767224f2d06cb4b65992dee77a859df",
    files=(
        DownloadFile(
            "config.json",
            2_263,
            "b0253ea6c0d3bea6b1e19e91a02acfd3b53f4467362efcb5a3e6b16c9b3a9b7e",
        ),
        DownloadFile(
            "model.bin",
            1_617_884_929,
            "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da",
        ),
        *COMMON_MODEL_FILES,
    ),
)

LARGE_MODEL = ModelManifest(
    key="large-v3",
    display_name="large-v3",
    repository="Systran/faster-whisper-large-v3",
    revision="edaa852ec7e145841d8ffdb056a99866b5f0a478",
    files=(
        DownloadFile(
            "config.json",
            2_394,
            "a9306624f5ec14270a014b647e5c316b6e03a662c369758d1b90697a7b0655b9",
        ),
        DownloadFile(
            "model.bin",
            3_087_284_237,
            "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1",
        ),
        *COMMON_MODEL_FILES,
    ),
)

MODELS = (TURBO_MODEL, LARGE_MODEL)

SETUP_DOWNLOAD_BYTES = ENGINE_ARCHIVE_SIZE + sum(
    item.size for model in MODELS for item in model.files
)
# 압축을 푼 엔진, 두 모델, 복구용 엔진 압축본을 합친 실측 설치 크기는 약 10.1GiB입니다.
SETUP_INSTALLED_BYTES = 10 * 1024**3
