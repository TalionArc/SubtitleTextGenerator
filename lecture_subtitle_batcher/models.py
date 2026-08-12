from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class JobMode(StrEnum):
    ADAPTIVE = "adaptive"
    REPROCESS = "reprocess"
    FULL_LARGE = "full_large"


class JobStatus(StrEnum):
    WAITING = "대기"
    RUNNING = "처리 중"
    COMPLETE = "완료"
    REVIEW = "완료 / 검토 필요"
    SKIPPED = "건너뜀"
    WARNING = "완료 / 경고"
    FAILED = "실패"
    CANCELLED = "취소됨"


@dataclass(slots=True)
class VideoEntry:
    path: Path
    root: Path
    size: int
    mtime_ns: int
    existing_subtitles: tuple[Path, ...] = ()
    status: JobStatus = JobStatus.WAITING
    stage: str = ""
    elapsed_seconds: float = 0.0
    message: str = ""

    @property
    def relative_parent(self) -> str:
        try:
            parent = self.path.parent.relative_to(self.root)
        except ValueError:
            parent = self.path.parent
        return "." if str(parent) in {"", "."} else str(parent)

    @property
    def has_subtitle(self) -> bool:
        return bool(self.existing_subtitles)


@dataclass(frozen=True, slots=True)
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None
    temperature: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Transcript:
    segments: list[Segment]
    duration: float
    language: str = "ko"
    source: str = ""

    @property
    def words(self) -> list[Word]:
        return [word for segment in self.segments for word in segment.words]


@dataclass(frozen=True, slots=True)
class RefineWindow:
    core_start: float
    core_end: float
    decode_start: float
    decode_end: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class JobRequest:
    video: Path
    root: Path
    mode: JobMode
    glossary: tuple[str, ...] = ()


@dataclass(slots=True)
class JobResult:
    video: Path
    status: JobStatus
    srt_path: Path | None = None
    message: str = ""
    uncertain_ratio: float = 0.0
    elapsed_seconds: float = 0.0
