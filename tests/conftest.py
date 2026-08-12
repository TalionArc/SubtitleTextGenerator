from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def clean_transcript_payload() -> dict[str, object]:
    return {
        "language": "ko",
        "duration": 6.0,
        "segments": [
            {
                "id": 0,
                "start": 0.2,
                "end": 2.2,
                "text": " 안녕하세요 강의를 시작합니다.",
                "avg_logprob": -0.2,
                "compression_ratio": 1.1,
                "no_speech_prob": 0.02,
                "temperature": 0.0,
                "words": [
                    {"start": 0.2, "end": 0.9, "word": " 안녕하세요", "probability": 0.96},
                    {"start": 1.0, "end": 1.5, "word": " 강의를", "probability": 0.94},
                    {"start": 1.6, "end": 2.2, "word": " 시작합니다.", "probability": 0.95},
                ],
            },
            {
                "id": 1,
                "start": 3.0,
                "end": 5.5,
                "text": " 오늘 주제는 테스트입니다.",
                "avg_logprob": -0.25,
                "compression_ratio": 1.2,
                "no_speech_prob": 0.01,
                "temperature": 0.0,
                "words": [
                    {"start": 3.0, "end": 3.5, "word": " 오늘", "probability": 0.97},
                    {"start": 3.6, "end": 4.2, "word": " 주제는", "probability": 0.96},
                    {"start": 4.3, "end": 5.5, "word": " 테스트입니다.", "probability": 0.93},
                ],
            },
        ],
    }


@pytest.fixture
def write_json():
    def writer(path: Path, payload: dict[str, object]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    return writer
