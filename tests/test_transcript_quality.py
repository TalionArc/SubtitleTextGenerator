from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from subtitle_text_generator.models import Segment, Transcript, Word
from subtitle_text_generator.quality import (
    build_refine_windows,
    merge_refined_transcript,
    segment_uncertainty,
    uncertain_ratio,
)
from subtitle_text_generator.transcript import load_transcript, transcript_is_valid


def test_load_standard_whisper_json(
    tmp_path: Path, clean_transcript_payload: dict[str, object], write_json
) -> None:
    transcript = load_transcript(write_json(tmp_path / "result.json", clean_transcript_payload))
    assert transcript_is_valid(transcript)
    assert len(transcript.segments) == 2
    assert transcript.words[0].text.strip() == "안녕하세요"
    assert transcript.duration == 6.0


def test_uncertainty_thresholds_and_window_merge() -> None:
    transcript = Transcript(
        segments=[
            Segment(2.0, 4.0, "첫 구간", [Word(2.0, 4.0, "첫 구간", 0.3)], avg_logprob=-1.0),
            Segment(6.5, 8.0, "둘째 구간", [Word(6.5, 8.0, "둘째 구간", 0.9)], compression_ratio=2.5),
            Segment(20.0, 22.0, "정상", [Word(20.0, 22.0, "정상", 0.9)], avg_logprob=-0.1),
        ],
        duration=30.0,
    )
    windows = build_refine_windows(transcript)
    assert len(windows) == 1
    assert windows[0].core_start == 1.5
    assert windows[0].core_end == 8.5
    assert windows[0].decode_start == 0.0
    assert windows[0].decode_end == 10.5
    assert uncertain_ratio(transcript, windows) > 0


def test_temperature_repetition_and_timestamps_are_flagged() -> None:
    segment = Segment(
        1.0,
        4.0,
        "반복 반복 반복 반복 반복 반복 반복 반복 반복",
        [Word(2.0, 1.5, "오류", 0.9)],
        temperature=0.2,
    )
    reasons = segment_uncertainty(segment)
    assert "temperature fallback" in reasons
    assert "반복 문구" in reasons
    assert "타임스탬프 이상" in reasons


def test_merge_replaces_only_core_words(
    tmp_path: Path, clean_transcript_payload: dict[str, object], write_json
) -> None:
    turbo_payload = deepcopy(clean_transcript_payload)
    first = turbo_payload["segments"][0]  # type: ignore[index]
    first["avg_logprob"] = -1.2  # type: ignore[index]
    turbo = load_transcript(write_json(tmp_path / "turbo.json", turbo_payload))
    windows = build_refine_windows(turbo)
    refined_payload = {
        "language": "ko",
        "duration": 6.0,
        "segments": [
            {
                "start": 0.2,
                "end": 2.2,
                "text": "안녕하세요 정확한 강의를 시작합니다.",
                "avg_logprob": -0.1,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
                "temperature": 0.0,
                "words": [
                    {"start": 0.2, "end": 0.8, "word": " 안녕하세요", "probability": 0.99},
                    {"start": 0.9, "end": 1.2, "word": " 정확한", "probability": 0.98},
                    {"start": 1.3, "end": 1.7, "word": " 강의를", "probability": 0.98},
                    {"start": 1.8, "end": 2.2, "word": " 시작합니다.", "probability": 0.98},
                ],
            }
        ],
    }
    refined = load_transcript(write_json(tmp_path / "refined.json", refined_payload))
    merged, replaced, unresolved = merge_refined_transcript(turbo, refined, windows)
    assert replaced == 1
    assert unresolved == 0
    assert "정확한" in merged.segments[0].text
    assert "오늘" in merged.segments[0].text
