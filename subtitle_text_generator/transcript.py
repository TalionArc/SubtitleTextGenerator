from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .models import Segment, Transcript, Word


class TranscriptError(ValueError):
    pass


def load_transcript(path: Path, *, source: str = "") -> Transcript:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TranscriptError(f"JSON 결과를 읽을 수 없습니다: {path.name}") from exc
    if not isinstance(payload, dict):
        raise TranscriptError("JSON 결과의 최상위 값이 객체가 아닙니다.")
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise TranscriptError("JSON 결과에 segments 배열이 없습니다.")

    segments: list[Segment] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        start = _number(raw.get("start"))
        end = _number(raw.get("end"))
        text = str(raw.get("text") or "").strip()
        if start is None or end is None or end <= start or not text:
            continue
        words = _parse_words(raw.get("words"), start, end, text)
        segments.append(
            Segment(
                start=max(start, 0.0),
                end=max(end, start),
                text=text,
                words=words,
                avg_logprob=_number(raw.get("avg_logprob")),
                compression_ratio=_number(raw.get("compression_ratio")),
                no_speech_prob=_number(raw.get("no_speech_prob")),
                temperature=_number(raw.get("temperature")),
                metadata={
                    key: raw[key]
                    for key in ("id", "seek", "tokens")
                    if key in raw
                },
            )
        )
    segments.sort(key=lambda segment: (segment.start, segment.end))
    duration = _number(payload.get("duration")) or 0.0
    if segments:
        duration = max(duration, max(segment.end for segment in segments))
    language = str(payload.get("language") or "ko")
    return Transcript(segments=segments, duration=duration, language=language, source=source)


def transcript_is_valid(transcript: Transcript) -> bool:
    if not transcript.segments:
        return False
    words = transcript_words(transcript)
    return any(word.text.strip() and word.end > word.start >= 0 for word in words)


def transcript_words(transcript: Transcript) -> list[Word]:
    words: list[Word] = []
    for segment in transcript.segments:
        if segment.words:
            words.extend(segment.words)
        else:
            words.extend(_fallback_words(segment.text, segment.start, segment.end))
    return sorted(words, key=lambda word: (word.start, word.end))


def transcript_from_words(words: list[Word], *, duration: float, source: str) -> Transcript:
    ordered = sorted(
        (word for word in words if word.text.strip() and word.end > word.start >= 0),
        key=lambda word: (word.start, word.end),
    )
    if not ordered:
        return Transcript([], duration=max(duration, 0.0), source=source)
    text = _join_tokens([word.text for word in ordered])
    segment = Segment(
        start=ordered[0].start,
        end=ordered[-1].end,
        text=text,
        words=ordered,
    )
    return Transcript([segment], duration=max(duration, ordered[-1].end), source=source)


def _parse_words(raw_words: Any, segment_start: float, segment_end: float, text: str) -> list[Word]:
    if not isinstance(raw_words, list):
        return _fallback_words(text, segment_start, segment_end)
    words: list[Word] = []
    for raw in raw_words:
        if not isinstance(raw, dict):
            continue
        start = _number(raw.get("start"))
        end = _number(raw.get("end"))
        token = str(raw.get("word") if "word" in raw else raw.get("text") or "")
        probability = _number(raw.get("probability"))
        if start is None or end is None or not token.strip() or end <= start:
            continue
        words.append(Word(max(start, 0.0), end, token, probability))
    return words or _fallback_words(text, segment_start, segment_end)


def _fallback_words(text: str, start: float, end: float) -> list[Word]:
    tokens = re.findall(r"\s*\S+", text)
    if not tokens:
        return []
    duration = max(end - start, 0.001)
    weights = [max(len(token.strip()), 1) for token in tokens]
    total = sum(weights)
    words: list[Word] = []
    cursor = start
    for index, (token, weight) in enumerate(zip(tokens, weights, strict=True)):
        next_cursor = end if index == len(tokens) - 1 else cursor + duration * weight / total
        words.append(Word(cursor, max(next_cursor, cursor + 0.001), token, None))
        cursor = next_cursor
    return words


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _join_tokens(tokens: list[str]) -> str:
    text = "".join(tokens).strip()
    return re.sub(r"\s+", " ", text)
