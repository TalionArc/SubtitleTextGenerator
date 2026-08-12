from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from .models import RefineWindow, Segment, Transcript, Word
from .transcript import transcript_from_words, transcript_words

AVG_LOGPROB_THRESHOLD = -0.80
COMPRESSION_RATIO_THRESHOLD = 2.20
NO_SPEECH_THRESHOLD = 0.55
MEAN_WORD_PROB_THRESHOLD = 0.65
LOW_WORD_PROB_THRESHOLD = 0.45
LOW_WORD_FRACTION_THRESHOLD = 0.25
CORE_PADDING_SECONDS = 0.5
CONTEXT_PADDING_SECONDS = 2.0
MERGE_GAP_SECONDS = 3.0
REVIEW_RATIO_THRESHOLD = 0.30

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def segment_uncertainty(segment: Segment) -> tuple[str, ...]:
    reasons: list[str] = []
    if segment.avg_logprob is not None and segment.avg_logprob < AVG_LOGPROB_THRESHOLD:
        reasons.append("낮은 평균 로그확률")
    if (
        segment.compression_ratio is not None
        and segment.compression_ratio > COMPRESSION_RATIO_THRESHOLD
    ):
        reasons.append("높은 압축률")
    if (
        segment.text.strip()
        and segment.no_speech_prob is not None
        and segment.no_speech_prob > NO_SPEECH_THRESHOLD
    ):
        reasons.append("발화/무음 충돌")
    if segment.temperature is not None and segment.temperature > 0.0:
        reasons.append("temperature fallback")

    probabilities = [
        word.probability for word in segment.words if word.probability is not None
    ]
    if probabilities:
        if sum(probabilities) / len(probabilities) < MEAN_WORD_PROB_THRESHOLD:
            reasons.append("낮은 단어 평균확률")
        low_fraction = sum(value < LOW_WORD_PROB_THRESHOLD for value in probabilities) / len(
            probabilities
        )
        if low_fraction >= LOW_WORD_FRACTION_THRESHOLD:
            reasons.append("낮은 확률 단어 다수")
    if _has_repetition(segment.text):
        reasons.append("반복 문구")
    if _has_timestamp_anomaly(segment):
        reasons.append("타임스탬프 이상")
    return tuple(dict.fromkeys(reasons))


def build_refine_windows(transcript: Transcript) -> list[RefineWindow]:
    flagged: list[tuple[float, float, tuple[str, ...]]] = []
    for segment in transcript.segments:
        reasons = segment_uncertainty(segment)
        if reasons:
            flagged.append((segment.start, segment.end, reasons))
    if not flagged:
        return []
    flagged.sort(key=lambda item: item[0])
    merged: list[tuple[float, float, list[str]]] = []
    for start, end, reasons in flagged:
        if merged and start - merged[-1][1] <= MERGE_GAP_SECONDS:
            old_start, old_end, old_reasons = merged[-1]
            merged[-1] = (old_start, max(old_end, end), [*old_reasons, *reasons])
        else:
            merged.append((start, end, list(reasons)))
    duration = max(transcript.duration, max(item[1] for item in merged))
    windows: list[RefineWindow] = []
    for start, end, reasons in merged:
        core_start = max(0.0, start - CORE_PADDING_SECONDS)
        core_end = min(duration, end + CORE_PADDING_SECONDS)
        windows.append(
            RefineWindow(
                core_start=core_start,
                core_end=core_end,
                decode_start=max(0.0, core_start - CONTEXT_PADDING_SECONDS),
                decode_end=min(duration, core_end + CONTEXT_PADDING_SECONDS),
                reasons=tuple(dict.fromkeys(reasons)),
            )
        )
    return windows


def uncertain_ratio(transcript: Transcript, windows: list[RefineWindow]) -> float:
    spoken = _union_duration((segment.start, segment.end) for segment in transcript.segments)
    if spoken <= 0:
        return 0.0
    uncertain = _union_duration((window.core_start, window.core_end) for window in windows)
    return min(uncertain / spoken, 1.0)


def merge_refined_transcript(
    turbo: Transcript, refined: Transcript, windows: list[RefineWindow]
) -> tuple[Transcript, int, int]:
    base_words = transcript_words(turbo)
    refined_words = transcript_words(refined)
    kept = list(base_words)
    replacements: list[Word] = []
    unresolved = 0
    replaced = 0
    for window in windows:
        candidates = [
            word
            for word in refined_words
            if window.core_start <= word.midpoint <= window.core_end
        ]
        if not candidates:
            unresolved += 1
            continue
        relevant_segments = [
            segment
            for segment in refined.segments
            if segment.end > window.core_start and segment.start < window.core_end
        ]
        if any(segment_uncertainty(segment) for segment in relevant_segments):
            unresolved += 1
        kept = [
            word
            for word in kept
            if not (window.core_start <= word.midpoint <= window.core_end)
        ]
        replacements.extend(candidates)
        replaced += 1
    merged_words = _deduplicate_words([*kept, *replacements])
    return (
        transcript_from_words(merged_words, duration=turbo.duration, source="adaptive"),
        replaced,
        unresolved,
    )


def _has_repetition(text: str) -> bool:
    tokens = [token.casefold() for token in _TOKEN_RE.findall(text)]
    if len(tokens) >= 3:
        consecutive = 1
        for previous, current in zip(tokens, tokens[1:], strict=False):
            consecutive = consecutive + 1 if previous == current else 1
            if consecutive >= 3:
                return True
    if len(tokens) >= 9:
        trigrams = Counter(tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2))
        if any(count >= 3 for count in trigrams.values()):
            return True
    compact = re.sub(r"\s+", "", text)
    return bool(re.search(r"(.{2,12})\1\1", compact))


def _has_timestamp_anomaly(segment: Segment) -> bool:
    if segment.end <= segment.start or segment.start < 0:
        return True
    previous_end = segment.start
    for word in segment.words:
        if word.end <= word.start or word.start < segment.start - 1.0 or word.end > segment.end + 1.0:
            return True
        if word.start < previous_end - 0.5:
            return True
        previous_end = max(previous_end, word.end)
    return False


def _union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((float(start), float(end)) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _deduplicate_words(words: list[Word]) -> list[Word]:
    ordered = sorted(words, key=lambda word: (word.start, word.end, word.text))
    result: list[Word] = []
    for word in ordered:
        if result:
            previous = result[-1]
            same_text = previous.text.strip().casefold() == word.text.strip().casefold()
            same_time = abs(previous.start - word.start) < 0.12 and abs(previous.end - word.end) < 0.12
            if same_text and same_time:
                if (word.probability or -1) > (previous.probability or -1):
                    result[-1] = word
                continue
        result.append(word)
    return result
