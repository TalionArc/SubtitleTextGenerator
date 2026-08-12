from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .models import Transcript, Word
from .transcript import transcript_words

MAX_LINE_CHARS = 22
MAX_CUE_CHARS = MAX_LINE_CHARS * 2
MIN_CUE_SECONDS = 1.0
MAX_CUE_SECONDS = 6.0
MIN_GAP_SECONDS = 0.05
PAUSE_BREAK_SECONDS = 0.8

_ENDING_PUNCTUATION = frozenset(".!?。！？")
_NO_SPACE_BEFORE = frozenset(",.!?;:%)]}。，！？：；、…")
_NO_SPACE_AFTER = frozenset("([{‘“")
_TIMING_LINE = re.compile(
    r"^(\d{2,}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2,}):(\d{2}):(\d{2}),(\d{3})$"
)


class SubtitleError(RuntimeError):
    pass


class ExistingSubtitleError(SubtitleError):
    pass


@dataclass(slots=True)
class Cue:
    start: float
    end: float
    text: str


def cues_from_transcript(transcript: Transcript) -> list[Cue]:
    words = _expand_long_words(transcript_words(transcript))
    if not words:
        return []
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current and _should_break(current, word):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    groups = _merge_short_groups(groups)
    raw = [
        Cue(group[0].start, group[-1].end, wrap_two_lines(join_word_text(group)))
        for group in groups
        if join_word_text(group)
    ]
    return _normalize_timings(raw)


def join_word_text(words: list[Word]) -> str:
    result = ""
    for word in words:
        raw = word.text
        token = raw.strip()
        if not token:
            continue
        if not result:
            result = token
        elif raw[:1].isspace():
            result += " " + token
        elif token[0] in _NO_SPACE_BEFORE or result[-1] in _NO_SPACE_AFTER:
            result += token
        elif _needs_space(result[-1], token[0]):
            result += " " + token
        else:
            result += token
    return re.sub(r"[ \t]+", " ", result).strip()


def wrap_two_lines(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= MAX_LINE_CHARS:
        return text
    if len(text) > MAX_CUE_CHARS:
        raise SubtitleError(f"자막 큐가 {MAX_CUE_CHARS}자를 초과했습니다.")
    possible = [index for index, char in enumerate(text) if char == " "]
    valid = [index for index in possible if index <= MAX_LINE_CHARS and len(text) - index - 1 <= MAX_LINE_CHARS]
    if valid:
        split_at = min(valid, key=lambda index: abs(index - len(text) / 2))
        return text[:split_at].rstrip() + "\n" + text[split_at + 1 :].lstrip()
    split_at = min(MAX_LINE_CHARS, len(text))
    return text[:split_at].rstrip() + "\n" + text[split_at:].lstrip()


def render_srt(cues: list[Cue]) -> str:
    blocks: list[str] = []
    for index, cue in enumerate(cues, 1):
        blocks.append(
            f"{index}\r\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\r\n{cue.text}"
        )
    return "\r\n\r\n".join(blocks) + ("\r\n" if blocks else "")


def validate_srt_text(text: str) -> None:
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n")
    blocks = [block for block in normalized.strip().split("\n\n") if block.strip()]
    if not blocks:
        raise SubtitleError("자막 내용이 비어 있습니다.")
    previous_end = -1.0
    for expected_index, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if len(lines) < 3 or lines[0].strip() != str(expected_index):
            raise SubtitleError(f"SRT 번호 형식 오류: {expected_index}")
        match = _TIMING_LINE.fullmatch(lines[1].strip())
        if not match:
            raise SubtitleError(f"SRT 시간 형식 오류: {lines[1] if len(lines) > 1 else ''}")
        start = _match_seconds(match.groups()[:4])
        end = _match_seconds(match.groups()[4:])
        if end <= start or start < previous_end - 0.001:
            raise SubtitleError(f"SRT 시간 순서 오류: {expected_index}")
        if not "".join(lines[2:]).strip() or len(lines[2:]) > 2:
            raise SubtitleError(f"SRT 본문 형식 오류: {expected_index}")
        if any(len(line) > MAX_LINE_CHARS for line in lines[2:]):
            raise SubtitleError(f"SRT 줄 길이 초과: {expected_index}")
        previous_end = end


def commit_srt(video: Path, cues: list[Cue], *, allow_replace: bool) -> Path:
    output = video.with_suffix(".srt")
    if output.exists() and not allow_replace:
        raise ExistingSubtitleError(f"이미 자막이 있습니다: {output.name}")
    text = render_srt(cues)
    validate_srt_text(text)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_bytes(text.encode("utf-8-sig"))
    try:
        written = temporary.read_text(encoding="utf-8-sig")
        validate_srt_text(written)
        backup = output.with_suffix(output.suffix + ".bak")
        moved_original = False
        if output.exists():
            backup.unlink(missing_ok=True)
            os.replace(output, backup)
            moved_original = True
        try:
            os.replace(temporary, output)
        except BaseException:
            if moved_original and backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
    finally:
        temporary.unlink(missing_ok=True)
    return output


def format_timestamp(seconds: float) -> str:
    milliseconds = max(round(seconds * 1000), 0)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _should_break(current: list[Word], word: Word) -> bool:
    previous = current[-1]
    candidate = join_word_text([*current, word])
    if word.start - previous.end >= PAUSE_BREAK_SECONDS:
        return True
    if join_word_text(current).endswith(tuple(_ENDING_PUNCTUATION)):
        return True
    if len(candidate) > MAX_CUE_CHARS:
        return True
    return word.end - current[0].start > MAX_CUE_SECONDS


def _merge_short_groups(groups: list[list[Word]]) -> list[list[Word]]:
    if len(groups) < 2:
        return groups
    result: list[list[Word]] = []
    index = 0
    while index < len(groups):
        group = groups[index]
        duration = group[-1].end - group[0].start
        if duration < MIN_CUE_SECONDS and index + 1 < len(groups):
            combined = [*group, *groups[index + 1]]
            if (
                len(join_word_text(combined)) <= MAX_CUE_CHARS
                and combined[-1].end - combined[0].start <= MAX_CUE_SECONDS
            ):
                result.append(combined)
                index += 2
                continue
        if duration < MIN_CUE_SECONDS and result:
            combined = [*result[-1], *group]
            if (
                len(join_word_text(combined)) <= MAX_CUE_CHARS
                and combined[-1].end - combined[0].start <= MAX_CUE_SECONDS
            ):
                result[-1] = combined
                index += 1
                continue
        result.append(group)
        index += 1
    return result


def _normalize_timings(cues: list[Cue]) -> list[Cue]:
    result: list[Cue] = []
    for index, cue in enumerate(cues):
        previous_end = result[-1].end if result else 0.0
        start = max(cue.start, previous_end + (MIN_GAP_SECONDS if result else 0.0))
        next_start = cues[index + 1].start if index + 1 < len(cues) else None
        limit = start + MAX_CUE_SECONDS
        if next_start is not None:
            limit = min(limit, max(next_start - MIN_GAP_SECONDS, start + 0.001))
        end = min(max(cue.end, start + MIN_CUE_SECONDS), limit)
        if end <= start:
            end = start + 0.001
        result.append(Cue(start, end, cue.text))
    return result


def _expand_long_words(words: list[Word]) -> list[Word]:
    result: list[Word] = []
    for word in words:
        stripped = word.text.strip()
        if len(stripped) <= MAX_CUE_CHARS:
            result.append(word)
            continue
        pieces = [stripped[index : index + MAX_CUE_CHARS] for index in range(0, len(stripped), MAX_CUE_CHARS)]
        duration = max(word.end - word.start, 0.001)
        for index, piece in enumerate(pieces):
            start = word.start + duration * index / len(pieces)
            end = word.start + duration * (index + 1) / len(pieces)
            result.append(Word(start, end, piece, word.probability))
    return result


def _needs_space(previous: str, current: str) -> bool:
    return previous.isascii() and previous.isalnum() and current.isascii() and current.isalnum()


def _match_seconds(values: tuple[str, ...]) -> float:
    hours, minutes, seconds, millis = (int(value) for value in values)
    return hours * 3600 + minutes * 60 + seconds + millis / 1000
