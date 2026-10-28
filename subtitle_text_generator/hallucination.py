from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass

from .models import Transcript, Word
from .transcript import transcript_from_words, transcript_words

# 말이 없거나 잡음뿐인 구간(쉬는 시간 잡담, 멀리서 들리는 소리)에서 Whisper가 지어내는 문장을 거릅니다.
# 문장 부호나 이 길이 이상의 쉼에서 말 덩어리를 나눕니다(TXT 줄 나누기와 같은 기준).
CHUNK_PAUSE_SECONDS = 1.5
# 알려진 헛소리 문구는 아래 조건을 모두 만족할 때만 실제로 한 말(영상 끝인사 등)로 보고 남깁니다.
# 강의 녹음의 헛소리는 90%가 단어 확률 평균 0.62 아래였고, 정상 문장은 중앙값이 0.81이었습니다.
# 확률이 높은 헛소리도 첫 낱말이 0.01처럼 아주 낮거나, 낱말 하나가 몇 초씩 늘어지거나,
# 앞뒤가 조용한 곳에 혼자 있거나, 바로 앞 헛소리에 이어 나왔습니다.
PHRASE_KEEP_MEAN_PROBABILITY = 0.70
PHRASE_KEEP_MIN_PROBABILITY = 0.30
PHRASE_MAX_WORD_SECONDS = 2.5
PHRASE_ISOLATION_SECONDS = 5.0
PHRASE_FOLLOW_SECONDS = 3.0
# 같은 낱말 묶음이 덩어리 안에서 연달아 이만큼 반복되면 한 번만 남깁니다(한 낱말은 5번, 두 낱말 이상은 4번).
# "이거 하지마, 이거 하지마, 이거 하지마"처럼 강의에서 실제로 세 번 되풀이하는 말은 남깁니다.
LOOP_SINGLE_WORD_REPEATS = 5
LOOP_PHRASE_REPEATS = 4
LOOP_MAX_WORDS = 8
# 똑같은 짧은 덩어리가 이 간격 안으로 연달아 이만큼 나오면 첫 번째만 남깁니다.
REPEATED_CHUNK_RUN = 4
REPEATED_CHUNK_MAX_WORDS = 6
REPEATED_CHUNK_MAX_GAP_SECONDS = 10.0

_ENDING_PUNCTUATION = frozenset(".!?…。！？")
_KEY_RE = re.compile(r"[^0-9A-Za-z가-힣]+")


@dataclass(frozen=True, slots=True)
class _PhraseFamily:
    vocabulary: frozenset[str]
    # 문구를 알아보게 하는 낱말. "있습니다", "다음"처럼 흔한 낱말은 빠집니다.
    distinctive: frozenset[str]
    anchors: frozenset[str]
    # "never": 실제 말로 나올 일이 없음, "video": 영상 끝인사로만 나올 수 있음, "any": 어디서나 나올 수 있음
    spoken_in: str


def _family(distinctive: str, common: str, anchors: str, spoken_in: str) -> _PhraseFamily:
    strong = frozenset(distinctive.split())
    return _PhraseFamily(strong | frozenset(common.split()), strong, frozenset(anchors.split()), spoken_in)


# 덩어리가 한 묶음의 낱말로만 이루어지고 그 묶음의 핵심 낱말을 포함하면 헛소리 문구로 봅니다.
# 일부만 나오거나 뒤섞인 꼴("자막 제공 및 자막 제공 및 광고를")까지 잡기 위해 문장 대신 낱말로 비교합니다.
# 강의 녹음에서는 실제 말 끝에 바로 붙어 확률도 높게 나온 "영상에서 만나요"가 있어,
# 녹음(TXT)에서는 영상 끝인사 문구를 조건 없이 지웁니다.
# 자막 제작 문구는 "자막 제공 및 광고를…"과 "자막 제공 및 협조해 주신 모든 분들께 진심으로 감사드립니다"
# 두 꼴이 섞여 나오고, 앞이 잘린 "제공 주신 모든 분들께…", "및 주신 모든 분들께…"도 나왔습니다.
# "진심으로 감사드립니다."처럼 실제 인사일 수 있는 덩어리는 자막·제공·및이 없으면 건드리지 않습니다.
_PHRASE_FAMILIES = (
    _family(
        "자막 한글자막 제공 광고를 포함하고 협조 협조해 협조해주신 분들께 분께 진심으로",
        "및 영상 있습니다 주신 주셔서 모든 감사드립니다 감사합니다",
        "자막 한글자막 제공 및 광고를 포함하고",
        "never",
    ),
    _family("영상에서 만나요", "다음 영상", "영상에서 만나요", "video"),
    _family("시청해 시청해주셔서 주셔서 감사합니다", "", "시청해 시청해주셔서", "video"),
    _family(
        "구독 구독과 좋아요 알림 알림설정 부탁드립니다 눌러주세요", "설정", "구독 구독과", "video"
    ),
    _family("잠시 쉬고", "있습니다", "쉬고", "any"),
    _family("우리는 돌아왔습니다", "그리고 다시", "돌아왔습니다", "any"),
    _family("시간에 뵙겠습니다", "다음", "뵙겠습니다", "any"),
)
# 문구를 알아보게 하는 낱말이 이만큼 있어야 조건 없이 뺍니다. "만나요.", "구독.", "자막 있습니다."처럼
# 한 낱말만 겹치는 덩어리는 실제 대답일 수 있어 확률·맥락으로 판단합니다.
_UNCONDITIONAL_DISTINCTIVE_WORDS = 2


@dataclass(frozen=True, slots=True)
class RemovedText:
    start: float
    text: str
    reason: str


def filter_hallucinations(
    transcript: Transcript, *, video: bool
) -> tuple[Transcript, list[RemovedText]]:
    """알려진 헛소리 문구와 같은 말 반복을 뺀 결과와 뺀 내용을 돌려줍니다."""
    words = transcript_words(transcript)
    removed: list[RemovedText] = []
    chunks: list[list[Word]] = []
    split = _split_chunks(words)
    # 헛소리 문구끼리는 서로를 "앞뒤의 말"로 쳐 주지 않도록, 조용한지는 문구가 아닌 덩어리로만 잽니다.
    spoken = [index for index, chunk in enumerate(split) if _phrase_family(chunk) is None]
    last_phrase_end: float | None = None
    for index, chunk in enumerate(split):
        gap_before, gap_after = _gaps_to_speech(split, spoken, index)
        follows_phrase = (
            last_phrase_end is not None and chunk[0].start - last_phrase_end <= PHRASE_FOLLOW_SECONDS
        )
        if _is_phrase_hallucination(chunk, video, gap_before, gap_after, follows_phrase):
            removed.append(RemovedText(chunk[0].start, _join(chunk), "헛소리 문구"))
            last_phrase_end = chunk[-1].end
            continue
        kept, loops = _collapse_loops(chunk)
        removed.extend(loops)
        if kept:
            chunks.append(kept)
    chunks = _drop_repeated_chunks(chunks, removed)
    if not removed:
        return transcript, []
    kept_words = [word for chunk in chunks for word in chunk]
    filtered = transcript_from_words(kept_words, duration=transcript.duration, source=transcript.source)
    removed.sort(key=lambda item: item.start)
    return filtered, removed


def _split_chunks(words: list[Word]) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current and word.start - current[-1].end >= CHUNK_PAUSE_SECONDS:
            chunks.append(current)
            current = []
        current.append(word)
        if word.text.strip()[-1:] in _ENDING_PUNCTUATION:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def _key(word: Word) -> str:
    return _KEY_RE.sub("", word.text).casefold()


def _phrase_family(chunk: list[Word]) -> _PhraseFamily | None:
    keys = [key for key in map(_key, chunk) if key]
    if not keys:
        return None
    for family in _PHRASE_FAMILIES:
        if all(key in family.vocabulary for key in keys) and any(key in family.anchors for key in keys):
            return family
    return None


def _gaps_to_speech(
    split: list[list[Word]], spoken: list[int], index: int
) -> tuple[float | None, float | None]:
    position = bisect_left(spoken, index)
    before = spoken[position - 1] if position > 0 else None
    after_position = position + 1 if position < len(spoken) and spoken[position] == index else position
    after = spoken[after_position] if after_position < len(spoken) else None
    chunk = split[index]
    gap_before = chunk[0].start - split[before][-1].end if before is not None else None
    gap_after = split[after][0].start - chunk[-1].end if after is not None else None
    return gap_before, gap_after


def _is_phrase_hallucination(
    chunk: list[Word],
    video: bool,
    gap_before: float | None,
    gap_after: float | None,
    follows_phrase: bool,
) -> bool:
    family = _phrase_family(chunk)
    if family is None:
        return False
    keys = [key for key in map(_key, chunk) if key]
    distinctive = len({key for key in keys if key in family.distinctive})
    if distinctive >= _UNCONDITIONAL_DISTINCTIVE_WORDS and (
        family.spoken_in == "never" or (family.spoken_in == "video" and not video)
    ):
        return True
    return not _looks_spoken(chunk, keys, family, gap_before, gap_after, follows_phrase)


def _looks_spoken(
    chunk: list[Word],
    keys: list[str],
    family: _PhraseFamily,
    gap_before: float | None,
    gap_after: float | None,
    follows_phrase: bool,
) -> bool:
    if max(keys.count(anchor) for anchor in family.anchors) >= 2 or follows_phrase:
        return False
    probabilities = [word.probability for word in chunk if word.probability is not None]
    if probabilities and (
        sum(probabilities) / len(probabilities) < PHRASE_KEEP_MEAN_PROBABILITY
        or min(probabilities) < PHRASE_KEEP_MIN_PROBABILITY
    ):
        return False
    if any(word.end - word.start > PHRASE_MAX_WORD_SECONDS for word in chunk):
        return False
    quiet_before = gap_before is None or gap_before >= PHRASE_ISOLATION_SECONDS
    quiet_after = gap_after is None or gap_after >= PHRASE_ISOLATION_SECONDS
    return not (quiet_before and quiet_after)


def _collapse_loops(chunk: list[Word]) -> tuple[list[Word], list[RemovedText]]:
    keys = [_key(word) for word in chunk]
    kept: list[Word] = []
    removed: list[RemovedText] = []
    index = 0
    while index < len(chunk):
        loop = _loop_at(keys, index)
        if loop is None:
            kept.append(chunk[index])
            index += 1
            continue
        size, repeats = loop
        kept.extend(chunk[index : index + size])
        removed.append(
            RemovedText(chunk[index].start, f"{_join(chunk[index : index + size])} ×{repeats}", "반복")
        )
        index += size * repeats
    return kept, removed


def _loop_at(keys: list[str], index: int) -> tuple[int, int] | None:
    for size in range(1, LOOP_MAX_WORDS + 1):
        unit = keys[index : index + size]
        if len(unit) < size or not all(unit):
            return None
        repeats = 1
        while keys[index + size * repeats : index + size * (repeats + 1)] == unit:
            repeats += 1
        needed = LOOP_SINGLE_WORD_REPEATS if size == 1 else LOOP_PHRASE_REPEATS
        if repeats >= needed:
            return size, repeats
    return None


def _drop_repeated_chunks(chunks: list[list[Word]], removed: list[RemovedText]) -> list[list[Word]]:
    result: list[list[Word]] = []
    index = 0
    while index < len(chunks):
        key = _chunk_key(chunks[index])
        end = index + 1
        if key:
            while (
                end < len(chunks)
                and _chunk_key(chunks[end]) == key
                and chunks[end][0].start - chunks[end - 1][-1].end <= REPEATED_CHUNK_MAX_GAP_SECONDS
            ):
                end += 1
        result.append(chunks[index])
        if end - index >= REPEATED_CHUNK_RUN:
            removed.append(
                RemovedText(chunks[index][0].start, f"{_join(chunks[index])} ×{end - index}", "반복")
            )
            index = end
        else:
            result.extend(chunks[index + 1 : end])
            index = end
    return result


def _chunk_key(chunk: list[Word]) -> tuple[str, ...]:
    keys = tuple(key for key in map(_key, chunk) if key)
    return keys if len(keys) <= REPEATED_CHUNK_MAX_WORDS else ()


def _join(words: list[Word]) -> str:
    return re.sub(r"\s+", " ", "".join(word.text for word in words)).strip()
