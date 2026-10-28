from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .models import Correction, Transcript, Word
from .transcript import transcript_from_words, transcript_words

# 고쳐 쓰기는 인식이 끝난 글자만 바꾸므로, 인식을 흔드는 용어집(최대 50개)보다 넉넉히 받습니다.
MAX_CORRECTIONS = 1000
MAX_WRONG_WORDS = 4
SUGGESTIONS_PER_TERM = 20
_ARROWS = ("→", "->", "=>")
_WORD_PARTS = re.compile(r"^([^0-9A-Za-z가-힣]*)(.*?)([^0-9A-Za-z가-힣]*)$", re.S)
_KEY_RE = re.compile(r"[^0-9A-Za-z가-힣]+")
_HANGUL_RE = re.compile(r"[가-힣]")
_TXT_TIMESTAMP = re.compile(r"^\[[0-9:.]+ - [0-9:.]+\]\s*")
_SRT_TIMING = re.compile(r"-->")
_ASS_TAGS = re.compile(r"\{[^}]*\}")

# 낱말 뒤에 붙어도 같은 낱말로 보는 조사·어미입니다. 두 개까지 이어 붙은 꼴(에서는, 들의)도 받습니다.
# 이 밖의 글자가 붙으면("자바"에 대한 "자바스크립트") 다른 낱말로 보고 고치지 않습니다.
_PARTICLE_TEXT = """
    은 는 이 가 을 를 의 에 에서 에게 에다 에다가 께 께서 한테 으로 로 와 과 도 만 까지 부터 처럼 보다 마다
    이나 나 이랑 랑 하고 야 이야 요 이요 라는 이라는 란 이란 라고 이라고 라면 이라면 인 인데 인가
    인지 이고 고 이며 며 이다 다 입니다 이에요 예요 이죠 죠 들 서 엔 에선
"""
_PARTICLES = frozenset(_PARTICLE_TEXT.split())
# 받침 있는 말 뒤의 꼴과 받침 없는 말 뒤의 꼴. 바른 꼴의 받침에 맞춰 바꿉니다("테스트를" → "시험을").
_PARTICLE_PAIRS = (
    ("은", "는"), ("을", "를"), ("이", "가"), ("과", "와"), ("으로", "로"), ("이랑", "랑"),
    ("이라는", "라는"), ("이란", "란"), ("이라고", "라고"), ("이라면", "라면"), ("이야", "야"),
    ("이에요", "예요"), ("이죠", "죠"), ("이다", "다"), ("이며", "며"), ("이고", "고"), ("이나", "나"),
    ("이요", "요"),
)  # fmt: skip
_PARTICLE_FORMS = {form: pair for pair in _PARTICLE_PAIRS for form in pair}
# 비슷한 꼴의 마지막 글자부터가 조사로 읽히면("모델을", "모델로", "텍스트라는") 다른 낱말에
# 조사가 붙은 것이라 제안하지 않습니다. 용어 자체가 조사 같은 글자로 끝나면("아나콘다") 이 규칙을 쓰지 않습니다.

_CHOSEONG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNGSEONG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONGSEONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


@dataclass(frozen=True, slots=True)
class Suggestion:
    term: str
    wrong: str
    count: int
    example: str


def parse_corrections(text: str) -> tuple[Correction, ...]:
    """'틀린 꼴 → 바른 꼴' 줄을 읽습니다. 화살표가 없거나 양쪽이 같은 줄은 건너뜁니다."""
    result: list[Correction] = []
    seen: set[tuple[str, ...]] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        arrow = next((item for item in _ARROWS if item in line), None)
        if not line or line.startswith("#") or arrow is None:
            continue
        wrong_text, right_text = line.split(arrow, 1)
        # 낱말 앞뒤의 문장 부호만 떼어 "e-mail"처럼 안에 든 부호는 그대로 맞춰 봅니다.
        wrong = tuple(core for core in (_split(part)[1] for part in wrong_text.split()) if core)
        right = " ".join(right_text.split())
        folded = tuple(part.casefold() for part in wrong)
        if (
            not wrong
            or not _key(right)
            or len(wrong) > MAX_WRONG_WORDS
            or folded in seen
            or wrong == tuple(_split(part)[1] for part in right.split())
        ):
            continue
        seen.add(folded)
        result.append(Correction(wrong, right))
        if len(result) >= MAX_CORRECTIONS:
            break
    return tuple(result)


def format_correction(wrong: str, right: str) -> str:
    return f"{wrong} → {right}"


def apply_corrections(
    transcript: Transcript, corrections: Sequence[Correction]
) -> tuple[Transcript, Counter[str]]:
    """틀린 꼴을 바른 꼴로 바꾼 결과와 항목별로 바꾼 횟수를 돌려줍니다."""
    if not corrections:
        return transcript, Counter()
    words, counts = correct_words(transcript_words(transcript), corrections)
    if not counts:
        return transcript, counts
    corrected = transcript_from_words(words, duration=transcript.duration, source=transcript.source)
    return corrected, counts


def correct_words(
    words: Sequence[Word], corrections: Sequence[Correction]
) -> tuple[list[Word], Counter[str]]:
    # 긴 틀린 꼴을 먼저 맞춰 봐서 "디지털 트원"이 "디지털"보다 앞서게 합니다.
    by_first: dict[str, list[Correction]] = {}
    for item in sorted(corrections, key=lambda c: (-len(c.wrong), -sum(map(len, c.wrong)))):
        by_first.setdefault(item.wrong[0].casefold(), []).append(item)
    parts = [_split(word.text) for word in words]
    result: list[Word] = []
    counts: Counter[str] = Counter()
    index = 0
    while index < len(words):
        core = parts[index][1].casefold()
        matched: tuple[Correction, str] | None = None
        for length in range(len(core), 0, -1):
            for item in by_first.get(core[:length], ()):
                rest = _match(parts, index, item)
                if rest is not None:
                    matched = (item, rest)
                    break
            if matched:
                break
        if matched is None:
            result.append(words[index])
            index += 1
            continue
        item, rest = matched
        size = len(item.wrong)
        original = words[index : index + size]
        replaced = _replacement(original, parts[index : index + size], item, rest)
        # 대소문자만 고치는 항목("opencv → OpenCV")이 이미 바른 꼴을 만나면 센 것으로 치지 않습니다.
        if [word.text for word in replaced] == [word.text for word in original]:
            result.extend(original)
        else:
            result.extend(replaced)
            counts[item.label] += 1
        index += size
    return result, counts


def find_soundalikes(
    terms: Iterable[str], lines: Iterable[str], *, skip: Iterable[str] = ()
) -> list[Suggestion]:
    """바른 용어와 자모 한두 개만 다른 꼴을 찾아 많이 나온 순서로 돌려줍니다.

    "모델"과 "모텔"처럼 진짜 낱말도 함께 걸리므로, 무엇을 고칠지는 사람이 고릅니다.
    """
    prepared = _prepare_terms(terms)
    if not prepared:
        return []
    skipped = {_key(item).casefold() for item in skip}
    skipped.update(term.folded for term in prepared)
    by_length: dict[int, list[_Term]] = {}
    for term in prepared:
        for length in term.lengths:
            by_length.setdefault(length, []).append(term)
    # 같은 낱말 묶음은 한 번만 비교하도록 먼저 셉니다. 여러 낱말 용어("디지털 트윈")만 묶음을 봅니다.
    max_span = max(term.words for term in prepared)
    sequences: Counter[tuple[str, ...]] = Counter()
    sequence_examples: dict[tuple[str, ...], str] = {}
    for line in lines:
        tokens = line.split()
        parts = [_split(token) for token in tokens]
        for start in range(len(tokens)):
            for span in range(1, max_span + 1):
                end = start + span
                if end > len(tokens) or not parts[end - 1][1]:
                    break
                if span > 1 and (parts[end - 2][2] or parts[end - 1][0].strip()):
                    break
                sequence = tuple(part[1] for part in parts[start:end])
                sequences[sequence] += 1
                if sequence not in sequence_examples:
                    sequence_examples[sequence] = _example(line, tokens[start])
    counts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], str] = {}
    for sequence, occurrences in sequences.items():
        prefix_length = sum(len(piece) for piece in sequence[:-1])
        joined = "".join(sequence)
        for length in range(prefix_length + 1, len(joined) + 1):
            candidates = by_length.get(length)
            if not candidates or not _particle_chain(joined[length:]):
                continue
            head = joined[:length]
            folded = head.casefold()
            head_jamo = _jamo(head)
            head_letters = set(head_jamo)
            for term in candidates:
                if term.words < len(sequence) or folded == term.folded:
                    continue
                # 자모 수 차이와 한쪽에만 있는 자모 수는 거리보다 클 수 없으므로, 이것으로 걸러도
                # 놓치는 꼴이 없습니다.
                if abs(len(head_jamo) - len(term.jamo)) > term.limit:
                    continue
                if sum(letter not in term.letters for letter in head_jamo) > term.limit:
                    continue
                if sum(letter not in head_letters for letter in term.jamo) > term.limit:
                    continue
                # 용어의 앞부분("텍스트뷰"의 "텍스트")이나 용어 뒤에 글자가 더 붙은 꼴은 다른 낱말입니다.
                if term.folded.startswith(folded) or folded.startswith(term.folded):
                    continue
                if term.key[-1] not in _PARTICLES and _particle_chain(head[-1] + joined[length:]):
                    continue
                if _distance(head_jamo, term.jamo) > term.limit:
                    continue
                wrong = " ".join([*sequence[:-1], sequence[-1][: length - prefix_length]])
                if _key(wrong).casefold() in skipped:
                    continue
                key = (term.text, wrong)
                counts[key] += occurrences
                examples.setdefault(key, sequence_examples[sequence])
    order = {term.text: number for number, term in enumerate(prepared)}
    result: list[Suggestion] = []
    per_term: Counter[str] = Counter()
    for (term_text, wrong), count in sorted(
        counts.items(), key=lambda item: (order[item[0][0]], -item[1], item[0][1])
    ):
        if per_term[term_text] >= SUGGESTIONS_PER_TERM:
            continue
        per_term[term_text] += 1
        result.append(Suggestion(term_text, wrong, count, examples[(term_text, wrong)]))
    return result


def output_lines(path: Path) -> list[str]:
    """결과 파일에서 타임스탬프·번호·서식을 뺀 글 줄만 읽습니다(TXT·SRT·VTT·ASS·SSA)."""
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return []
    styled = path.suffix.casefold() in {".ass", ".ssa"}
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if styled:
            # Dialogue: 레이어,시작,끝,스타일,이름,여백 세 개,효과,대사
            if not line.startswith("Dialogue:"):
                continue
            line = _ASS_TAGS.sub("", line.split(",", 9)[-1])
            line = " ".join(line.replace("\\N", " ").replace("\\n", " ").split())
        else:
            line = _TXT_TIMESTAMP.sub("", line)
            if line.isdigit() or _SRT_TIMING.search(line) or line.upper() == "WEBVTT":
                continue
            if line.startswith("NOTE"):
                continue
        if line:
            lines.append(line)
    return lines


@dataclass(frozen=True, slots=True, eq=False)
class _Term:
    text: str
    key: str
    folded: str
    jamo: tuple[str, ...]
    letters: frozenset[str]
    words: int
    limit: int
    lengths: tuple[int, ...]


def _prepare_terms(terms: Iterable[str]) -> list[_Term]:
    prepared: list[_Term] = []
    seen: set[str] = set()
    for raw in terms:
        text = " ".join(raw.split())
        key = _key(text)
        # 한 글자 용어와 한글이 없는 용어는 비슷한 진짜 낱말이 너무 많아 찾지 않습니다.
        if len(key) < 2 or not _HANGUL_RE.search(key) or key.casefold() in seen:
            continue
        seen.add(key.casefold())
        limit = 1 if len(key) <= 2 else 2 if len(key) <= 4 else 3
        # 한 글자를 통째로 넣거나 빼면 자모가 두 개 이상 바뀌므로, 거리 2부터 글자 수 ±1도 봅니다
        # ("파이썬"과 "파썬"). 연음("사람"과 "살암")은 글자 수가 같아 그대로 잡힙니다.
        spread = limit // 2
        lengths = tuple(range(max(len(key) - spread, 2), len(key) + spread + 1))
        jamo = tuple(_jamo(key))
        prepared.append(
            _Term(text, key, key.casefold(), jamo, frozenset(jamo), len(text.split()), limit, lengths)
        )
    return prepared


def _match(parts: list[tuple[str, str, str]], index: int, item: Correction) -> str | None:
    size = len(item.wrong)
    if index + size > len(parts):
        return None
    for offset in range(size - 1):
        lead, core, trail = parts[index + offset]
        if core.casefold() != item.wrong[offset].casefold() or trail:
            return None
        if offset and lead.strip():
            return None
    lead, core, _ = parts[index + size - 1]
    if size > 1 and lead.strip():
        return None
    last = item.wrong[-1].casefold()
    if not core.casefold().startswith(last):
        return None
    rest = core[len(last) :]
    return rest if _particle_chain(rest) else None


def _replacement(
    matched: Sequence[Word], parts: Sequence[tuple[str, str, str]], item: Correction, rest: str
) -> list[Word]:
    right_words = item.right.split()
    trail = parts[-1][2]
    rest = _fit_particles(right_words[-1], rest)
    if len(right_words) == len(matched):
        return [
            Word(
                word.start,
                word.end,
                part[0] + text + (rest + trail if number == len(matched) - 1 else ""),
                word.probability,
            )
            for number, (word, part, text) in enumerate(zip(matched, parts, right_words, strict=True))
        ]
    # 낱말 수가 다르면 원래 낱말들이 차지하던 시간 그대로 한 덩어리로 둡니다. 시간을 쪼개면
    # 아주 짧은 낱말에서 뒤 낱말과 순서가 뒤바뀔 수 있습니다.
    probabilities = [word.probability for word in matched if word.probability is not None]
    probability = min(probabilities) if probabilities else None
    text = parts[0][0] + item.right + rest + trail
    return [Word(matched[0].start, matched[-1].end, text, probability)]


def _fit_particles(word: str, rest: str) -> str:
    """바로 붙는 첫 조사를 바른 꼴의 마지막 글자 받침에 맞춥니다."""
    code = ord(word[-1]) - 0xAC00 if word else -1
    if not rest or not 0 <= code < 11172:
        return rest
    final = code % 28
    for size in range(len(rest), 0, -1):
        pair = _PARTICLE_FORMS.get(rest[:size])
        if pair is None or not _particle_chain(rest[size:]):
            continue
        with_final, without_final = pair
        # ㄹ 받침 뒤에는 "으로" 대신 "로"를 씁니다("기술로").
        chosen = without_final if not final or (final == 8 and with_final == "으로") else with_final
        return chosen + rest[size:]
    return rest


def _split(text: str) -> tuple[str, str, str]:
    match = _WORD_PARTS.match(text)
    if match is None:
        return text, "", ""
    return match.group(1), match.group(2), match.group(3)


def _key(text: str) -> str:
    return _KEY_RE.sub("", text)


def _particle_chain(rest: str) -> bool:
    if not rest or rest in _PARTICLES:
        return True
    return any(rest[:cut] in _PARTICLES and rest[cut:] in _PARTICLES for cut in range(1, len(rest)))


def _jamo(text: str) -> list[str]:
    result: list[str] = []
    for character in text:
        code = ord(character) - 0xAC00
        if 0 <= code < 11172:
            result.append(_CHOSEONG[code // 588])
            result.append(_JUNGSEONG[code % 588 // 28])
            if code % 28:
                result.append(_JONGSEONG[code % 28])
        else:
            result.append(character.casefold())
    return result


def _distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, 1):
        current = [row]
        for column, right_item in enumerate(right, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _example(line: str, token: str, width: int = 70) -> str:
    position = max(line.find(token), 0)
    start = max(position - width // 3, 0)
    snippet = line[start : start + width]
    return ("…" if start else "") + snippet + ("…" if start + width < len(line) else "")
