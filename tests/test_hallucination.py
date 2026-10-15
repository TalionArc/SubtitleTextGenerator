from __future__ import annotations

from subtitle_text_generator.hallucination import filter_hallucinations
from subtitle_text_generator.models import Segment, Transcript, Word
from subtitle_text_generator.subtitles import render_txt
from subtitle_text_generator.transcript import transcript_words


def _transcript(*runs: tuple[float, str, float]) -> Transcript:
    """(시작 시각, 띄어 쓴 문장, 단어 확률)마다 단어를 0.4초씩 이어 붙입니다."""
    segments: list[Segment] = []
    for start, sentence, probability in runs:
        words: list[Word] = []
        cursor = start
        for token in sentence.split():
            words.append(Word(cursor, cursor + 0.4, f" {token}", probability))
            cursor += 0.4
        segments.append(Segment(start, cursor, sentence, words))
    return Transcript(segments, duration=max(segment.end for segment in segments) + 10)


def _text(transcript: Transcript) -> str:
    return render_txt(transcript, include_timestamps=False)


def test_clean_transcript_is_returned_unchanged() -> None:
    transcript = _transcript((0.0, "오늘은 3장 내용을 봅니다.", 0.9), (3.0, "질문 있나요?", 0.9))
    filtered, removed = filter_hallucinations(transcript, video=False)
    assert filtered is transcript
    assert removed == []


def test_known_phrase_is_removed_from_recordings_even_when_confident() -> None:
    transcript = _transcript(
        (0.0, "코드만 캡처하면 됩니다.", 0.95),
        (2.0, "영상에서 만나요.", 0.95),
        (4.0, "다음 시간에는 객체를 봅니다.", 0.95),
    )
    filtered, removed = filter_hallucinations(transcript, video=False)
    assert _text(filtered) == "코드만 캡처하면 됩니다.\r\n다음 시간에는 객체를 봅니다.\r\n"
    assert [item.reason for item in removed] == ["헛소리 문구"]


def test_video_keeps_a_confident_closing_line_attached_to_speech() -> None:
    transcript = _transcript(
        (0.0, "오늘 준비한 내용은 여기까지입니다.", 0.95),
        (2.5, "다음 영상에서 만나요!", 0.92),
    )
    filtered, removed = filter_hallucinations(transcript, video=True)
    assert filtered is transcript
    assert removed == []


def test_video_still_removes_closing_lines_that_look_made_up() -> None:
    low_first_word = _transcript((0.0, "설명은 여기까지입니다.", 0.95), (2.0, "다음 영상에서 만나요.", 0.95))
    low_first_word.segments[1].words[0] = Word(2.0, 2.4, " 다음", 0.01)
    isolated = _transcript((0.0, "설명은 여기까지입니다.", 0.95), (30.0, "다음 영상에서 만나요.", 0.95))
    stretched = _transcript((0.0, "설명은 여기까지입니다.", 0.95), (2.0, "만나요.", 0.95))
    stretched.segments[1].words[0] = Word(2.0, 7.0, " 만나요.", 0.97)
    repeated = _transcript((0.0, "설명은 여기까지입니다.", 0.95), (2.0, "영상에서 만나요 영상에서 만나요.", 0.95))
    for transcript in (low_first_word, isolated, stretched, repeated):
        filtered, removed = filter_hallucinations(transcript, video=True)
        assert _text(filtered) == "설명은 여기까지입니다.\r\n"
        assert len(removed) == 1


def test_subtitle_credit_is_removed_even_from_videos() -> None:
    transcript = _transcript(
        (0.0, "설명은 여기까지입니다.", 0.95),
        (2.0, "자막 제공 및 자막 제공 및 광고를 포함하고 있습니다.", 0.95),
    )
    filtered, _ = filter_hallucinations(transcript, video=True)
    assert _text(filtered) == "설명은 여기까지입니다.\r\n"
    spoken = _transcript((0.0, "이 화면 아래쪽에 나오는 게 뭐죠?", 0.95), (2.5, "자막.", 0.95))
    filtered, removed = filter_hallucinations(spoken, video=True)
    assert filtered is spoken
    assert removed == []


def test_phrase_words_inside_a_real_sentence_are_kept() -> None:
    transcript = _transcript(
        (0.0, "다음 영상에서 보여 드릴게요.", 0.5),
        (2.0, "잠시 쉬고 하겠습니다.", 0.5),
        (4.0, "다음.", 0.5),
    )
    filtered, removed = filter_hallucinations(transcript, video=False)
    assert filtered is transcript
    assert removed == []


def test_follow_up_fragment_after_a_removed_phrase_is_removed() -> None:
    transcript = _transcript(
        (0.0, "설명은 여기까지입니다.", 0.95),
        (2.0, "다음 영상에서 만나요.", 0.4),
        (3.5, "뵙겠습니다.", 0.95),
    )
    filtered, removed = filter_hallucinations(transcript, video=True)
    assert _text(filtered) == "설명은 여기까지입니다.\r\n"
    assert len(removed) == 2


def test_word_loops_collapse_to_one_occurrence() -> None:
    transcript = _transcript(
        (0.0, "목이 든다 " + "더 길게, " * 9, 0.6),
        (10.0, "저, " * 30 + "그래서요.", 0.6),
    )
    filtered, removed = filter_hallucinations(transcript, video=False)
    # 지운 반복 자리는 빈 시간으로 남아 TXT 줄이 나뉩니다.
    assert _text(filtered) == "목이 든다 더 길게,\r\n저,\r\n그래서요.\r\n"
    assert [item.text for item in removed] == ["더 길게, ×9", "저, ×30"]
    kept = transcript_words(filtered)
    assert kept[2].start == 0.8


def test_natural_short_repeats_are_kept() -> None:
    transcript = _transcript(
        (0.0, "네 네 네 맞아요.", 0.9),
        (3.0, "아니 아니 아니 아니 그게 아니라.", 0.9),
        (6.0, "너 이거 하지마, 이거 하지마, 이거 하지마, 다 이렇게 말하죠.", 0.9),
        (12.0, "잠시만요.", 0.9),
        (14.0, "잠시만요.", 0.9),
        (16.0, "잠시만요.", 0.9),
    )
    filtered, removed = filter_hallucinations(transcript, video=False)
    assert filtered is transcript
    assert removed == []


def test_same_short_line_repeated_in_a_row_keeps_the_first() -> None:
    transcript = _transcript(
        (0.0, "식전.", 0.8),
        (2.0, "식전.", 0.8),
        (4.0, "식전.", 0.8),
        (6.0, "식전.", 0.8),
        (8.0, "밥 먹고 왔어요.", 0.8),
        (10.0, "네.", 0.8),
        (12.0, "네.", 0.8),
    )
    filtered, removed = filter_hallucinations(transcript, video=False)
    assert _text(filtered) == "식전.\r\n밥 먹고 왔어요.\r\n네.\r\n네.\r\n"
    assert [item.text for item in removed] == ["식전. ×4"]


def test_repeated_phrase_loop_inside_a_chunk_is_removed_entirely() -> None:
    transcript = _transcript((0.0, "설명은 끝입니다.", 0.9), (2.0, "잠시 쉬고, " * 5, 0.9))
    filtered, _ = filter_hallucinations(transcript, video=False)
    assert _text(filtered) == "설명은 끝입니다.\r\n"


def test_single_word_answers_are_kept_in_recordings() -> None:
    transcript = _transcript(
        (0.0, "그럼 금요일 세 시에 봬요.", 0.95),
        (2.5, "네.", 0.95),
        (3.5, "만나요.", 0.95),
        (5.0, "요금은 어떤 방식이에요?", 0.95),
        (7.0, "구독.", 0.95),
        (8.0, "월 만 원입니다.", 0.95),
    )
    filtered, removed = filter_hallucinations(transcript, video=False)
    assert filtered is transcript
    assert removed == []


def test_subtitle_word_in_a_real_answer_is_kept() -> None:
    transcript = _transcript(
        (0.0, "자막 있어요?", 0.95), (2.0, "자막 있습니다.", 0.95), (4.0, "좋네요.", 0.95)
    )
    filtered, removed = filter_hallucinations(transcript, video=True)
    assert filtered is transcript
    assert removed == []


def test_far_apart_identical_lines_are_not_a_loop() -> None:
    transcript = _transcript((0.0, "네.", 0.9), (60.0, "네.", 0.9), (120.0, "네.", 0.9), (180.0, "네.", 0.9))
    filtered, removed = filter_hallucinations(transcript, video=False)
    assert filtered is transcript
    assert removed == []


def test_removed_phrases_do_not_count_as_nearby_speech() -> None:
    transcript = _transcript(
        (0.0, "설명은 여기까지입니다.", 0.95),
        (40.0, "다음 영상에서 만나요.", 0.95),
        (42.0, "자막 제공 및 광고를 포함하고 있습니다.", 0.95),
    )
    filtered, removed = filter_hallucinations(transcript, video=True)
    assert _text(filtered) == "설명은 여기까지입니다.\r\n"
    assert len(removed) == 2
