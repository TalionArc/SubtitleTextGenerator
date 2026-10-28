from __future__ import annotations

from pathlib import Path

from subtitle_text_generator.corrections import (
    apply_corrections,
    find_soundalikes,
    output_lines,
    parse_corrections,
)
from subtitle_text_generator.models import Correction, Segment, Transcript, Word
from subtitle_text_generator.subtitles import render_txt
from subtitle_text_generator.transcript import transcript_words


def _transcript(sentence: str) -> Transcript:
    words = [
        Word(index * 0.5, index * 0.5 + 0.4, f" {token}", 0.9)
        for index, token in enumerate(sentence.split())
    ]
    return Transcript([Segment(0.0, words[-1].end, sentence, words)], duration=30.0)


def _text(transcript: Transcript) -> str:
    return render_txt(transcript, include_timestamps=False).strip()


def test_parse_reads_arrows_and_skips_unusable_lines() -> None:
    corrections = parse_corrections(
        "파이선 → 파이썬\n"
        "디지털 트원 -> 디지털 트윈\n"
        "opencv => OpenCV\n"
        "# 메모\n"
        "화살표 없음\n"
        "→ 파이썬\n"
        "파이썬 → 파이썬\n"
        "파이선 → 파이손\n"
        "한 두 세 네 다섯 → 너무 김\n"
    )
    assert corrections == (
        Correction(("파이선",), "파이썬"),
        Correction(("디지털", "트원"), "디지털 트윈"),
        Correction(("opencv",), "OpenCV"),
    )


def test_wrong_word_is_replaced_with_particles_and_punctuation_kept() -> None:
    transcript = _transcript("파이선은 파이선. 파이선의, 파이선스 그리고 오파이선")
    corrected, counts = apply_corrections(transcript, parse_corrections("파이선 → 파이썬"))
    assert _text(corrected) == "파이썬은 파이썬.\r\n파이썬의, 파이선스 그리고 오파이선"
    assert counts == {"파이선 → 파이썬": 3}


def test_longer_word_starting_with_the_wrong_form_is_kept() -> None:
    transcript = _transcript("자바를 배우고 자바스크립트도 씁니다")
    corrected, counts = apply_corrections(transcript, parse_corrections("자바 → Java"))
    assert _text(corrected) == "Java를 배우고 자바스크립트도 씁니다"
    assert counts == {"자바 → Java": 1}


def test_particles_follow_the_final_consonant_of_the_right_form() -> None:
    transcript = _transcript("테스트를 테스트는 테스트로 테스트입니다 테스트와")
    corrected, _ = apply_corrections(transcript, parse_corrections("테스트 → 시험"))
    assert _text(corrected) == "시험을 시험은 시험으로 시험입니다 시험과"
    back, _ = apply_corrections(corrected, parse_corrections("시험 → 기술"))
    assert _text(back) == "기술을 기술은 기술로 기술입니다 기술과"
    plain, _ = apply_corrections(back, parse_corrections("기술 → 카메라"))
    assert _text(plain) == "카메라를 카메라는 카메라로 카메라입니다 카메라와"
    polite, _ = apply_corrections(_transcript("시험이요 테스트요"), parse_corrections("시험 → 테스트"))
    assert _text(polite) == "테스트요 테스트요"
    polite, _ = apply_corrections(polite, parse_corrections("테스트 → 시험"))
    assert _text(polite) == "시험이요 시험이요"


def test_multi_word_and_different_word_counts_are_replaced() -> None:
    transcript = _transcript("디지털 트원, 그리고 디지털트원을 씁니다")
    corrected, counts = apply_corrections(
        transcript, parse_corrections("디지털 트원 → 디지털 트윈\n디지털트원 → 디지털 트윈")
    )
    assert _text(corrected) == "디지털 트윈, 그리고 디지털 트윈을 씁니다"
    assert counts == {"디지털 트원 → 디지털 트윈": 1, "디지털트원 → 디지털 트윈": 1}
    words = transcript_words(corrected)
    assert words[3].text == " 디지털 트윈을"
    assert (words[3].start, words[3].end) == (1.5, 1.9)


def test_one_word_becoming_several_keeps_word_order() -> None:
    words = [Word(0.0, 0.02, " 에스브이엠", 0.9), Word(0.02, 0.04, " 다음", 0.9)]
    transcript = Transcript([Segment(0.0, 0.04, "에스브이엠 다음", words)], duration=1.0)
    corrected, _ = apply_corrections(transcript, parse_corrections("에스브이엠 → 서포트 벡터 머신"))
    assert _text(corrected) == "서포트 벡터 머신 다음"


def test_wrong_form_with_inner_punctuation_matches() -> None:
    assert parse_corrections("e-mail → 이메일") == (Correction(("e-mail",), "이메일"),)
    corrected, counts = apply_corrections(_transcript("e-mail은 보냈습니다"), parse_corrections("e-mail → 이메일"))
    assert _text(corrected) == "이메일은 보냈습니다"
    assert counts == {"e-mail → 이메일": 1}


def test_latin_wrong_form_matches_without_case() -> None:
    corrected, _ = apply_corrections(_transcript("OPENCV를 씁니다"), parse_corrections("opencv → OpenCV"))
    assert _text(corrected) == "OpenCV를 씁니다"
    already = _transcript("OpenCV를 씁니다")
    unchanged, counts = apply_corrections(already, parse_corrections("opencv → OpenCV"))
    assert unchanged is already
    assert not counts


def test_words_split_by_punctuation_are_not_joined() -> None:
    transcript = _transcript("디지털, 트원")
    corrected, counts = apply_corrections(transcript, parse_corrections("디지털 트원 → 디지털 트윈"))
    assert corrected is transcript
    assert not counts


def test_soundalikes_list_misheard_forms_and_skip_particles() -> None:
    lines = [
        "파이선을 설치합니다. 파이선의 버전",
        "파이썬은 바른 꼴입니다",
        "모델을 만들고 모델로 봅니다. 모덴링도 봅니다",
        "디지털 트원과 디지털트원",
        "도카는 이미 고쳐 쓰기에 있습니다",
    ]
    found = find_soundalikes(
        ["파이썬", "모델링", "디지털 트윈", "도커", "A", "OpenCV"], lines, skip=["도카"]
    )
    assert [(item.term, item.wrong, item.count) for item in found] == [
        ("파이썬", "파이선", 2),
        ("모델링", "모덴링", 1),
        ("디지털 트윈", "디지털 트원", 1),
        ("디지털 트윈", "디지털트원", 1),
    ]
    assert found[0].example == "파이선을 설치합니다. 파이선의 버전"


def test_soundalikes_cover_liaison_letter_case_and_a_dropped_syllable() -> None:
    lines = [
        "살암이 왔다",
        "ai모댈을 봅니다",
        "파썬으로 짭니다",
        "Ai모델도 봅니다",
        "텍스트를 고치고 텍스트뷰어를 엽니다",
    ]
    found = find_soundalikes(["사람", "AI모델", "파이썬", "텍스트뷰"], lines)
    # 용어의 앞부분("텍스트")과 용어 뒤에 글자가 붙은 꼴("텍스트뷰어")은 다른 낱말이라 빼고,
    # 대소문자만 다른 꼴("Ai모델")도 뺍니다.
    assert [(item.term, item.wrong) for item in found] == [
        ("사람", "살암"),
        ("AI모델", "ai모댈"),
        ("파이썬", "파썬"),
    ]


def test_output_lines_drop_timestamps_and_srt_numbers(tmp_path: Path) -> None:
    txt = tmp_path / "강의 1강.txt"
    txt.write_text(
        chr(0xFEFF) + "[00:00:01.000 - 00:00:02.000] 첫 줄\r\n두 번째 줄\r\n", encoding="utf-8"
    )
    srt = tmp_path / "강의 1강.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n자막 줄\n\n", encoding="utf-8")
    assert output_lines(txt) == ["첫 줄", "두 번째 줄"]
    assert output_lines(srt) == ["자막 줄"]
    assert output_lines(tmp_path / "없음.txt") == []
    ass = tmp_path / "강의 1강.ass"
    ass.write_text(
        "[Script Info]\nTitle: 강의\n[Events]\nFormat: Layer, Start, End, Style, Text\n"
        "Style: Default,Arial,20\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\i1}첫 대사{\\i0}\\N둘째 줄, 쉼표\n",
        encoding="utf-8",
    )
    assert output_lines(ass) == ["첫 대사 둘째 줄, 쉼표"]
