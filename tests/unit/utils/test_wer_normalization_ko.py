"""Korean WER/CER normalization.

Two properties: the reference (intended TTS text) and the STT transcript
normalize to the same string when they say the same thing, and names and
ordinary words are not turned into numbers.
"""

import pytest

from eva.metrics.diagnostic.stt_wer import STTWERMetric
from eva.utils.wer_normalization import normalize_text
from tests.unit.metrics.conftest import make_metric_context


def _ko(s: str) -> str:
    return normalize_text(s, "ko")


@pytest.mark.parametrize(
    "spoken,written",
    [
        # the day counter 일 must not be read as the digit 1
        ("삼월 이십오일", "3월 25일"),
        ("삼월 이십일일", "3월 21일"),
        ("삼월 이십일", "3월 20일"),
        ("삼월 일일", "3월 1일"),
        ("십이월 삼십일일", "12월 31일"),
        ("이천이십육년 삼월 이십오일", "2026년 3월 25일"),
        # irregular month names
        ("유월", "6월"),
        ("시월 삼일", "10월 3일"),
        ("십일월", "11월"),
    ],
)
def test_dates(spoken: str, written: str):
    assert _ko(spoken) == _ko(written)


def test_date_output_form():
    assert _ko("삼월 이십오일") == "3월 25일"


@pytest.mark.parametrize(
    "spoken,written",
    [
        # 만-grouping, Arabic digits combined with units, thousands separators
        ("만 이천 원", "12,000원"),
        ("만 이천 원", "12000원"),
        ("1만 2천 원", "12000원"),
        ("1만 2천 원", "12,000 원"),
        ("삼만 오천 원", "35000원"),
        ("천 원", "1000원"),
        ("백 명", "100명"),
        ("이십 분", "20분"),
    ],
)
def test_amounts(spoken: str, written: str):
    assert _ko(spoken) == _ko(written)


@pytest.mark.parametrize(
    "spoken,written",
    [
        ("공일공 일이삼사 오육칠팔", "010-1234-5678"),
        ("공일공 일이삼사", "010 1234"),
        ("일 이 삼", "123"),
        ("번호는 일 이 삼입니다", "번호는 123입니다"),
        ("일이삼사", "1234"),
    ],
)
def test_digit_strings(spoken: str, written: str):
    assert _ko(spoken) == _ko(written)


@pytest.mark.parametrize(
    "spoken,written",
    [
        ("열두 시", "12시"),
        ("열두 시 삼십 분", "12시 30분"),
        ("세 시", "3시"),
        ("네 명", "4명"),
        ("한 개", "1개"),
        ("두 시간", "2시간"),
        ("두 개를", "2개를"),
        ("세 명이요", "3명이요"),
        ("세 대가", "3대가"),
        ("스무 살", "20살"),
        ("스물한 살", "21살"),
        ("열 시", "10시"),
    ],
)
def test_native_counters(spoken: str, written: str):
    assert _ko(spoken) == _ko(written)


@pytest.mark.parametrize(
    "text",
    [
        # surnames 천 (Cheon), 백 (Baek), 조 (Jo)
        "천민수",
        "천 씨",
        "백지수",
        "백 선생님",
        "조 씨",
        # ordinary words that contain number syllables
        "만나서 반갑습니다",
        "조금만 기다려 주세요",
        "천천히 말씀해 주세요",
        "백화점",
        "만약에",
        "열심히 하겠습니다",
        "열차",
        "인터넷",
        "넷플릭스",
        "리셋",
        "하나님",
        "이 책",
        "일하다",
        "이번 주",
        "사회",
        "공원",
        "사원증",
        "이분이 담당자예요",
        "구분",
        "이동",
        "공사",
        "사이에",
        "일일이 확인",
        "이삼일",
        "네 알겠습니다",
        "네, 시간 있어요",
        "에 대한 시간",
        "한국",
        "세계",
        # 한/두/세/네 before a word that only starts like a counter
        "네 개인정보 확인해 주세요",
        "네 시작할게요",
        "네 가지고 있어요",
        "네 번호는 알려드릴게요",
        "한 번호를 알려주세요",
        "두 대리님",
        "세대 차이",
    ],
)
def test_names_and_ordinary_words_unchanged(text: str):
    assert not any(c.isdigit() for c in _ko(text)), _ko(text)


def test_surname_output():
    assert _ko("천민수") == "천민수"
    assert _ko("천 씨") == "천 씨"
    assert _ko("백지수") == "백지수"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("십", "10"),
        ("삼백이십일", "321"),
        ("이천이십사", "2024"),
        ("만이천삼백사십오", "12345"),
        ("일억", "100000000"),
        ("열다섯", "15"),
        ("스물하나", "21"),
        ("열 하나", "11"),
        ("아흔아홉", "99"),
        ("고객은 스물하나입니다", "고객은 21입니다"),
    ],
)
def test_existing_conversions_kept(text: str, expected: str):
    assert _ko(text) == expected


def test_fillers_only_removed_as_whole_words():
    assert _ko("음 다음 주에 음식을") == "다음 주에 음식을"
    assert _ko("영어로 열어 주세요") == "영어로 열어 주세요"


def test_spaces_between_words_kept():
    assert _ko("예약을 변경하고 싶어요") == "예약을 변경하고 싶어요"


@pytest.mark.asyncio
async def test_metric_scores_spoken_date_as_correct():
    metric = STTWERMetric(config={"language": "ko"})
    ctx = make_metric_context(
        intended_user_turns={1: "삼월 이십오일에 예약하고 싶어요"},
        transcribed_user_turns={1: "3월 25일에 예약하고 싶어요"},
    )
    result = await metric.compute(ctx)
    assert result.details["cer"] == 0.0
