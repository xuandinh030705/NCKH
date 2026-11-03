"""Kiểm tra sinh đề có seed: tái lập, 3 biến thể, đáp số do code tính."""
from __future__ import annotations

from tutor.problemgen import TOPIC_GRADES, generate, summary


def test_deterministic() -> None:
    a = generate(20, 42)
    b = generate(20, 42)
    assert [x.id for x in a] == [x.id for x in b]
    assert [x.text for x in a] == [x.text for x in b]
    assert [x.answer for x in a] == [x.answer for x in b]


def test_different_seed_differs() -> None:
    a = generate(20, 42)
    b = generate(20, 43)
    assert [x.text for x in a] != [x.text for x in b]


def test_variants_and_lang() -> None:
    for item in generate(30, 7, style="mixed"):
        assert set(item.variants) == {"vi", "vi_nodiac", "en"}
        assert item.text == item.variants[item.lang]
        assert item.with_lang("en").lang == "en"
        assert item.with_lang("vi").text == item.variants["vi"]


def test_answers_positive_and_finite() -> None:
    import math

    for item in generate(200, 11):
        for value in item.answers():
            assert math.isfinite(value)
            assert value > 0


def test_parts_only_for_ratio_topics() -> None:
    for item in generate(120, 5):
        if item.parts is not None:
            assert item.topic in ("sum_ratio", "diff_ratio")
            assert item.parts["big"] is not None
            assert item.parts["small"] is not None
            assert item.parts["total_parts"] is not None


def test_grades_and_topics_filter() -> None:
    items = generate(30, 3, grades=[2], topics=["linear_add", "linear_sub"])
    assert items
    assert all(x.grade == 2 for x in items)
    assert all(x.topic in ("linear_add", "linear_sub") for x in items)


def test_summary_counts() -> None:
    items = generate(24, 1)
    counts = summary(items)
    assert sum(counts.values()) == 24
    assert set(counts) <= set(TOPIC_GRADES)


def test_answer_text() -> None:
    for item in generate(20, 9):
        text = item.answer_text()
        assert text
        for value in item.answers():
            token = str(int(value)) if abs(value - round(value)) < 1e-9 else f"{value:.2f}".rstrip("0").rstrip(".").replace(".", ",")
            assert token in text
