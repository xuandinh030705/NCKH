"""Kiểm tra guard: làm sạch, phát hiện vi phạm, câu dự phòng hợp lệ."""
from __future__ import annotations

from pathlib import Path

from tutor.config import Config
from tutor.guard import clean_model_output, fallback, validate, violation_names
from tutor.mathcheck import solve

ROOT = Path(__file__).resolve().parent.parent
CFG = Config.load(ROOT / "config.yaml")

PROBLEM = "Tìm x, biết: x + 18 = 45"
SUM_RATIO = ("Một cửa hàng có 36 kg gạo, trong đó số lượng loại lớn gấp 3 lần "
             "số lượng loại nhỏ. Hỏi mỗi loại có bao nhiêu kg gạo?")
SUM_DIFF = ("Hai bạn An và Bình có tổng cộng 24 cái kẹo. "
            "An có nhiều hơn Bình 6 cái kẹo. Hỏi mỗi bạn có bao nhiêu cái kẹo?")


def ctx(problem: str = PROBLEM, mode: str = "1", turn: int = 0, revealed: bool = False) -> dict:
    return {
        "mode": mode,
        "lang": "vi",
        "turn_index": turn,
        "problem": problem,
        "numbers": [],
        "student_text": "",
        "solution": solve(problem),
        "revealed_answer": revealed,
    }


def test_clean_removes_think() -> None:
    raw = "thinking nháp...</think>Chào em! Ta xét 45 - 18 nhé?"
    assert clean_model_output(raw) == "Chào em! Ta xét 45 - 18 nhé?"


def test_clean_removes_control_tokens() -> None:
    assert clean_model_output("<|im_end|>Chào em<|im_end|>") == "Chào em"


def test_clean_english_starter_keeps_vietnamese() -> None:
    raw = "Okay, let's see. The student is stuck. Chào em! Ta xét 45 - 18 nhé?"
    out = clean_model_output(raw)
    assert out.startswith("Chào em!")
    assert "Okay" not in out


def test_clean_pure_english_gives_default() -> None:
    out = clean_model_output("The problem is: x + 18 = 45.")
    assert out.startswith("Chào em!")


def test_banned_word() -> None:
    v = validate("Đây là phương trình cần giải.", ctx())
    assert "banned_word" in violation_names(v)


def test_vague_question() -> None:
    v = validate("Em làm gì tiếp theo?", ctx())
    assert "vague" in violation_names(v)


def test_praise_turn1() -> None:
    v = validate("Thầy khen em làm đúng rồi! Em hãy tính 45 - 18 nhé?", ctx())
    assert "praise_turn1" in violation_names(v)


def test_too_many_questions() -> None:
    v = validate("45 - 18 bằng mấy? 18 + 27 bằng mấy? 45 + 18 bằng mấy?", ctx())
    assert "too_many_questions" in violation_names(v)


def test_question_without_problem_number() -> None:
    v = validate("Em hãy tính tiếp nhé?", ctx())
    assert "question_without_number" in violation_names(v)


def test_answer_leak() -> None:
    v = validate("Đáp số là 27 nhé!", ctx())
    assert "answer_leak" in violation_names(v)


def test_answer_allowed_after_student_solved() -> None:
    v = validate("Đúng rồi em! 45 - 18 = 27.", ctx(turn=1, revealed=True))
    assert "answer_leak" not in violation_names(v)


def test_mode3_requires_answer_line() -> None:
    v = validate("Bài giải: 45 - 18 = 27.", ctx(mode="3"))
    assert "mode3_answer_line" in violation_names(v)
    assert validate("Bài giải:\nĐáp số: 27", ctx(mode="3")) == []


def test_mode2_requires_parts() -> None:
    v = validate("Số lớn gấp 3 lần số bé.", ctx(problem=SUM_RATIO, mode="2"))
    assert "mode2_parts" in violation_names(v)
    ok = validate("Số lớn 3 phần, số bé 1 phần, tổng 4 phần. "
                  "Em hãy tính 36 : 4 nhé?", ctx(problem=SUM_RATIO, mode="2"))
    assert ok == []


def test_guard_disabled() -> None:
    cfg = CFG.with_overrides({"guard.enabled": False})
    assert validate("Đây là phương trình.", ctx(), cfg) == []
    # thẻ think vẫn bị chặn vì là lỗi kỹ thuật
    assert "think_tag" in violation_names(validate("...</think>Chào em", ctx(), cfg))


def test_fallback_mode1_valid() -> None:
    out = fallback(ctx())
    assert out
    assert validate(out, ctx()) == []
    assert "45" in out or "18" in out


def test_fallback_mode2_valid() -> None:
    c = ctx(problem=SUM_RATIO, mode="2")
    out = fallback(c)
    assert "phần" in out.lower()
    assert validate(out, c) == []


def test_fallback_mode3_valid() -> None:
    c = ctx(problem=SUM_DIFF, mode="3")
    out = fallback(c)
    assert "Đáp số" in out
    assert "15" in out and "9" in out
    assert validate(out, c) == []


def test_violation_names_unique_sorted() -> None:
    v = validate("phương trình? Em làm gì tiếp theo?", ctx())
    names = violation_names(v)
    assert names == sorted(set(names))
