"""Kiểm tra bộ phân tích/giải toán và kiểm tra phép tính."""
from __future__ import annotations

import pytest

from tutor.mathcheck import (
    arith_claims,
    fmt_num,
    hidden_hint,
    numbers_in,
    parse_problem,
    solve,
)


def test_linear_add() -> None:
    sol = solve("Tìm x, biết: x + 18 = 45")
    assert sol is not None
    assert sol.kind == "linear"
    assert sol.answer == 27.0
    assert sol.x == 27.0
    assert sol.steps and sol.steps[0].value == 27.0


def test_linear_mul() -> None:
    sol = solve("Tìm x, biết: x * 4 = 36")
    assert sol is not None and sol.answer == 9.0


def test_linear_two_step() -> None:
    sol = solve("Tìm x, biết: x * 3 - 4 = 11")
    assert sol is not None and sol.answer == 5.0
    assert len(sol.steps) == 2


def test_sum_diff_notebook() -> None:
    sol = solve("Hai bạn An và Bình có tổng cộng 24 cái kẹo. "
                "An có nhiều hơn Bình 6 cái kẹo. Hỏi mỗi bạn có bao nhiêu cái kẹo?")
    assert sol is not None
    assert sol.kind == "sum_diff"
    assert sol.answer == (15.0, 9.0)
    assert sol.unit == "cái kẹo"


def test_motion_notebook() -> None:
    sol = solve("Một ô tô đi quãng đường dài 120 km trong thời gian 2 giờ 30 phút. "
                "Hãy tính vận tốc của ô tô.")
    assert sol is not None
    assert sol.kind == "motion_v"
    assert abs(float(sol.answer) - 48.0) < 1e-9
    assert sol.unit == "km/giờ"


def test_sum_ratio_parts() -> None:
    sol = solve("Một cửa hàng có 36 kg gạo, trong đó số lượng loại lớn gấp 3 lần "
                "số lượng loại nhỏ. Hỏi mỗi loại có bao nhiêu kg gạo?")
    assert sol is not None and sol.parts is not None
    assert sol.parts["big"] == 3.0 and sol.parts["small"] == 1.0
    assert sol.parts["total_parts"] == 4.0


def test_unknown_returns_none() -> None:
    assert solve("Hôm nay trời đẹp quá") is None
    p = parse_problem("Hôm nay trời đẹp quá")
    assert p is None or p.kind == "unknown"


def test_numbers_in() -> None:
    assert numbers_in("Tìm x, biết: x + 18 = 45") == [18.0, 45.0]
    assert numbers_in("120 : 2,5") == [120.0, 2.5]


def test_fmt_num() -> None:
    assert fmt_num(5) == "5"
    assert fmt_num(2.5) == "2,5"
    assert fmt_num(None) == "?"


def test_arith_claims() -> None:
    good = arith_claims("Ta có 45 - 18 = 27.")
    assert good and good[0].ok
    bad = arith_claims("Ta có 45 - 18 = 28.")
    assert bad and not bad[0].ok
    # phương trình có x không bị coi là phép tính số học
    assert arith_claims("x * 3 = 15") == []
    assert arith_claims("15 : 3 = 5")[0].ok


def test_arith_claims_chuoi_nhieu_phep_tinh() -> None:
    """Chuỗi 'a x b : c = d' phải đọc trọn chuỗi, không cắt thành 'b : c = d'."""
    claims = arith_claims("500 x 10 : 100 = 50")
    assert len(claims) == 1
    assert claims[0].ok

    wrong = arith_claims("500 x 10 : 100 = 60")
    assert wrong and not wrong[0].ok

    # 'x' giữa hai chữ số là dấu nhân; 'x' đứng riêng vẫn là ẩn số.
    assert arith_claims("2 + 3 x 4 = 14")[0].ok
    assert arith_claims("x = 12 : 3 = 4")[0].ok
    assert arith_claims("x * 3 = 15") == []

    # Số âm ở vế phải vẫn đọc được.
    assert arith_claims("5 - 8 = -3")[0].ok

    # Câu lời giải có số ở cuối không được dính vào phép tính.
    mixed = arith_claims(
        "- 10% của 500\n  500 x 10 : 100 = 50\n- Số còn lại\n  500 - 50 = 450"
    )
    assert [c.ok for c in mixed] == [True, True]


def test_hidden_hint_default_off() -> None:
    assert hidden_hint("Tìm x, biết: x + 18 = 45", "45 - 18 = 27") == ""


def test_hidden_hint_on() -> None:
    from tutor.config import Config

    cfg = Config.load().with_overrides({"math.hidden_hint": True})
    hint = hidden_hint("Tìm x, biết: x + 18 = 45", "45 - 18 = 27", cfg)
    assert "27" in hint
    assert "ĐÚNG" in hint


@pytest.mark.parametrize("style", ["vi", "mixed"])
def test_problemgen_round_trip(style: str) -> None:
    from tutor.problemgen import generate

    items = generate(120, 2026, style=style)
    for item in items:
        sol = solve(item.text)
        assert sol is not None, (item.lang, item.text)
        if isinstance(item.answer, tuple):
            assert isinstance(sol.answer, tuple)
            assert len(sol.answer) == len(item.answer)
            assert all(abs(a - b) < 1e-6 for a, b in zip(sol.answer, item.answer))
        else:
            assert abs(float(sol.answer) - float(item.answer)) < 1e-6
