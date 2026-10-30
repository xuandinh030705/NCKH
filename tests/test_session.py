"""Kiểm tra phiên học: lịch sử, reset khi đổi chế độ, retry + fallback."""
from __future__ import annotations

from pathlib import Path

import pytest

from tutor.config import Config
from tutor.engine import Engine
from tutor.session import TutorSession

ROOT = Path(__file__).resolve().parent.parent
PROBLEM = "Tìm x, biết: x + 18 = 45"
SUM_DIFF = ("Hai bạn An và Bình có tổng cộng 24 cái kẹo. "
            "An có nhiều hơn Bình 6 cái kẹo. Hỏi mỗi bạn có bao nhiêu cái kẹo?")


def _session(behavior: str = "auto", mode: str = "1", **overrides) -> TutorSession:
    cfg = Config.load(ROOT / "config.yaml").with_overrides(
        {"engine.backend": "mock", "engine.mock_behavior": behavior, **overrides}
    )
    engine = Engine(cfg)
    engine.load()
    return TutorSession(engine, mode=mode, cfg=cfg)


def test_first_turn_clean() -> None:
    session = _session()
    res = session.send(PROBLEM)
    assert res.reply
    assert res.violations == []
    assert res.retries == 0
    assert session.turn_index == 1
    assert len(session.history) == 3
    assert session.history[0]["role"] == "system"
    assert session.history[1] == {"role": "user", "content": PROBLEM}


def test_multi_turn_keeps_history() -> None:
    session = _session()
    session.send(PROBLEM)
    res = session.send("Em tính 45 - 18 = 27 ạ.")
    assert res.violations == []
    assert session.turn_index == 2
    assert len(session.history) == 5


def test_history_trimmed() -> None:
    session = _session()
    session.send(PROBLEM)
    for _ in range(10):
        session.send("Em tính 45 - 18 = 27 ạ.")
    assert len(session.history) <= 7  # system + tối đa 6 tin
    assert session.history[0]["role"] == "system"


def test_set_mode_resets() -> None:
    session = _session()
    session.send(PROBLEM)
    session.set_mode("3")
    assert session.turn_index == 0
    assert session.problem == ""
    assert len(session.history) == 1
    assert session.mode == "3"


def test_reset() -> None:
    session = _session()
    session.send(PROBLEM)
    session.reset()
    assert session.turn_index == 0 and session.problem == ""
    assert len(session.history) == 1


def test_violating_backend_falls_back() -> None:
    session = _session("violating")
    res = session.send(PROBLEM)
    assert res.used_fallback is True
    assert res.retries == 2
    assert res.violations == []
    assert res.violations_before  # model vi phạm trước khi guard xử lý
    assert "Chào em!" in res.reply


def test_raw_mode_skips_guard() -> None:
    session = _session("violating")
    res = session.send(PROBLEM, raw=True)
    assert res.violations == []
    assert res.used_fallback is False
    assert "khen em" in res.reply.lower()


def test_mode3_has_answer_line() -> None:
    session = _session(mode="3")
    res = session.send(SUM_DIFF)
    assert res.violations == []
    assert "Đáp số" in res.reply
    assert "15" in res.reply and "9" in res.reply


def test_empty_message_raises() -> None:
    session = _session()
    with pytest.raises(ValueError):
        session.send("   ")


def test_stats_line() -> None:
    session = _session()
    session.send(PROBLEM)
    line = session.stats_line()
    assert "1 lượt" in line
    assert "retry" in line


def test_language_detected_from_problem() -> None:
    session = _session()
    session.send("The student needs to find x: x + 18 = 45")
    assert session.lang == "en"
    assert "English" in session.system_prompt or "ADDITIONAL" in session.system_prompt
