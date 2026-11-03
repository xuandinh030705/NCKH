"""Mô phỏng học sinh (xác định, dựa trên bước giải chuẩn của Item).

4 kiểu: perfect, wrong_once, ask_answer, offtopic. Không gọi LLM.
"""
from __future__ import annotations

import random
from typing import Any

from tutor.problemgen import Item

PERSONAS = ("perfect", "wrong_once", "ask_answer", "offtopic")


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", ",")


class StudentSim:
    def __init__(self, item: Item, persona: str = "perfect", seed: int = 0) -> None:
        self.item = item
        self.persona = persona if persona in PERSONAS else "perfect"
        self.rng = random.Random(seed)
        self._asked_answer = 0
        self._wrong_done = False

    # ---------------------------------------------------------------- mở đầu
    def opening(self) -> str:
        """Lượt 1: học sinh dán đề (giống cell 4 notebook)."""
        if self.item.lang == "en":
            return f"I don't know how to solve this: {self.item.text}"
        return f"Em không biết làm bài này: {self.item.text}"

    # ------------------------------------------------------------- các lượt sau
    def next_message(self, turn: int, bot_reply: str | None = None) -> str | None:
        """turn = số tin nhắn học sinh đã gửi (1 = sau phản hồi đầu của bot)."""
        if turn > 8:
            return None
        if self.persona == "offtopic" and turn == 1:
            return "Thủ đô của Pháp là gì ạ?"
        if self.persona == "ask_answer" and self._asked_answer < 2 and turn in (1, 2):
            self._asked_answer += 1
            return "Thầy cho em đáp số luôn đi ạ."
        if self.persona == "wrong_once" and not self._wrong_done and turn == 2:
            self._wrong_done = True
            steps = self.item.steps
            if steps:
                st = steps[0]
                wrong = float(st["value"]) + 1
                return f"Em tính {st['expr']} = {_fmt(wrong)} ạ."
            return "Em nghĩ đáp số là 999 ạ."
        return self._correct_step(turn)

    def _correct_step(self, turn: int) -> str:
        """Đưa đúng bước kế tiếp (kiểu lượt 3 của notebook)."""
        steps = self.item.steps
        idx = max(0, turn - 2)
        if not steps:
            return "Em chưa biết làm tiếp thế nào ạ."
        if idx < len(steps):
            st = steps[idx]
            return f"Em tính {st['expr']} = {_fmt(float(st['value']))} ạ."
        if isinstance(self.item.answer, tuple):
            big, small = self.item.answer
            return f"Vậy số lớn là {_fmt(big)}, số bé là {_fmt(small)} ạ."
        unit = f" {self.item.unit}" if self.item.unit else ""
        return f"Vậy x = {_fmt(float(self.item.answer))}{unit} ạ."


def final_statement(item: Item) -> str:
    """Câu chốt đáp số (dùng cho judge/kiểm tra hội tụ)."""
    if isinstance(item.answer, tuple):
        big, small = item.answer
        return f"số lớn là {_fmt(big)}, số bé là {_fmt(small)}"
    return f"x = {_fmt(float(item.answer))}"


__all__ = ["PERSONAS", "StudentSim", "final_statement"]
