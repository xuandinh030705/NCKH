"""Phiên học đa lượt: system prompt 3 chế độ, guard + retry + fallback, lịch sử ngắn."""
from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .engine import Engine, GenResult
from .guard import Violation, clean_model_output, fallback, validate
from .input_guard import check_input_safety
from .lang import detect
from .mathcheck import (
    arith_claims,
    detect_contradiction,
    detect_missing_data,
    evaluate_student_answer,
    hidden_hint,
    numbers_in,
    solve,
)
from .prompts import DEFAULT_MODE, build_system_from_cfg, mode_key


@dataclass
class TurnResult:
    reply: str
    raw: str
    violations: list[Violation]
    retries: int
    gen: GenResult
    violations_before: list[Violation] = field(default_factory=list)
    used_fallback: bool = False
    turn_index: int = 0

    @property
    def clean(self) -> bool:
        return not self.violations


class TutorSession:
    """Một phiên học. Đổi chế độ ⇒ reset (giống dropdown của notebook)."""

    def __init__(self, engine: Engine, mode: str = DEFAULT_MODE, cfg: Config | None = None) -> None:
        self.engine = engine
        self.cfg = cfg or engine.cfg
        self.mode = mode_key(mode)
        self.lang = str(self.cfg.get("lang", "vi") or "vi")
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        self.problem: str = ""
        self.turn_index = 0
        self.history: list[dict[str, str]] = [
            {"role": "system", "content": build_system_from_cfg(self.mode, self.lang, self.cfg)}
        ]
        self.stats: dict[str, Any] = {
            "turns": 0, "retries": 0, "fallbacks": 0,
            "violations_before": 0, "violations_after": 0,
            "gen_tokens": 0, "prompt_tokens": 0, "total_s": 0.0,
        }

    def set_mode(self, mode: str) -> None:
        self.mode = mode_key(mode)
        self.lang = str(self.cfg.get("lang", "vi") or "vi")
        self.reset()

    # ------------------------------------------------------------- gửi 1 câu
    def send(self, text: str, stream_cb=None, *, raw: bool = False,
             max_tokens: int | None = None) -> TurnResult:
        text = (text or "").strip()
        if not text:
            raise ValueError("Tin nhắn rỗng")

        # 1. Tầng Input Guard: lọc jailbreak, trích xuất prompt, y tế, ngoài phạm vi
        intercepted, safe_reply, reason = check_input_safety(text, mode=self.mode)
        if intercepted and safe_reply:
            if stream_cb:
                stream_cb(safe_reply)
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": safe_reply})
            self._trim_history()
            self.turn_index += 1
            self.stats["turns"] += 1
            return TurnResult(
                reply=safe_reply,
                raw=safe_reply,
                violations=[],
                retries=0,
                gen=GenResult(text=safe_reply),
                turn_index=self.turn_index - 1,
            )

        if self.turn_index == 0:
            self.problem = text
            self.lang = detect(text)
            # dựng lại system prompt theo ngôn ngữ của đề
            self.history[0] = {
                "role": "system",
                "content": build_system_from_cfg(self.mode, self.lang, self.cfg),
            }

            # 2. Kiểm tra đề bài thiếu dữ kiện hoặc mâu thuẫn ở lượt 0
            missing_flag, missing_msg = detect_missing_data(text)
            if missing_flag and missing_msg:
                if stream_cb:
                    stream_cb(missing_msg)
                self.history.append({"role": "user", "content": text})
                self.history.append({"role": "assistant", "content": missing_msg})
                self._trim_history()
                self.turn_index += 1
                self.stats["turns"] += 1
                return TurnResult(
                    reply=missing_msg,
                    raw=missing_msg,
                    violations=[],
                    retries=0,
                    gen=GenResult(text=missing_msg),
                    turn_index=self.turn_index - 1,
                )

            contra_flag, contra_msg = detect_contradiction(text)
            if contra_flag and contra_msg:
                if stream_cb:
                    stream_cb(contra_msg)
                self.history.append({"role": "user", "content": text})
                self.history.append({"role": "assistant", "content": contra_msg})
                self._trim_history()
                self.turn_index += 1
                self.stats["turns"] += 1
                return TurnResult(
                    reply=contra_msg,
                    raw=contra_msg,
                    violations=[],
                    retries=0,
                    gen=GenResult(text=contra_msg),
                    turn_index=self.turn_index - 1,
                )

        solution = solve(self.problem) if self.problem else None

        # 3. Chế độ 2: sơ đồ đoạn thẳng trực quan ở lượt 0
        if self.mode == "2" and self.turn_index == 0:
            from .diagram import is_diagram_suitable, generate_ascii_diagram
            suitable, diagram_advice = is_diagram_suitable(text, solution)
            reply_text = diagram_advice if not suitable else generate_ascii_diagram(text, solution)
            if stream_cb:
                stream_cb(reply_text)
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply_text})
            self._trim_history()
            self.turn_index += 1
            self.stats["turns"] += 1
            return TurnResult(
                reply=reply_text,
                raw=reply_text,
                violations=[],
                retries=0,
                gen=GenResult(text=reply_text),
                turn_index=self.turn_index - 1,
            )

        revealed = self._student_revealed(text, solution)
        student_eval = None
        if self.turn_index > 0:
            student_eval = evaluate_student_answer(text, self.problem, solution, mode=self.mode, turn=self.turn_index)
            if getattr(student_eval, "verdict", "") in ("FINAL_CORRECT", "CORRECT"):
                revealed = True

        ctx: dict[str, Any] = {
            "mode": self.mode,
            "lang": self.lang,
            "turn_index": self.turn_index,
            "problem": self.problem,
            "numbers": numbers_in(self.problem),
            "student_text": text,
            "solution": solution,
            "revealed_answer": revealed,
            "student_eval": student_eval,
        }

        user_content = text
        if student_eval and student_eval.has_answer:
            if student_eval.verdict in ("WRONG", "CALC_WRONG", "FINAL_WRONG", "INTERMEDIATE_WRONG"):
                user_content += (
                    f"\n\n[GHI CHÚ HỆ THỐNG: Học sinh vừa đưa ra kết quả/phép tính SAI ({student_eval.detail}). "
                    f"Thầy tuyệt đối KHÔNG khen đúng, mà hãy nhẹ nhàng nhắc học sinh tính lại cẩn thận!]"
                )
            elif student_eval.verdict in ("CORRECT", "FINAL_CORRECT", "INTERMEDIATE_CORRECT"):
                user_content += (
                    f"\n\n[GHI CHÚ HỆ THỐNG: Học sinh vừa tính ĐÚNG ({student_eval.detail}). "
                    f"Thầy hãy khen ngợi ngắn gọn và hướng dẫn bước tiếp theo!]"
                )

        hint = hidden_hint(self.problem, text, self.cfg)
        if hint:
            user_content = f"{user_content}\n\n{hint}"

        base_messages = self.history + [{"role": "user", "content": user_content}]
        messages = list(base_messages)
        max_retries = int(self.cfg.get("session.max_retries", 2))
        boost = float(self.cfg.get("session.retry_temp_boost", 0.1))
        decode = self.cfg.section("decode")
        temperature = float(decode.get("temperature", 0.25))
        max_tokens = int(max_tokens or decode.get("max_new_tokens_chat", 300))

        last_gen: GenResult | None = None
        violations_before: list[Violation] = []
        violations: list[Violation] = []
        retries = 0
        cleaned = ""
        raw_text = ""

        for attempt in range(max_retries + 1):
            last_gen = self.engine.generate(
                messages,
                lang=self.lang,
                mode=self.mode,
                stream_cb=stream_cb if attempt == 0 else None,
                max_new_tokens=max_tokens,
                temperature=temperature + boost * attempt,
            )
            raw_text = last_gen.text
            cleaned = clean_model_output(raw_text)
            if raw:  # --stream: hiển thị thô, không kiểm duyệt, không retry
                violations = []
                break
            violations = validate(cleaned, ctx, self.cfg)
            if attempt == 0:
                violations_before = list(violations)
            if not violations:
                break
            if attempt < max_retries:
                retries = attempt + 1
                messages = base_messages + [
                    {"role": "assistant", "content": cleaned},
                    {"role": "user", "content": self._retry_note(violations)},
                ]

        used_fallback = False
        if violations:
            cleaned = fallback(ctx, self.cfg)
            used_fallback = True
            # câu fallback phải hợp lệ; nếu không, ghi nhận để eval thấy
            violations = validate(cleaned, ctx, self.cfg)

        # lưu lịch sử (không lưu phần nhắc retry)
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": cleaned})
        self._trim_history()
        self.turn_index += 1

        self.stats["turns"] += 1
        self.stats["retries"] += retries
        self.stats["fallbacks"] += int(used_fallback)
        self.stats["violations_before"] += len(violations_before)
        self.stats["violations_after"] += len(violations)
        if last_gen is not None:
            self.stats["gen_tokens"] += last_gen.gen_tokens
            self.stats["prompt_tokens"] += last_gen.prompt_tokens
            self.stats["total_s"] += last_gen.total_s

        return TurnResult(
            reply=cleaned,
            raw=raw_text,
            violations=violations,
            retries=retries,
            gen=last_gen or GenResult(text=""),
            violations_before=violations_before,
            used_fallback=used_fallback,
            turn_index=self.turn_index - 1,
        )

    # ------------------------------------------------------------------ phụ
    def _retry_note(self, violations: list[Violation]) -> str:
        rules = ", ".join(sorted({v.rule for v in violations}))
        return (
            "Câu trả lời vừa rồi vi phạm quy tắc: "
            f"{rules}. Hãy trả lời lại thật ngắn gọn, đúng quy tắc: chỉ hỏi ĐÚNG 1 câu có số liệu "
            "trong đề, không dùng từ bị cấm, không khen ở lượt đầu, không đưa đáp số cuối."
        )

    def _trim_history(self) -> None:
        keep = int(self.cfg.get("session.max_history_turns", 6))
        if len(self.history) > keep + 1:
            self.history = [self.history[0]] + self.history[-keep:]

    def _student_revealed(self, text: str, solution) -> bool:
        """Học sinh đã tự tính ra đáp số cuối chưa?"""
        if solution is None:
            return False
        answers = solution.answers()
        for claim in arith_claims(text):
            if claim.ok and any(abs(claim.correct - a) < 1e-9 for a in answers):
                return True
        low = text.lower()
        for a in answers:
            token = str(int(a)) if abs(a - round(a)) < 1e-9 else str(a).replace(".", ",")
            if f"x = {token}" in low or f"x={token}" in low or f"đáp số {token}" in low:
                return True
        # Biểu thức có ngoặc (ví dụ "(24 + 6) : 2 = 15") không khớp regex đơn giản
        # của arith_claims; nếu học sinh viết số đúng kèm dấu '=' / câu chốt thì coi
        # như đã tự tính ra.
        if re.search(r"=|vậy|đáp số|số lớn|số bé", low):
            nums = numbers_in(text)
            if any(abs(n - a) < 1e-9 for n in nums for a in answers):
                return True
        return False

    # --------------------------------------------------------------- tiện ích
    @property
    def system_prompt(self) -> str:
        return self.history[0]["content"]

    def stats_line(self) -> str:
        s = self.stats
        turns = max(1, s["turns"])
        return (f"{s['turns']} lượt | retry {s['retries']} ({s['retries'] / turns:.0%}) | "
                f"fallback {s['fallbacks']} | vi phạm trước guard {s['violations_before']}")


__all__ = ["TurnResult", "TutorSession"]
