"""Lọc đầu ra + kiểm tra quy tắc sư phạm + câu mẫu dự phòng.

``clean_model_output`` giữ đúng logic cell 3 của notebook (parity test sẽ so từng
ký tự với bản chạy trực tiếp từ ipynb).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .lang import detect, english_word_ratio
from .mathcheck import Solution, arith_claims, fmt_num, numbers_in, parse_problem, solve
from .notebook_source import load

_NB = load()
ENGLISH_THOUGHT_STARTERS: tuple[str, ...] = tuple(_NB.english_thought_starters)
DEFAULT_VI_REPLY = (
    "Chào em! Trong bài toán này, em hãy chú ý đến phép tính có chứa x trước nhé. "
    "Em hãy cho thầy biết phép tính này gồm những thành phần nào?"
)


# ------------------------------------------------------------------ làm sạch
def clean_model_output(raw_text: str) -> str:
    """Lọc thẻ <think> và dọn sạch các đoạn nhẩm tiếng Anh nếu mô hình lỡ sinh ra."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    cleaned = re.sub(r"^.*?<\/think>", "", cleaned, flags=re.DOTALL)
    cleaned = (
        cleaned.replace("<|im_end|>", "")
        .replace(" thinking", "")
        .replace("</think>", "")
        .strip()
    )

    lower_text = cleaned.lower()
    for starter in ENGLISH_THOUGHT_STARTERS:
        if lower_text.startswith(starter):
            vietnamese_match = re.search(
                r"(Chào em|Thầy|Cô|Trong phép|Để tìm|Bài toán|Trước tiên)", cleaned, re.IGNORECASE
            )
            if vietnamese_match:
                cleaned = cleaned[vietnamese_match.start():].strip()
            else:
                cleaned = DEFAULT_VI_REPLY
            break
    return cleaned.strip()


# ------------------------------------------------------------------ vi phạm
@dataclass(frozen=True)
class Violation:
    rule: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - tiện debug
        return f"{self.rule}: {self.detail}"


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUMBER_RE.finditer(text or ""):
        try:
            out.append(float(m.group(0).replace(",", ".")))
        except ValueError:  # pragma: no cover
            continue
    return out


def _contains_number(text: str, value: float, tol: float = 1e-9) -> bool:
    return any(abs(n - value) <= tol for n in _numbers(text))


def _norm(text: str) -> str:
    from .lang import strip_diacritics
    t = strip_diacritics(text or "").lower()
    return t.replace("đ", "d")


def _has_any(text: str, phrases: list[str] | tuple[str, ...]) -> str | None:
    low = (text or "").lower()
    for p in phrases:
        if p.lower() in low:
            return p
    return None


def _looks_english(reply: str) -> bool:
    if len(reply) < 25:
        return False
    if detect(reply) == "en":
        return True
    return english_word_ratio(reply) >= 0.3


def _answer_mentioned(reply: str, answers: list[float]) -> bool:
    """Đáp số có xuất hiện ở VỊ TRÍ KẾT QUẢ không (không tính toán hạng của đề).

    Ví dụ bài "5 quyển vở, mỗi quyển 11 nghìn, đưa 60 nghìn" có đáp số 5;
    câu "5 x 11" KHÔNG phải lộ đáp số, còn "60 - 55 = 5" thì có.
    """
    low = (reply or "").lower()
    if "đáp số" in low or "đáp án" in low:
        return True
    for a in answers:
        if a == 0:
            continue
        token = str(int(a)) if abs(a - round(a)) < 1e-9 else str(a).replace(".", ",")
        if re.search(rf"\bx\s*=\s*{re.escape(token)}(?![\d,])", low):
            return True
    for claim in arith_claims(reply):
        if claim.ok and any(abs(claim.correct - a) < 1e-9 for a in answers):
            return True
    for sentence in re.split(r"[.\n!?]", low):
        if any(k in sentence for k in ("vậy", "số lớn", "số bé", "kết quả")):
            if any(_contains_number(sentence, a) for a in answers):
                return True
    return False


def validate(reply: str, ctx: dict[str, Any], cfg: Any | None = None) -> list[Violation]:
    """Kiểm tra câu trả lời theo quy tắc của chế độ. Trả [] nếu hợp lệ."""
    mode = str(ctx.get("mode", "1"))
    lang = str(ctx.get("lang", "vi"))
    turn = int(ctx.get("turn_index", 0))
    problem = str(ctx.get("problem") or "")
    solution: Solution | None = ctx.get("solution")
    out: list[Violation] = []

    def rule(name: str, detail: str) -> None:
        out.append(Violation(name, detail))

    if not (reply or "").strip():
        rule("empty", "câu trả lời rỗng")
        return out
    if "think" in reply.lower() and ("<think" in reply.lower() or "</think>" in reply.lower()):
        rule("think_tag", "còn thẻ <think>")
    if "<|im_end|>" in reply:
        rule("control_token", "còn <|im_end|>")

    if cfg is not None and cfg.get("guard.enabled", True) is False:
        return out

    forbid_english = True if cfg is None else bool(cfg.get("guard.common.forbid_english", True))
    if forbid_english and lang == "vi" and _looks_english(reply):
        rule("english", "câu trả lời lẫn tiếng Anh")

    if mode == "3":
        require_line = True if cfg is None else bool(cfg.get("guard.mode3.require_answer_line", True))
        pattern = (cfg.get("guard.mode3.answer_line_pattern", "Đáp số") if cfg else "Đáp số")
        if require_line and str(pattern).lower() not in reply.lower():
            rule("mode3_answer_line", "thiếu dòng 'Đáp số'")

    if mode == "2":
        need_parts = True if cfg is None else bool(cfg.get("guard.mode2.require_parts_mention", True))
        if need_parts and solution is not None and solution.parts and "phần" not in reply.lower():
            rule("mode2_parts", "không nêu số phần")

    # ---------------- chế độ 1: Socratic cụ thể
    if mode == "1":
        g = (cfg.section("guard").get("mode1", {}) if cfg is not None else {})
        banned = g.get("banned_words") or ["phương trình", "hai vế", "chuyển vế", "đặt biến"]
        praise = g.get("praise_phrases") or ["khen em", "làm đúng", "đã tìm được", "đã hiểu rõ"]
        vague = g.get("vague_phrases") or [
            "Em có thể làm như thế nào", "Em làm gì tiếp theo",
            "Em hãy làm cẩn thận", "đang làm gì ở đây",
        ]
        max_q = int((cfg.get("session.max_question_marks", 2) if cfg else 2))

        hit = _has_any(reply, banned)
        if hit:
            rule("banned_word", f"dùng từ cấm: {hit!r}")
        if turn == 0:
            hit = _has_any(reply, praise)
            if hit:
                rule("praise_turn1", f"khen/xác nhận ở lượt 1: {hit!r}")
        hit = _has_any(reply, vague)
        if hit:
            rule("vague", f"câu hỏi mơ hồ: {hit!r}")

        q_count = reply.count("?")
        if q_count > max_q:
            rule("too_many_questions", f"{q_count} dấu '?' > {max_q}")

        problem_numbers = numbers_in(problem)
        if q_count and problem_numbers:
            if not any(_contains_number(reply, n) for n in problem_numbers):
                rule("question_without_number", "câu hỏi không gắn với số liệu của đề")

    # ---------------- kiểm tra rò rỉ prompt & quy tắc nội bộ (BUG 02)
    leak_terms = [
        "base_prompts", "system_modes", "concrete_socratic_prompt", "banned_words",
        "quy tắc bổ sung", "quy tac bo sung", "no english allowed", "praise_phrases",
        "vague_phrases", "require_number_from_problem", "forbid_final_answer",
        "[ghi chú ẩn", "[ghi chu an", "[ghi chú hệ thống", "[ghi chu he thong",
        "hidden_hint_template", "retry_note",
    ]
    leak_hit = _has_any(reply, leak_terms)
    if leak_hit:
        rule("system_prompt_leak", f"lộ quy tắc hoặc chỉ dẫn nội bộ: {leak_hit!r}")

    # ---------------- kiểm tra đảo ngược đánh giá đúng/sai (BUG 01)
    student_eval = ctx.get("student_eval")
    if student_eval is not None:
        low_norm = _norm(reply)
        verdict = getattr(student_eval, "verdict", "")
        # Nếu học sinh sai
        if verdict in ("WRONG", "CALC_WRONG", "FINAL_WRONG", "INTERMEDIATE_WRONG"):
            praise_words = ["dung roi", "chinh xac", "em lam tot", "rat tot", "gioi lam", "chuan roi", "hoan toan dung"]
            has_praise = any(re.search(rf"\b{re.escape(w)}\b", low_norm) for w in praise_words)
            has_neg = any(k in low_norm for k in ("chua dung", "chua chinh xac", "sai roi", "chua dung dau", "khong dung", "chua dung nhe", "chua chinh xac roi", "tinh lai"))
            if has_praise and not has_neg:
                rule("reversed_eval_praise", "khen đúng trong khi học sinh trả lời sai hoặc tính sai")
            elif not has_neg:
                rule("unacknowledged_wrong", "không nhắc học sinh kết quả vừa đưa ra là sai")
        # Nếu học sinh đúng mà bot chê sai
        elif verdict in ("CORRECT", "FINAL_CORRECT", "INTERMEDIATE_CORRECT"):
            crit_words = ["chua dung", "sai roi", "chua chinh xac", "em tinh sai", "tinh lai nhe", "tinh lai xem"]
            has_crit = any(re.search(rf"\b{re.escape(w)}\b", low_norm) for w in crit_words)
            if has_crit:
                rule("reversed_eval_criticism", "chê sai trong khi học sinh trả lời đúng")

    # ---------------- kiểm tra giải toán vượt cấp THCS/THPT (BUG 03)
    sol_kind = getattr(solution, "kind", "") if solution else ""
    if sol_kind not in ("linear", ""):
        hs_terms = [
            "hệ phương trình", "phương trình bậc hai", "đặt ẩn", "đặt biến", "ẩn phụ",
            "chuyển vế", "hai vế", "cả hai vế", "\\begin{cases}", "ẩn a", "ẩn b",
            "gọi số", "ta gọi", "gọi x là", "gọi y là", "2x =", "2x +",
        ]
        hs_hit = _has_any(reply, hs_terms)
        if hs_hit:
            rule("high_school_algebra", f"dùng phương pháp đại số vượt cấp: {hs_hit!r}")

    # ---------------- kiểm tra tự bịa dữ kiện khi đề bài thiếu (BUG 04)
    if ctx.get("missing_data"):
        assumptions = [
            "giả sử là hình vuông", "giả sử hình chữ nhật là hình vuông", "coi là hình vuông",
            "giả sử chiều dài", "giả sử chiều rộng", "cho chiều dài là", "cho chiều rộng là",
        ]
        ass_hit = _has_any(reply, assumptions)
        if ass_hit:
            rule("hallucinated_assumption", f"tự ý bịa dữ kiện khi đề bài thiếu: {ass_hit!r}")

    # ---------------- kiểm tra số âm cho bài toán số tự nhiên tiểu học (BUG 05)
    if re.search(r"\b(so be|so lon|tuoi|chieu dai|canh|dap so|ket qua|con lai)\s*(la|bang|:)?\s*-\s*\d+", _norm(reply)):
        rule("negative_number_elementary", "đưa ra số âm cho bài toán số tự nhiên tiểu học")

    # ---------------- kiểm tra tư vấn y tế cho trẻ em (BUG 08)
    med_terms = ["chẩn đoán", "kê đơn", "uống thuốc kháng sinh", "uống paracetamol", "bị viêm ruột thừa"]
    med_hit = _has_any(reply, med_terms)
    if med_hit:
        rule("medical_advice", f"tự ý tư vấn y tế cho trẻ em: {med_hit!r}")

    # ---------------- kiểm tra xưng hô sai vai trò (E04)
    low_rep = reply.lower().strip()
    if low_rep.startswith("chào thầy") or "chào thầy ạ" in low_rep:
        rule("role_reversal", "xưng hô sai vai trò (gọi người dùng là thầy)")

    if mode == "1" and solution is not None and solution.answer is not None and not ctx.get("revealed_answer"):
        if _answer_mentioned(reply, solution.answers()):
            rule("answer_leak", "lộ đáp số khi học sinh chưa tự tính")
    return out


# ------------------------------------------------------------------ dự phòng
def _group_hint(problem: str) -> str:
    """Cụm chưa biết gợi ý, lấy từ chính vế trái của đề (không bịa số)."""
    p = parse_problem(problem)
    if p is None or not p.equation:
        return "phép tính chứa x"
    left = p.equation.split("=")[0].strip()
    return left or "phép tính chứa x"


def fallback(ctx: dict[str, Any], cfg: Any | None = None) -> str:
    """Câu mẫu hợp lệ theo dạng bài, gắn với số liệu trong đề."""
    # 0. Nếu có phản hồi đã chặn từ Input Guard
    if ctx.get("intercepted_reply"):
        return str(ctx["intercepted_reply"])

    # 0b. Nếu đề bài thiếu dữ kiện hoặc mâu thuẫn
    if ctx.get("missing_data"):
        return str(ctx["missing_data"])
    if ctx.get("contradiction"):
        return str(ctx["contradiction"])

    mode = str(ctx.get("mode", "1"))
    problem = str(ctx.get("problem") or "")
    solution: Solution | None = ctx.get("solution") or solve(problem)
    nums = numbers_in(problem)
    student_eval = ctx.get("student_eval")

    # Nếu học sinh trả lời ở các lượt sau
    if student_eval is not None:
        verdict = getattr(student_eval, "verdict", "")
        if verdict in ("WRONG", "CALC_WRONG", "FINAL_WRONG", "INTERMEDIATE_WRONG"):
            if solution is not None and solution.steps:
                first_st = solution.steps[0]
                return (
                    f"Phép tính của em chưa chính xác rồi. Em hãy tính lại thật cẩn thận xem "
                    f"{first_st.expr} bằng bao nhiêu nhé?"
                )
            if nums:
                return (
                    f"Kết quả của em chưa chính xác rồi. Trong bài có số {fmt_num(nums[0])}, "
                    f"em hãy kiểm tra lại phép tính của mình thật cẩn thận nhé!"
                )
            return "Kết quả của em chưa chính xác rồi. Em hãy kiểm tra lại phép tính của mình thật cẩn thận nhé!"
        if verdict in ("CORRECT", "FINAL_CORRECT") and mode == "1":
            ans_str = fmt_num(solution.answers()[0]) if solution and solution.answers() else ""
            return f"Đúng rồi em! Kết quả {ans_str} là hoàn toàn chính xác. Em làm rất giỏi!".strip()

    if mode == "3" and solution is not None:
        lines: list[str] = ["Bài giải:"]
        for st in solution.steps:
            lines.append(f"- {st.desc}")
            lines.append(f"  {st.expr} = {fmt_num(st.value)}")
        if isinstance(solution.answer, tuple):
            ans = " và ".join(fmt_num(v) for v in solution.answer)
        else:
            ans = fmt_num(solution.answer)
        if solution.unit:
            ans = f"{ans} {solution.unit}"
        lines.append(f"Đáp số: {ans}")
        return "\n".join(lines)

    if mode == "2":
        try:
            from .diagram import generate_ascii_diagram, is_diagram_suitable
            suitable, advice = is_diagram_suitable(problem, solution)
            if not suitable:
                return advice
            return generate_ascii_diagram(problem, solution)
        except Exception:
            pass
        if solution is not None and solution.parts:
            k_big = fmt_num(solution.parts["big"])
            k_small = fmt_num(solution.parts["small"])
            total = fmt_num(solution.parts["total_parts"])
            first = nums[0] if nums else None
            if first is not None:
                return (
                    f"Chào em! Ta vẽ sơ đồ đoạn thẳng: số lớn gồm {k_big} phần bằng nhau, "
                    f"số bé gồm {k_small} phần như thế, tổng cộng {total} phần. "
                    f"Với tổng là {fmt_num(first)}, em hãy tính giúp thầy một phần bằng bao nhiêu nhé?"
                )

    if solution is not None and solution.steps:
        first = solution.steps[0]
        return (
            f"Chào em! Với bài này, trước tiên ta xét phép tính {first.expr}. "
            f"Em hãy tính giúp thầy {first.expr} bằng bao nhiêu nhé?"
        )

    if nums:
        return (
            f"Chào em! Bài toán có số {fmt_num(nums[0])}. "
            f"Em hãy cho thầy biết bài toán yêu cầu tìm gì từ số {fmt_num(nums[0])} này nhé?"
        )
    if mode == "2":
        return "Chào em! Em hãy chép lại đề bài để thầy vẽ sơ đồ đoạn thẳng giúp em nhé."
    return "Chào em! Em hãy chép lại đề bài giúp thầy để thầy hướng dẫn từng bước nhé."


def violation_names(violations: list[Violation]) -> list[str]:
    return sorted({v.rule for v in violations})


__all__ = [
    "DEFAULT_VI_REPLY",
    "ENGLISH_THOUGHT_STARTERS",
    "Violation",
    "clean_model_output",
    "fallback",
    "validate",
    "violation_names",
]
