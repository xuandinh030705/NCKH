"""Judge độc lập (rule-based) — KHÔNG import ``tutor.guard``.

Mục đích: chấm câu trả lời của gia sư theo 9 tiêu chí B1–B9, hoàn toàn tách khỏi
bộ lọc/guard của runtime. Nếu guard bị sửa để "cho đạt" thì judge vẫn phát hiện.

B1  ngôn ngữ: tiếng Việt (khi ``lang="vi"``), không còn thẻ think/control token
B2  không dùng từ cấm của tiểu học (phương trình / hai vế / chuyển vế / đặt biến)
B3  không có câu hỏi mơ hồ ("em làm gì tiếp theo", "em hãy làm cẩn thận"...)
B4  câu hỏi ở chế độ 1 phải gắn với ít nhất một con số của đề
B5  không lộ đáp số cuối khi học sinh chưa tự tính ra
B6  không khen/xác nhận ở lượt đầu (học sinh chưa làm gì)
B7  chế độ 2 phải nêu số phần bằng nhau (chỉ áp dụng cho bài có ``parts``)
B8  chế độ 3 phải có dòng "Đáp số" và đúng đáp án
B9  mọi phép tính "a op b = c" trong câu trả lời phải đúng

``None`` = tiêu chí không áp dụng cho lượt đó (bị loại khỏi mẫu số).
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

LABELS: dict[str, str] = {
    "B1": "ngôn ngữ tiếng Việt, không thẻ think/control",
    "B2": "không dùng từ cấm tiểu học",
    "B3": "không câu hỏi mơ hồ",
    "B4": "câu hỏi gắn với số liệu của đề",
    "B5": "không lộ đáp số khi học sinh chưa tính ra",
    "B6": "không khen ở lượt đầu",
    "B7": "chế độ 2 nêu số phần bằng nhau",
    "B8": "chế độ 3 có dòng Đáp số đúng",
    "B9": "phép tính trong câu trả lời đúng",
}

TOL = 1e-6

# --------------------------------------------------------------- nhận diện chữ
_VI_CHARS = set(
    "ăâđêôơư"
    "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)
_EN_STOP = {
    "the", "is", "are", "a", "an", "of", "to", "and", "in", "that", "this",
    "we", "you", "your", "let", "solve", "student", "teacher", "answer",
    "step", "first", "then", "how", "what", "with", "for", "from", "it",
    "be", "can", "should", "need", "will", "have", "has", "do", "does",
}
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_NUM_RE = re.compile(r"(?<!\d)(\d+(?:[.,]\d+)?)(?!\d)")
_MATH_CHARS = set("0123456789+-*/:×÷(),. \t")


def _to_float(raw: str) -> float:
    s = raw.strip().replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    return float(s)


def numbers_in(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        try:
            out.append(_to_float(m.group(1)))
        except ValueError:  # pragma: no cover - regex đã chặn
            continue
    return out


def _has_value(numbers: Iterable[float], value: float, tol: float = TOL) -> bool:
    return any(abs(n - value) <= tol for n in numbers)


# --------------------------------------------------------------- phép tính
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Load,
)


def _safe_eval(expr: str) -> float | None:
    src = expr.replace(":", "/").replace("×", "*").replace("÷", "/").replace("−", "-")
    src = re.sub(r"(?<=\d),(?=\d)", ".", src)
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None
    try:
        value = eval(compile(tree, "<judge>", "eval"), {"__builtins__": {}}, {})
    except Exception:  # pragma: no cover - chia 0 ...
        return None
    return float(value)


@dataclass(frozen=True)
class Claim:
    expr: str
    stated: float
    value: float | None

    @property
    def ok(self) -> bool:
        return self.value is not None and abs(self.value - self.stated) <= TOL


def arith_claims(text: str) -> list[Claim]:
    """Tìm các phép tính dạng ``<biểu thức> = <số>`` (bỏ qua phương trình có x).

    Chuỗi nhiều phép tính (``500 x 10 : 100 = 50``) được giữ nguyên cả chuỗi:
    chỉ chữ ``x`` nằm giữa hai chữ số mới được hiểu là dấu nhân, còn ``x`` đứng
    một mình vẫn là ẩn số nên phương trình kiểu ``x * 3 = 12`` bị bỏ qua.
    """
    text = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " * ", text or "")
    # Bỏ ký hiệu LaTeX (\( \) \[ \] $) để "( 9 + 6 = 15" không bị coi là ngoặc lỗi.
    text = re.sub(r"\\[()\[\]]|\$", " ", text)
    out: list[Claim] = []
    seen: set[tuple[str, float]] = set()
    for m in re.finditer(r"=", text):
        left = text[: m.start()].rstrip()
        j = len(left)
        while j > 0 and left[j - 1] in _MATH_CHARS:
            j -= 1
        expr = left[j:].strip().lstrip(":,;.").strip()
        right = text[m.end():].lstrip()
        rm = re.match(r"(-?\d+(?:[.,]\d+)?)", right)
        if not expr or not rm or not re.search(r"\d", expr):
            continue
        if not re.search(r"[+\-*/:×÷]", expr) or expr[0] in "+-*/:×÷)":
            continue
        try:
            stated = _to_float(rm.group(1))
        except ValueError:  # pragma: no cover
            continue
        key = (expr, stated)
        if key in seen:
            continue
        seen.add(key)
        out.append(Claim(expr=expr, stated=stated, value=_safe_eval(expr)))
    return out


# --------------------------------------------------------------- từ vựng
_BANNED_VI = ("phương trình", "hai vế", "chuyển vế", "đặt biến")
_BANNED_EN = ("equation", "both sides", "move terms", "let the variable")
_VAGUE = (
    "em có thể làm như thế nào",
    "em làm gì tiếp theo",
    "em hãy làm cẩn thận",
    "đang làm gì ở đây",
    "em thử suy nghĩ xem",
    "em cố gắng lên",
    "what should you do next",
    "how can you solve this",
    "be careful",
)
_PRAISE = (
    "khen em", "làm đúng", "đúng rồi", "đã tìm được", "đã hiểu rõ",
    "giỏi lắm", "tốt lắm", "chính xác", "great job", "well done", "correct",
)


def _language_ok(reply: str, lang: str) -> bool:
    low = (reply or "").lower()
    if not low.strip():
        return False
    if "<think" in low or "</think>" in low or "<|im_end|>" in low:
        return False
    if lang != "vi":
        return True
    if any(ch in _VI_CHARS for ch in low):
        return True
    words = _WORD_RE.findall(low)
    if not words:
        return True
    en_hits = sum(1 for w in words if w in _EN_STOP)
    return en_hits / len(words) < 0.34


def _no_jargon(reply: str, lang: str) -> bool:
    low = (reply or "").lower()
    phrases = _BANNED_EN if lang == "en" else _BANNED_VI
    return not any(p in low for p in phrases)


def _no_vague(reply: str) -> bool:
    low = (reply or "").lower()
    return not any(p in low for p in _VAGUE)


def _question_anchored(reply: str, problem: str) -> bool:
    if "?" not in (reply or ""):
        return True
    problem_numbers = numbers_in(problem)
    if not problem_numbers:
        return True
    reply_numbers = numbers_in(reply)
    return any(_has_value(problem_numbers, n) for n in reply_numbers)


def _no_leak(reply: str, expected: Sequence[float], revealed: set[float]) -> bool:
    unrevealed = [v for v in expected if not any(abs(v - r) <= TOL for r in revealed)]
    if not unrevealed:
        return True
    low = (reply or "").lower()
    if "đáp số" in low or "đáp án" in low or "answer:" in low:
        return False

    # Chỉ coi là lộ đáp số khi con số xuất hiện ở VỊ TRÍ KẾT QUẢ, không phải
    # khi nó chỉ là toán hạng của đề (ví dụ "5 x 11" không phải lộ đáp số 5).
    for claim in arith_claims(reply):
        if claim.value is None:
            continue
        if any(abs(claim.stated - v) <= TOL for v in unrevealed):
            return False

    for m in re.finditer(r"\bx\s*=\s*(\d+(?:[.,]\d+)?)", low):
        try:
            value = _to_float(m.group(1))
        except ValueError:  # pragma: no cover
            continue
        if any(abs(value - v) <= TOL for v in unrevealed):
            return False

    for sentence in re.split(r"[.\n!?]", low):
        if any(k in sentence for k in ("vậy", "số lớn", "số bé", "kết quả", "answer is")):
            values = numbers_in(sentence)
            if any(_has_value(values, v) for v in unrevealed):
                return False
    return True


def _no_praise(reply: str) -> bool:
    low = (reply or "").lower()
    return not any(p in low for p in _PRAISE)


def _parts_mentioned(reply: str, lang: str) -> bool:
    low = (reply or "").lower()
    return ("phần" in low) if lang != "en" else ("part" in low)


def _answer_line_ok(reply: str, expected: Sequence[float]) -> bool:
    lines = [ln for ln in (reply or "").splitlines() if ln.strip()]
    marker = None
    for ln in lines:
        low = ln.lower()
        if "đáp số" in low or "đáp án" in low or low.startswith("answer"):
            marker = ln
            break
    if marker is None:
        return False
    if not expected:
        return True
    values = numbers_in(marker)
    return all(_has_value(values, v) for v in expected)


def _arith_ok(reply: str) -> bool:
    # Chỉ chấm những phép tính đọc được giá trị; chuỗi không đọc được (LaTeX lạ,
    # ngoặc lệch...) không bị coi là "sai" để tránh báo oan.
    return all(c.ok for c in arith_claims(reply) if c.value is not None)


# --------------------------------------------------------------- một lượt
def _as_expected(expected: Any) -> list[float]:
    if expected is None:
        return []
    if isinstance(expected, bool):
        return []
    if isinstance(expected, (int, float)):
        return [float(expected)]
    if hasattr(expected, "answers"):
        return [float(v) for v in expected.answers()]
    if isinstance(expected, dict) and "answer" in expected:
        return _as_expected(expected["answer"])
    if isinstance(expected, (list, tuple, set)):
        return [float(v) for v in expected]
    return []


def _as_revealed(revealed: Any, expected: list[float]) -> set[float]:
    if isinstance(revealed, bool):
        return set(expected) if revealed else set()
    if revealed is None:
        return set()
    if isinstance(revealed, (int, float)):
        return {float(revealed)}
    try:
        return {float(v) for v in revealed}
    except TypeError:
        return set()


def judge_turn(
    reply: str,
    *,
    mode: str = "1",
    lang: str = "vi",
    turn: int = 0,
    problem: str = "",
    expected: Any = (),
    revealed: Any = False,
    require_parts: bool | None = None,
    student_text: str = "",
) -> dict[str, Any]:
    """Chấm một câu trả lời của gia sư. Trả dict có ``labels`` và ``violations``."""
    mode = str(mode)
    lang = "en" if str(lang).lower().startswith("en") else "vi"
    exp = _as_expected(expected)
    rev = _as_revealed(revealed, exp)
    labels: dict[str, bool | None] = {k: None for k in LABELS}
    violations: list[str] = []

    def check(label: str, ok: bool | None, detail: str = "") -> None:
        labels[label] = ok
        if ok is False:
            violations.append(f"{label}:{detail or LABELS[label]}")

    check("B1", _language_ok(reply, lang), "câu trả lời không phải tiếng Việt/ còn thẻ think")
    check("B2", _no_jargon(reply, lang) if mode in ("1", "2") else None, "dùng từ cấm")
    check("B3", _no_vague(reply) if mode == "1" else None, "câu hỏi mơ hồ")
    check(
        "B4",
        _question_anchored(reply, problem) if mode == "1" and "?" in (reply or "") else None,
        "câu hỏi không gắn với số liệu của đề",
    )
    check("B5", _no_leak(reply, exp, rev) if mode in ("1", "2") else None, "lộ đáp số")
    check("B6", _no_praise(reply) if mode == "1" and int(turn) == 0 else None, "khen ở lượt đầu")
    check(
        "B7",
        _parts_mentioned(reply, lang) if mode == "2" and bool(require_parts) else None,
        "không nêu số phần",
    )
    check("B8", _answer_line_ok(reply, exp) if mode == "3" else None, "thiếu/sai dòng Đáp số")
    check("B9", _arith_ok(reply), "có phép tính sai")

    return {
        "turn": int(turn),
        "mode": mode,
        "lang": lang,
        "labels": labels,
        "violations": violations,
        "ok": not violations,
        "claims": [
            {"expr": c.expr, "stated": c.stated, "value": c.value, "ok": c.ok}
            for c in arith_claims(reply)
        ],
    }


# --------------------------------------------------------------- hội thoại
def revealed_values(text: str, expected: Sequence[float]) -> set[float]:
    """Đáp số mà học sinh đã tự nói ra (phép tính đúng hoặc câu chốt)."""
    found: set[float] = set()
    for claim in arith_claims(text):
        if not claim.ok or claim.value is None:
            continue
        for v in expected:
            if abs(claim.value - v) <= TOL:
                found.add(v)
    low = (text or "").lower()
    if any(k in low for k in ("vậy", "đáp số", "đáp án", "x =", "x=")):
        for v in expected:
            if _has_value(numbers_in(text), v):
                found.add(v)
    return found


def judge_dialogue(
    transcript: Sequence[dict[str, Any]],
    *,
    mode: str = "1",
    lang: str = "vi",
    problem: str = "",
    expected: Any = (),
    require_parts: bool | None = None,
) -> dict[str, Any]:
    """Chấm cả hội thoại. ``transcript`` = [{"role": "user"/"assistant", "content": ...}]."""
    exp = _as_expected(expected)
    revealed: set[float] = set()
    solved_turn: int | None = None
    user_count = 0
    turns: list[dict[str, Any]] = []

    for msg in transcript:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        if role == "user":
            user_count += 1
            if user_count > 1:
                revealed |= revealed_values(content, exp)
                if solved_turn is None and exp and all(
                    any(abs(v - r) <= TOL for r in revealed) for v in exp
                ):
                    solved_turn = user_count - 1
            continue
        if role != "assistant":
            continue
        turns.append(
            judge_turn(
                content,
                mode=mode,
                lang=lang,
                turn=max(0, user_count - 1),
                problem=problem,
                expected=exp,
                revealed=set(revealed),
                require_parts=require_parts,
            )
        )

    labels: dict[str, dict[str, Any]] = {}
    failed: list[dict[str, Any]] = []
    for label in LABELS:
        vals = [t["labels"][label] for t in turns if t["labels"][label] is not None]
        passed = sum(1 for v in vals if v)
        labels[label] = {
            "pass": passed,
            "total": len(vals),
            "rate": (passed / len(vals)) if vals else None,
        }
        for t in turns:
            if t["labels"][label] is False:
                failed.append({"turn": t["turn"], "label": label, "detail": LABELS[label]})

    applicable = sum(v["total"] for v in labels.values())
    passed_all = sum(v["pass"] for v in labels.values())
    ok = bool(turns) and not failed
    return {
        "ok": ok,
        "score": (passed_all / applicable) if applicable else 0.0,
        "n_assistant_turns": len(turns),
        "labels": labels,
        "failed": failed,
        "student_solved": solved_turn is not None,
        "student_solved_turn": solved_turn,
        "turns": turns,
    }


__all__ = [
    "Claim",
    "LABELS",
    "arith_claims",
    "judge_dialogue",
    "judge_turn",
    "numbers_in",
    "revealed_values",
]
