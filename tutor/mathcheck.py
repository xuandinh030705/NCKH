"""Phân tích đề, giải bài và kiểm tra phép tính — KHÔNG dùng eval chuỗi thô.

Hỗ trợ:
- "Tìm x, biết: ..." với *, ×, :, ÷, /, −, thiếu dấu cách ("x*3-4=11").
- Toán lời văn: Tổng–Hiệu, Tổng–Tỉ, Hiệu–Tỉ, chuyển động đều, phần trăm, hai bước.
- ``arith_claims``: bắt mọi phát biểu "a op b = c" trong câu trả lời của bot.

Mọi phép toán dùng ``fractions.Fraction`` để tránh sai số nhị phân.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .lang import strip_diacritics

# ------------------------------------------------------------------ kiểu dữ liệu
Number = float | int


@dataclass(frozen=True)
class Step:
    """Một bước giải chuẩn (dùng cho fallback, judge, mode 2)."""

    desc: str
    expr: str
    value: float | None = None


@dataclass
class Solution:
    kind: str
    answer: float | tuple[float, ...] | None
    unit: str = ""
    x: float | None = None
    steps: list[Step] = field(default_factory=list)
    parts: dict[str, float] | None = None
    equations: list[str] = field(default_factory=list)

    def answers(self) -> list[float]:
        if self.answer is None:
            return []
        if isinstance(self.answer, tuple):
            return [float(v) for v in self.answer]
        return [float(self.answer)]


@dataclass
class Problem:
    text: str
    kind: str
    numbers: list[float] = field(default_factory=list)
    unit: str = ""
    equation: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Claim:
    expr: str
    stated: float
    correct: float

    @property
    def ok(self) -> bool:
        return abs(self.stated - self.correct) <= 1e-9


# ------------------------------------------------------------------- tiện ích
_OP_NORMALISE = {
    "×": "*", "✕": "*", "·": "*", "⋅": "*", "x": "*", "X": "*",
    "÷": "/", "−": "-", "–": "-", "—": "-", ":": "/",
}
_NUM = r"\d+(?:[.,]\d+)?"
_REL_TOL = 1e-9


def normalise_ops(text: str) -> str:
    """Chuẩn hoá ký hiệu toán mà KHÔNG biến biến x thành dấu nhân."""
    t = text.replace("\u00a0", " ").replace("−", "-").replace("–", "-").replace("—", "-")
    t = t.replace("×", "*").replace("✕", "*").replace("÷", "/")
    return t


def to_float(raw: str) -> float:
    """'2,5' -> 2.5 ; '2.5' -> 2.5 ; '1 000' -> 1000."""
    s = raw.strip().replace(" ", "")
    if "," in s and "." in s:  # 1.234,5 kiểu Việt
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    return float(s)


def fmt_num(value: float | None, *, comma: bool = True) -> str:
    """Hiển thị số gọn: 5 -> '5'; 2.5 -> '2,5' (chuẩn Việt)."""
    if value is None:
        return "?"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    s = f"{value:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", ",") if comma else s


def numbers_in(text: str) -> list[float]:
    """Mọi số trong câu (bỏ số thứ tự lớp 'lớp 4'? vẫn giữ, guard chỉ cần giao)."""
    out: list[float] = []
    for m in re.finditer(_NUM, text or ""):
        try:
            out.append(to_float(m.group(0)))
        except ValueError:  # pragma: no cover
            continue
    return out


def _clean_number(raw: str) -> Fraction:
    return Fraction(str(to_float(raw))).limit_denominator(10**6)


# ------------------------------------------------------- biểu thức tuyến tính
class LinearError(ValueError):
    pass


_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Pow, ast.Call,
)


def _parse_expr(text: str) -> ast.Expression:
    s = text.strip()
    if not s:
        raise LinearError("biểu thức rỗng")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as exc:
        raise LinearError(f"không đọc được biểu thức {s!r}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise LinearError(f"ký tự không cho phép: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in {"x", "X"}:
            raise LinearError(f"biến lạ: {node.id}")
        if isinstance(node, ast.Call):  # chặn mọi lời gọi hàm
            raise LinearError("không cho phép gọi hàm")
        if isinstance(node, ast.Pow) and not (
            isinstance(node.right, ast.Constant) and node.right.value == 1
        ):
            raise LinearError("chỉ hỗ trợ bậc 1")
    return tree


def _linear_form(node: ast.AST) -> tuple[Fraction, Fraction]:
    """Trả (a, b) sao cho biểu thức = a*x + b."""
    if isinstance(node, ast.Expression):
        return _linear_form(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(0), Fraction(str(node.value))
    if isinstance(node, ast.Name) and node.id in {"x", "X"}:
        return Fraction(1), Fraction(0)
    if isinstance(node, ast.UnaryOp):
        a, b = _linear_form(node.operand)
        return (-a, -b) if isinstance(node.op, ast.USub) else (a, b)
    if isinstance(node, ast.BinOp):
        la, lb = _linear_form(node.left)
        ra, rb = _linear_form(node.right)
        if isinstance(node.op, ast.Add):
            return la + ra, lb + rb
        if isinstance(node.op, ast.Sub):
            return la - ra, lb - rb
        if isinstance(node.op, ast.Mult):
            if la == 0:
                return ra * lb, rb * lb
            if ra == 0:
                return la * rb, lb * rb
            raise LinearError("biểu thức không tuyến tính")
        if isinstance(node.op, ast.Div):
            if ra != 0:
                raise LinearError("không chia cho biểu thức chứa x")
            if rb == 0:
                raise LinearError("chia cho 0")
            return la / rb, lb / rb
        if isinstance(node.op, ast.Pow):
            if ra == 0 and rb == 1:
                return la, lb
            raise LinearError("chỉ hỗ trợ bậc 1")
    raise LinearError(f"nút không hỗ trợ: {type(node).__name__}")


def _decompose(node: ast.AST) -> tuple[list[tuple[str, Fraction]], Fraction | None]:
    """Phân rã f(x) thành chuỗi phép toán áp lên x.

    Trả (chain, hằng_số_nếu_không_chứa_x). Mỗi phần tử chain: (op, c).
    """
    if isinstance(node, ast.Expression):
        return _decompose(node.body)
    if isinstance(node, ast.Name) and node.id in {"x", "X"}:
        return [], None
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return [], Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        chain, const = _decompose(node.operand)
        if const is not None:
            return [], -const
        return chain + [("neg", Fraction(0))], None
    if isinstance(node, ast.BinOp):
        lc, lconst = _decompose(node.left)
        rc, rconst = _decompose(node.right)
        if isinstance(node.op, ast.Add):
            if lconst is not None and rconst is not None:
                return [], lconst + rconst
            if lconst is not None:
                return rc + [("+", lconst)], None
            return lc + [("+", rconst if rconst is not None else Fraction(0))], None
        if isinstance(node.op, ast.Sub):
            if lconst is not None and rconst is not None:
                return [], lconst - rconst
            if rconst is not None:
                return lc + [("-", rconst)], None
            if lconst is not None:
                return rc + [("rsub", lconst)], None
        if isinstance(node.op, ast.Mult):
            if lconst is not None and rconst is not None:
                return [], lconst * rconst
            if lconst is not None:
                return rc + [("*", lconst)], None
            if rconst is not None:
                return lc + [("*", rconst)], None
        if isinstance(node.op, ast.Div):
            if rconst is not None and rconst != 0:
                return lc + [("/", rconst)], None
            if lconst is not None and rconst is None:
                return rc + [("rdiv", lconst)], None
    raise LinearError("không phân rã được")


def _solve_chain(chain: list[tuple[str, Fraction]], target: Fraction) -> tuple[Fraction, list[Step]]:
    """Đi ngược chuỗi phép toán để tìm x, sinh các bước giải tiếng Việt."""
    steps: list[Step] = []
    value = target
    for op, c in reversed(chain):
        if op == "+":
            steps.append(Step(f"Bớt {fmt_num(float(c))} ở cả hai bên", f"{fmt_num(float(value))} - {fmt_num(float(c))}",
                              float(value - c)))
            value = value - c
        elif op == "-":
            steps.append(Step(f"Thêm {fmt_num(float(c))} vào cả hai bên", f"{fmt_num(float(value))} + {fmt_num(float(c))}",
                              float(value + c)))
            value = value + c
        elif op == "*":
            steps.append(Step(f"Chia cả hai bên cho {fmt_num(float(c))}", f"{fmt_num(float(value))} : {fmt_num(float(c))}",
                              float(value / c)))
            value = value / c
        elif op == "/":
            steps.append(Step(f"Nhân cả hai bên với {fmt_num(float(c))}", f"{fmt_num(float(value))} x {fmt_num(float(c))}",
                              float(value * c)))
            value = value * c
        elif op == "rsub":
            steps.append(Step("x là số trừ, lấy số bị trừ trừ đi hiệu", f"{fmt_num(float(c))} - {fmt_num(float(value))}",
                              float(c - value)))
            value = c - value
        elif op == "rdiv":
            steps.append(Step("x là số chia, lấy số bị chia chia cho thương", f"{fmt_num(float(c))} : {fmt_num(float(value))}",
                              float(c / value)))
            value = c / value
        elif op == "neg":
            value = -value
    return value, steps


def parse_linear(text: str) -> Problem | None:
    """Nhận diện dạng 'Tìm x, biết: ...' / 'Find x: ...' (kể cả không dấu) hoặc phương trình trực tiếp."""
    t = normalise_ops(text)
    low = _norm(t)
    has_lead = bool(re.search(r"(tim|find)\s*x", low))
    has_eq_x = bool(re.search(r"(?<![a-zA-Z0-9])[xX](?![a-zA-Z0-9])", t)) and ("=" in t)
    if not (has_lead or has_eq_x):
        return None
    # Cắt bỏ phần dẫn "Tìm x, biết:" / "Find x:" (trên chuỗi đã bỏ dấu) rồi xử lý dấu ':'
    leads = list(re.finditer(r"(?:biet|find\s*x|tim\s*x)", low))
    body = low[leads[-1].end():] if leads else low
    body = body.lstrip(" :,.").strip()
    if "=" not in body:
        return None
    body = body.replace(":", "/")
    # '3x' -> '3*x' (nhưng không đụng tới 'x' đứng một mình)
    body = re.sub(r"(\d)\s*([xX])", r"\1*\2", body)
    left, _, right = body.partition("=")
    left = re.sub(r"[^0-9xX+\-*/().\s]", " ", left).strip()
    right = re.sub(r"[^0-9xX+\-*/().\s]", " ", right).strip()
    if not left or not right:
        return None
    solved = _solve_linear_equation(left, right)
    if solved is None:
        return None
    x, _steps = solved
    if not math.isfinite(float(x)):
        return None
    return Problem(
        text=text,
        kind="linear",
        equation=f"{left} = {right}",
        numbers=numbers_in(text),
        data={"x": float(x)},
    )


def _solve_linear_equation(left: str, right: str) -> tuple[Fraction, list[Step]] | None:
    """Giải f(x) = g(x) bậc 1, kể cả dạng có x ở mẫu (24 : x = 6)."""
    # 1) Nghiệm chính xác bằng dạng tuyến tính hai vế
    x_exact: Fraction | None = None
    try:
        la, lb = _linear_form(_parse_expr(left))
        ra, rb = _linear_form(_parse_expr(right))
        if la != ra:
            x_exact = (rb - lb) / (la - ra)
    except LinearError:
        pass

    # 2) Các bước giải: ưu tiên vế chứa x, vế kia là hằng số
    steps: list[Step] = []
    try:
        lchain, lconst = _decompose(_parse_expr(left))
        rchain, rconst = _decompose(_parse_expr(right))
        if lconst is None and rconst is not None:
            x_steps, steps = _solve_chain(lchain, rconst)
            x_exact = x_exact if x_exact is not None else x_steps
        elif rconst is None and lconst is not None:
            x_steps, steps = _solve_chain(rchain, lconst)
            x_exact = x_exact if x_exact is not None else x_steps
    except (LinearError, ZeroDivisionError):
        steps = []

    if x_exact is None:
        return None
    if not steps:
        steps = [Step("Tìm x", f"x = {fmt_num(float(x_exact))}", float(x_exact))]
    return x_exact, steps


# ------------------------------------------------------------- bài lời văn
_UNIT_WORDS = ("cái kẹo", "viên bi", "quả cam", "quả bóng", "cái bút", "quyển vở",
               "quyển truyện", "tuổi", "ki-lô-gam", "kg", "mét", "km", "giờ",
               "phút", "nghìn đồng", "đồng", "lít", "cái", "quả")
_TOTAL_WORDS = ("tổng cộng", "tổng số", "tất cả", "cả hai", "tổng", "in total", "total")
_DIFF_WORDS = ("nhiều hơn", "ít hơn", "hơn", "kém", "more", "less", "difference", "older")


def _norm(text: str) -> str:
    """Chữ thường, bỏ dấu — để so khớp cả tiếng Việt không dấu và tiếng Anh."""
    text = strip_diacritics(text or "").lower()
    return text.replace("đ", "d")


def _kw_index(low: str, key: str) -> int:
    """Vị trí từ khoá theo ranh giới từ (tránh 'hieu' khớp trong 'nhieu')."""
    k = _norm(key)
    if not k:
        return -1
    m = re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", low)
    return m.start() if m else -1


def _kw(low: str, *keys: str) -> bool:
    """Khớp từ khoá (có ranh giới từ) trên chuỗi đã bỏ dấu."""
    return any(_kw_index(low, k) >= 0 for k in keys)


def _find_unit(text: str) -> str:
    low = _norm(text)
    for u in _UNIT_WORDS:
        if _norm(u) in low:
            return u
    return ""


def _two_numbers_after(text: str, *keywords: str) -> list[float]:
    """Các số xuất hiện ngay sau từ khoá đầu tiên tìm được (text đã bỏ dấu)."""
    low = _norm(text)
    for kw in keywords:
        idx = _kw_index(low, kw)
        if idx >= 0:
            nums = numbers_in(low[idx + len(_norm(kw)):])
            if nums:
                return nums
    return []


def _parse_time_hours(text: str) -> float | None:
    """'2 giờ 30 phút' -> 2.5 ; '30 phút' -> 0.5 ; '2.5 hours' -> 2.5."""
    low = _norm(text)
    m = re.search(rf"({_NUM})\s*(?:gio|hours?|h)\s*(?:({_NUM})\s*(?:phut|minutes?))?", low)
    if m:
        h = to_float(m.group(1))
        if m.group(2):
            h += to_float(m.group(2)) / 60.0
        return h
    m = re.search(rf"({_NUM})\s*(?:phut|minutes?)", low)
    if m:
        return to_float(m.group(1)) / 60.0
    return None


def parse_problem(text: str) -> Problem | None:
    """Phân loại đề (vi có dấu / vi không dấu / en). None nếu không nhận dạng được."""
    if not text or not text.strip():
        return None
    linear = parse_linear(text)
    if linear:
        return linear

    low = _norm(text)
    nums = numbers_in(text)
    unit = _find_unit(text)

    # --- chuyển động đều
    if _kw(low, "van toc", "speed", "quang duong", "distance", "thoi gian",
           "how long", "how far"):
        s_m = re.search(rf"({_NUM})\s*km", low)
        v_m = re.search(rf"({_NUM})\s*km\s*/\s*(?:gio|h|hour)", low)
        t = _parse_time_hours(text)
        distance = to_float(s_m.group(1)) if s_m else None
        speed = to_float(v_m.group(1)) if v_m else None
        if _kw(low, "how far", "quang duong bao nhieu", "di duoc bao nhieu", "travel"):
            kind = "motion_s"
        elif _kw(low, "how long", "bao lau", "tinh thoi gian", "het bao nhieu thoi gian"):
            kind = "motion_t"
        else:
            kind = "motion_v"
        return Problem(text=text, kind=kind, numbers=nums, unit=unit,
                       data={"s": distance, "v": speed, "t": t})

    # --- phần trăm
    if "%" in text or _kw(low, "phan tram", "percent"):
        pct_m = re.search(rf"({_NUM})\s*%", text)
        if pct_m:
            total = nums[0] if nums else None
            return Problem(text=text, kind="percent", numbers=nums, unit=unit,
                           data={"pct": to_float(pct_m.group(1)), "total": total,
                                 "ask_left": _kw(low, "con lai", "left")})

    # --- tỉ số (gấp ... lần / ... times hoặc tỉ số dạng a/b)
    frac_ratio = re.search(r"(?:ti so|ratio)\s*(?:la|cua hai so la|:)?\s*(\d+)\s*[/:]\s*(\d+)", low)
    ratio = re.search(rf"({_NUM})\s*(?:lan|times)", low)
    if frac_ratio or ratio:
        if frac_ratio:
            num_a, num_b = float(frac_ratio.group(1)), float(frac_ratio.group(2))
            small_p, big_p = min(num_a, num_b), max(num_a, num_b)
            # k = big / small
            k = big_p / small_p if small_p > 0 else 1.0
            tot_parts = small_p + big_p
            diff_parts = big_p - small_p
        else:
            k = to_float(ratio.group(1))
            small_p, big_p = 1.0, k
            tot_parts = k + 1.0
            diff_parts = k - 1.0

        has_total = _kw(low, *_TOTAL_WORDS)
        is_diff = (not has_total) and _kw(low, "hon", "kem", "hieu", "more", "older", "difference")
        base = nums[0] if nums else None
        if base is not None:
            if is_diff:
                return Problem(text=text, kind="diff_ratio", numbers=nums, unit=unit,
                               data={"diff": base, "k": k, "small_parts": small_p, "big_parts": big_p, "diff_parts": diff_parts})
            return Problem(text=text, kind="sum_ratio", numbers=nums, unit=unit,
                           data={"total": base, "k": k, "small_parts": small_p, "big_parts": big_p, "total_parts": tot_parts})

    # --- tổng - hiệu
    if _kw(low, *_TOTAL_WORDS) and _kw(low, *_DIFF_WORDS):
        total = nums[0] if nums else None
        got = _two_numbers_after(text, "nhieu hon", "it hon", "hon", "more", "difference", "less")
        diff = got[0] if got else (nums[1] if len(nums) > 1 else None)
        if total is not None and diff is not None and diff < total:
            return Problem(text=text, kind="sum_diff", numbers=nums, unit=unit,
                           data={"total": total, "diff": diff})

    # --- hai bước mua bán
    if _kw(low, "mua", "buys", "buy", "tra lai", "change") and len(nums) >= 3:
        return Problem(text=text, kind="twostep", numbers=nums, unit="nghìn đồng", data={})

    return Problem(text=text, kind="unknown", numbers=nums, unit=unit)


def solve(problem: str | Problem, **_: Any) -> Solution | None:
    """Giải đề đã nhận dạng. Trả None nếu không đủ dữ kiện."""
    p = parse_problem(problem) if isinstance(problem, str) else problem
    if p is None:
        return None
    unit = p.unit
    d = p.data

    if p.kind == "linear":
        try:
            body = p.equation or ""
            left, _, right = normalise_ops(body).partition("=")
            left = re.sub(r"[^0-9xX+\-*/().\s]", " ", left).strip()
            right = re.sub(r"[^0-9xX+\-*/().\s]", " ", right).strip()
            solved = _solve_linear_equation(left, right)
            if solved is None:
                return None
            x, steps = solved
        except (LinearError, ZeroDivisionError):
            return None
        xf = float(x)
        return Solution(kind="linear", answer=xf, x=xf, unit="", steps=steps,
                        equations=[p.equation or ""])

    if p.kind == "sum_diff":
        total, diff = Fraction(str(d["total"])), Fraction(str(d["diff"]))
        big = (total + diff) / 2
        small = (total - diff) / 2
        steps = [
            Step("Số lớn bằng (tổng + hiệu) : 2", f"({fmt_num(float(total))} + {fmt_num(float(diff))}) : 2", float(big)),
            Step("Số bé bằng (tổng - hiệu) : 2", f"({fmt_num(float(total))} - {fmt_num(float(diff))}) : 2", float(small)),
        ]
        return Solution(kind="sum_diff", answer=(float(big), float(small)), unit=unit, steps=steps)

    if p.kind == "sum_ratio":
        total = Fraction(str(d["total"]))
        small_p = Fraction(str(d.get("small_parts", 1)))
        big_p = Fraction(str(d.get("big_parts", d["k"])))
        tot_parts = small_p + big_p
        unit_val = total / tot_parts
        small = unit_val * small_p
        big = unit_val * big_p
        steps = [
            Step("Tổng số phần bằng nhau", f"{fmt_num(float(small_p))} + {fmt_num(float(big_p))}", float(tot_parts)),
            Step(f"Số bé ({fmt_num(float(small_p))} phần)", f"{fmt_num(float(total))} : {fmt_num(float(tot_parts))} x {fmt_num(float(small_p))}" if small_p != 1 else f"{fmt_num(float(total))} : {fmt_num(float(tot_parts))}", float(small)),
            Step(f"Số lớn ({fmt_num(float(big_p))} phần)", f"{fmt_num(float(total))} - {fmt_num(float(small))}", float(big)),
        ]
        return Solution(kind="sum_ratio", answer=(float(big), float(small)), unit=unit, steps=steps,
                        parts={"big": float(big_p), "small": float(small_p), "total_parts": float(tot_parts)})

    if p.kind == "diff_ratio":
        diff, k = Fraction(str(d["diff"])), Fraction(str(d["k"]))
        small = diff / (k - 1) if k != 1 else None
        if small is None:
            return None
        big = small + diff
        steps = [
            Step("Hiệu số phần bằng nhau", f"{fmt_num(float(k))} - 1", float(k - 1)),
            Step("Số bé (1 phần)", f"{fmt_num(float(diff))} : {fmt_num(float(k - 1))}", float(small)),
            Step("Số lớn", f"{fmt_num(float(small))} + {fmt_num(float(diff))}", float(big)),
        ]
        return Solution(kind="diff_ratio", answer=(float(big), float(small)), unit=unit, steps=steps,
                        parts={"big": float(k), "small": 1.0, "total_parts": float(k)})

    if p.kind in ("motion_v", "motion_s", "motion_t"):
        s, v, t = d.get("s"), d.get("v"), d.get("t")
        if p.kind == "motion_v" and s is not None and t:
            ans, expr, desc = s / t, f"{fmt_num(s)} : {fmt_num(t)}", "Vận tốc = quãng đường : thời gian"
        elif p.kind == "motion_s" and v is not None and t:
            ans, expr, desc = v * t, f"{fmt_num(v)} x {fmt_num(t)}", "Quãng đường = vận tốc x thời gian"
        elif p.kind == "motion_t" and s is not None and v:
            ans, expr, desc = s / v, f"{fmt_num(s)} : {fmt_num(v)}", "Thời gian = quãng đường : vận tốc"
        else:
            return None
        unit = {"motion_v": "km/giờ", "motion_s": "km", "motion_t": "giờ"}[p.kind]
        return Solution(kind=p.kind, answer=float(ans), unit=unit,
                        steps=[Step(desc, expr, float(ans))])

    if p.kind == "percent":
        pct, total = Fraction(str(d["pct"])), Fraction(str(d["total"]))
        part = total * pct / 100
        if d.get("ask_left"):
            ans = total - part
            steps = [Step(f"{fmt_num(float(pct))}% của {fmt_num(float(total))}", f"{fmt_num(float(total))} x {fmt_num(float(pct))} : 100", float(part)),
                     Step("Số còn lại", f"{fmt_num(float(total))} - {fmt_num(float(part))}", float(ans))]
        else:
            ans = part
            steps = [Step(f"{fmt_num(float(pct))}% của {fmt_num(float(total))}", f"{fmt_num(float(total))} x {fmt_num(float(pct))} : 100", float(ans))]
        return Solution(kind="percent", answer=float(ans), unit=unit, steps=steps)

    if p.kind == "twostep":
        nums = p.numbers
        if len(nums) >= 3:
            n, price, paid = nums[0], nums[1], nums[2]
            cost = n * price
            left = paid - cost
            steps = [Step("Số tiền phải trả", f"{fmt_num(n)} x {fmt_num(price)}", cost),
                     Step("Số tiền trả lại", f"{fmt_num(paid)} - {fmt_num(cost)}", left)]
            return Solution(kind="twostep", answer=left, unit="nghìn đồng", steps=steps)
        return None

    return None


# ---------------------------------------------------------- kiểm tra phép tính
# Chuỗi nhiều phép tính (vd "500 x 10 : 100 = 50") phải được đọc trọn chuỗi;
# nếu chỉ bắt "a op b = c" thì phần đuôi bị cắt và sinh ra báo sai oan.
_CLAIM_EXPR = rf"{_NUM}(?:\s*[+\-*/x×÷:]\s*{_NUM})+"
_CLAIM_RE = re.compile(
    rf"(?<![\d.,])(-?\s*{_CLAIM_EXPR})\s*=\s*(-?\s*{_NUM})(?!\d)(?![.,]\d)"
)

_ARITH_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Load,
)


def _eval_arith(expr: str) -> float | None:
    """Tính giá trị biểu thức số học (không có ẩn số), trả None nếu không đọc được."""
    s = normalise_ops(expr)
    s = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", s)
    s = re.sub(r"(?<=\d),(?=\d)", ".", s).replace(":", "/").replace(" ", "")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ARITH_NODES):
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None
    try:
        return float(eval(compile(tree, "<claim>", "eval"), {"__builtins__": {}}, {}))
    except Exception:  # pragma: no cover - chia 0
        return None


def arith_claims(text: str) -> list[Claim]:
    """Tìm mọi phát biểu '<chuỗi phép tính> = <số>' và kiểm tra đúng/sai."""
    out: list[Claim] = []
    src = normalise_ops(text or "")
    for m in _CLAIM_RE.finditer(src):
        expr_raw, stated_raw = m.group(1), m.group(2)
        correct = _eval_arith(expr_raw)
        if correct is None:
            continue
        try:
            stated = to_float(stated_raw)
        except ValueError:  # pragma: no cover
            continue
        out.append(Claim(expr=m.group(0).strip(), stated=stated, correct=correct))
    return out


# --------------------------------------------------------------- ghi chú ẩn
def hidden_hint(problem: str, student_text: str, cfg: Any | None = None) -> str:
    """Ghi chú ẩn (mặc định TẮT) chèn vào lượt user cuối để model bám đáp số."""
    if cfg is None or not bool(cfg.get("math.hidden_hint", False)):
        return ""
    sol = solve(problem) if problem else None
    if sol is None or sol.answer is None:
        return ""
    if isinstance(sol.answer, tuple):
        answer_txt = " và ".join(fmt_num(v) for v in sol.answer)
    else:
        answer_txt = fmt_num(sol.answer) + (f" {sol.unit}" if sol.unit else "")
    claims = arith_claims(student_text)
    verdict = "chưa có phép tính"
    if claims:
        last = claims[-1]
        verdict = f"'{last.expr}' {'ĐÚNG' if last.ok else 'SAI'}"
    tpl = (cfg.get("math.hidden_hint_template") if cfg is not None else None) or (
        "[GHI CHÚ ẨN - không nhắc lại với học sinh] Đáp số đúng: {answer}. "
        "Phép tính gần nhất của học sinh: {verdict}."
    )
    return tpl.format(answer=answer_txt, verdict=verdict)


# =========================================================================
# KIỂM TRA ĐỀ BÀI: THIẾU DỮ KIỆN & MÂU THUẪN (BUG 04, BUG 05)
# =========================================================================
def detect_missing_data(text: str) -> tuple[bool, str]:
    """Phát hiện đề bài bị thiếu dữ kiện không thể giải được (BUG 04 - C07, CLI-03)."""
    low = _norm(text)
    nums = numbers_in(text)

    # 1. Hình chữ nhật:
    if "hinh chu nhat" in low:
        has_chu_vi = "chu vi" in low or "nua chu vi" in low
        asks_dien_tich = any(k in low for k in (
            "tinh dien tich", "dien tich la bao nhieu", "dien tich bang bao nhieu",
            "dien tich hinh chu nhat", "dien tich cua hinh chu nhat"
        ))
        asks_chu_vi = any(k in low for k in ("tinh chu vi", "chu vi la bao nhieu", "chu vi bang bao nhieu"))
        has_dimensions = any(k in low for k in (
            "chieu dai", "chieu rong", "dai hon", "rong bang", "gap", "ti so",
            "hon chieu rong", "kem chieu dai", "nua chu vi"
        ))

        # Chỉ có chu vi mà hỏi diện tích, không có kích thước khác (len(nums) <= 1)
        if has_chu_vi and (asks_dien_tich or "dien tich" in low) and not has_dimensions and len(nums) <= 1:
            return (
                True,
                "Đề bài này còn thiếu dữ kiện rồi em nhé. Để tính diện tích hình chữ nhật, "
                "chúng ta cần biết cả số đo chiều dài và chiều rộng (hoặc mối liên hệ giữa chúng). "
                "Đề bài hiện tại chỉ mới cho biết chu vi nên chưa đủ để tính diện tích. "
                "Em hãy kiểm tra lại xem đề bài có cho thêm chiều dài hoặc chiều rộng không nhé!"
            )

        # Chỉ có diện tích mà hỏi chu vi, không có kích thước khác
        if "dien tich" in low and asks_chu_vi and not has_dimensions and len(nums) <= 1:
            return (
                True,
                "Đề bài này còn thiếu dữ kiện rồi em nhé. Để tính chu vi hình chữ nhật, "
                "chúng ta cần biết cả chiều dài và chiều rộng. Đề bài chỉ mới cho biết diện tích "
                "mà chưa có số đo của cạnh nào. Em hãy kiểm tra lại đề bài nhé!"
            )

    # 2. Chuyển động đều thiếu dữ kiện:
    if any(k in low for k in ("van toc", "quang duong", "thoi gian", "chuyen dong", "o to", "xe may", "xe dap")):
        asks_v = any(k in low for k in ("tinh van toc", "van toc la bao nhieu", "van toc cua"))
        asks_s = any(k in low for k in ("tinh quang duong", "quang duong dai bao nhieu", "di duoc bao xa"))
        asks_t = any(k in low for k in ("tinh thoi gian", "bao lau", "het bao nhieu thoi gian"))
        has_rel = any(k in low for k in ("gap", "ti so", "hon", "kem", "nghi"))

        if (asks_v or asks_s or asks_t) and len(nums) == 1 and not has_rel:
            return (
                True,
                "Đề bài toán chuyển động này còn thiếu dữ kiện rồi em nhé. Trong chuyển động đều, "
                "ta cần ít nhất hai trong ba đại lượng (quãng đường, vận tốc, thời gian) để tìm đại lượng còn lại. "
                "Em hãy kiểm tra lại đề bài xem có dữ kiện nào bị thiếu không nhé!"
            )

    # 3. Trung bình cộng thiếu số lượng:
    if "trung binh cong" in low and ("tong" in low or "la" in low) and len(nums) == 1:
        if not any(k in low for k in ("hai so", "ba so", "bon so", "nam so", "2 so", "3 so", "4 so", "5 so")):
            return (
                True,
                "Đề bài này còn thiếu dữ kiện về số lượng các số hạng cần tính trung bình cộng rồi em nhé. "
                "Em hãy kiểm tra lại xem đề bài có nêu rõ là trung bình cộng của mấy số không nhé!"
            )

    return (False, "")


def detect_contradiction(text: str) -> tuple[bool, str]:
    """Phát hiện đề bài có dữ kiện mâu thuẫn hoặc vô lý (BUG 05 - F10, I14)."""
    low = _norm(text)
    nums = numbers_in(text)

    # 1. Tổng - Hiệu: Hiệu > Tổng
    has_total = any(k in low for k in ("tong hai so", "tong cua hai so", "tong so", "tong bang", "tong la"))
    has_diff = any(k in low for k in ("hieu hai so", "hieu cua hai so", "hieu la", "hieu bang", "hon nhau", "kem nhau", "nhieu hon"))

    if has_total and has_diff and len(nums) >= 2:
        after_total = _two_numbers_after(text, "tong hai so la", "tong cua hai so la", "tong so la", "tong la", "tong bang", "tong")
        after_diff = _two_numbers_after(text, "hieu hai so la", "hieu cua hai so la", "hieu so la", "hieu la", "hieu bang", "hieu", "nhieu hon", "it hon")
        t_val = after_total[0] if after_total else nums[0]
        d_val = after_diff[0] if after_diff else nums[1]

        if t_val > 0 and d_val > 0 and d_val > t_val:
            return (
                True,
                f"Đề bài có dữ kiện chưa hợp lý rồi em nhé: Trong bài toán tìm hai số tự nhiên, "
                f"hiệu của hai số ({fmt_num(d_val)}) không thể lớn hơn tổng của hai số đó ({fmt_num(t_val)}) được "
                f"(vì nếu hiệu lớn hơn tổng thì số bé sẽ là số âm, không phù hợp với chương trình tiểu học). "
                f"Em hãy kiểm tra lại đề bài xem có bị ghi nhầm số liệu không nhé!"
            )

    return (False, "")


# =========================================================================
# ĐÁNH GIÁ CÂU TRẢ LỜI CỦA HỌC SINH (BUG 01 - C03)
# =========================================================================
@dataclass
class StudentAnswerEval:
    has_answer: bool = False
    verdict: str = "UNKNOWN"  # "FINAL_CORRECT", "FINAL_WRONG", "INTERMEDIATE_CORRECT", "INTERMEDIATE_WRONG", "CALC_WRONG", "CALC_CORRECT", "QUESTION", "UNKNOWN"
    is_final: bool = False
    claimed_val: float | None = None
    expected_val: float | None = None
    claims: list[Claim] = field(default_factory=list)
    detail: str = ""


def evaluate_student_answer(
    student_text: str,
    problem_text: str,
    solution: Solution | None = None,
    mode: str = "1",
    turn: int = 0,
) -> StudentAnswerEval:
    """Đánh giá câu trả lời của học sinh bằng logic Python độc lập (chống đảo ngược đúng/sai)."""
    text = (student_text or "").strip()
    if not text:
        return StudentAnswerEval(verdict="UNKNOWN")

    low = _norm(text)
    claims = arith_claims(text)
    sol = solution or solve(problem_text)
    expected_answers = sol.answers() if sol else []

    # 1. Kiểm tra nếu học sinh đặt câu hỏi hoặc chưa hiểu
    question_kw = ("chua hieu", "khong hieu", "lam the nao", "huong dan", "la gi", "sao lai", "thay oi giup", "kho qua")
    if any(k in low for k in question_kw) and not claims and "=" not in text:
        return StudentAnswerEval(has_answer=False, verdict="QUESTION", detail="Học sinh đặt câu hỏi hoặc chưa hiểu")

    # 2. Kiểm tra nếu có phép tính cụ thể: a op b = c
    if claims:
        bad_claims = [c for c in claims if not c.ok]
        if bad_claims:
            first_bad = bad_claims[0]
            return StudentAnswerEval(
                has_answer=True,
                verdict="CALC_WRONG",
                claimed_val=first_bad.stated,
                expected_val=first_bad.correct,
                claims=claims,
                detail=f"Phép tính '{first_bad.expr}' sai (kết quả đúng là {first_bad.correct:g})",
            )
        # Tất cả phép tính đều đúng số học
        for c in claims:
            if any(abs(c.correct - a) < 1e-9 for a in expected_answers):
                return StudentAnswerEval(
                    has_answer=True,
                    verdict="FINAL_CORRECT",
                    is_final=True,
                    claimed_val=c.correct,
                    expected_val=c.correct,
                    claims=claims,
                    detail=f"Phép tính '{c.expr}' đưa ra đáp số đúng {c.correct:g}",
                )
        return StudentAnswerEval(
            has_answer=True,
            verdict="INTERMEDIATE_CORRECT",
            claims=claims,
            detail=f"Phép tính '{claims[0].expr}' chính xác",
        )

    # 3. Kiểm tra dạng gán giá trị: x = 8 / x = 6 / kết quả là 8
    x_match = re.search(r"\b[xX]\s*=\s*(-?\d+(?:[.,]\d+)?)", text)
    if x_match:
        val = to_float(x_match.group(1))
        if expected_answers:
            if any(abs(val - a) < 1e-9 for a in expected_answers):
                return StudentAnswerEval(
                    has_answer=True,
                    verdict="FINAL_CORRECT",
                    is_final=True,
                    claimed_val=val,
                    expected_val=expected_answers[0],
                    detail=f"Học sinh trả lời x = {val:g} ĐÚNG",
                )
            return StudentAnswerEval(
                has_answer=True,
                verdict="FINAL_WRONG",
                is_final=True,
                claimed_val=val,
                expected_val=expected_answers[0],
                detail=f"Học sinh trả lời x = {val:g} SAI (đáp số đúng là {expected_answers[0]:g})",
            )

    # 4. Kiểm tra nếu học sinh trả lời số trực tiếp (ví dụ: "8", "8 ạ", "kết quả là 8", "em ra 8")
    nums = numbers_in(text)
    if nums and expected_answers:
        last_num = nums[-1]
        is_closing = any(k in low for k in ("ket qua", "dap an", "dap so", "em ra", "em tinh ra", "bang", "la", "so lon", "so be"))
        if is_closing or len(nums) == 1:
            if any(abs(last_num - a) < 1e-9 for a in expected_answers):
                return StudentAnswerEval(
                    has_answer=True,
                    verdict="FINAL_CORRECT",
                    is_final=True,
                    claimed_val=last_num,
                    expected_val=expected_answers[0],
                    detail=f"Học sinh trả lời {last_num:g} ĐÚNG đáp số cuối",
                )
            # Kiểm tra xem có khớp bước trung gian nào không
            if sol and sol.steps:
                for idx, st in enumerate(sol.steps):
                    if st.value is not None and abs(last_num - st.value) < 1e-9:
                        return StudentAnswerEval(
                            has_answer=True,
                            verdict="INTERMEDIATE_CORRECT",
                            claimed_val=last_num,
                            expected_val=st.value,
                            detail=f"Học sinh tính đúng bước trung gian: {st.desc} = {last_num:g}",
                        )
            return StudentAnswerEval(
                has_answer=True,
                verdict="FINAL_WRONG",
                is_final=True,
                claimed_val=last_num,
                expected_val=expected_answers[0],
                detail=f"Học sinh trả lời {last_num:g} SAI (đáp số đúng là {expected_answers[0]:g})",
            )

    return StudentAnswerEval(verdict="UNKNOWN")


__all__ = [
    "Claim",
    "Problem",
    "Solution",
    "Step",
    "StudentAnswerEval",
    "arith_claims",
    "detect_contradiction",
    "detect_missing_data",
    "evaluate_student_answer",
    "fmt_num",
    "hidden_hint",
    "normalise_ops",
    "numbers_in",
    "parse_problem",
    "solve",
    "to_float",
]
