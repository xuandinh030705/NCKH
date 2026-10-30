"""Sinh bài toán tiểu học có seed — đáp số do CODE tính, không do LLM.

9 dạng bài theo yêu cầu đề tài, mỗi Item có 3 biến thể ngôn ngữ (vi / vi không dấu / en),
bước giải chuẩn, thông tin số phần (cho chế độ 2), độ khó 1–3.
"""
from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from .lang import strip_diacritics
from .mathcheck import fmt_num


# ------------------------------------------------------------------ kiểu dữ liệu
@dataclass(frozen=True)
class Item:
    id: str
    grade: int
    topic: str
    lang: str
    text: str
    variants: dict[str, str]
    answer: float | tuple[float, ...]
    unit: str = ""
    steps: tuple[dict[str, object], ...] = ()
    parts: dict[str, float] | None = None
    difficulty: int = 1

    def with_lang(self, lang: str) -> "Item":
        text = self.variants.get(lang, self.text)
        return replace(self, lang=lang, text=text)

    def answers(self) -> list[float]:
        return list(self.answer) if isinstance(self.answer, tuple) else [float(self.answer)]

    def answer_text(self) -> str:
        if isinstance(self.answer, tuple):
            return " và ".join(fmt_num(v) for v in self.answer) + (f" {self.unit}" if self.unit else "")
        return fmt_num(self.answer) + (f" {self.unit}" if self.unit else "")


TOPIC_GRADES: dict[str, tuple[int, ...]] = {
    "linear_add": (2,),
    "linear_sub": (2,),
    "linear_mul": (3,),
    "linear_div": (3,),
    "linear_two_step": (3, 4),
    "sum_diff": (4,),
    "sum_ratio": (4,),
    "diff_ratio": (4,),
    "motion": (5,),
    "twostep_word": (2, 3),
    "percent": (5,),
}


def _steps(*rows: tuple[str, str, float]) -> tuple[dict[str, object], ...]:
    return tuple({"desc": d, "expr": e, "value": v} for d, e, v in rows)


def _time_text(hours: float, lang: str) -> str:
    whole = int(hours)
    mins = int(round((hours - whole) * 60))
    if lang == "en":
        if whole == 0:
            return f"{mins} minutes"
        return f"{whole} hours {mins} minutes" if mins else f"{whole} hours"
    if lang == "vi_nodiac":
        if whole == 0:
            return f"{mins} phut"
        return f"{whole} gio {mins} phut" if mins else f"{whole} gio"
    if whole == 0:
        return f"{mins} phút"
    return f"{whole} giờ {mins} phút" if mins else f"{whole} giờ"


UNIT_EN: dict[str, str] = {
    "cái kẹo": "candies", "viên bi": "marbles", "quả cam": "oranges",
    "quả bóng": "balls", "cái bút": "pens", "kg gạo": "kg of rice",
    "tuổi": "years old", "quyển truyện": "books", "nghìn đồng": "thousand dong",
    "quyển vở": "notebooks", "kg": "kg", "km": "km", "km/giờ": "km/h", "giờ": "hours",
}


def _variants(vi: str, en: str) -> dict[str, str]:
    return {"vi": vi, "vi_nodiac": strip_diacritics(vi), "en": en}


# --------------------------------------------------------------- từng dạng bài
def _gen_linear(kind: str, rng: random.Random, grade: int) -> dict:
    if kind == "linear_add":
        x = rng.randint(3, 60)
        a = rng.randint(2, 39)
        b = x + a
        vi = f"Tìm x, biết: x + {a} = {b}"
        en = f"Find x: x + {a} = {b}"
        steps = _steps(("Lấy tổng trừ đi số hạng đã biết", f"{b} - {a}", x))
    elif kind == "linear_sub":
        x = rng.randint(10, 99)
        a = rng.randint(2, min(50, x - 1))
        b = x - a
        vi = f"Tìm x, biết: x - {a} = {b}"
        en = f"Find x: x - {a} = {b}"
        steps = _steps(("Lấy hiệu cộng với số trừ", f"{b} + {a}", x))
    elif kind == "linear_mul":
        a = rng.randint(2, 9)
        x = rng.randint(2, 10)
        b = a * x
        vi = f"Tìm x, biết: x * {a} = {b}"
        en = f"Find x: x * {a} = {b}"
        steps = _steps(("Lấy tích chia cho thừa số đã biết", f"{b} : {a}", x))
    elif kind == "linear_div":
        a = rng.randint(2, 9)
        q = rng.randint(2, 10)
        x = a * q
        vi = f"Tìm x, biết: x : {a} = {q}"
        en = f"Find x: x : {a} = {q}"
        steps = _steps(("Lấy thương nhân với số chia", f"{q} x {a}", x))
    else:  # linear_two_step
        shape = rng.choice(("mul_sub", "mul_add", "div_add"))
        a = rng.randint(2, 9)
        if shape == "mul_sub":
            x = rng.randint(3, 12)
            b = rng.randint(2, 20)
            c = a * x - b
            vi = f"Tìm x, biết: x * {a} - {b} = {c}"
            en = f"Find x: x * {a} - {b} = {c}"
            steps = _steps(("Tìm nhóm chưa biết (x * a)", f"{c} + {b}", a * x),
                           ("Tìm x", f"{a * x} : {a}", x))
        elif shape == "mul_add":
            x = rng.randint(3, 12)
            b = rng.randint(2, 20)
            c = a * x + b
            vi = f"Tìm x, biết: x * {a} + {b} = {c}"
            en = f"Find x: x * {a} + {b} = {c}"
            steps = _steps(("Tìm nhóm chưa biết (x * a)", f"{c} - {b}", a * x),
                           ("Tìm x", f"{a * x} : {a}", x))
        else:
            x = rng.randint(3, 12) * a
            b = rng.randint(2, 20)
            c = x // a + b
            vi = f"Tìm x, biết: x : {a} + {b} = {c}"
            en = f"Find x: x : {a} + {b} = {c}"
            steps = _steps(("Tìm nhóm chưa biết (x : a)", f"{c} - {b}", x // a),
                           ("Tìm x", f"{x // a} x {a}", x))
    return {"text_vi": vi, "text_en": en, "answer": float(x), "unit": "",
            "steps": steps, "difficulty": 1 if kind in ("linear_add", "linear_sub", "linear_mul", "linear_div") else 2,
            "parts": None}


def _gen_sum_diff(rng: random.Random) -> dict:
    small = rng.randint(5, 40)
    diff = rng.randint(2, 18)
    big = small + diff
    total = big + small
    unit = rng.choice(("cái kẹo", "viên bi", "quả cam"))
    unit_en = UNIT_EN[unit]
    vi = (f"Hai bạn An và Bình có tổng cộng {total} {unit}. An có nhiều hơn Bình {diff} {unit}. "
          f"Hỏi mỗi bạn có bao nhiêu {unit}?")
    en = (f"An and Binh have {total} {unit_en} in total. An has {diff} more {unit_en} than Binh. "
          f"How many {unit_en} does each of them have?")
    steps = _steps(("Số lớn = (tổng + hiệu) : 2", f"({total} + {diff}) : 2", big),
                   ("Số bé = (tổng - hiệu) : 2", f"({total} - {diff}) : 2", small))
    return {"text_vi": vi, "text_en": en, "answer": (float(big), float(small)), "unit": unit,
            "steps": steps, "difficulty": 2, "parts": None}


def _gen_sum_ratio(rng: random.Random) -> dict:
    k = rng.randint(2, 5)
    small = rng.randint(4, 30)
    total = small * (k + 1)
    big = total - small
    unit = rng.choice(("kg gạo", "quả bóng", "cái bút"))
    unit_en = UNIT_EN[unit]
    vi = (f"Một cửa hàng có {total} {unit}, trong đó số lượng loại lớn gấp {k} lần "
          f"số lượng loại nhỏ. Hỏi mỗi loại có bao nhiêu {unit}?")
    en = (f"A shop has {total} {unit_en}; the larger kind is {k} times the smaller kind. "
          f"How many {unit_en} of each kind are there?")
    steps = _steps(("Tổng số phần bằng nhau", f"1 + {k}", k + 1),
                   ("Số bé (1 phần)", f"{total} : {k + 1}", small),
                   ("Số lớn", f"{small} x {k}", big))
    return {"text_vi": vi, "text_en": en, "answer": (float(big), float(small)), "unit": unit,
            "steps": steps, "difficulty": 3,
            "parts": {"big": float(k), "small": 1.0, "total_parts": float(k + 1)}}


def _gen_diff_ratio(rng: random.Random) -> dict:
    k = rng.randint(2, 5)
    small = rng.randint(4, 20)
    diff = small * (k - 1)
    big = small + diff
    unit = rng.choice(("tuổi", "cái kẹo", "quyển truyện"))
    unit_en = UNIT_EN[unit]
    vi = (f"Mẹ hơn con {diff} {unit}. Tuổi mẹ gấp {k} lần tuổi con. "
          f"Hỏi mỗi người bao nhiêu {unit}?")
    en = (f"A mother is {diff} {unit_en} older than her child. Her age is {k} times the child's age. "
          f"How old is each of them?")
    if unit != "tuổi":
        vi = (f"Lan có nhiều hơn Hoa {diff} {unit}. Số {unit} của Lan gấp {k} lần số {unit} của Hoa. "
              f"Hỏi mỗi bạn có bao nhiêu {unit}?")
        en = (f"Lan has {diff} more {unit_en} than Hoa. Lan has {k} times as many {unit_en} as Hoa. "
              f"How many {unit_en} does each of them have?")
    steps = _steps(("Hiệu số phần bằng nhau", f"{k} - 1", k - 1),
                   ("Số bé (1 phần)", f"{diff} : {k - 1}", small),
                   ("Số lớn", f"{small} + {diff}", big))
    return {"text_vi": vi, "text_en": en, "answer": (float(big), float(small)), "unit": unit,
            "steps": steps, "difficulty": 3,
            "parts": {"big": float(k), "small": 1.0, "total_parts": float(k)}}


def _gen_motion(rng: random.Random) -> dict:
    v = rng.choice((20, 30, 40, 45, 50, 60))
    t = rng.choice((0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
    s = v * t
    ask = rng.choice(("v", "s", "t"))
    t_vi = _time_text(t, "vi")
    t_en = _time_text(t, "en")
    if ask == "v":
        vi = f"Một ô tô đi quãng đường dài {fmt_num(s)} km trong thời gian {t_vi}. Hãy tính vận tốc của ô tô."
        en = f"A car travels {fmt_num(s)} km in {t_en}. Find its speed."
        ans, unit = v, "km/giờ"
        steps = _steps(("Vận tốc = quãng đường : thời gian", f"{fmt_num(s)} : {fmt_num(t)}", v))
    elif ask == "s":
        vi = (f"Một ô tô đi với vận tốc {v} km/giờ trong {t_vi}. "
              f"Hỏi ô tô đi được quãng đường bao nhiêu ki-lô-mét?")
        en = f"A car drives at {v} km/h for {t_en}. How far does it travel?"
        ans, unit = s, "km"
        steps = _steps(("Quãng đường = vận tốc x thời gian", f"{v} x {fmt_num(t)}", s))
    else:
        vi = (f"Một ô tô đi quãng đường dài {fmt_num(s)} km với vận tốc {v} km/giờ. "
              f"Hỏi ô tô đi hết bao nhiêu thời gian?")
        en = f"A car travels {fmt_num(s)} km at {v} km/h. How long does the trip take?"
        ans, unit = t, "giờ"
        steps = _steps(("Thời gian = quãng đường : vận tốc", f"{fmt_num(s)} : {v}", t))
    return {"text_vi": vi, "text_en": en, "answer": float(ans), "unit": unit,
            "steps": steps, "difficulty": 2, "parts": None}


def _gen_twostep(rng: random.Random) -> dict:
    n = rng.randint(3, 9)
    price = rng.randint(3, 12)
    paid = n * price + rng.choice((5, 10, 15, 20))
    cost = n * price
    left = paid - cost
    vi = (f"Lan mua {n} quyển vở, mỗi quyển giá {price} nghìn đồng. Lan đưa cô bán hàng "
          f"{paid} nghìn đồng. Hỏi cô bán hàng phải trả lại Lan bao nhiêu nghìn đồng?")
    en = (f"Lan buys {n} notebooks at {price} thousand dong each. She pays {paid} thousand dong. "
          f"How much change does she get?")
    steps = _steps(("Số tiền phải trả", f"{n} x {price}", cost),
                   ("Số tiền trả lại", f"{paid} - {cost}", left))
    return {"text_vi": vi, "text_en": en, "answer": float(left), "unit": "nghìn đồng",
            "steps": steps, "difficulty": 2, "parts": None}


def _gen_percent(rng: random.Random) -> dict:
    pct = rng.choice((10, 20, 25, 50))
    total = 100 * rng.randint(1, 6)
    part = total * pct // 100
    ask_left = rng.random() < 0.6
    vi = (f"Một cửa hàng có {total} kg gạo, đã bán {pct}% số gạo. "
          f"Hỏi cửa hàng còn lại bao nhiêu ki-lô-gam gạo?") if ask_left else (
          f"Một cửa hàng có {total} kg gạo, đã bán {pct}% số gạo. "
          f"Hỏi cửa hàng đã bán bao nhiêu ki-lô-gam gạo?")
    en = (f"A shop has {total} kg of rice and has sold {pct}% of it. "
          f"How many kg are left?") if ask_left else (
          f"A shop has {total} kg of rice and has sold {pct}% of it. "
          f"How many kg were sold?")
    if ask_left:
        ans = total - part
        steps = _steps((f"{pct}% của {total}", f"{total} x {pct} : 100", part),
                       ("Số gạo còn lại", f"{total} - {part}", ans))
    else:
        ans = part
        steps = _steps((f"{pct}% của {total}", f"{total} x {pct} : 100", ans))
    return {"text_vi": vi, "text_en": en, "answer": float(ans), "unit": "kg",
            "steps": steps, "difficulty": 2, "parts": None}


_GENERATORS = {
    "linear_add": lambda rng, g: _gen_linear("linear_add", rng, g),
    "linear_sub": lambda rng, g: _gen_linear("linear_sub", rng, g),
    "linear_mul": lambda rng, g: _gen_linear("linear_mul", rng, g),
    "linear_div": lambda rng, g: _gen_linear("linear_div", rng, g),
    "linear_two_step": lambda rng, g: _gen_linear("linear_two_step", rng, g),
    "sum_diff": lambda rng, g: _gen_sum_diff(rng),
    "sum_ratio": lambda rng, g: _gen_sum_ratio(rng),
    "diff_ratio": lambda rng, g: _gen_diff_ratio(rng),
    "motion": lambda rng, g: _gen_motion(rng),
    "twostep_word": lambda rng, g: _gen_twostep(rng),
    "percent": lambda rng, g: _gen_percent(rng),
}


# ------------------------------------------------------------------ API chính
def generate(
    n: int = 10,
    seed: int = 0,
    grades: Sequence[int] | None = None,
    topics: Iterable[str] | None = None,
    style: str = "mixed",
) -> list[Item]:
    """Sinh ``n`` bài. Tái lập được 100% với cùng (n, seed, grades, topics)."""
    if n < 0:
        raise ValueError("n phải >= 0")
    rng = random.Random(seed)
    grades_set = set(grades or (2, 3, 4, 5))
    topic_pool = [t for t in (topics or _GENERATORS) if t in _GENERATORS]
    if not topic_pool:
        topic_pool = list(_GENERATORS)
    langs = ["vi"] if style == "vi" else (["en"] if style == "en" else ["vi", "vi_nodiac", "en"])

    # chia đều topic để mỗi dạng có mặt
    plan: list[str] = []
    while len(plan) < n:
        batch = topic_pool[:]
        rng.shuffle(batch)
        plan.extend(batch)
    plan = plan[:n]

    items: list[Item] = []
    for i, topic in enumerate(plan):
        allowed = [g for g in TOPIC_GRADES[topic] if g in grades_set]
        grade = rng.choice(allowed) if allowed else rng.choice(TOPIC_GRADES[topic])
        data = _GENERATORS[topic](rng, grade)
        lang = langs[i % len(langs)]
        variants = _variants(data["text_vi"], data["text_en"])
        items.append(
            Item(
                id=f"g{grade}-{topic}-{seed}-{i:04d}",
                grade=grade,
                topic=topic,
                lang=lang,
                text=variants[lang],
                variants=variants,
                answer=data["answer"],
                unit=data["unit"],
                steps=data["steps"],
                parts=data["parts"],
                difficulty=data["difficulty"],
            )
        )
    return items


def summary(items: Sequence[Item]) -> dict[str, int]:
    """Đếm theo topic (tiện cho báo cáo)."""
    out: dict[str, int] = {}
    for it in items:
        out[it.topic] = out.get(it.topic, 0) + 1
    return out


__all__ = ["Item", "TOPIC_GRADES", "generate", "strip_diacritics", "summary"]
