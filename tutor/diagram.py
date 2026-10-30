"""diagram.py – Bộ sinh sơ đồ đoạn thẳng ASCII chuẩn mực cho Chế độ 2.

Phân loại dạng toán phù hợp vẽ sơ đồ (Tổng–Tỉ, Hiệu–Tỉ, Tổng–Hiệu, Gấp số lần)
và từ chối vẽ sơ đồ cho dạng không phù hợp (Thể tích, biểu thức số học, đổi đơn vị).
"""
from __future__ import annotations

import re
from typing import Any

from .lang import strip_diacritics


def _norm(text: str) -> str:
    text = strip_diacritics(text or "").lower()
    return text.replace("đ", "d")


# Các dạng bài KHÔNG phù hợp với sơ đồ đoạn thẳng
UNSUITABLE_KEYWORDS = (
    "the tich", "volume", "m3", "dm3", "cm3", "lit", "khoi",
    "doi don vi", "chuyen doi",
    "dien tich hinh chu nhat", "dien tich hinh vuong", "dien tich hinh tron",
    "dien tich hinh thang", "chu vi hinh tron",
)


def is_diagram_suitable(problem_text: str, solution: Any | None = None) -> tuple[bool, str]:
    """Kiểm tra xem bài toán có phù hợp vẽ sơ đồ đoạn thẳng không.

    Trả về (suitable: bool, reason_or_advice: str).
    """
    low = _norm(problem_text)

    # 1. Kiểm tra từ khoá không phù hợp
    for kw in UNSUITABLE_KEYWORDS:
        if kw in low:
            return (
                False,
                f"Dạng toán này (liên quan đến {kw}) không phù hợp để giải bằng phương pháp sơ đồ đoạn thẳng. "
                "Em hãy chuyển sang Chế độ 1 (Gợi mở từng bước) hoặc Chế độ 3 (Bài giải mẫu đầy đủ) "
                "để thầy hướng dẫn cách giải chuẩn nhất nhé!"
            )

    # 2. Biểu thức tìm x thuần tuý không cần sơ đồ đoạn thẳng
    if re.search(r"^[xX\s0-9+\-*/:=().]+$", problem_text.strip()):
        return (
            False,
            "Bài toán tìm x dạng biểu thức số học này phù hợp nhất với phương pháp phân tích thành phần "
            "trong phép tính. Em hãy chuyển sang Chế độ 1 hoặc Chế độ 3 để thầy hướng dẫn từng bước nhé!"
        )

    # 3. Các dạng phù hợp: Tổng-Tỉ, Hiệu-Tỉ, Tổng-Hiệu, Gấp nhiều lần, Hơn kém
    if solution is not None and getattr(solution, "kind", None) in ("sum_ratio", "diff_ratio", "sum_diff"):
        return (True, "")

    suitable_indicators = (
        "tong", "hieu", "ti so", "gap", "lan", "nhieu hon", "it hon",
        "hon", "kem", "so lon", "so be", "tuoi me", "tuoi con", "vien bi",
        "cai keo", "thung thu nhat", "thung thu hai", "lop 4a", "lop 4b",
    )
    if any(ind in low for ind in suitable_indicators):
        return (True, "")

    return (True, "")


def generate_ascii_diagram(
    problem_text: str,
    solution: Any | None = None,
    parts_data: dict[str, float] | None = None,
) -> str:
    """Tạo sơ đồ đoạn thẳng trực quan dạng ASCII và lời hướng dẫn chuẩn sư phạm."""
    # Segment block: |----|
    seg = "|----|"

    if solution is not None and getattr(solution, "kind", None) == "sum_ratio":
        parts = getattr(solution, "parts", None) or parts_data or {}
        k_big = int(parts.get("big", 3))
        k_small = int(parts.get("small", 1))
        tot_parts = k_big + k_small
        answers = solution.answers()
        total_val = sum(answers) if answers else None

        line_small = "Số bé:  " + "|----" * k_small + "|"
        line_big   = "Số lớn: " + "|----" * k_big + "|"
        total_str = f"Tổng hai số: {total_val:g}" if total_val is not None else "Tổng hai số"

        return (
            "Chào em! Bài toán này ta giải bằng phương pháp sơ đồ đoạn thẳng như sau:\n\n"
            "Ta có sơ đồ đoạn thẳng:\n"
            f"{line_small}\n"
            f"{line_big}\n"
            f"{total_str}\n\n"
            f"Theo sơ đồ:\n"
            f"- Số bé gồm {k_small} phần bằng nhau.\n"
            f"- Số lớn gồm {k_big} phần bằng nhau như thế.\n"
            f"- Tổng số phần bằng nhau là: {k_small} + {k_big} = {tot_parts} (phần).\n\n"
            f"Em hãy cho thầy biết: Muốn tìm giá trị của 1 phần, ta lấy {total_str.lower()} "
            f"chia cho tổng số phần bằng nhau là {tot_parts}. Em hãy tính giúp thầy giá trị của 1 phần nhé?"
        )

    if solution is not None and getattr(solution, "kind", None) == "diff_ratio":
        parts = getattr(solution, "parts", None) or parts_data or {}
        k_big = int(parts.get("big", 3))
        k_small = int(parts.get("small", 1))
        diff_parts = k_big - k_small

        line_small = "Số bé:  " + "|----" * k_small + "|"
        line_big   = "Số lớn: " + "|----" * k_big + "|"

        return (
            "Chào em! Ta giải bài toán bằng sơ đồ đoạn thẳng như sau:\n\n"
            "Ta có sơ đồ đoạn thẳng:\n"
            f"{line_small}\n"
            f"{line_big}\n\n"
            f"Theo sơ đồ:\n"
            f"- Số bé gồm {k_small} phần bằng nhau.\n"
            f"- Số lớn gồm {k_big} phần bằng nhau như thế.\n"
            f"- Hiệu số phần bằng nhau là: {k_big} - {k_small} = {diff_parts} (phần).\n\n"
            f"Em hãy thử tính giá trị của 1 phần bằng cách lấy hiệu hai số chia cho {diff_parts} nhé?"
        )

    if solution is not None and getattr(solution, "kind", None) == "sum_diff":
        answers = solution.answers()
        total_val = sum(answers) if answers else None
        diff_val = abs(answers[0] - answers[1]) if len(answers) >= 2 else None

        diff_str = f" (phần hơn: {diff_val:g})" if diff_val else " (phần hơn)"
        line_small = "Số bé:  |----------|"
        line_big   = f"Số lớn: |----------|---{diff_str}---|"
        tot_str = f"Tổng: {total_val:g}" if total_val else "Tổng hai số"

        return (
            "Chào em! Ta biểu diễn bài toán Tổng – Hiệu bằng sơ đồ đoạn thẳng:\n\n"
            "Ta có sơ đồ:\n"
            f"{line_small}\n"
            f"{line_big}\n"
            f"{tot_str}\n\n"
            "Nhìn vào sơ đồ, nếu ta bớt phần hơn của số lớn đi thì hai số sẽ bằng nhau và bằng hai lần số bé.\n"
            "Em hãy cho thầy biết hai lần số bé bằng bao nhiêu nhé?"
        )

    # Sơ đồ mặc định cho bài toán tỉ số chung
    return (
        "Chào em! Ta có sơ đồ đoạn thẳng:\n"
        "Số bé:  |----|\n"
        "Số lớn: |----|----|----|\n\n"
        "Em hãy quan sát sơ đồ và cho thầy biết mỗi đại lượng gồm mấy phần bằng nhau nhé?"
    )
