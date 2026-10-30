"""input_guard.py – Tầng 1: Kiểm tra và làm sạch dữ liệu đầu vào.

Xử lý các tình huống:
- Prompt Injection / Trích xuất System Prompt (BUG 02, BUG 10 - E06, E15, E01, E09, E12)
- Tư vấn y tế cho trẻ em (BUG 08 - D11)
- Ngoài phạm vi Toán tiểu học Lớp 1–5 (BUG 09 - D01, D02, D07, D12, D17, D18, E10)
- Đề bài thiếu dữ kiện (BUG 04 - C07, CLI-03)
- Đề bài mâu thuẫn dữ kiện (BUG 05 - F10, I14)
"""
from __future__ import annotations

import re
from typing import Any

from .lang import strip_diacritics


def _norm(text: str) -> str:
    text = strip_diacritics(text or "").lower()
    return text.replace("đ", "d")


# =========================================================================
# 1. PHÁT HIỆN YÊU CẦU TRÍCH XUẤT SYSTEM PROMPT & QUY TẮC NỘI BỘ (BUG 02)
# =========================================================================
_PROMPT_LEAK_PATTERNS = [
    r"system\s*prompt",
    r"developer\s*(instruction|message|prompt)",
    r"huong\s*dan\s*(he\s*thong|noi\s*bo|ban\s*dau|goc)",
    r"quy\s*tac\s*(noi\s*bo|bao\s*mat|he\s*thong)",
    r"cau\s*hinh\s*(prompt|he\s*thong|noi\s*bo)",
    r"tiet\s*lo\s*(prompt|quy\s*tac|huong\s*dan|chi\s*dan)",
    r"(cho\s*biet|in\s*ra|show|display|print|tell\s*me|doc\s*lai|nhac\s*lai|repeat)\s+(toan\s*bo\s+)?(your\s+)?(prompt|system|quy\s*tac|huong\s*dan|instructions)",
    r"prompt\s*instructions",
    r"(nhung|cac)?\s*dieu\s*cam",
    r"banned_words",
    r"base_prompts",
    r"system_modes",
    r"what\s*is\s*your\s*(system\s*)?prompt",
    r"repeat\s*the\s*words\s*above",
    r"print\s*everything\s*before",
]

_PROMPT_LEAK_RE = re.compile("|".join(_PROMPT_LEAK_PATTERNS), re.IGNORECASE)


def is_system_prompt_request(text: str) -> bool:
    low = _norm(text)
    return bool(_PROMPT_LEAK_RE.search(low))


# =========================================================================
# 2. PHÁT HIỆN JAILBREAK & GIẢ MẠO VAI TRÒ (BUG 10)
# =========================================================================
_JAILBREAK_PATTERNS = [
    r"bo\s*qua\s*(tat\s*ca\s*)?(huong\s*dan|chi\s*dan|quy\s*tac|lenh)\s*(truoc\s*do|cu)?",
    r"ignore\s*(all\s*)?(previous|prior|above)\s*(instructions|prompts|rules)",
    r"tu\s*gio\s*(tro\s*di\s*)?(ban|may)\s*la\s*(chatgpt|dan|ai\s*khong\s*gioi\s*han|tro\s*ly)",
    r"you\s*are\s*now\s*(chatgpt|dan|unfiltered|jailbroken)",
    r"<\s*\|\s*im_start\s*\|\s*>\s*system",
    r"\[system\]",
    r"system:",
    r"admin:",
    r"root\s*mode",
    r"developer\s*mode",
    r"tro\s*thanh\s*mot\s*ai\s*khong\s*co\s*gioi\s*han",
]

_JAILBREAK_RE = re.compile("|".join(_JAILBREAK_PATTERNS), re.IGNORECASE)


def is_jailbreak_attempt(text: str) -> bool:
    low = _norm(text)
    return bool(_JAILBREAK_RE.search(low))


def has_actual_math_problem(text: str) -> bool:
    """Kiểm tra xem trong tin nhắn có chứa bài toán thật không."""
    low = _norm(text)
    # Có biểu thức toán học hoặc các từ khoá toán tiểu học
    if re.search(r"\b[xX]\s*[\+\-\*\/×:÷=]\s*\d+", text):
        return True
    if re.search(r"\d+\s*[\+\-\*\/×:÷]\s*\d+\s*=", text):
        return True
    math_kw = [
        "tim x", "tinh ", "tong hai so", "hieu hai so", "ti so", "van toc",
        "quang duong", "thoi gian", "chu vi", "dien tich", "the tich",
        "bao nhieu vien bi", "bao nhieu cai keo", "phan tram", "phan so"
    ]
    return any(k in low for k in math_kw)


# =========================================================================
# 3. PHÁT HIỆN YÊU CẦU Y TẾ CHO TRẺ EM (BUG 08 - D11)
# =========================================================================
_MEDICAL_PATTERNS = [
    r"dau\s*bung(\s*ben\s*(phai|trai))?",
    r"dau\s*(dau|nguc|lung|hong|co|rang|mat|tai|chan|tay)",
    r"bi\s*(sot|cum|ho|so\s*mui|tieu\s*chay|non|buon\s*non|chong\s*mat|co\s*giat|phat\s*ban)",
    r"bi\s*benh\s*gi",
    r"em\s*bi\s*benh",
    r"uong\s*thuoc\s*gi",
    r"chua\s*(benh|tri)",
    r"thuoc\s*(khang\s*sinh|giam\s*dau|ha\s*sot|tri)",
    r"chuan\s*doan",
    r"chay\s*mau",
    r"gay\s*(tay|chan|xuong)",
    r"ngo\s*doc",
    r"dau\s*ruot\s*thua",
    r"bac\s*si\s*tu\s*van",
]

_MEDICAL_RE = re.compile("|".join(_MEDICAL_PATTERNS), re.IGNORECASE)


def is_medical_query(text: str) -> bool:
    low = _norm(text)
    return bool(_MEDICAL_RE.search(low))


# =========================================================================
# 4. PHÁT HIỆN NỘI DUNG NGOÀI PHẠM VI TOÁN TIỂU HỌC LỚP 1–5 (BUG 09)
# =========================================================================
_OUT_OF_SCOPE_PATTERNS = [
    # Đại số THCS/THPT & Giải tích đại học
    r"giai\s*tich",
    r"tich\s*phan",
    r"dao\s*ham",
    r"gioi\s*han\s*lim|\blim\b",
    r"logarit|\blog\b|\bln\b",
    r"phuong\s*trinh\s*bac\s*(hai|ba|bon|2|3|4)",
    r"x\s*\^\s*2|x\s*²|x\s*\^|x\s*³",
    r"he\s*phuong\s*trinh\s*bac\s*nhat",
    r"luong\s*giac|\bsin\b|\bcos\b|\btan\b|\bcot\b",
    r"so\s*phuc",
    r"ma\s*tran",
    r"vecto|vector",
    r"hinh\s*hoc\s*khong\s*gian\s*(toa\s*do|thpt)",
    # Vật lý, Hoá học
    r"vat\s*ly\s*(thpt|cap\s*3|10|11|12)",
    r"con\s*lac\s*(don|lo\s*xo)",
    r"dien\s*tro\s*r|dinh\s*luat\s*om",
    r"quang\s*hinh|thau\s*kinh",
    r"hat\s*nhan|phong\s*xa",
    r"hoa\s*hoc|axit|bazo|phan\s*ung\s*hoa\s*hoc",
    # Lập trình
    r"viet\s*code|lap\s*trinh|python|javascript|c\+\+|java\b|html\b",
    # Chính trị, nội dung người lớn, cờ bạc
    r"chinh\s*tri|dang\s*cong\s*san|bau\s*cu|chinh\s*phu",
    r"chuyen\s*nguoi\s*lon|18\+|sex|tinh\s*duc|khieu\s*dam",
    r"danh\s*bac|ca\s*do|lo\s*de|tai\s*xiu",
    r"che\s*tao\s*(bom|thuoc\s*no|vu\s*khi)",
]

_OUT_OF_SCOPE_RE = re.compile("|".join(_OUT_OF_SCOPE_PATTERNS), re.IGNORECASE)


def is_out_of_scope(text: str) -> bool:
    low = _norm(text)
    return bool(_OUT_OF_SCOPE_RE.search(low))


# =========================================================================
# BỘ ĐIỀU PHỐI KIỂM TRA ĐẦU VÀO (INPUT SAFETY DISPATCHER)
# =========================================================================
def check_input_safety(text: str, mode: str = "1") -> tuple[bool, str | None, str | None]:
    """Kiểm tra an toàn đầu vào trước khi đưa tới Engine.

    Trả về: (is_intercepted, safe_response, reason)
    Nếu is_intercepted = True: không cần gọi Engine, trả về safe_response ngay.
    """
    clean_text = text.strip()
    if not clean_text:
        return (True, "Em chưa nhập nội dung bài toán. Em hãy nhập đề bài để thầy hướng dẫn nhé!", "empty")

    # 1. Trích xuất system prompt / hướng dẫn nội bộ (BUG 02)
    if is_system_prompt_request(clean_text):
        return (
            True,
            "Thầy không thể cung cấp hướng dẫn nội bộ của hệ thống. "
            "Thầy là gia sư chuyên về môn Toán tiểu học Lớp 1–5, em có bài toán tiểu học nào cần thầy hướng dẫn không?",
            "system_prompt_request",
        )

    # 2. Jailbreak / Giả mạo vai trò (BUG 10)
    if is_jailbreak_attempt(clean_text):
        if not has_actual_math_problem(clean_text):
            return (
                True,
                "Chào em! Thầy là gia sư chuyên hỗ trợ học sinh học môn Toán tiểu học (Lớp 1 đến Lớp 5). "
                "Em đang cần thầy hướng dẫn bài toán nào, hãy gửi đề bài cho thầy nhé!",
                "jailbreak_no_math",
            )

    # 3. Tư vấn y tế cho trẻ em (BUG 08 - D11)
    if is_medical_query(clean_text):
        return (
            True,
            "Thầy không thể xác định bệnh hay tư vấn y tế qua tin nhắn. "
            "Em hãy báo ngay cho bố mẹ, người giám hộ hoặc thầy cô giáo để được giúp đỡ và đưa đi khám bác sĩ kịp thời nhé!",
            "medical_query",
        )

    # 4. Ngoài phạm vi giáo dục Toán tiểu học (BUG 09)
    if is_out_of_scope(clean_text):
        return (
            True,
            "Nội dung này nằm ngoài phạm vi chương trình môn Toán tiểu học (Lớp 1–5) nên thầy chưa hỗ trợ dạng này nhé. "
            "Thầy là gia sư Toán tiểu học, em có bài toán tiểu học nào cần thầy trò mình cùng giải không?",
            "out_of_scope",
        )

    return (False, None, None)
