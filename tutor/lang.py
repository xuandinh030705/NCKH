"""Nhận diện ngôn ngữ cho đề bài và câu trả lời (vi / en).

Tiếng Việt không dấu vẫn phải nhận là "vi" — đây là yêu cầu bắt buộc của đề tài.
"""
from __future__ import annotations

import re

# Ký tự có dấu đặc trưng tiếng Việt (bao gồm cả chữ hoa)
_VI_DIACRITIC = re.compile(
    r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
    r"ĂÂĐÊÔƠƯÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ]"
)

# Từ tiếng Việt viết không dấu (đủ phổ biến để tách khỏi tiếng Anh)
_VI_NODIAC = {
    "khong", "cua", "em", "thay", "co", "bai", "toan", "tim", "biet", "tinh", "hoi",
    "bao", "nhieu", "mot", "hai", "ba", "bon", "nam", "sau", "bay", "tam", "chin",
    "muoi", "tram", "nghin", "lop", "hoc", "sinh", "giao", "vien", "phep", "cong",
    "tru", "nhan", "chia", "bang", "vay", "the", "nay", "do", "la", "va", "voi",
    "trong", "ngoai", "tren", "duoi", "sau", "truoc", "moi", "ca", "cai", "keo",
    "ban", "chung", "minh", "di", "quang", "duong", "van", "toc", "thoi", "gian",
    "gio", "phut", "km", "me", "con", "ong", "ba", "anh", "chi", "tong", "hieu",
    "ti", "so", "phan", "tram", "tang", "giam", "mua", "ban", "con", "lai",
}

# Từ tiếng Anh hay gặp trong đề/bài giải
_EN_WORDS = {
    "the", "is", "are", "how", "what", "calculate", "find", "solve", "and", "of",
    "student", "teacher", "answer", "step", "first", "then", "let", "we", "need",
    "to", "you", "your", "car", "distance", "speed", "time", "hour", "hours",
    "total", "more", "less", "each", "many", "much", "there", "this", "that",
    "with", "for", "from", "have", "has", "will", "can", "should", "please",
}

_WORD = re.compile(r"[A-Za-zÀ-ỹĐđ]+", re.UNICODE)


def strip_diacritics(text: str) -> str:
    """'Tìm x, biết' -> 'Tim x, biet' (giữ nguyên số và dấu câu)."""
    import unicodedata

    text = (text or "").replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def detect(text: str) -> str:
    """Trả "vi" hoặc "en". Mặc định "vi" khi không đủ dấu hiệu."""
    t = (text or "").strip()
    if not t:
        return "vi"
    if _VI_DIACRITIC.search(t):
        return "vi"
    words = _tokens(t)
    if not words:
        return "vi"
    vi_hits = sum(1 for w in words if w in _VI_NODIAC)
    en_hits = sum(1 for w in words if w in _EN_WORDS)
    # Có dấu hiệu tiếng Việt không dấu rõ ràng thì ưu tiên tiếng Việt
    if vi_hits >= 2 and vi_hits >= en_hits:
        return "vi"
    if en_hits >= 3 and en_hits > vi_hits * 2:
        return "en"
    return "vi"


def is_english(text: str) -> bool:
    """True nếu câu được xem là tiếng Anh (dùng cho guard)."""
    return detect(text) == "en"


def has_vietnamese_diacritics(text: str) -> bool:
    return bool(_VI_DIACRITIC.search(text or ""))


def english_word_ratio(text: str) -> float:
    """Tỉ lệ từ tiếng Anh trên tổng số từ (0..1) — phục vụ guard chống lẫn tiếng Anh."""
    words = _tokens(text)
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _EN_WORDS)
    return hits / len(words)


__all__ = ["detect", "english_word_ratio", "has_vietnamese_diacritics", "is_english"]
