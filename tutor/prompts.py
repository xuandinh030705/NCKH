"""System prompt 3 chế độ — TRÍCH NGUYÊN VĂN từ notebook, chỉ nối thêm appendix.

Nguyên tắc bất biến:
- ``BASE_PROMPTS`` đúng bằng ``SYSTEM_MODES`` của cell cuối notebook.
- Mọi chỉnh sửa là *appendix* nối sau, có ghi chép trong docs/PROMPT_CHANGELOG.md.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from .notebook_source import NotebookError, load

# --------------------------------------------------------------------- nguồn
try:
    _NB = load()
    MODE_NAMES: dict[str, str] = dict(_NB.mode_names)
    BASE_PROMPTS: dict[str, str] = {
        key: _NB.system_modes[name] for key, name in _NB.mode_names.items()
    }
    DEMO_DIALOGUE: tuple[str, ...] = _NB.demo_dialogue
    SOURCE = "reference/nckh-math.ipynb (trích trực tiếp bằng json+ast)"
except NotebookError:  # pragma: no cover - chỉ xảy ra khi người dùng xoá notebook
    from . import _nb_snapshot as _snap

    MODE_NAMES = dict(_snap.MODE_NAMES)
    BASE_PROMPTS = dict(_snap.BASE_PROMPTS)
    DEMO_DIALOGUE = tuple(_snap.DEMO_DIALOGUE)
    SOURCE = "tutor/_nb_snapshot.py (bản chụp dự phòng)"

DEFAULT_MODE = "1"
MODE_KEYS: tuple[str, ...] = tuple(sorted(BASE_PROMPTS))


def mode_key(name_or_key: str) -> str:
    """Chuẩn hoá '1' / '1. Gia sư...' / tên đầy đủ -> khoá '1'."""
    s = str(name_or_key).strip()
    if s in BASE_PROMPTS:
        return s
    for key, name in MODE_NAMES.items():
        if s == name or s.lower() == name.lower():
            return key
    if s[:1] in BASE_PROMPTS:
        return s[:1]
    raise ValueError(f"Chế độ không hợp lệ: {name_or_key!r}. Chọn 1, 2 hoặc 3.")


def mode_name(mode: str) -> str:
    return MODE_NAMES[mode_key(mode)]


# ------------------------------------------------------------------ appendix
# Dòng chống tiếng Anh lấy ý từ cell PROMPT_MODES trước của notebook.
_VI_LANGUAGE_RULE = (
    "- BẮT BUỘC DÙNG 100% TIẾNG VIỆT. TUYỆT ĐỐI KHÔNG SUY NGHĨ HOẶC TRẢ LỜI "
    "BẰNG TIẾNG ANH (No English allowed)."
)
_EN_LANGUAGE_RULE = (
    "- You MUST answer 100% in English. Do NOT reply in Vietnamese or any other language."
)

_VI_RULE_REMINDER = (
    "- KHÔNG dùng các từ: 'phương trình', 'hai vế', 'chuyển vế', 'đặt biến'.\n"
    "- Mỗi lượt chỉ hỏi tối đa 1 câu; câu hỏi phải chứa ít nhất một con số lấy từ đề.\n"
    "- Không mở đầu bằng lời khen khi học sinh chưa làm gì."
)
_EN_RULE_REMINDER = (
    "- Never use the words: 'equation', 'both sides', 'move terms', 'let the variable'.\n"
    "- Ask at most ONE question per turn, and it must contain a number taken from the problem.\n"
    "- Do not start with praise before the student has done anything."
)

_MODE2_VI = (
    "- Nêu rõ: số lớn/số bé vẽ mấy phần bằng nhau, tổng (hoặc hiệu) ứng với bao nhiêu phần, "
    "và tổng số phần bằng nhau là bao nhiêu."
)
_MODE2_EN = (
    "- State clearly how many equal parts each quantity gets, which total/difference maps to "
    "the parts, and the total number of equal parts."
)
_MODE3_VI = "- Kết thúc bài giải bằng đúng một dòng 'Đáp số: ...' kèm đơn vị (nếu đề có đơn vị)."
_MODE3_EN = "- End the solution with exactly one line 'Answer: ...' including the unit if any."

APPENDIX: dict[tuple[str, str], str] = {}
for _m in ("1", "2", "3"):
    for _lang in ("vi", "en"):
        _rule = _VI_LANGUAGE_RULE if _lang == "vi" else _EN_LANGUAGE_RULE
        _extra = ""
        if _m == "1":
            _extra = _VI_RULE_REMINDER if _lang == "vi" else _EN_RULE_REMINDER
        elif _m == "2":
            _extra = _MODE2_VI if _lang == "vi" else _MODE2_EN
        elif _m == "3":
            _extra = _MODE3_VI if _lang == "vi" else _MODE3_EN
        _head = "QUY TẮC BỔ SUNG (BẮT BUỘC):" if _lang == "vi" else "ADDITIONAL RULES (MANDATORY):"
        APPENDIX[(_m, _lang)] = "\n".join(p for p in (_head, _rule, _extra) if p)


def _example_lines() -> str:
    """Lấy nguyên văn các dòng 'Ví dụ:' có sẵn trong prompt gốc (không gõ lại)."""
    base = BASE_PROMPTS["1"]
    found = re.findall(r"-\s*Ví dụ:\s*'(.+?)'", base)
    return "\n".join(f"- Ví dụ: '{line}'" for line in found)


@lru_cache(maxsize=32)
def build_system(
    mode: str = DEFAULT_MODE,
    lang: str = "vi",
    *,
    append_language: bool = True,
    append_reminder: bool = True,
    few_shot: bool = False,
) -> str:
    """Prompt gốc (nguyên văn) + appendix theo chế độ/ngôn ngữ."""
    key = mode_key(mode)
    lang = "en" if str(lang).lower().startswith("en") else "vi"
    parts = [BASE_PROMPTS[key]]
    block: list[str] = []
    if append_language:
        block.append(_VI_LANGUAGE_RULE if lang == "vi" else _EN_LANGUAGE_RULE)
    if append_reminder:
        if key == "1":
            block.append(_VI_RULE_REMINDER if lang == "vi" else _EN_RULE_REMINDER)
        elif key == "2":
            block.append(_MODE2_VI if lang == "vi" else _MODE2_EN)
        elif key == "3":
            block.append(_MODE3_VI if lang == "vi" else _MODE3_EN)
    if block:
        head = "QUY TẮC BỔ SUNG (BẮT BUỘC):" if lang == "vi" else "ADDITIONAL RULES (MANDATORY):"
        parts.append(head + "\n" + "\n".join(block))
    if few_shot and key == "1":
        examples = _example_lines()
        if examples:
            title = "VÍ DỤ CÁCH HỎI ĐÚNG (bám số liệu):" if lang == "vi" else "GOOD QUESTION EXAMPLES:"
            parts.append(f"{title}\n{examples}")
    return "\n\n".join(p for p in parts if p)


def build_system_from_cfg(mode: str, lang: str, cfg: Any | None = None) -> str:
    """Như build_system nhưng đọc cờ từ config.yaml."""
    if cfg is None:
        return build_system(mode, lang)
    return build_system(
        mode,
        lang,
        append_language=bool(cfg.get("prompts.append_language_rule", True)),
        append_reminder=bool(cfg.get("prompts.append_rule_reminder", True)),
        few_shot=bool(cfg.get("prompts.few_shot", False)),
    )


__all__ = [
    "APPENDIX",
    "BASE_PROMPTS",
    "DEFAULT_MODE",
    "DEMO_DIALOGUE",
    "MODE_KEYS",
    "MODE_NAMES",
    "SOURCE",
    "build_system",
    "build_system_from_cfg",
    "mode_key",
    "mode_name",
]
