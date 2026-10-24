"""Trích nội dung chuẩn từ notebook Kaggle bằng json + ast.

Không có chuỗi nào của notebook bị gõ lại trong mã nguồn: mọi prompt, tham số và
đề test đều được đọc trực tiếp từ ``reference/nckh-math.ipynb``.

Cách dùng::

    from tutor.notebook_source import load
    nb = load()
    nb.system_modes["1"]        # nguyên văn CONCRETE_SOCRATIC_PROMPT
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOK = PROJECT_ROOT / "reference" / "nckh-math.ipynb"


class NotebookError(RuntimeError):
    """Không đọc được notebook nguồn."""


def _eval_const(node: ast.AST, ns: dict[str, Any]) -> Any:
    """Đánh giá biểu thức hằng số an toàn (không dùng eval chuỗi thô)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in ns:
            return ns[node.id]
        raise NotebookError(f"Không tìm thấy biến {node.id!r} trong notebook")
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            if k is None:  # {**other}
                out.update(_eval_const(v, ns))
            else:
                out[_eval_const(k, ns)] = _eval_const(v, ns)
        return out
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        vals = [_eval_const(e, ns) for e in node.elts]
        return vals if isinstance(node, ast.List) else tuple(vals)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_const(node.left, ns) + _eval_const(node.right, ns)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_const(node.operand, ns)
    raise NotebookError(f"Biểu thức không hỗ trợ: {ast.dump(node)[:120]}")


def _walk_assigns(tree: ast.AST) -> list[ast.Assign]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]


def _find_assign(tree: ast.AST, name: str, *, last: bool = True) -> ast.Assign | None:
    """Tìm phép gán ``name = ...`` (mặc định lấy lần gán cuối, giống Python)."""
    found = [
        n
        for n in _walk_assigns(tree)
        if any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    ]
    if not found:
        return None
    return found[-1] if last else found[0]


def _as_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


@dataclass(frozen=True)
class NotebookData:
    """Toàn bộ dữ liệu lấy từ notebook."""

    system_modes: dict[str, str]                # "1".."3" -> prompt nguyên văn
    mode_names: dict[str, str]                 # "1" -> "1. Gia sư gợi mở ..."
    prompt_modes_versions: list[dict[str, str]]  # cả 2 lần định nghĩa PROMPT_MODES
    concrete_socratic_prompt: str
    english_thought_starters: tuple[str, ...]
    elementary_tests: tuple[dict[str, str], ...]
    demo_dialogue: tuple[str, ...]             # 3 lượt hội thoại mẫu
    default_mode: str
    default_user_text: str
    decode: dict[str, float] = field(default_factory=dict)
    max_new_tokens_chat: int = 300
    max_new_tokens_single: int = 400
    max_seq_length: int = 4096
    enable_thinking: bool = False

    def mode_key(self, name_or_key: str) -> str:
        """Nhận '1', '1. Gia sư...' hoặc tên đầy đủ -> trả khoá '1'."""
        s = str(name_or_key).strip()
        if s in self.system_modes:
            return s
        for k, name in self.mode_names.items():
            if s == name or s.lower() == name.lower():
                return k
        if s and s[0] in self.system_modes:
            return s[0]
        raise NotebookError(f"Chế độ không hợp lệ: {name_or_key!r}")


def _extract(ipynb: Path) -> NotebookData:
    try:
        raw = json.loads(ipynb.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise NotebookError(f"Thiếu notebook nguồn: {ipynb}") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - lỗi file hỏng
        raise NotebookError(f"Notebook không phải JSON hợp lệ: {ipynb}") from exc

    cells = [_as_source(c) for c in raw.get("cells", []) if c.get("cell_type") == "code"]
    ns: dict[str, Any] = {}
    modes_versions: list[dict[str, str]] = []
    elementary: list[dict[str, str]] | None = None
    starters: tuple[str, ...] | None = None
    defaults: dict[str, Any] = {}
    chat_calls: list[dict[str, Any]] = []

    for src in cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:  # %%capture, !pip ... -> bỏ qua cell lệnh hệ thống
            continue

        # Hằng chuỗi / dict hằng -> nạp vào namespace
        for name in ("CONCRETE_SOCRATIC_PROMPT", "SYSTEM_MODES", "PROMPT_MODES"):
            node = _find_assign(tree, name)
            if node is None:
                continue
            try:
                value = _eval_const(node.value, ns)
            except NotebookError:
                continue
            ns[name] = value
            if name == "PROMPT_MODES":
                modes_versions.append(dict(value))

        node = _find_assign(tree, "english_thought_starters")
        if node is not None:
            starters = tuple(_eval_const(node.value, ns))

        node = _find_assign(tree, "elementary_tests")
        if node is not None:
            elementary = [dict(x) for x in _eval_const(node.value, ns)]

        # Tham số mặc định của run_pedagogical_inference(...)
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            if fn.name == "run_pedagogical_inference":
                args = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
                defs = fn.args.defaults + fn.args.kw_defaults
                names = args[-len(defs):] if defs else []
                for name, d in zip(names, defs):
                    if d is not None:
                        try:
                            defaults[name] = _eval_const(d, ns)
                        except NotebookError:
                            pass

        # max_new_tokens trong các lời gọi .generate(...)
        for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
            func = call.func
            fname = getattr(func, "attr", getattr(func, "id", ""))
            if fname != "generate":
                continue
            kw = {}
            for k in call.keywords:
                if k.arg:
                    try:
                        kw[k.arg] = _eval_const(k.value, ns)
                    except NotebookError:
                        pass
            if kw:
                chat_calls.append(kw)

    if "SYSTEM_MODES" not in ns:
        raise NotebookError("Không tìm thấy SYSTEM_MODES trong notebook")
    system_modes = {str(k): str(v) for k, v in ns["SYSTEM_MODES"].items()}
    mode_names = {}
    for k in system_modes:
        key = str(k).split(".")[0].strip()
        mode_names[key] = str(k)
    if "CONCRETE_SOCRATIC_PROMPT" not in ns:
        raise NotebookError("Không tìm thấy CONCRETE_SOCRATIC_PROMPT")
    if elementary is None:
        raise NotebookError("Không tìm thấy elementary_tests (MỤC 6)")
    if starters is None:
        raise NotebookError("Không tìm thấy english_thought_starters")

    demo = _extract_demo_dialogue(cells)
    default_user_text = _extract_widget_default(cells) or demo[0] if demo else ""

    # max_new_tokens: 300 cho hội thoại (MultiTurnMathSession) / 400 cho chạy lẻ
    chat_tokens = sorted({int(c["max_new_tokens"]) for c in chat_calls if "max_new_tokens" in c})
    max_chat = 300 if 300 in chat_tokens else (chat_tokens[0] if chat_tokens else 300)
    max_single = int(defaults.get("max_new_tokens", 400))

    decode = {
        "temperature": float(defaults.get("temp", 0.25)),
        "top_p": float(next((c.get("top_p") for c in chat_calls if "top_p" in c), 0.85)),
        "repetition_penalty": float(
            next((c.get("repetition_penalty") for c in chat_calls if "repetition_penalty" in c), 1.15)
        ),
    }

    return NotebookData(
        system_modes=system_modes,
        mode_names=mode_names,
        prompt_modes_versions=modes_versions,
        concrete_socratic_prompt=str(ns["CONCRETE_SOCRATIC_PROMPT"]),
        english_thought_starters=starters,
        elementary_tests=tuple(elementary),
        demo_dialogue=demo,
        default_mode=mode_names["1"],
        default_user_text=default_user_text,
        decode=decode,
        max_new_tokens_chat=max_chat,
        max_new_tokens_single=max_single,
        max_seq_length=4096,
        enable_thinking=False,
    )


def _extract_demo_dialogue(cells: list[str]) -> tuple[str, ...]:
    """Lấy 3 câu người dùng của MỤC 3 qua ast (send_message('<chuỗi>'))."""
    out: list[str] = []
    for src in cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
            fname = getattr(call.func, "attr", getattr(call.func, "id", ""))
            if fname != "send_message" or not call.args:
                continue
            arg = call.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.append(arg.value)
    return tuple(out[:3])


def _extract_widget_default(cells: list[str]) -> str:
    """Giá trị mặc định của ô nhập trong phòng học (w_user_text)."""
    for src in cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
            fname = getattr(call.func, "attr", getattr(call.func, "id", ""))
            if fname != "Text":
                continue
            for k in call.keywords:
                if k.arg == "value" and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str):
                    return k.value.value
    return ""


@lru_cache(maxsize=4)
def load(path: str | Path | None = None) -> NotebookData:
    """Đọc notebook (có cache). ``path=None`` dùng reference/nckh-math.ipynb."""
    p = Path(path) if path else DEFAULT_NOTEBOOK
    return _extract(p)


def extract_text_constants(ipynb: Path | None = None) -> dict[str, Any]:
    """Bản dict tiện cho test parity / script snapshot."""
    nb = load(ipynb)
    return {
        "system_modes": nb.system_modes,
        "english_thought_starters": list(nb.english_thought_starters),
        "elementary_tests": list(nb.elementary_tests),
        "demo_dialogue": list(nb.demo_dialogue),
        "decode": nb.decode,
        "max_new_tokens_chat": nb.max_new_tokens_chat,
        "max_new_tokens_single": nb.max_new_tokens_single,
        "max_seq_length": nb.max_seq_length,
    }


__all__ = [
    "DEFAULT_NOTEBOOK",
    "NotebookData",
    "NotebookError",
    "PROJECT_ROOT",
    "extract_text_constants",
    "load",
]
