"""Parity: mọi prompt/hàm/tham số phải khớp notebook gốc, không được gõ lại.

Test này thực thi trực tiếp cell 3 và cell 6 của ``reference/nckh-math.ipynb``
(cell 6 với stub ipywidgets/IPython) rồi so với thư viện ``tutor``.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tutor import guard, prompts
from tutor.config import Config
from tutor.notebook_source import DEFAULT_NOTEBOOK, load

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ notebook
def _code_cells() -> list[str]:
    raw = json.loads(DEFAULT_NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", ""))
        for cell in raw.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def _cell_with(marker: str) -> str:
    for src in _code_cells():
        if marker in src:
            return src
    raise AssertionError(f"Không thấy cell chứa {marker!r}")


def _exec(src: str, ns: dict | None = None) -> dict:
    ns = ns if ns is not None else {}
    exec(compile(src, "<notebook>", "exec"), ns)  # noqa: S102 - test cố ý chạy notebook
    return ns


class _Widget:
    """Stub tối thiểu cho ipywidgets."""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.value = kwargs.get("value")
        self.args = args

    def __getattr__(self, name):
        return lambda *a, **k: None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def notebook_cell6(monkeypatch) -> dict:
    widgets = types.ModuleType("ipywidgets")
    for name in ("Dropdown", "Text", "Button", "Output", "HBox", "VBox", "Layout"):
        setattr(widgets, name, _Widget)
    ipython = types.ModuleType("IPython")
    display_mod = types.ModuleType("IPython.display")
    display_mod.display = lambda *a, **k: None
    display_mod.HTML = lambda *a, **k: None
    display_mod.Markdown = lambda *a, **k: None
    display_mod.clear_output = lambda *a, **k: None
    ipython.display = display_mod
    monkeypatch.setitem(sys.modules, "ipywidgets", widgets)
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display_mod)
    return _exec(_cell_with("SYSTEM_MODES = {"))


# ------------------------------------------------------------------ prompts
def test_base_prompts_equal_notebook() -> None:
    nb = load()
    expected = {key: nb.system_modes[name] for key, name in nb.mode_names.items()}
    assert prompts.BASE_PROMPTS == expected
    assert prompts.MODE_NAMES == nb.mode_names
    assert list(prompts.MODE_NAMES) == ["1", "2", "3"]


def test_concrete_socratic_prompt_verbatim() -> None:
    nb = load()
    assert prompts.BASE_PROMPTS["1"] == nb.concrete_socratic_prompt
    assert prompts.BASE_PROMPTS["1"].startswith("Bạn là một gia sư dạy Toán tiểu học")


def test_system_modes_executed_from_cell6(notebook_cell6: dict) -> None:
    system_modes = notebook_cell6["SYSTEM_MODES"]
    assert len(system_modes) == 3
    for key, name in prompts.MODE_NAMES.items():
        assert system_modes[name] == prompts.BASE_PROMPTS[key]
    assert notebook_cell6["CONCRETE_SOCRATIC_PROMPT"] == prompts.BASE_PROMPTS["1"]
    assert notebook_cell6["w_user_text"].value == load().default_user_text


def test_demo_dialogue_and_starters_verbatim() -> None:
    nb = load()
    assert prompts.DEMO_DIALOGUE == nb.demo_dialogue
    assert guard.ENGLISH_THOUGHT_STARTERS == nb.english_thought_starters
    assert len(prompts.DEMO_DIALOGUE) == 3


def test_appendix_only_appends() -> None:
    for key, base in prompts.BASE_PROMPTS.items():
        built = prompts.build_system(key, "vi")
        assert built.startswith(base)
        assert len(built) > len(base)
        assert built == prompts.build_system(key, "vi")  # cached, ổn định


# ---------------------------------------------------------- clean_model_output
_CLEAN_CASES = [
    "",
    "   ",
    "Chào em! Trong bài x + 18 = 45, em hãy tính 45 - 18 nhé?",
    "thinking về bài toán...</think>Chào em! Ta xét 45 - 18 nhé?",
    "Chào em! Ta xét 45 - 18 nhé?</think>",
    "<|im_end|>Chào em! Ta xét 45 - 18 nhé?<|im_end|>",
    "Okay, let's see. The student is confused. Chào em! Ta xét 45 - 18 nhé?",
    "First, let's solve the equation. Trong phép trừ, 45 - 18 = 27.",
    "The problem is: x + 18 = 45.",
    "Let me see. We need to solve for x.",
    "Hmm, how do I start?",
    "Starting with the numbers, 45 - 18 = 27.",
    "okay, let's see\nChào em! Trong bài có số 18.",
    "  Đáp số: 27  ",
]


def test_clean_model_output_parity() -> None:
    ns = _exec(_cell_with("def clean_model_output"))
    nb_clean = ns["clean_model_output"]
    for raw in _CLEAN_CASES:
        assert guard.clean_model_output(raw) == nb_clean(raw), f"lệch ở: {raw!r}"


def test_default_reply_verbatim() -> None:
    ns = _exec(_cell_with("def clean_model_output"))
    nb_clean = ns["clean_model_output"]
    assert guard.DEFAULT_VI_REPLY == nb_clean("Okay, let's see. The student is confused.")
    assert guard.clean_model_output("Okay, let's see. The student is confused.") == guard.DEFAULT_VI_REPLY


# ------------------------------------------------------------------ tham số
def test_decode_params_match_notebook() -> None:
    cfg = Config.load(ROOT / "config.yaml")
    nb = load()
    assert float(cfg.get("decode.temperature")) == nb.decode["temperature"]
    assert float(cfg.get("decode.top_p")) == nb.decode["top_p"]
    assert float(cfg.get("decode.repetition_penalty")) == nb.decode["repetition_penalty"]
    assert int(cfg.get("decode.max_new_tokens_chat")) == nb.max_new_tokens_chat == 300
    assert int(cfg.get("decode.max_new_tokens_single")) == nb.max_new_tokens_single == 400
    assert cfg.get("engine.backend") in ("llama", "mock")
    stop = cfg.get("decode.stop") or []
    assert "<|im_end|>" in stop and "<|endoftext|>" in stop


def test_history_and_mode_contract() -> None:
    cfg = Config.load(ROOT / "config.yaml")
    assert int(cfg.get("session.max_history_turns")) == 6
    assert int(cfg.get("session.max_retries")) >= 1


# ------------------------------------------------------------------ dữ liệu
def test_snapshot_matches_notebook() -> None:
    from tutor import _nb_snapshot as snap

    nb = load()
    assert snap.MODE_NAMES == nb.mode_names
    assert snap.BASE_PROMPTS == {k: nb.system_modes[n] for k, n in nb.mode_names.items()}
    assert snap.DEMO_DIALOGUE == nb.demo_dialogue
    assert snap.ENGLISH_THOUGHT_STARTERS == nb.english_thought_starters
    assert snap.DECODE == nb.decode


def test_frozen_regression_matches_notebook() -> None:
    from tutor.mathcheck import solve

    path = ROOT / "data" / "frozen_regression.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 5
    nb = load()
    by_text = {t["prompt"]: t for t in nb.elementary_tests}
    for row in rows[:4]:
        assert row["text"] in by_text
        sol = solve(row["text"])
        assert sol is not None
        answer = list(sol.answer) if isinstance(sol.answer, tuple) else sol.answer
        assert row["answer"] == answer
        assert row["unit"] == sol.unit
        assert row["source"] == "notebook"
    demo = rows[4]
    assert demo["id"] == "nb-demo"
    assert demo["text"] == nb.default_user_text
    assert demo["dialogue"] == list(nb.demo_dialogue)
    assert demo["answer"] == 5.0


def test_generated_snapshot_is_current() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "extract_notebook.py"), "--check", "--quiet"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
