"""Tự kiểm tra harness đánh giá: judge phải bắt được lỗi, mock phải sạch, ngưỡng không bị nới."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from eval import judge, runner
from eval.student_sim import PERSONAS
from tutor.config import Config
from tutor.engine import MockBackend
from tutor.notebook_source import load as load_notebook

ROOT = Path(__file__).resolve().parent.parent
PROBLEM = "Tìm x, biết: x * 3 - 4 = 11"
EXPECTED = [5.0]


def _raw(behavior: str, mode: str = "1") -> str:
    backend = MockBackend(behavior)
    msgs = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": PROBLEM},
    ]
    return backend.generate(msgs, mode=mode).text


# ------------------------------------------------------------------ judge
def test_judge_accepts_good_reply() -> None:
    res = judge.judge_turn(
        _raw("auto"), mode="1", lang="vi", turn=0, problem=PROBLEM, expected=EXPECTED
    )
    assert res["ok"], res["violations"]
    assert all(v is not False for v in res["labels"].values())


@pytest.mark.parametrize(
    "behavior,label",
    [
        ("violating", "B3"),
        ("violating", "B6"),
        ("vague", "B3"),
        ("english", "B1"),
        ("english", "B5"),
        ("think", "B1"),
        ("leak", "B5"),
    ],
)
def test_judge_catches_bad_replies(behavior: str, label: str) -> None:
    res = judge.judge_turn(
        _raw(behavior), mode="1", lang="vi", turn=0, problem=PROBLEM, expected=EXPECTED
    )
    assert res["labels"][label] is False, (behavior, label, res)
    assert not res["ok"]


def test_judge_arithmetic() -> None:
    good = judge.judge_turn("Ta có 45 - 18 = 27 và 15 : 3 = 5.", mode="3", expected=[27, 5])
    bad = judge.judge_turn("Ta có 45 - 18 = 28.", mode="3", expected=[27])
    assert good["labels"]["B9"] is True
    assert bad["labels"]["B9"] is False
    # phương trình có x không bị coi là phép tính sai
    eq = judge.judge_turn("Bài có x * 3 - 4 = 11.", mode="3", expected=[5])
    assert eq["labels"]["B9"] is True


def test_judge_arithmetic_chuoi_nhieu_phep_tinh() -> None:
    """'500 x 10 : 100 = 50' phải đọc trọn chuỗi (từng bị cắt thành '10 : 100')."""
    good = judge.judge_turn(
        "Bài giải:\n- 10% của 500\n  500 x 10 : 100 = 50\n"
        "- Số còn lại\n  500 - 50 = 450\nĐáp số: 450 ki-lô-gam",
        mode="3",
        expected=[450],
    )
    assert good["labels"]["B9"] is True, good

    bad = judge.judge_turn("Bài giải:\n  500 x 10 : 100 = 60\nĐáp số: 450", mode="3", expected=[450])
    assert bad["labels"]["B9"] is False

    # LaTeX \( \) và dấu ':' của câu dẫn không được làm hỏng phép tính.
    latex = judge.judge_turn(
        "Kiểm tra lại: \\( 9 + 15 = 24 \\).\nĐáp số: 15 và 9", mode="3", expected=[15, 9]
    )
    assert latex["labels"]["B9"] is True, latex
    assert latex["labels"]["B8"] is True


def test_judge_mode3_answer_line() -> None:
    ok = judge.judge_turn("Bài giải:\nĐáp số: 15 và 9 cái kẹo", mode="3", expected=[15, 9])
    bad = judge.judge_turn("Bài giải:\nĐáp số: 14 và 9 cái kẹo", mode="3", expected=[15, 9])
    missing = judge.judge_turn("Bài giải: 15 và 9", mode="3", expected=[15, 9])
    assert ok["labels"]["B8"] is True
    assert bad["labels"]["B8"] is False
    assert missing["labels"]["B8"] is False


def test_judge_does_not_import_guard() -> None:
    import ast

    src = (ROOT / "eval" / "judge.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.append(module)
            imported += [f"{module}.{alias.name}" for alias in node.names]
    assert not any(m == "tutor.guard" or m.startswith("tutor.guard.") for m in imported)
    assert "guard" not in imported


def test_judge_dialogue_marks_student_solved() -> None:
    transcript = [
        {"role": "user", "content": f"Em không biết làm bài này: {PROBLEM}"},
        {"role": "assistant", "content": "Chào em! Ta xét 11 + 4 trước nhé?"},
        {"role": "user", "content": "Em tính 11 + 4 = 15 ạ."},
        {"role": "assistant", "content": "Đúng rồi em! Vậy 15 : 3 bằng mấy?"},
        {"role": "user", "content": "Em tính 15 : 3 = 5 ạ."},
        {"role": "assistant", "content": "Chính xác! Vậy x = 5."},
    ]
    res = judge.judge_dialogue(
        transcript, mode="1", lang="vi", problem=PROBLEM, expected=EXPECTED
    )
    assert res["student_solved"] is True
    assert res["student_solved_turn"] == 2
    assert res["n_assistant_turns"] == 3


# ------------------------------------------------------------------ runner
def test_runner_mock_all_modes(tmp_path: Path) -> None:
    cfg = Config.load(ROOT / "config.yaml")
    for mode in ("1", "2", "3"):
        run_dir = runner.run(
            cfg, n=6, seed=99, mode=mode, backend="mock",
            out_dir=tmp_path / f"run_{mode}",
        )
        assert (run_dir / "results.jsonl").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "REPORT.md").exists()
        assert (run_dir / "meta.json").exists()

        lines = [ln for ln in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if ln]
        assert len(lines) == 6
        records = [json.loads(ln) for ln in lines]
        assert {r["persona"] for r in records} <= set(PERSONAS)
        assert all(r["mode"] == mode for r in records)
        for rec in records:
            assert rec["transcript"]
            assert rec["labels"]
            assert rec["expected"]

        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["n_dialogues"] == 6
        assert summary["backend"] == "mock"
        assert summary["evidence"] == "[MOCK]"
        assert summary["clean_dialogue_rate"] == 1.0, summary["labels"]
        assert summary["violations_after_guard"] == 0
        assert summary["gates"]["status"] == "skipped"

        report = (run_dir / "REPORT.md").read_text(encoding="utf-8")
        assert "[MOCK]" in report
        assert "KHÔNG phải đo thật" in report


def test_summarize_recomputes(tmp_path: Path) -> None:
    cfg = Config.load(ROOT / "config.yaml")
    run_dir = runner.run(cfg, n=3, seed=7, mode="1", backend="mock", out_dir=tmp_path / "r")
    first = runner.summarize(run_dir)
    second = runner.summarize(run_dir)
    assert first["n_dialogues"] == second["n_dialogues"] == 3
    assert first["clean_dialogue_rate"] == second["clean_dialogue_rate"]
    assert first["labels"] == second["labels"]


def test_gate_logic() -> None:
    th = runner.load_thresholds(Config.load(ROOT / "config.yaml"))
    base = {
        "backend": "llama",
        "n_dialogues": 100,
        "clean_dialogue_rate": 0.95,
        "fallback_rate": 0.01,
        "retry_rate": 0.02,
        "violations_after_guard": 0,
        "avg_ttft_s": 0.3,
        "avg_tok_per_s": 40.0,
        "hardware": {"has_cuda": True},
        "labels": {k: {"pass": 100, "total": 100, "rate": 1.0} for k in judge.LABELS},
    }
    ok = runner.evaluate_gates(base, th)
    assert ok["status"] == "pass", ok

    bad = json.loads(json.dumps(base))
    bad["clean_dialogue_rate"] = 0.50
    bad["labels"]["B5"]["rate"] = 0.50
    res = runner.evaluate_gates(bad, th)
    assert res["status"] == "fail"
    assert "clean_dialogue_rate" in res["checks"]
    assert res["checks"]["label_B5"]["status"] == "fail"

    mock = json.loads(json.dumps(base))
    mock["backend"] = "mock"
    assert runner.evaluate_gates(mock, th)["status"] == "skipped"

    small = json.loads(json.dumps(base))
    small["n_dialogues"] = 10
    assert runner.evaluate_gates(small, th)["status"] == "skipped"


# ------------------------------------------------------------------ ngưỡng
def test_thresholds_not_loosened() -> None:
    data = yaml.safe_load((ROOT / "eval" / "thresholds.yaml").read_text(encoding="utf-8"))
    gates = data["gates"]
    assert gates["clean_dialogue_rate"] >= 0.90
    assert gates["fallback_rate_max"] <= 0.10
    assert gates["retry_rate_max"] <= 0.20
    assert gates["violations_after_guard_max"] <= 0
    assert gates["avg_ttft_s_max"] <= 1.0
    assert gates["min_tok_per_s"] >= 12.0
    floors = gates["label_min_rate"]
    expected = {"B1": 0.99, "B2": 0.99, "B3": 0.98, "B4": 0.95, "B5": 0.98,
                "B6": 0.99, "B7": 0.90, "B8": 0.95, "B9": 0.99}
    for key, floor in expected.items():
        assert floors[key] >= floor, f"ngưỡng {key} bị nới lỏng"
    assert int(data["min_dialogues_for_gate"]) >= 100
    assert data["require_real_backend"] is True


# ------------------------------------------------------------------ dữ liệu
def test_frozen_regression_judges_cleanly() -> None:
    """4 đề notebook + hội thoại mẫu: mọi đề phải giải được và judge chấp nhận câu mẫu."""
    path = ROOT / "data" / "frozen_regression.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 5
    for row in rows:
        expected = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
        if isinstance(row["answer"], list):
            ans_txt = " và ".join(str(v) for v in row["answer"])
        else:
            ans_txt = str(row["answer"])
        res = judge.judge_dialogue(
            [{"role": "assistant", "content": f"Bài giải:\nĐáp số: {ans_txt}"}],
            mode="3", lang=row["lang"], problem=row["text"], expected=expected,
        )
        assert res["labels"]["B8"]["pass"] == 1, (row["id"], res)


def test_notebook_problems_solvable() -> None:
    from tutor.mathcheck import solve

    nb = load_notebook()
    for test in nb.elementary_tests:
        sol = solve(test["prompt"])
        assert sol is not None and sol.answer is not None, test["prompt"]
