"""Chạy đánh giá hội thoại: sinh đề -> mô phỏng học sinh -> judge độc lập -> báo cáo.

Kết quả nằm ở ``eval/runs/<timestamp>_<mode>_<backend>/``:
- ``results.jsonl`` : 1 dòng / hội thoại
- ``meta.json``     : thông tin lượt chạy (config, seed, phần cứng)
- ``summary.json``  : tổng hợp + trạng thái cổng chất lượng
- ``REPORT.md``     : báo cáo đọc được, có nhãn chứng cứ

Nhãn chứng cứ: ``[ĐO THẬT trên máy này]`` (backend llama) / ``[MOCK]``.
Không có kết luận tốc độ nào từ backend mock.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

from tutor.config import Config
from tutor.engine import Engine
from tutor.hardware import describe, detect
from tutor.problemgen import Item, generate
from tutor.session import TutorSession

from .judge import LABELS, judge_dialogue, revealed_values
from .student_sim import PERSONAS, StudentSim

RUNS_DIR = Path(__file__).resolve().parent / "runs"


# ------------------------------------------------------------------ tiện ích
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp() -> str:
    return _now().strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def _evidence(backend: str) -> str:
    return "[ĐO THẬT trên máy này]" if backend == "llama" else "[MOCK]"


def load_thresholds(cfg: Config) -> dict[str, Any]:
    rel = cfg.get("eval.thresholds", "eval/thresholds.yaml") or "eval/thresholds.yaml"
    path = cfg.resolve(rel)
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file ngưỡng: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} phải là mapping YAML")
    return data


def _student_done(text: str, expected: Sequence[float]) -> bool:
    if not expected:
        return False
    got = revealed_values(text, expected)
    return all(any(abs(v - g) <= 1e-6 for g in got) for v in expected)


def _turn_timing(gen: Any) -> dict[str, float]:
    return {
        "total_s": round(float(getattr(gen, "total_s", 0.0)), 4),
        "ttft_s": round(float(getattr(gen, "ttft_s", 0.0)), 4),
        "gen_tokens": int(getattr(gen, "gen_tokens", 0)),
        "tok_per_s": round(float(getattr(gen, "tok_per_s", 0.0)), 3),
    }


# ------------------------------------------------------------------ 1 hội thoại
def _run_dialogue(
    engine: Engine,
    cfg: Config,
    item: Item,
    persona: str,
    *,
    mode: str,
    seed: int,
    max_turns: int,
) -> dict[str, Any]:
    session = TutorSession(engine, mode=mode, cfg=cfg)
    sim = StudentSim(item, persona=persona, seed=seed)
    expected = item.answers()

    transcript: list[dict[str, Any]] = []
    violations_before: list[str] = []
    violations_after: list[str] = []
    retries = 0
    fallbacks = 0
    gen_tokens = 0
    total_s = 0.0
    ttft_values: list[float] = []
    t_start = time.time()

    message = sim.opening()
    for turn in range(max_turns):
        result = session.send(message)
        transcript.append({"role": "user", "content": message})
        transcript.append(
            {
                "role": "assistant",
                "content": result.reply,
                "raw": result.raw,
                "turn": turn,
                "retries": result.retries,
                "used_fallback": result.used_fallback,
                "violations_before": [v.rule for v in result.violations_before],
                "violations_after": [v.rule for v in result.violations],
                "timing": _turn_timing(result.gen),
            }
        )
        violations_before += [v.rule for v in result.violations_before]
        violations_after += [v.rule for v in result.violations]
        retries += result.retries
        fallbacks += int(result.used_fallback)
        gen_tokens += int(result.gen.gen_tokens)
        total_s += float(result.gen.total_s)
        ttft_values.append(float(result.gen.ttft_s))

        if _student_done(message, expected) and turn >= 1:
            break
        nxt = sim.next_message(turn + 1, result.reply)
        if nxt is None:
            break
        message = nxt

    judged = judge_dialogue(
        transcript,
        mode=mode,
        lang=item.lang,
        problem=item.text,
        expected=expected,
        require_parts=item.parts is not None,
    )
    return {
        "id": f"{mode}-{item.id}-{persona}",
        "mode": mode,
        "persona": persona,
        "item_id": item.id,
        "grade": item.grade,
        "topic": item.topic,
        "problem": item.text,
        "lang": item.lang,
        "expected": expected,
        "unit": item.unit,
        "transcript": transcript,
        "violations_before_guard": violations_before,
        "violations_after_guard": violations_after,
        "timings": {
            "wall_s": round(time.time() - t_start, 4),
            "gen_total_s": round(total_s, 4),
            "gen_tokens": gen_tokens,
            "tok_per_s": round(gen_tokens / total_s, 3) if total_s > 0 else 0.0,
            "ttft_s": round(sum(ttft_values) / len(ttft_values), 4) if ttft_values else 0.0,
            "turns": len(ttft_values),
        },
        "retries": retries,
        "fallbacks": fallbacks,
        "labels": judged["labels"],
        "judge": {
            "ok": judged["ok"],
            "score": round(float(judged["score"]), 4),
            "failed": judged["failed"],
            "student_solved": judged["student_solved"],
            "student_solved_turn": judged["student_solved_turn"],
        },
    }


# ------------------------------------------------------------------ tổng hợp
def summarize(run_dir: Path) -> dict[str, Any]:
    """Đọc results.jsonl + meta.json, tính lại toàn bộ chỉ số và cổng chất lượng."""
    run_dir = Path(run_dir)
    records: list[dict[str, Any]] = []
    path = run_dir / "results.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    meta: dict[str, Any] = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    n = len(records)
    turns = sum(int(r.get("timings", {}).get("turns", 0)) for r in records)
    gen_tokens = sum(int(r.get("timings", {}).get("gen_tokens", 0)) for r in records)
    gen_total_s = sum(float(r.get("timings", {}).get("gen_total_s", 0.0)) for r in records)
    ttft_weighted = sum(
        float(r.get("timings", {}).get("ttft_s", 0.0)) * int(r.get("timings", {}).get("turns", 0))
        for r in records
    )
    retries = sum(int(r.get("retries", 0)) for r in records)
    fallbacks = sum(int(r.get("fallbacks", 0)) for r in records)
    violations_before = sum(len(r.get("violations_before_guard", [])) for r in records)
    violations_after = sum(len(r.get("violations_after_guard", [])) for r in records)

    labels: dict[str, dict[str, Any]] = {}
    for key in LABELS:
        passed = sum(int(r.get("labels", {}).get(key, {}).get("pass", 0)) for r in records)
        total = sum(int(r.get("labels", {}).get(key, {}).get("total", 0)) for r in records)
        labels[key] = {
            "pass": passed,
            "total": total,
            "rate": round(passed / total, 4) if total else None,
        }

    clean = sum(1 for r in records if r.get("judge", {}).get("ok"))
    solved = sum(1 for r in records if r.get("judge", {}).get("student_solved"))
    per_persona: dict[str, dict[str, Any]] = {}
    for r in records:
        p = str(r.get("persona", "?"))
        node = per_persona.setdefault(p, {"n": 0, "clean": 0})
        node["n"] += 1
        node["clean"] += int(bool(r.get("judge", {}).get("ok")))
    for node in per_persona.values():
        node["clean_rate"] = round(node["clean"] / node["n"], 4) if node["n"] else None

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "created_at": meta.get("created_at", ""),
        "mode": meta.get("mode", ""),
        "backend": meta.get("backend", ""),
        "seed": meta.get("seed"),
        "n_requested": meta.get("n_requested", n),
        "n_dialogues": n,
        "n_turns": turns,
        "labels": labels,
        "clean_dialogues": clean,
        "clean_dialogue_rate": round(clean / n, 4) if n else 0.0,
        "student_solved_rate": round(solved / n, 4) if n else 0.0,
        "fallback_rate": round(fallbacks / turns, 4) if turns else 0.0,
        "retry_rate": round(retries / turns, 4) if turns else 0.0,
        "violations_before_guard": violations_before,
        "violations_after_guard": violations_after,
        "avg_ttft_s": round(ttft_weighted / turns, 4) if turns else 0.0,
        "avg_tok_per_s": round(gen_tokens / gen_total_s, 3) if gen_total_s > 0 else 0.0,
        "gen_tokens": gen_tokens,
        "gen_total_s": round(gen_total_s, 3),
        "per_persona": per_persona,
        "hardware": meta.get("hardware", {}),
        "config_sha256_16": meta.get("config_sha256_16", ""),
        "evidence": _evidence(str(meta.get("backend", ""))),
        "thresholds": meta.get("thresholds", {}),
    }

    gates = evaluate_gates(summary, summary["thresholds"])
    summary["gates"] = gates
    return summary


def evaluate_gates(summary: dict[str, Any], th: dict[str, Any]) -> dict[str, Any]:
    """Chấm cổng. Chỉ chấm khi backend thật + đủ mẫu (tránh kết luận từ [MOCK])."""
    gates = th.get("gates", {}) if isinstance(th, dict) else {}
    backend = str(summary.get("backend", ""))
    n = int(summary.get("n_dialogues", 0) or 0)
    min_n = int(th.get("min_dialogues_for_gate", 100) or 100)

    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, value: Any, threshold: Any, ok: bool, note: str = "") -> None:
        checks[name] = {
            "value": value,
            "threshold": threshold,
            "status": "pass" if ok else "fail",
            "note": note,
        }

    def skip_all(reason: str) -> dict[str, Any]:
        return {"status": "skipped", "reason": reason, "checks": {}}

    if bool(th.get("require_real_backend", True)) and backend != "llama":
        return skip_all(f"backend={backend or '?'} — [MOCK] không dùng để chấm cổng")
    if n < min_n:
        return skip_all(f"n={n} < min_dialogues_for_gate={min_n} — [CHƯA ĐO] đủ mẫu")

    add(
        "clean_dialogue_rate",
        summary.get("clean_dialogue_rate", 0.0),
        gates.get("clean_dialogue_rate", 1.0),
        float(summary.get("clean_dialogue_rate", 0.0)) >= float(gates.get("clean_dialogue_rate", 1.0)),
    )
    for key, floor in (gates.get("label_min_rate") or {}).items():
        node = summary.get("labels", {}).get(key, {})
        rate = node.get("rate")
        if rate is None:
            checks[f"label_{key}"] = {
                "value": None, "threshold": floor, "status": "skipped",
                "note": "không có lượt áp dụng nhãn",
            }
            continue
        add(f"label_{key}", rate, floor, float(rate) >= float(floor))

    add(
        "fallback_rate",
        summary.get("fallback_rate", 0.0),
        gates.get("fallback_rate_max", 1.0),
        float(summary.get("fallback_rate", 0.0)) <= float(gates.get("fallback_rate_max", 1.0)),
        "càng thấp càng tốt",
    )
    add(
        "retry_rate",
        summary.get("retry_rate", 0.0),
        gates.get("retry_rate_max", 1.0),
        float(summary.get("retry_rate", 0.0)) <= float(gates.get("retry_rate_max", 1.0)),
        "càng thấp càng tốt",
    )
    add(
        "violations_after_guard",
        summary.get("violations_after_guard", 0),
        gates.get("violations_after_guard_max", 0),
        int(summary.get("violations_after_guard", 0)) <= int(gates.get("violations_after_guard_max", 0)),
    )

    hw = summary.get("hardware", {}) or {}
    has_cuda = bool(hw.get("has_cuda", False))
    need_cuda = bool(gates.get("speed_gate_requires_cuda", True))
    if need_cuda and not has_cuda:
        checks["avg_ttft_s"] = {
            "value": summary.get("avg_ttft_s"), "threshold": gates.get("avg_ttft_s_max"),
            "status": "skipped", "note": "máy không có CUDA — không chấm tốc độ",
        }
        checks["avg_tok_per_s"] = {
            "value": summary.get("avg_tok_per_s"), "threshold": gates.get("min_tok_per_s"),
            "status": "skipped", "note": "máy không có CUDA — không chấm tốc độ",
        }
    else:
        add(
            "avg_ttft_s",
            summary.get("avg_ttft_s", 0.0),
            gates.get("avg_ttft_s_max", 99.0),
            float(summary.get("avg_ttft_s", 0.0)) <= float(gates.get("avg_ttft_s_max", 99.0)),
            "càng thấp càng tốt",
        )
        add(
            "avg_tok_per_s",
            summary.get("avg_tok_per_s", 0.0),
            gates.get("min_tok_per_s", 0.0),
            float(summary.get("avg_tok_per_s", 0.0)) >= float(gates.get("min_tok_per_s", 0.0)),
            "càng cao càng tốt",
        )

    failed = [k for k, v in checks.items() if v["status"] == "fail"]
    return {
        "status": "fail" if failed else "pass",
        "reason": ("các cổng không đạt: " + ", ".join(failed)) if failed else "tất cả cổng đạt",
        "checks": checks,
    }


# ------------------------------------------------------------------ báo cáo
def _pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def write_report(run_dir: Path, summary: dict[str, Any]) -> Path:
    ev = summary.get("evidence", "[MOCK]")
    hw = summary.get("hardware", {}) or {}
    gates = summary.get("gates", {}) or {}
    lines: list[str] = []
    lines.append(f"# BÁO CÁO ĐÁNH GIÁ — chế độ {summary.get('mode', '?')}")
    lines.append("")
    lines.append(f"- **Chứng cứ:** {ev}")
    lines.append(f"- **Thời điểm (UTC):** {summary.get('created_at', '?')}")
    lines.append(f"- **Backend:** `{summary.get('backend', '?')}`")
    lines.append(f"- **Seed:** {summary.get('seed')} | **Số hội thoại:** {summary.get('n_dialogues')} | "
                 f"**Số lượt:** {summary.get('n_turns')}")
    lines.append(f"- **Phần cứng:** {hw.get('description', '[CHƯA ĐO]')}")
    lines.append(f"- **Config sha256/16:** `{summary.get('config_sha256_16', '?')}`")
    lines.append("")
    lines.append("## 1. Kết quả tổng")
    lines.append("")
    lines.append("| Chỉ số | Giá trị |")
    lines.append("|---|---|")
    lines.append(f"| Hội thoại sạch (không vi phạm nhãn nào) | {_pct(summary.get('clean_dialogue_rate'))} "
                 f"({summary.get('clean_dialogues')}/{summary.get('n_dialogues')}) |")
    lines.append(f"| Học sinh tự giải ra đáp số | {_pct(summary.get('student_solved_rate'))} |")
    lines.append(f"| Tỉ lệ fallback | {_pct(summary.get('fallback_rate'))} |")
    lines.append(f"| Tỉ lệ phải thử lại | {_pct(summary.get('retry_rate'))} |")
    lines.append(f"| Vi phạm trước guard | {summary.get('violations_before_guard')} |")
    lines.append(f"| Vi phạm sau guard | {summary.get('violations_after_guard')} |")
    lines.append(f"| TTFT trung bình | {summary.get('avg_ttft_s')} s {ev} |")
    lines.append(f"| Tốc độ sinh | {summary.get('avg_tok_per_s')} tok/s {ev} |")
    lines.append("")
    lines.append("## 2. Nhãn B1–B9 (judge độc lập)")
    lines.append("")
    lines.append("| Nhãn | Mô tả | Đạt | Tổng | Tỉ lệ |")
    lines.append("|---|---|---|---|---|")
    for key, desc in LABELS.items():
        node = summary.get("labels", {}).get(key, {})
        lines.append(f"| {key} | {desc} | {node.get('pass', 0)} | {node.get('total', 0)} | "
                     f"{_pct(node.get('rate'))} |")
    lines.append("")
    lines.append("## 3. Cổng chất lượng")
    lines.append("")
    if gates.get("status") == "skipped":
        lines.append(f"- **BỎ QUA:** {gates.get('reason', '')}")
    else:
        lines.append(f"- **Trạng thái:** `{gates.get('status')}` — {gates.get('reason', '')}")
        lines.append("")
        lines.append("| Cổng | Giá trị | Ngưỡng | Trạng thái | Ghi chú |")
        lines.append("|---|---|---|---|---|")
        for name, node in (gates.get("checks") or {}).items():
            lines.append(f"| {name} | {node.get('value')} | {node.get('threshold')} | "
                         f"{node.get('status')} | {node.get('note', '')} |")
    lines.append("")
    lines.append("## 4. Theo kiểu học sinh")
    lines.append("")
    lines.append("| Persona | Số hội thoại | Sạch | Tỉ lệ |")
    lines.append("|---|---|---|---|")
    for name, node in (summary.get("per_persona") or {}).items():
        lines.append(f"| {name} | {node.get('n')} | {node.get('clean')} | {_pct(node.get('clean_rate'))} |")
    lines.append("")
    lines.append("## 5. Ghi chú về chứng cứ")
    lines.append("")
    if summary.get("backend") == "llama":
        lines.append("- Số liệu tốc độ và chất lượng được đo thật trên máy này bằng llama.cpp/GGUF.")
        lines.append("- Không suy rộng tốc độ sang máy khác; hãy chạy lại `scripts/benchmark.py`.")
    else:
        lines.append("- Đây là lượt chạy **[MOCK]**: câu trả lời do backend giả lập, "
                     "số liệu tốc độ KHÔNG phải đo thật và không được dùng để kết luận.")
        lines.append("- Dùng `python scripts/autotest.py --backend llama` để có [ĐO THẬT trên máy này].")
    lines.append("")
    path = run_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ------------------------------------------------------------------ API chính
def run(
    config: str | Path | Config | None = None,
    n: int = 100,
    seed: int = 1234,
    mode: str = "1",
    backend: str | None = None,
    *,
    max_turns: int | None = None,
    lang: str = "vi",
    out_dir: str | Path | None = None,
) -> Path:
    """Chạy ``n`` hội thoại cho một chế độ, ghi kết quả và trả thư mục run."""
    cfg = config if isinstance(config, Config) else Config.load(config)
    if backend:
        cfg = cfg.with_overrides({"engine.backend": str(backend).lower()})
    backend = str(cfg.get("engine.backend", "llama")).lower()
    if backend not in ("llama", "mock"):
        raise ValueError(f"backend không hợp lệ: {backend!r} (llama|mock)")

    max_turns = int(max_turns or cfg.get("eval.max_turns", 8) or 8)
    grades = cfg.get("eval.grades", [2, 3, 4, 5]) or [2, 3, 4, 5]
    items = generate(int(n), int(seed), grades=grades, style=lang)
    rng = random.Random(int(seed))
    personas = [PERSONAS[i % len(PERSONAS)] for i in range(len(items))]
    rng.shuffle(personas)

    engine = Engine(cfg)
    engine.load()

    stamp = _stamp()
    run_dir = Path(out_dir) if out_dir else (RUNS_DIR / f"{stamp}_{mode}_{backend}")
    run_dir.mkdir(parents=True, exist_ok=True)

    hw = detect()
    th = load_thresholds(cfg)
    meta = {
        "created_at": _now().isoformat(timespec="seconds"),
        "mode": str(mode),
        "backend": backend,
        "seed": int(seed),
        "n_requested": int(n),
        "max_turns": max_turns,
        "lang": lang,
        "config": str(cfg.source),
        "config_sha256_16": _sha256(Path(cfg.source)),
        "hardware": {
            "gpu_name": hw.gpu_name,
            "vram_free_gb": hw.vram_free_gb,
            "vram_total_gb": hw.vram_total_gb,
            "ram_gb": hw.ram_gb,
            "cpu_phys": hw.cpu_phys,
            "has_cuda": hw.has_cuda,
            "description": describe(hw),
        },
        "engine": {
            "load_s": round(float(engine.load_s), 3),
            "profile": engine.profile.as_dict() if engine.profile else None,
        },
        "thresholds": th,
        "evidence": _evidence(backend),
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results_path = run_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as fh:
        for item, persona in zip(items, personas):
            record = _run_dialogue(
                engine, cfg, item, persona, mode=str(mode), seed=int(seed), max_turns=max_turns
            )
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

    engine.close()
    summary = summarize(run_dir)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(run_dir, summary)
    return run_dir


__all__ = [
    "RUNS_DIR",
    "evaluate_gates",
    "load_thresholds",
    "run",
    "summarize",
    "write_report",
]
