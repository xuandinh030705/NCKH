"""Dò lưới tham số ENGINE + APPENDIX prompt (KHÔNG đụng ``decode.*``).

Lý do: ``temperature/top_p/repetition_penalty/max_new_tokens`` là bất biến của đề tài
(phải khớp notebook). Muốn đổi phải sửa notebook và ghi vào PROMPT_CHANGELOG.

Ví dụ::

    python scripts/autotune.py --backend mock --n 6 --max-configs 3   # thử plumbing
    python scripts/autotune.py --backend llama --n 12 --mode 1
    python scripts/autotune.py --backend llama --n 12 --apply          # ghi config.yaml

Kết quả luôn ghi ``config.best.yaml``; ``--apply`` mới ghi đè ``config.yaml``
(có backup + ghi docs/TUNING_LOG.md).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.runner import run  # noqa: E402
from tutor.config import Config  # noqa: E402

BEST = ROOT / "config.best.yaml"
LOG = ROOT / "docs" / "TUNING_LOG.md"


def _candidates(cfg: Config) -> list[dict[str, Any]]:
    n_ctx = int(cfg.get("model.n_ctx", 2048) or 2048)
    n_ctx_max = int(cfg.get("model.n_ctx_max", n_ctx) or n_ctx)
    small_ctx = max(1024, min(1536, n_ctx_max))

    engine_grid: list[dict[str, Any]] = [
        {},
        {"model.n_batch": 256, "model.n_ubatch": 128},
        {"model.kv_quant": "q8_0"},
        {"model.kv_quant": "f16"},
        {"model.flash_attn": True},
        {"model.flash_attn": False},
    ]
    if small_ctx < n_ctx:
        engine_grid.append({"model.n_ctx": small_ctx})

    prompt_grid: list[dict[str, Any]] = [
        {"prompts.few_shot": True},
        {"prompts.append_rule_reminder": False},
        {"prompts.append_language_rule": False},
        {"prompts.few_shot": True, "prompts.append_rule_reminder": True},
    ]
    return engine_grid + prompt_grid


def _score(summary: dict[str, Any], backend: str) -> tuple:
    speed = float(summary.get("avg_tok_per_s", 0.0)) if backend == "llama" else 0.0
    return (
        float(summary.get("clean_dialogue_rate", 0.0)),
        -float(summary.get("fallback_rate", 0.0)),
        -float(summary.get("retry_rate", 0.0)),
        speed,
    )


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(base))  # deep copy
    for dotted, value in overrides.items():
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def _append_log(entry: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(entry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Autotune engine + prompt appendix")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mode", default="1")
    parser.add_argument("--backend", default=None, choices=["llama", "mock"])
    parser.add_argument("--max-configs", type=int, default=8)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    backend = args.backend or str(cfg.get("engine.backend", "llama"))
    if args.apply and backend != "llama":
        print("TỪ CHỐI --apply: kết quả [MOCK] không được dùng để ghi đè cấu hình.")
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = ROOT / "eval" / "runs" / f"autotune_{stamp}_{backend}"
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for idx, overrides in enumerate(_candidates(cfg)[: max(1, args.max_configs)]):
        candidate_cfg = cfg.with_overrides(overrides)
        try:
            run_dir = run(
                candidate_cfg,
                n=args.n,
                seed=args.seed,
                mode=args.mode,
                backend=backend,
                out_dir=out_root / f"cand_{idx:02d}",
            )
        except Exception as exc:
            print(f"[{idx}] LỖI: {exc}")
            continue
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        score = _score(summary, backend)
        results.append(
            {
                "index": idx,
                "overrides": overrides,
                "score": list(score),
                "clean_dialogue_rate": summary.get("clean_dialogue_rate"),
                "fallback_rate": summary.get("fallback_rate"),
                "retry_rate": summary.get("retry_rate"),
                "avg_tok_per_s": summary.get("avg_tok_per_s"),
                "run_dir": str(run_dir),
            }
        )
        print(f"[{idx}] clean={summary.get('clean_dialogue_rate')} "
              f"fallback={summary.get('fallback_rate')} retry={summary.get('retry_rate')} "
              f"tok/s={summary.get('avg_tok_per_s')} overrides={overrides or '{}'}")

    if not results:
        print("Không có cấu hình nào chạy được.")
        return 2

    best = max(results, key=lambda r: tuple(r["score"]))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": backend,
        "evidence": "[ĐO THẬT trên máy này]" if backend == "llama" else "[MOCK]",
        "n": args.n,
        "seed": args.seed,
        "mode": args.mode,
        "best": best,
        "all": results,
        "note": "decode.* giữ nguyên theo notebook; chỉ tối ưu engine + appendix prompt.",
    }
    BEST.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"\nĐã ghi {BEST.relative_to(ROOT)}")
    print(f"Tốt nhất: {best['overrides'] or '{}'} | score={best['score']}")

    if args.apply:
        overrides = best["overrides"]
        if not overrides:
            print("Cấu hình tốt nhất là mặc định — không cần ghi đè.")
            return 0
        raw = yaml.safe_load(Path(cfg.source).read_text(encoding="utf-8")) or {}
        backup = Path(cfg.source).with_suffix(f".yaml.bak.{stamp}")
        shutil.copy2(cfg.source, backup)
        merged = _merge(raw, overrides)
        Path(cfg.source).write_text(
            yaml.safe_dump(merged, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        _append_log(
            f"\n## {stamp} — autotune ({payload['evidence']})\n\n"
            f"- Lệnh: `python scripts/autotune.py --backend {backend} --n {args.n} "
            f"--mode {args.mode} --apply`\n"
            f"- Ghi đè: `{json.dumps(overrides, ensure_ascii=False)}`\n"
            f"- Score: `{best['score']}` (clean, -fallback, -retry, tok/s)\n"
            f"- Backup: `{backup.name}`\n"
            f"- Lưu ý: không thay đổi `decode.*` (bất biến notebook).\n"
        )
        print(f"Đã ghi đè {cfg.source} (backup: {backup.name}) và ghi {LOG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
