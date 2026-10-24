"""Đo tốc độ thật: thời gian nạp, TTFT, tok/s. Xuất JSON + in bảng.

Ví dụ::

    python scripts/benchmark.py --backend llama --reps 3 --max-tokens 128
    python scripts/benchmark.py --backend mock --reps 1

Chỉ backend ``llama`` mới cho số liệu [ĐO THẬT trên máy này]; ``mock`` là [MOCK].
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tutor.config import Config  # noqa: E402
from tutor.engine import Engine  # noqa: E402
from tutor.hardware import describe, detect  # noqa: E402
from tutor.notebook_source import load as load_notebook  # noqa: E402
from tutor.prompts import build_system_from_cfg  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp/GGUF")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default=None, choices=["llama", "mock"])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--mode", default="1", choices=["1", "2", "3"])
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    backend = args.backend or str(cfg.get("engine.backend", "llama"))
    cfg = cfg.with_overrides({"engine.backend": backend})
    evidence = "[ĐO THẬT trên máy này]" if backend == "llama" else "[MOCK]"

    nb = load_notebook()
    prompt = args.prompt or nb.default_user_text
    messages = [
        {"role": "system", "content": build_system_from_cfg(args.mode, "vi", cfg)},
        {"role": "user", "content": prompt},
    ]

    hw = detect()
    engine = Engine(cfg, model_path=args.model)
    t0 = time.time()
    engine.load()
    load_s = time.time() - t0

    runs = []
    for i in range(max(1, args.reps)):
        res = engine.generate(messages, lang="vi", mode=args.mode, max_new_tokens=args.max_tokens)
        runs.append(
            {
                "rep": i + 1,
                "ttft_s": round(res.ttft_s, 4),
                "total_s": round(res.total_s, 4),
                "gen_tokens": res.gen_tokens,
                "tok_per_s": round(res.tok_per_s, 3),
                "prompt_tokens": res.prompt_tokens,
                "text": res.text,
            }
        )
    engine.close()

    tok = [r["tok_per_s"] for r in runs]
    ttft = [r["ttft_s"] for r in runs]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": backend,
        "evidence": evidence,
        "hardware": {
            "description": describe(hw),
            "gpu_name": hw.gpu_name,
            "vram_free_gb": hw.vram_free_gb,
            "vram_total_gb": hw.vram_total_gb,
            "ram_gb": hw.ram_gb,
            "cpu_phys": hw.cpu_phys,
            "has_cuda": hw.has_cuda,
        },
        "config": str(cfg.source),
        "profile": engine.profile.as_dict() if engine.profile else None,
        "model": str(cfg.model_path(args.model)),
        "mode": args.mode,
        "max_tokens": args.max_tokens,
        "reps": max(1, args.reps),
        "load_s": round(load_s, 3),
        "ttft_s_mean": round(statistics.mean(ttft), 4) if ttft else 0.0,
        "ttft_s_min": round(min(ttft), 4) if ttft else 0.0,
        "tok_per_s_mean": round(statistics.mean(tok), 3) if tok else 0.0,
        "tok_per_s_min": round(min(tok), 3) if tok else 0.0,
        "tok_per_s_max": round(max(tok), 3) if tok else 0.0,
        "runs": runs,
        "note": (
            "Số liệu đo thật trên máy này; không suy rộng sang máy khác."
            if backend == "llama"
            else "Số liệu [MOCK] do backend giả lập — KHÔNG dùng để kết luận tốc độ."
        ),
    }

    out = Path(args.out) if args.out else (
        ROOT / "eval" / "runs" / f"bench_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{backend}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== Benchmark {backend} {evidence} ===")
    print(f"Phần cứng: {report['hardware']['description']}")
    print(f"Nạp model: {report['load_s']} s")
    print(f"TTFT: trung bình {report['ttft_s_mean']} s | min {report['ttft_s_min']} s")
    print(f"Tốc độ: trung bình {report['tok_per_s_mean']} tok/s | "
          f"min {report['tok_per_s_min']} | max {report['tok_per_s_max']}")
    print(f"Profile: {report['profile']}")
    print(f"Đã ghi {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
