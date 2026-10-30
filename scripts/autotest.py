"""Chạy đánh giá thật hoặc mock, in bảng tổng hợp.

Ví dụ::

    python scripts/autotest.py --backend mock --n 12 --mode all
    python scripts/autotest.py --backend llama --n 100 --mode 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.runner import run  # noqa: E402
from tutor.config import Config  # noqa: E402


def _print_summary(summary: dict) -> None:
    print(f"\n=== Chế độ {summary.get('mode')} | {summary.get('backend')} | "
          f"{summary.get('evidence')} ===")
    print(f"Hội thoại: {summary.get('n_dialogues')} | lượt: {summary.get('n_turns')} | "
          f"sạch: {summary.get('clean_dialogues')}/{summary.get('n_dialogues')} "
          f"({float(summary.get('clean_dialogue_rate', 0)) * 100:.1f}%)")
    print(f"Fallback: {float(summary.get('fallback_rate', 0)) * 100:.1f}% | "
          f"retry: {float(summary.get('retry_rate', 0)) * 100:.1f}% | "
          f"vi phạm sau guard: {summary.get('violations_after_guard')}")
    print(f"TTFT: {summary.get('avg_ttft_s')} s | tốc độ: {summary.get('avg_tok_per_s')} tok/s "
          f"{summary.get('evidence')}")
    rows = []
    for key, node in (summary.get("labels") or {}).items():
        rate = node.get("rate")
        rows.append(f"{key}={(rate * 100):.1f}%" if rate is not None else f"{key}=—")
    print("Nhãn: " + " ".join(rows))
    gates = summary.get("gates") or {}
    print(f"Cổng: {gates.get('status')} — {gates.get('reason')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Autotest chatbot gia sư toán")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", default="all", help="1 | 2 | 3 | all")
    parser.add_argument("--backend", default=None, choices=["llama", "mock"],
                        help="mặc định theo config.yaml")
    parser.add_argument("--lang", default="vi", choices=["vi", "vi_nodiac", "en", "mixed"])
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--strict", action="store_true", help="exit 1 nếu cổng không đạt")
    args = parser.parse_args(argv)

    try:
        cfg = Config.load(args.config)
    except Exception as exc:
        print(f"LỖI cấu hình: {exc}")
        return 2

    n = int(args.n if args.n is not None else cfg.get("eval.n_dialogues", 100) or 100)
    seed = int(args.seed if args.seed is not None else cfg.get("eval.seed", 1234) or 1234)

    modes = ["1", "2", "3"] if args.mode == "all" else [args.mode]
    if any(m not in ("1", "2", "3") for m in modes):
        print("LỖI: --mode phải là 1, 2, 3 hoặc all")
        return 2

    backend = args.backend or str(cfg.get("engine.backend", "llama"))

    summaries = []
    for mode in modes:
        try:
            run_dir = run(
                cfg,
                n=n,
                seed=seed,
                mode=mode,
                backend=backend,
                max_turns=args.max_turns,
                lang=args.lang,
            )
        except Exception as exc:
            print(f"\nLỖI khi chạy chế độ {mode}: {exc}")
            return 2
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        summaries.append(summary)
        _print_summary(summary)
        print(f"Thư mục: {run_dir}")

    if args.strict:
        bad = [s for s in summaries if (s.get("gates") or {}).get("status") == "fail"]
        if bad:
            print(f"\n{len(bad)} lượt chạy KHÔNG ĐẠT cổng.")
            return 1
        skipped = [s for s in summaries if (s.get("gates") or {}).get("status") == "skipped"]
        if skipped:
            print(f"\n{len(skipped)} lượt chạy bỏ qua cổng (mock/chưa đủ mẫu) — "
                  "không kết luận đạt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
