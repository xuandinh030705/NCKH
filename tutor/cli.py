"""CLI terminal: REPL, --once, --stream, các lệnh /mode /reset /stats /demo /help /exit."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .config import Config
from .engine import Engine, EngineError
from .notebook_source import load as load_notebook
from .prompts import DEFAULT_MODE, DEMO_DIALOGUE, MODE_NAMES, mode_key
from .session import TutorSession


# ------------------------------------------------------------------- tiện ích
def _force_utf8() -> None:
    """Windows: ép UTF-8 để không lỗi khi in tiếng Việt/emoji."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    if os.name == "nt":
        try:
            os.system("")  # bật ANSI escape trên cmd cũ
        except Exception:
            pass


def _hint_chcp() -> None:
    if os.name != "nt":
        return
    try:
        enc = (sys.stdout.encoding or "").lower()
    except Exception:
        enc = ""
    if enc not in ("utf-8", "utf8"):
        print("MẸO: nếu chữ tiếng Việt bị lỗi, chạy 'chcp 65001' trước khi mở chương trình.")


def _fmt_timing(gen: Any, retries: int) -> str:
    if gen is None:
        return ""
    return (f"{gen.total_s:.1f}s | {gen.tok_per_s:.1f} tok/s | "
            f"TTFT {gen.ttft_s:.2f}s | {gen.gen_tokens} token | retry {retries}")


def _print_violations(result: Any) -> None:
    if result.used_fallback:
        print("  (đã dùng câu mẫu dự phòng vì model vi phạm quy tắc)")
    if result.violations:
        names = ", ".join(sorted({v.rule for v in result.violations}))
        print(f"  CẢNH BÁO: còn vi phạm sau guard: {names}")


# ----------------------------------------------------------------------- demo
def _run_demo(session: TutorSession, cfg: Config) -> None:
    nb = load_notebook()
    print("\n=== MỤC 3: hội thoại 3 lượt (x * 3 - 4 = 11) ===")
    session.set_mode("1")
    for i, msg in enumerate(DEMO_DIALOGUE, start=1):
        print(f"\nHọc sinh (lượt {i}): {msg}")
        res = session.send(msg)
        print(f"Gia sư:\n{res.reply}")
        print(f"  [{_fmt_timing(res.gen, res.retries)}]")

    print("\n=== MỤC 6: 4 đề lớp 2–5 (chế độ 1, mỗi đề một phiên) ===")
    for item in nb.elementary_tests:
        session.set_mode("1")
        print(f"\n[{item['grade']} - {item['topic']}]")
        print(f"Đề: {item['prompt']}")
        res = session.send(item["prompt"])
        print(f"Gia sư:\n{res.reply}")
        print(f"  [{_fmt_timing(res.gen, res.retries)}]")
    session.set_mode("1")


# ----------------------------------------------------------------------- REPL
def _repl(session: TutorSession, cfg: Config, *, stream: bool) -> int:
    print("=" * 68)
    print("GIA SƯ TOÁN TIỂU HỌC — gõ /help để xem lệnh, /exit để thoát")
    print(f"Chế độ: {MODE_NAMES[session.mode]}")
    print("=" * 68)
    while True:
        try:
            text = input("\nBạn: ").strip()
        except EOFError:
            print("\nTạm biệt em!")
            return 0
        except KeyboardInterrupt:
            print("\n(Ctrl+C: gõ /exit để thoát)")
            continue
        if not text:
            continue
        if text.startswith("/"):
            cmd, _, arg = text.partition(" ")
            cmd = cmd.lower()
            if cmd in ("/exit", "/quit", "/q"):
                print("Tạm biệt em!")
                return 0
            if cmd == "/help":
                _print_help()
                continue
            if cmd == "/mode":
                try:
                    session.set_mode(arg.strip() or DEFAULT_MODE)
                    print(f"Đã chuyển sang: {MODE_NAMES[session.mode]} (phiên mới)")
                except ValueError as exc:
                    print(f"Lỗi: {exc}")
                continue
            if cmd == "/reset":
                session.reset()
                print("Đã làm mới phiên học.")
                continue
            if cmd == "/stats":
                print(session.stats_line())
                continue
            if cmd == "/demo":
                try:
                    _run_demo(session, cfg)
                except KeyboardInterrupt:
                    print("\n(đã ngắt demo)")
                continue
            print(f"Lệnh không rõ: {cmd}. Gõ /help.")
            continue

        try:
            if stream:
                print("Gia sư: ", end="", flush=True)
                res = session.send(text, stream_cb=lambda s: print(s, end="", flush=True), raw=True)
                print()
            else:
                res = session.send(text)
                print(f"\nGia sư:\n{res.reply}")
            print(f"  [{_fmt_timing(res.gen, res.retries)}]")
            _print_violations(res)
        except KeyboardInterrupt:
            print("\n(đã ngắt lượt này, phiên học vẫn tiếp tục)")
            continue
        except EngineError as exc:
            print(f"\nLỖI: {exc}")
            continue
    return 0


def _print_help() -> None:
    print(
        "\nLệnh:\n"
        "  /mode 1|2|3  đổi chế độ (1 gợi mở, 2 sơ đồ đoạn thẳng, 3 bài giải mẫu)\n"
        "  /reset       làm mới phiên\n"
        "  /stats       thống kê phiên hiện tại\n"
        "  /demo        chạy lại MỤC 3 và MỤC 6 của notebook\n"
        "  /help        trợ giúp\n"
        "  /exit        thoát\n"
        "Ctrl+C ngắt lượt đang sinh, không thoát chương trình."
    )


# ----------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tutor", description="Chatbot gia sư Toán tiểu học")
    p.add_argument("--once", metavar="TEXT", help="chạy một câu rồi thoát")
    p.add_argument("--mode", default=DEFAULT_MODE, help="chế độ 1|2|3 (mặc định 1)")
    p.add_argument("--stream", action="store_true", help="streaming thô, không kiểm duyệt")
    p.add_argument("--model", help="đường dẫn file .gguf (ghi đè config)")
    p.add_argument("--config", help="đường dẫn config.yaml")
    p.add_argument("--verbose", action="store_true", help="in chi tiết lỗi")
    p.add_argument("--mock", action="store_true", help="dùng backend mock (không cần model)")
    return p


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    args = build_parser().parse_args(argv)

    try:
        cfg = Config.load(args.config)
    except FileNotFoundError as exc:
        print(f"LỖI: {exc}")
        return 2

    overrides: dict[str, Any] = {}
    if args.mock:
        overrides["engine.backend"] = "mock"
    if args.verbose:
        overrides["engine.verbose"] = True
    if overrides:
        cfg = cfg.with_overrides(overrides)

    try:
        mode = mode_key(args.mode)
    except ValueError as exc:
        print(f"LỖI: {exc}")
        return 2

    try:
        engine = Engine(cfg, model_path=args.model)
        if engine.backend != "mock":
            print("Đang nạp model...", flush=True)
        engine.load()
        if engine.backend != "mock":
            prof = engine.profile
            print(f"Nạp xong sau {engine.load_s:.1f}s | n_ctx={prof.n_ctx} | "
                  f"n_gpu_layers={prof.n_gpu_layers} | kv={prof.kv_quant}")
    except EngineError as exc:
        print(f"\nLỖI: {exc}")
        return 2
    except Exception as exc:  # pragma: no cover - lỗi bất ngờ
        if args.verbose:
            raise
        print(f"\nLỖI không mong đợi: {exc}\nChạy lại với --verbose để xem chi tiết.")
        return 2

    session = TutorSession(engine, mode=mode, cfg=cfg)

    if args.once:
        try:
            single = int(cfg.get("decode.max_new_tokens_single", 400))
            res = session.send(args.once, max_tokens=single)
            print(res.reply)
            print(f"[{_fmt_timing(res.gen, res.retries)}]")
            _print_violations(res)
            return 0
        except KeyboardInterrupt:
            print("\n(đã ngắt)")
            return 130
        except EngineError as exc:
            print(f"LỖI: {exc}")
            return 2

    _hint_chcp()
    try:
        return _repl(session, cfg, stream=args.stream)
    except KeyboardInterrupt:
        print("\nTạm biệt em!")
        return 0
    finally:
        engine.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
