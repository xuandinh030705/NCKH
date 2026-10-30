"""Tải GGUF từ Hugging Face và kiểm tra file.

Ví dụ::

    python scripts/download_model.py
    python scripts/download_model.py --repo unsloth/Qwen3-4B-Instruct-2507-GGUF \
        --file Qwen3-4B-Instruct-2507-Q4_K_M.gguf
    python scripts/download_model.py --check-only

Không dùng torch/transformers. Chỉ cần ``huggingface_hub``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tutor.config import Config  # noqa: E402


def _human(size: int) -> str:
    gb = size / 1e9
    return f"{gb:.2f} GB"


def _valid_gguf(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "không tồn tại"
    size = path.stat().st_size
    if size < 50 * 1024 * 1024:
        return False, f"quá nhỏ ({_human(size)})"
    try:
        with path.open("rb") as fh:
            magic = fh.read(4)
    except OSError as exc:
        return False, f"không đọc được: {exc}"
    if magic != b"GGUF":
        return False, "không phải file GGUF (magic khác)"
    return True, _human(size)


def main(argv: list[str] | None = None) -> int:
    try:
        cfg = Config.load()
        default_repo = str(cfg.get("model.repo", "") or "")
        default_file = str(cfg.get("model.file", "") or "")
        default_dir = cfg.resolve("models")
    except Exception:
        default_repo, default_file = "", ""
        default_dir = ROOT / "models"

    parser = argparse.ArgumentParser(description="Tải model GGUF cho math-tutor-cli")
    parser.add_argument("--repo", default=default_repo, help="repo Hugging Face (org/name)")
    parser.add_argument("--file", default=default_file, help="tên file .gguf trong repo")
    parser.add_argument("--revision", default=None, help="nhánh/tag (mặc định main)")
    parser.add_argument("--out-dir", default=str(default_dir), help="thư mục lưu file")
    parser.add_argument("--check-only", action="store_true", help="chỉ kiểm tra file đã có")
    parser.add_argument("--force", action="store_true", help="tải lại dù file đã có")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    if not args.file:
        print("LỖI: thiếu --file (hoặc model.file trong config.yaml)")
        return 2
    target = out_dir / Path(args.file).name

    ok, detail = _valid_gguf(target)
    if args.check_only:
        if ok:
            print(f"OK: {target} ({detail})")
            return 0
        print(f"THIẾU/HỎNG: {target} — {detail}")
        return 1

    if ok and not args.force:
        print(f"Đã có {target} ({detail}) — bỏ qua tải. Dùng --force để tải lại.")
        return 0

    if not args.repo:
        print("LỖI: thiếu --repo (hoặc model.repo trong config.yaml)")
        return 2

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError:
        print("LỖI: chưa cài huggingface_hub. Chạy: pip install huggingface_hub")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Đang tải {args.repo}/{args.file} -> {out_dir} ...")
    try:
        downloaded = hf_hub_download(
            repo_id=args.repo,
            filename=args.file,
            revision=args.revision,
            local_dir=str(out_dir),
        )
    except Exception as exc:
        print(f"LỖI tải model: {exc}")
        print("Kiểm tra tên repo/file và kết nối mạng. Xem thêm models/README.md.")
        return 2

    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != target.resolve() and downloaded_path.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(downloaded_path.read_bytes())

    ok, detail = _valid_gguf(target)
    if not ok:
        print(f"LỖI: file tải về không hợp lệ: {target} — {detail}")
        return 2
    print(f"Xong: {target} ({detail})")
    print("Chạy thử: python -m tutor --once \"Tìm x, biết: x + 18 = 45\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
