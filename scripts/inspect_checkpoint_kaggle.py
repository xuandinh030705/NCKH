#!/usr/bin/env python3
"""Soi checkpoint LoRA (hoac model da merge) tren Kaggle - KHONG can GPU, KHONG import torch.

Muc dich: tra loi cau hoi "checkpoint-30 duoc fine-tune tren base nao?" truoc khi
quyet dinh merge LoRA. Script chi doc file JSON va liet ke file, nen chay duoc ca
tren may khong co CUDA.

Cach dung tren Kaggle:
    !python inspect_checkpoint_kaggle.py
    !python inspect_checkpoint_kaggle.py --path /kaggle/input/<dataset>/checkpoint-30

Quy uoc in an: tieng Viet khong dau trong cac dong KET_LUAN/JSON de copy-paste an toan
qua moi terminal; phan mo ta van giu tieng Viet co dau (stdout duoc ep UTF-8).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --- Duong dan mac dinh theo notebook reference/nckh-math.ipynb (cell 2) ---
DEFAULT_CHECKPOINT = (
    "/kaggle/input/models/aegold/math-nckh/transformers/default/1/checkpoint-30"
)
DEFAULT_FALLBACK = "/kaggle/input/models/aegold/math-nckh/transformers/default/1"

# Base duoc phep merge LoRA cua du an nay.
EXPECTED_BASE_MARKER = "qwen3-4b-instruct-2507"
WRONG_BASE_MARKER = "qwen3-8b"

VERDICT_OK = "KET_LUAN: BASE_KHOP -> duoc phep merge LoRA"
VERDICT_8B = "KET_LUAN: BASE_LA_8B -> KHONG dung LoRA, dung GGUF goc + prompt"
VERDICT_OTHER = "KET_LUAN: BASE_KHAC ({name}) -> KHONG dung LoRA"
VERDICT_UNKNOWN = (
    "KET_LUAN: BASE_KHONG_XAC_DINH -> KHONG dung LoRA (khong doc duoc ten base)"
)

ADAPTER_FILE = "adapter_config.json"
CONFIG_FILE = "config.json"


def _force_utf8() -> None:
    """Ep stdout/stderr sang UTF-8 de khong crash tren Windows/Kaggle."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Stream khong ho tro (vd. pytest capture) -> bo qua.
                pass


def read_json(path: Path) -> dict[str, Any] | None:
    """Doc file JSON, tra None neu thieu/hong (khong bao gio raise)."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"[CANH_BAO] Khong doc duoc JSON {path}: {exc}")
        return None
    if not isinstance(data, dict):
        print(f"[CANH_BAO] {path} khong phai JSON object, bo qua.")
        return None
    return data


def resolve_checkpoint_path(raw_path: str) -> Path:
    """Ap dung fallback y het notebook: neu checkpoint-30 khong ton tai thi lui ve thu muc cha."""
    path = Path(raw_path).expanduser()
    if path.exists():
        return path
    fallback = Path(DEFAULT_FALLBACK)
    if fallback.exists() and path.resolve() != fallback.resolve():
        print(f"[CANH_BAO] Khong thay {path}")
        print(f"[CANH_BAO] Dung duong dan du phong (nhu notebook): {fallback}")
        return fallback
    return path


def find_config_json(ckpt: Path) -> Path | None:
    """Tim config.json: trong checkpoint, roi toi 2 cap thu muc cha (kieu Kaggle model)."""
    candidates: list[Path] = [ckpt / CONFIG_FILE]
    parent = ckpt.parent
    for _ in range(2):
        candidates.append(parent / CONFIG_FILE)
        if parent.parent == parent:
            break
        parent = parent.parent
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def collect_files(root: Path) -> list[tuple[str, int]]:
    """Liet ke moi file duoi root (de quy), sap xep theo duong dan tuong doi."""
    if root.is_file():
        return [(root.name, root.stat().st_size)]
    files: list[tuple[str, int]] = []
    try:
        for entry in sorted(root.rglob("*")):
            if entry.is_file():
                try:
                    files.append((entry.relative_to(root).as_posix(), entry.stat().st_size))
                except OSError as exc:
                    print(f"[CANH_BAO] Khong doc duoc kich thuoc {entry}: {exc}")
    except OSError as exc:
        print(f"[CANH_BAO] Khong duyet duoc thu muc {root}: {exc}")
    return files


def human_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.3f} MB"


@dataclass
class Inspection:
    """Ket qua soi checkpoint, du de in ra va dump JSON."""

    requested_path: str
    path: str
    exists: bool
    has_adapter: bool
    adapter_config: dict[str, Any] = field(default_factory=dict)
    config_path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    files: list[tuple[str, int]] = field(default_factory=list)
    total_bytes: int = 0
    base_model_name: str | None = None
    base_source: str = "khong_co"
    verdict: str = VERDICT_UNKNOWN


def _adapter_extra_keys(adapter: dict[str, Any]) -> list[str]:
    """Cac khoa adapter khac ngoai nhom da in rieng."""
    known = {"base_model_name_or_path", "r", "lora_alpha", "target_modules"}
    return sorted(k for k in adapter if k not in known)


def _base_from_adapter(adapter: dict[str, Any]) -> str | None:
    value = adapter.get("base_model_name_or_path")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _base_from_config(config: dict[str, Any]) -> str | None:
    """Voi model da merge: config.json thuong con _name_or_path cua base."""
    for key in ("_name_or_path", "base_model_name_or_path", "name_or_path"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def decide_verdict(base_model_name: str | None) -> str:
    """So khop ten base (lower, strip) voi base mong doi cua du an."""
    if not base_model_name:
        return VERDICT_UNKNOWN
    name = base_model_name.strip().lower()
    if EXPECTED_BASE_MARKER in name:
        return VERDICT_OK
    if WRONG_BASE_MARKER in name:
        return VERDICT_8B
    return VERDICT_OTHER.format(name=base_model_name.strip())


def inspect(path: Path, requested: str) -> Inspection:
    """Thu thap toan bo thong tin can thiet, khong import torch."""
    info = Inspection(requested_path=requested, path=str(path), exists=path.exists(), has_adapter=False)

    if not info.exists:
        info.verdict = VERDICT_UNKNOWN
        return info

    adapter_path = path / ADAPTER_FILE if path.is_dir() else None
    adapter = read_json(adapter_path) if adapter_path and adapter_path.is_file() else None
    if adapter is not None:
        info.has_adapter = True
        info.adapter_config = adapter

    config_path = find_config_json(path) if path.is_dir() else None
    config = read_json(config_path) if config_path else None
    if config is not None:
        info.config_path = str(config_path)
        info.config = config

    info.files = collect_files(path)
    info.total_bytes = sum(size for _, size in info.files)

    # Uu tien ten base ghi trong adapter_config.json, sau do moi den config.json.
    base = _base_from_adapter(info.adapter_config)
    if base:
        info.base_source = ADAPTER_FILE
    else:
        base = _base_from_config(info.config)
        if base:
            info.base_source = config_path.name if config_path else CONFIG_FILE
    info.base_model_name = base
    info.verdict = decide_verdict(base)
    return info


def _get(config: dict[str, Any], key: str) -> Any:
    """Lay gia tri, tra 'None' string neu thieu de in cho ro rang."""
    value = config.get(key)
    return "None" if value is None else value


def print_report(info: Inspection) -> None:
    """In bao cao dang nguoi doc duoc."""
    print("=" * 72)
    print("SOI CHECKPOINT / MODEL - math-tutor-cli (CNTT-2026-35)")
    print("=" * 72)
    print(f"Duong dan yeu cau : {info.requested_path}")
    print(f"Duong dan dung    : {info.path}")
    print(f"Ton tai           : {'CO' if info.exists else 'KHONG'}")
    print(f"Co LoRA adapter   : {'CO (' + ADAPTER_FILE + ')' if info.has_adapter else 'KHONG (coi nhu model da merge)'}")

    print("-" * 72)
    print("[1] adapter_config.json")
    if info.has_adapter:
        adapter = info.adapter_config
        print(f"  base_model_name_or_path : {_get(adapter, 'base_model_name_or_path')}")
        print(f"  r                       : {_get(adapter, 'r')}")
        print(f"  lora_alpha              : {_get(adapter, 'lora_alpha')}")
        print(f"  target_modules          : {_get(adapter, 'target_modules')}")
        extras = _adapter_extra_keys(adapter)
        if extras:
            print("  cac khoa khac:")
            for key in extras:
                print(f"    - {key} = {adapter[key]}")
        else:
            print("  cac khoa khac           : (khong co)")
    else:
        print("  (khong co file adapter_config.json)")

    print("-" * 72)
    print("[2] config.json (kien truc model)")
    if info.config_path:
        print(f"  nguon file              : {info.config_path}")
        config = info.config
        print(f"  hidden_size             : {_get(config, 'hidden_size')}")
        print(f"  num_hidden_layers       : {_get(config, 'num_hidden_layers')}")
        print(f"  model_type              : {_get(config, 'model_type')}")
        print(f"  architectures           : {_get(config, 'architectures')}")
        print(f"  max_position_embeddings : {_get(config, 'max_position_embeddings')}")
        print(f"  torch_dtype             : {_get(config, 'torch_dtype')}")
    else:
        print("  (khong tim thay config.json trong checkpoint hoac 2 cap cha)")

    print("-" * 72)
    print("[3] File trong thu muc")
    if info.files:
        width = max(len(name) for name, _ in info.files)
        for name, size in info.files:
            print(f"  {name.ljust(width)}  {human_mb(size):>12}")
        print(f"  {'TONG CONG'.ljust(width)}  {human_mb(info.total_bytes):>12}")
        print(f"  So file: {len(info.files)}")
    else:
        print("  (khong co file nao hoac khong doc duoc thu muc)")

    print("-" * 72)
    print("[4] Ten base dung de quyet dinh")
    print(f"  base_model_name         : {info.base_model_name or 'None'}")
    print(f"  nguon base              : {info.base_source}")
    print("=" * 72)
    # Dong quyet dinh LUON in cuoi cung, de khong bi lan trong log.
    print(info.verdict)
    print("=" * 72)


def build_summary(info: Inspection) -> dict[str, Any]:
    """Dong JSON gon de ca nhom copy-paste lai cho nhau."""
    adapter = info.adapter_config
    config = info.config
    return {
        "requested_path": info.requested_path,
        "path": info.path,
        "exists": info.exists,
        "has_adapter": info.has_adapter,
        "base_model_name_or_path": info.base_model_name,
        "base_source": info.base_source,
        "r": adapter.get("r"),
        "lora_alpha": adapter.get("lora_alpha"),
        "target_modules": adapter.get("target_modules"),
        "adapter_other_keys": _adapter_extra_keys(adapter),
        "config_path": info.config_path,
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "torch_dtype": config.get("torch_dtype"),
        "file_count": len(info.files),
        "total_size_mb": round(info.total_bytes / (1024 * 1024), 3),
        "files": [{"name": name, "size_mb": round(size / (1024 * 1024), 3)} for name, size in info.files],
        "verdict": info.verdict,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Soi checkpoint LoRA / model merged tren Kaggle (chi doc JSON, khong can torch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_CHECKPOINT,
        help="Thu muc checkpoint (mac dinh theo notebook, co fallback ve thu muc cha).",
    )
    parser.add_argument(
        "--no-files",
        action="store_true",
        help="Bo qua buoc liet ke file (nhanh hon voi checkpoint lon).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    _force_utf8()
    args = parse_args(argv)

    path = resolve_checkpoint_path(args.path)
    if not path.exists():
        print("=" * 72)
        print("[LOI] Khong tim thay checkpoint.")
        print(f"  Duong dan yeu cau: {args.path}")
        print("  Goi y: kiem tra lai dataset da Add Data tren Kaggle chua,")
        print("  hoac truyen --path /kaggle/input/<dataset>/checkpoint-30")
        print("=" * 72)
        info = Inspection(requested_path=args.path, path=str(path), exists=False, has_adapter=False)
        info.verdict = VERDICT_UNKNOWN
        print(info.verdict)
        print("JSON_SUMMARY: " + json.dumps(build_summary(info), ensure_ascii=False, separators=(",", ":")))
        return 1

    info = inspect(path, args.path)
    if args.no_files:
        info.files = []
        info.total_bytes = 0
    print_report(info)
    print("JSON_SUMMARY: " + json.dumps(build_summary(info), ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
