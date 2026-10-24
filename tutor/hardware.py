"""Phát hiện phần cứng và chọn profile nạp model (VRAM trống, KV cache, thread)."""
from __future__ import annotations

import ctypes
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# GGUF metadata value types -> (struct format, kích thước cố định)
_GGUF_SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


@dataclass(frozen=True)
class HwInfo:
    vram_free_gb: float
    vram_total_gb: float
    cpu_phys: int
    ram_gb: float
    gpu_name: str = ""

    @property
    def has_cuda(self) -> bool:
        return self.vram_total_gb > 0


@dataclass(frozen=True)
class Profile:
    n_gpu_layers: int
    n_ctx: int
    kv_quant: str
    flash_attn: bool
    n_threads: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "kv_quant": self.kv_quant,
            "flash_attn": self.flash_attn,
            "n_threads": self.n_threads,
        }


# ------------------------------------------------------------------ phần cứng
def _ram_gb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / 1e9
    except Exception:
        pass
    if os.name == "nt":
        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemStatus()
        stat.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        return stat.ullTotalPhys / 1e9
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9  # type: ignore[attr-defined]
    except Exception:
        return 0.0


def _cpu_phys() -> int:
    try:
        import psutil  # type: ignore

        return psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except Exception:
        return max(1, (os.cpu_count() or 4) // 2)


def _gpu_info() -> tuple[float, float, str]:
    """(vram_free_gb, vram_total_gb, tên GPU) — (0,0,'') nếu không có NVIDIA."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "ignore")
            return mem.free / 1e9, mem.total / 1e9, str(name)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return 0.0, 0.0, ""


def detect() -> HwInfo:
    """Không bao giờ ném lỗi: thiếu driver/pynvml thì coi như CPU-only."""
    free, total, name = _gpu_info()
    return HwInfo(
        vram_free_gb=round(free, 2),
        vram_total_gb=round(total, 2),
        cpu_phys=_cpu_phys(),
        ram_gb=round(_ram_gb(), 2),
        gpu_name=name,
    )


# -------------------------------------------------------------- đọc GGUF meta
def read_gguf_meta(path: str | Path) -> dict[str, Any]:
    """Đọc metadata GGUF (không nạp trọng số). Lỗi -> dict rỗng."""
    p = Path(path)
    meta: dict[str, Any] = {"file_size_gb": round(p.stat().st_size / 1e9, 3) if p.exists() else 0.0}
    try:
        with p.open("rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return meta
            version, = struct.unpack("<I", f.read(4))
            _tensor_count, kv_count = struct.unpack("<QQ", f.read(16))

            def read_string() -> str:
                n, = struct.unpack("<Q", f.read(8))
                return f.read(n).decode("utf-8", "ignore")

            def read_value(vtype: int) -> Any:
                if vtype == _GGUF_STRING:
                    return read_string()
                if vtype == _GGUF_ARRAY:
                    itype, = struct.unpack("<I", f.read(4))
                    count, = struct.unpack("<Q", f.read(8))
                    if itype == _GGUF_STRING:
                        return [read_string() for _ in range(count)]
                    if itype in _GGUF_SCALARS:
                        fmt, size = _GGUF_SCALARS[itype]
                        raw = f.read(size * count)
                        return list(struct.unpack(f"<{count}{fmt[1:]}", raw))
                    return []
                if vtype in _GGUF_SCALARS:
                    fmt, size = _GGUF_SCALARS[vtype]
                    return struct.unpack(fmt, f.read(size))[0]
                return None

            for _ in range(int(kv_count)):
                key = read_string()
                vtype, = struct.unpack("<I", f.read(4))
                value = read_value(vtype)
                if key.startswith(("general.", "qwen", "llama", "tokenizer.chat_template")):
                    meta[key] = value
    except Exception:
        return meta

    arch = meta.get("general.architecture", "")
    prefix = str(arch)
    for out_key, suffix in (
        ("n_layer", "block_count"),
        ("n_head", "attention.head_count"),
        ("n_head_kv", "attention.head_count_kv"),
        ("n_embd", "embedding_length"),
        ("n_ctx_train", "context_length"),
    ):
        value = meta.get(f"{prefix}.{suffix}")
        if value is not None:
            meta[out_key] = value
    meta["arch"] = arch
    return meta


# ------------------------------------------------------------------- profile
def estimate_kv_gb_per_1k(model_meta: dict[str, Any], kv_quant: str = "f16") -> float:
    """Ước lượng KV cache cho 1000 token, dựa trên metadata thật của GGUF."""
    n_layer = int(model_meta.get("n_layer") or 0)
    n_head = int(model_meta.get("n_head") or 0)
    n_kv = int(model_meta.get("n_head_kv") or n_head or 0)
    n_embd = int(model_meta.get("n_embd") or 0)
    if not (n_layer and n_head and n_embd):
        return 0.0
    head_dim = n_embd // max(1, n_head)
    bytes_per = 1.0625 if str(kv_quant).lower() in ("q8_0", "q8") else 2.0
    # 2 = K và V
    return n_layer * n_kv * head_dim * 2 * bytes_per * 1000 / 1e9


def choose_profile(hw: HwInfo, model_meta: dict[str, Any], cfg: Any | None = None) -> Profile:
    """Chọn n_gpu_layers/n_ctx/KV theo VRAM TRỐNG. Không đủ chỗ thì offload một phần."""
    get = (cfg.get if cfg is not None else (lambda key, default=None: default))
    n_ctx = int(get("model.n_ctx", 2048) or 2048)
    n_ctx_max = int(get("model.n_ctx_max", n_ctx) or n_ctx)
    buffer_gb = float(get("model.gpu_buffer_gb", 0.55) or 0.55)
    want_layers = get("model.n_gpu_layers", "auto")
    want_kv = str(get("model.kv_quant", "auto") or "auto").lower()
    want_fa = get("model.flash_attn", "auto")
    n_layer = int(model_meta.get("n_layer") or 0)
    weights_gb = float(model_meta.get("file_size_gb") or 0.0) * 1.02

    kv_quant = "f16"
    if want_kv in ("q8_0", "q8"):
        kv_quant = "q8_0"
    elif want_kv == "auto":
        kv_quant = "q8_0" if hw.has_cuda and hw.vram_free_gb < 4.5 else "f16"

    raw_threads = get("model.n_threads", "auto")
    threads = hw.cpu_phys if raw_threads in (None, "auto", 0, "0") else int(raw_threads)

    if not hw.has_cuda:
        return Profile(n_gpu_layers=0, n_ctx=n_ctx, kv_quant=kv_quant,
                       flash_attn=False, n_threads=threads)

    # thử giảm n_ctx nếu thiếu VRAM
    for ctx in sorted({n_ctx, min(n_ctx_max, 2048), 1536}, reverse=True):
        kv_gb = estimate_kv_gb_per_1k(model_meta, kv_quant) * ctx / 1000
        need = weights_gb + kv_gb + buffer_gb
        if need <= hw.vram_free_gb * 0.97:
            return Profile(n_gpu_layers=-1, n_ctx=ctx, kv_quant=kv_quant,
                           flash_attn=bool(want_fa) if isinstance(want_fa, bool) else True,
                           n_threads=threads)

    # không đủ: offload một phần theo tỉ lệ
    kv_gb = estimate_kv_gb_per_1k(model_meta, kv_quant) * n_ctx / 1000
    avail = hw.vram_free_gb * 0.95 - kv_gb - buffer_gb
    if want_layers not in (None, "auto"):
        layers = int(want_layers)
    elif avail <= 0 or n_layer <= 0:
        layers = 0
    else:
        layers = max(0, min(n_layer, int(n_layer * (avail / max(weights_gb, 1e-6)))))
        if layers >= n_layer:
            layers = n_layer - 1
    return Profile(n_gpu_layers=layers, n_ctx=n_ctx, kv_quant=kv_quant,
                   flash_attn=bool(want_fa) if isinstance(want_fa, bool) else True,
                   n_threads=threads)


def describe(hw: HwInfo) -> str:
    """Chuỗi mô tả ngắn để in cho người dùng."""
    if hw.has_cuda:
        return (f"{hw.gpu_name} | VRAM trống {hw.vram_free_gb:.1f}/{hw.vram_total_gb:.1f} GB | "
                f"RAM {hw.ram_gb:.0f} GB | {hw.cpu_phys} core")
    return f"CPU-only | RAM {hw.ram_gb:.0f} GB | {hw.cpu_phys} core"


__all__ = [
    "HwInfo", "Profile", "choose_profile", "describe", "detect",
    "estimate_kv_gb_per_1k", "read_gguf_meta",
]
