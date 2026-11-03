"""Kiểm tra phát hiện phần cứng + chọn profile + đọc metadata GGUF."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from tutor.config import Config
from tutor.hardware import (
    HwInfo,
    choose_profile,
    describe,
    estimate_kv_gb_per_1k,
    read_gguf_meta,
)

ROOT = Path(__file__).resolve().parent.parent
CFG = Config.load(ROOT / "config.yaml")

META = {
    "n_layer": 36,
    "n_head": 32,
    "n_head_kv": 8,
    "n_embd": 2560,
    "file_size_gb": 2.33,
}


def test_estimate_kv_f16() -> None:
    kv = estimate_kv_gb_per_1k(META, "f16")
    # 36 layer * 8 kv head * 80 head_dim * 2 (K,V) * 2 byte * 1000 / 1e9
    assert math.isclose(kv, 0.09216, rel_tol=1e-6)


def test_estimate_kv_q8() -> None:
    assert estimate_kv_gb_per_1k(META, "q8_0") < estimate_kv_gb_per_1k(META, "f16")


def test_estimate_kv_missing_meta() -> None:
    assert estimate_kv_gb_per_1k({}) == 0.0


def test_choose_profile_cpu_only() -> None:
    hw = HwInfo(vram_free_gb=0.0, vram_total_gb=0.0, cpu_phys=8, ram_gb=16.0)
    prof = choose_profile(hw, META, CFG)
    assert prof.n_gpu_layers == 0
    assert prof.flash_attn is False
    assert prof.n_ctx == int(CFG.get("model.n_ctx"))


def test_choose_profile_full_gpu() -> None:
    hw = HwInfo(vram_free_gb=8.0, vram_total_gb=8.0, cpu_phys=8, ram_gb=32.0,
                gpu_name="Test GPU")
    prof = choose_profile(hw, META, CFG)
    assert prof.n_gpu_layers == -1
    assert prof.kv_quant == "f16"
    assert prof.n_ctx >= 1536


def test_choose_profile_tight_vram() -> None:
    hw = HwInfo(vram_free_gb=3.0, vram_total_gb=3.0, cpu_phys=8, ram_gb=16.0,
                gpu_name="Tight GPU")
    prof = choose_profile(hw, META, CFG)
    assert prof.kv_quant == "q8_0"
    assert 0 <= prof.n_gpu_layers < 36


def test_describe() -> None:
    hw = HwInfo(vram_free_gb=5.9, vram_total_gb=6.0, cpu_phys=12, ram_gb=32.0,
                gpu_name="RTX 4050")
    text = describe(hw)
    assert "RTX 4050" in text and "VRAM" in text
    cpu = describe(HwInfo(0.0, 0.0, 8, 16.0))
    assert "CPU-only" in cpu


def test_read_gguf_meta_missing() -> None:
    meta = read_gguf_meta(ROOT / "models" / "khong-co.gguf")
    assert meta.get("file_size_gb") == 0.0


@pytest.mark.skipif(
    not (ROOT / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf").exists(),
    reason="chưa tải model",
)
def test_read_gguf_meta_real_model() -> None:
    meta = read_gguf_meta(ROOT / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    assert meta.get("arch") == "qwen3"
    assert meta.get("n_layer") == 36
    assert meta.get("n_embd") == 2560
    assert meta.get("tokenizer.chat_template")
    assert meta.get("file_size_gb", 0) > 2.0
