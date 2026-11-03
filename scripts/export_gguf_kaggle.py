#!/usr/bin/env python3
"""Merge LoRA -> GGUF tren Kaggle, roi chay lai 4 de frozen cua notebook.

CHI dung khi buoc 0 (inspect_checkpoint_kaggle.py) ket luan base cua checkpoint la
Qwen3-4B-Instruct-2507. Neu base la Qwen3-8B thi KHONG merge: dung GGUF goc + prompt.

Script nay chi chay tren Kaggle (co unsloth/torch). Moi import nang (unsloth, torch)
nam trong ham, nen `--help` va pytest collection khong keo torch vao.

Vi du:
    !python export_gguf_kaggle.py --quant q4_k_m
    !python export_gguf_kaggle.py --adapter /kaggle/input/.../checkpoint-30 --skip-frozen
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# --- Mac dinh theo notebook reference/nckh-math.ipynb (cell 2) ---
DEFAULT_ADAPTER = "/kaggle/input/models/aegold/math-nckh/transformers/default/1/checkpoint-30"
DEFAULT_OUTDIR = "/kaggle/working/gguf"
DEFAULT_FROZEN_OUT = "/kaggle/working/frozen_outputs.jsonl"
DEFAULT_NOTEBOOK_NAME = "nckh-math.ipynb"

# Tham so lay dung theo notebook (cell 3: run_pedagogical_inference, cell 5, cell 6).
SAMPLING: dict[str, Any] = {
    "max_new_tokens": 300,
    "temperature": 0.25,
    "top_p": 0.85,
    "repetition_penalty": 1.15,
}
ENABLE_THINKING = False
MAX_SEQ_LENGTH = 2048

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"

# Ten file GGUF xuat ra, khop voi ten trong models/README.md.
OUTPUT_STEM = "Qwen3-4B-Instruct-2507"


# --------------------------------------------------------------------------- #
# Tien ich chung
# --------------------------------------------------------------------------- #
def _force_utf8() -> None:
    """Ep stdout/stderr sang UTF-8 de in tieng Viet tren Windows/Kaggle khong loi."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def _print_files(root: Path) -> list[Path]:
    """In moi file .gguf trong root kem dung luong, tra danh sach da in."""
    found: list[Path] = []
    if not root.exists():
        return found
    for path in sorted(root.rglob("*.gguf")):
        try:
            print(f"  {path}  {_mb(path.stat().st_size)}")
            found.append(path)
        except OSError as exc:
            print(f"  [CANH_BAO] Khong doc duoc {path}: {exc}")
    return found


# --------------------------------------------------------------------------- #
# Trich xuat tu notebook (nguon su that duy nhat) - json + ast, khong go tay
# --------------------------------------------------------------------------- #
def notebook_cells(notebook_path: Path) -> list[str]:
    """Doc ipynb, tra ve danh sach source cua tung cell (da noi chuoi)."""
    with notebook_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cells = payload.get("cells", [])
    result: list[str] = []
    for cell in cells:
        source = cell.get("source", "")
        if isinstance(source, list):
            result.append("".join(str(part) for part in source))
        else:
            result.append(str(source))
    return result


def _assigned_literal(cells: Sequence[str], name: str) -> Any | None:
    """Tim `name = <literal>` trong cac cell va literal_eval gia tri."""
    for source in cells:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Cell co %%capture / !pip ... -> khong parse duoc, bo qua.
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        continue
    return None


def _function_source(cells: Sequence[str], name: str) -> str | None:
    """Lay nguyen van source cua mot ham trong notebook (de doi chieu logic)."""
    for source in cells:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                segment = ast.get_source_segment(source, node)
                if segment:
                    return segment
    return None


@dataclass
class FrozenAssets:
    """Du lieu lay tu notebook de chay benchmark frozen."""

    prompts: list[dict[str, str]]
    system_prompt: str
    notebook_clean: Callable[[str], str] | None
    notebook_path: Path


def _notebook_candidates(explicit: str | None) -> list[Path]:
    """Cac noi co the tim thay notebook, uu tien tham so dong lenh."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get("AGON_NOTEBOOK")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    here = Path(__file__).resolve().parent
    candidates.append(here.parent / "reference" / DEFAULT_NOTEBOOK_NAME)
    candidates.append(here / DEFAULT_NOTEBOOK_NAME)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        try:
            candidates.extend(sorted(kaggle_input.rglob(DEFAULT_NOTEBOOK_NAME)))
        except OSError:
            pass
    return candidates


def find_notebook(explicit: str | None = None) -> Path | None:
    for candidate in _notebook_candidates(explicit):
        if candidate.is_file():
            return candidate
    return None


def load_frozen_assets(explicit_notebook: str | None = None) -> FrozenAssets:
    """Doc 4 de frozen + CONCRETE_SOCRATIC_PROMPT tu notebook.

    Khong bao gio tu bia prompt: thieu notebook thi bao loi ro rang.
    """
    notebook_path = find_notebook(explicit_notebook)
    if notebook_path is None:
        raise FileNotFoundError(
            "Khong tim thay notebook nckh-math.ipynb. Dat file vao "
            "/kaggle/input/<dataset>/nckh-math.ipynb hoac truyen --notebook <duong dan>. "
            "Script khong tu bia 4 de frozen."
        )

    cells = notebook_cells(notebook_path)
    prompts = _assigned_literal(cells, "elementary_tests")
    system_prompt = _assigned_literal(cells, "CONCRETE_SOCRATIC_PROMPT")

    if not isinstance(prompts, list) or len(prompts) != 4:
        raise ValueError(
            f"Notebook {notebook_path} khong co list 'elementary_tests' gom 4 de. "
            "Khong the chay frozen benchmark."
        )
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError(
            f"Notebook {notebook_path} khong co 'CONCRETE_SOCRATIC_PROMPT'. "
            "Khong the chay frozen benchmark."
        )

    notebook_clean: Callable[[str], str] | None = None
    source = _function_source(cells, "clean_model_output")
    if source:
        namespace: dict[str, Any] = {"re": re}
        try:
            exec(compile(source, str(notebook_path), "exec"), namespace)
            candidate = namespace.get("clean_model_output")
            if callable(candidate):
                notebook_clean = candidate  # type: ignore[assignment]
        except (SyntaxError, TypeError) as exc:
            print(f"[CANH_BAO] Khong nap duoc clean_model_output tu notebook: {exc}")

    return FrozenAssets(
        prompts=[dict(item) for item in prompts],
        system_prompt=system_prompt,
        notebook_clean=notebook_clean,
        notebook_path=notebook_path,
    )


# --------------------------------------------------------------------------- #
# clean_model_output: ban sao byte-identical cua cell 3 (KHONG import tutor)
# --------------------------------------------------------------------------- #
def clean_model_output(raw_text: str) -> str:
    """Loc the <think> va don cac doan nham tieng Anh - y het notebook."""
    cleaned = re.sub(r" thinking.*?</think>", "", raw_text, flags=re.DOTALL)
    cleaned = re.sub(r"^.*?<\/think>", "", cleaned, flags=re.DOTALL)

    cleaned = (
        cleaned.replace("<|im_end|>", "")
        .replace(" thinking", "")
        .replace("</think>", "")
        .strip()
    )

    english_thought_starters = [
        "okay, let's see",
        "first, let's",
        "the student is",
        "the problem is",
        "let me see",
        "we need to solve",
        "hmm, how do i",
        "starting with",
    ]

    lower_text = cleaned.lower()
    for starter in english_thought_starters:
        if lower_text.startswith(starter):
            vietnamese_match = re.search(
                r"(Chào em|Thầy|Cô|Trong phép|Để tìm|Bài toán|Trước tiên)",
                cleaned,
                re.IGNORECASE,
            )
            if vietnamese_match:
                cleaned = cleaned[vietnamese_match.start() :].strip()
            else:
                cleaned = (
                    "Chào em! Trong bài toán này, em hãy chú ý đến phép tính có chứa x trước nhé. "
                    "Em hãy cho thầy biết phép tính này gồm những thành phần nào?"
                )
            break

    return cleaned.strip()


def _pick_clean_fn(assets: FrozenAssets) -> Callable[[str], str]:
    """Uu tien ban local, doi chieu voi notebook; lech thi dung ban notebook."""
    if assets.notebook_clean is None:
        return clean_model_output
    probes = [
        "Chào em! Thầy hướng dẫn nhé.",
        "Okay, let's see. Chào em! Trong phép trừ có số trừ là 4.",
        "okay, let's see the problem. Nothing Vietnamese here.",
        " thinkingnham lung tung</think>Thầy chào em nhé.<|im_end|>",
        "abc</think>Chào em, ta tính 15 chia 3.",
    ]
    for probe in probes:
        try:
            if assets.notebook_clean(probe) != clean_model_output(probe):
                print(
                    "[CANH_BAO] clean_model_output local lech ban notebook -> "
                    "dung ban notebook lam chuan."
                )
                return assets.notebook_clean
        except Exception as exc:  # noqa: BLE001 - khong de benchmark chet vi probe
            print(f"[CANH_BAO] Doi chieu clean_model_output loi: {exc}")
            return assets.notebook_clean
    return clean_model_output


# --------------------------------------------------------------------------- #
# llama.cpp: tim / clone / build / convert
# --------------------------------------------------------------------------- #
def _llama_cpp_candidates(explicit: str | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    here = Path(__file__).resolve().parent
    candidates.extend(
        [
            Path("/kaggle/working/llama.cpp"),
            Path("/kaggle/tmp/llama.cpp"),
            Path.cwd() / "llama.cpp",
            here.parent / "llama.cpp",
            Path("/opt/llama.cpp"),
            Path("/usr/local/llama.cpp"),
        ]
    )
    return candidates


def find_llama_cpp(explicit: str | None = None) -> Path | None:
    for candidate in _llama_cpp_candidates(explicit):
        if (candidate / "convert_hf_to_gguf.py").is_file():
            return candidate
    return None


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int | None = None) -> None:
    """In nguyen van lenh roi chay, loi thi raise RuntimeError co ngu canh."""
    printable = " ".join(str(part) for part in cmd)
    print(f"[CMD] {printable}")
    try:
        completed = subprocess.run(
            [str(part) for part in cmd],
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Khong tim thay lenh: {cmd[0]} ({exc})") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"Lenh that bai (exit {completed.returncode}): {printable}")


def ensure_llama_cpp(workdir: Path, explicit: str | None = None) -> Path:
    """Tim llama.cpp; khong co thi clone ve workdir."""
    found = find_llama_cpp(explicit)
    if found is not None:
        return found

    target = workdir / "llama.cpp"
    if (target / "convert_hf_to_gguf.py").is_file():
        return target

    if shutil.which("git") is None:
        raise RuntimeError(
            "Khong co llama.cpp va cung khong co git de clone. Hay clone thu cong:\n"
            f"  git clone --depth 1 {LLAMA_CPP_REPO} {target}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(target)])
    if not (target / "convert_hf_to_gguf.py").is_file():
        raise RuntimeError(f"Clone xong nhung thieu convert_hf_to_gguf.py trong {target}")
    return target


def find_quantize_binary(llama_dir: Path, explicit: str | None = None) -> Path | None:
    """Tim binary llama-quantize o nhieu vi tri (env, PATH, build/bin, ...)."""
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate
    env_bin = os.environ.get("LLAMA_QUANTIZE")
    if env_bin and Path(env_bin).is_file():
        return Path(env_bin)

    names = ("llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe")
    search_dirs = [
        llama_dir,
        llama_dir / "build" / "bin",
        llama_dir / "build" / "bin" / "Release",
        llama_dir / "build",
        Path("/usr/local/bin"),
        Path("/kaggle/working/llama.cpp/build/bin"),
    ]
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    for name in names:
        located = shutil.which(name)
        if located:
            return Path(located)
    return None


def ensure_quantize_binary(llama_dir: Path, explicit: str | None = None) -> Path:
    """Tim llama-quantize, neu thieu thi thu build bang cmake; van thieu -> bao loi ro."""
    found = find_quantize_binary(llama_dir, explicit)
    if found is not None:
        return found

    print("[INFO] Chua co llama-quantize, thu build bang cmake (co the mat vai phut)...")
    if shutil.which("cmake") is not None:
        try:
            _run(
                [
                    "cmake",
                    "-S",
                    str(llama_dir),
                    "-B",
                    str(llama_dir / "build"),
                    "-DLLAMA_CURL=OFF",
                    "-DGGML_NATIVE=OFF",
                    "-DCMAKE_BUILD_TYPE=Release",
                ]
            )
            _run(
                [
                    "cmake",
                    "--build",
                    str(llama_dir / "build"),
                    "--config",
                    "Release",
                    "-j",
                    "--target",
                    "llama-quantize",
                ]
            )
        except RuntimeError as exc:
            print(f"[CANH_BAO] Build that bai: {exc}")
    else:
        print("[CANH_BAO] Khong co cmake tren may.")

    found = find_quantize_binary(llama_dir, explicit)
    if found is None:
        raise RuntimeError(
            "Khong tim thay binary llama-quantize. Chay thu cong:\n"
            f"  cmake -S {llama_dir} -B {llama_dir / 'build'} -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release\n"
            f"  cmake --build {llama_dir / 'build'} -j --target llama-quantize\n"
            "hoac dat bien moi truong LLAMA_QUANTIZE=/duong/dan/llama-quantize"
        )
    return found


def convert_and_quantize(
    merged_dir: Path,
    outdir: Path,
    quant: str,
    llama_dir: Path,
    quantize_bin: Path | None,
) -> Path:
    """Chuyen merged HF -> GGUF f16, roi quantize (hoac giu f16 neu quant=f16/bf16)."""
    outdir.mkdir(parents=True, exist_ok=True)
    outtype = "f16" if quant.lower() not in {"f16", "bf16"} else quant.lower()
    f16_path = outdir / f"{OUTPUT_STEM}-F16.gguf"

    _run(
        [
            sys.executable,
            str(llama_dir / "convert_hf_to_gguf.py"),
            str(merged_dir),
            "--outfile",
            str(f16_path),
            "--outtype",
            outtype,
        ]
    )
    if not f16_path.is_file():
        raise RuntimeError(f"Convert xong nhung khong thay {f16_path}")

    if outtype in {"f16", "bf16"}:
        print(f"[INFO] Giu nguyen {outtype.upper()}, bo qua buoc quantize.")
        return f16_path

    if quantize_bin is None:
        raise RuntimeError("Thieu binary llama-quantize de quantize.")
    final_path = outdir / f"{OUTPUT_STEM}-{quant.upper()}.gguf"
    _run([str(quantize_bin), str(f16_path), str(final_path), quant.upper()])
    if not final_path.is_file():
        raise RuntimeError(f"Quantize xong nhung khong thay {final_path}")
    return final_path


# --------------------------------------------------------------------------- #
# Xuat GGUF bang unsloth (import nang chi o day)
# --------------------------------------------------------------------------- #
def load_model(adapter: str) -> tuple[Any, Any]:
    """Nap base + LoRA bang unsloth (chi chay tren Kaggle/GPU)."""
    from unsloth import FastLanguageModel  # noqa: PLC0415 - co y import trong ham

    print(f"[INFO] Nap model tu: {adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    return model, tokenizer


def _quantize_kwarg_name(save_fn: Callable[..., Any]) -> str | None:
    """Doi chieu signature cua save_pretrained_gguf de tim ten tham so quantize."""
    import inspect  # noqa: PLC0415

    try:
        params = inspect.signature(save_fn).parameters
    except (TypeError, ValueError):
        return None
    for name in ("quantization_method", "quant_method", "quant_type", "quantization"):
        if name in params:
            return name
    return None


def export_gguf(
    model: Any,
    tokenizer: Any,
    outdir: Path,
    quant: str,
    *,
    llama_cpp: str | None = None,
    quantize_bin_path: str | None = None,
) -> tuple[list[str], Path | None]:
    """Thu 3 duong: goi truc tiep -> do signature -> merged 16bit + llama.cpp.

    Tra ve (danh sach lenh da chay, thu muc merged neu co).
    """
    commands: list[str] = []
    outdir.mkdir(parents=True, exist_ok=True)

    # Cach 1: goi dung API unsloth nhu notebook/README.
    try:
        print(f"[INFO] save_pretrained_gguf(quantization_method={quant!r})")
        model.save_pretrained_gguf(str(outdir), tokenizer, quantization_method=quant)
        commands.append(
            f"model.save_pretrained_gguf({str(outdir)!r}, tokenizer, quantization_method={quant!r})"
        )
        return commands, None
    except Exception as exc:  # noqa: BLE001 - version unsloth co the raise nhieu kieu
        print(f"[CANH_BAO] Cach 1 that bai: {type(exc).__name__}: {exc}")

    # Cach 2: do signature de tim ten tham so phu hop voi version unsloth.
    save_fn = getattr(model, "save_pretrained_gguf", None)
    if callable(save_fn):
        kwarg = _quantize_kwarg_name(save_fn)
        if kwarg and kwarg != "quantization_method":
            try:
                print(f"[INFO] Thu lai voi tham so {kwarg}={quant!r}")
                save_fn(str(outdir), tokenizer, **{kwarg: quant})
                commands.append(
                    f"model.save_pretrained_gguf({str(outdir)!r}, tokenizer, {kwarg}={quant!r})"
                )
                return commands, None
            except Exception as exc:  # noqa: BLE001
                print(f"[CANH_BAO] Cach 2 that bai: {type(exc).__name__}: {exc}")

    # Cach 3: merged 16bit + llama.cpp (convert + quantize).
    print("[INFO] Chuyen sang fallback: merged_16bit + llama.cpp")
    merged_dir = outdir / "merged_16bit"
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    commands.append(
        f"model.save_pretrained_merged({str(merged_dir)!r}, tokenizer, save_method='merged_16bit')"
    )

    workdir = outdir.parent if outdir.parent != outdir else outdir
    llama_dir = ensure_llama_cpp(workdir, llama_cpp)
    quantize_bin: Path | None = None
    if quant.lower() not in {"f16", "bf16"}:
        quantize_bin = ensure_quantize_binary(llama_dir, quantize_bin_path)
    final_path = convert_and_quantize(merged_dir, outdir, quant, llama_dir, quantize_bin)
    commands.append(f"convert_hf_to_gguf.py -> {final_path}")
    return commands, merged_dir


# --------------------------------------------------------------------------- #
# Benchmark frozen (4 de cua notebook)
# --------------------------------------------------------------------------- #
def ensure_merged_dir(model: Any, tokenizer: Any, outdir: Path) -> Path:
    """Dam bao co thu muc merged_16bit de nap lai bang duong merged (khong con LoRA)."""
    merged_dir = outdir / "merged_16bit"
    if (merged_dir / "config.json").is_file():
        print(f"[INFO] Dung lai merged model co san: {merged_dir}")
        return merged_dir
    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Luu merged_16bit de benchmark: {merged_dir}")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    return merged_dir


def load_merged_for_inference(merged_dir: Path) -> tuple[Any, Any]:
    """Nap model da merge de sinh van ban (unsloth van dung de toi uu toc do)."""
    from unsloth import FastLanguageModel  # noqa: PLC0415

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(merged_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    try:
        FastLanguageModel.for_inference(model)
    except Exception as exc:  # noqa: BLE001 - khong chet neu version khong co
        print(f"[CANH_BAO] for_inference khong chay duoc: {exc}")
    return model, tokenizer


def _device() -> str:
    import torch  # noqa: PLC0415

    return "cuda" if torch.cuda.is_available() else "cpu"


def _free_memory() -> None:
    """Giai phong RAM/VRAM truoc khi nap lai model merged (tranh OOM tren Kaggle)."""
    import gc  # noqa: PLC0415

    gc.collect()
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
            torch_module.cuda.ipc_collect()
    except Exception as exc:  # noqa: BLE001 - khong quan trong bang viec chay tiep
        print(f"[CANH_BAO] Khong don duoc VRAM: {exc}")


def run_frozen_benchmark(
    model: Any,
    tokenizer: Any,
    assets: FrozenAssets,
    out_path: Path,
) -> list[dict[str, Any]]:
    """Chay 4 de frozen theo dung prompt/tham so notebook, ghi JSONL + in bang."""
    import torch  # noqa: PLC0415

    clean_fn = _pick_clean_fn(assets)
    device = _device()
    print(f"[INFO] Chay frozen benchmark tren: {device}")
    print(f"[INFO] Notebook nguon: {assets.notebook_path}")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(assets.prompts, start=1):
        prompt = str(item.get("prompt", ""))
        messages = [
            {"role": "system", "content": assets.system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=SAMPLING["max_new_tokens"],
                temperature=SAMPLING["temperature"],
                top_p=SAMPLING["top_p"],
                repetition_penalty=SAMPLING["repetition_penalty"],
                do_sample=True,
            )
        elapsed = time.perf_counter() - started

        gen_tokens = output_ids[0][input_len:]
        raw = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        cleaned = clean_fn(raw)
        n_tokens = int(gen_tokens.shape[0])
        rows.append(
            {
                "id": f"frozen-{index:02d}",
                "grade": str(item.get("grade", "")),
                "topic": str(item.get("topic", "")),
                "prompt": prompt,
                "raw": raw,
                "cleaned": cleaned,
                "seconds": round(elapsed, 3),
                "gen_tokens": n_tokens,
                "tok_per_s": round(n_tokens / elapsed, 2) if elapsed > 0 else 0.0,
            }
        )
        print(f"[OK] {rows[-1]['id']} | {elapsed:.2f}s | {n_tokens} tokens")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = {
                "id": row["id"],
                "grade": row["grade"],
                "topic": row["topic"],
                "prompt": row["prompt"],
                "raw": row["raw"],
                "cleaned": row["cleaned"],
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"[INFO] Da ghi {len(rows)} dong vao: {out_path}")

    print("-" * 78)
    print(f"{'ID':<11}{'Lop':<9}{'Chu de':<32}{'Giay':>7} {'Token':>7} {'tok/s':>9}")
    print("-" * 78)
    for row in rows:
        topic = row["topic"][:30]
        print(
            f"{row['id']:<11}{row['grade']:<9}{topic:<32}"
            f"{row['seconds']:>7.2f} {row['gen_tokens']:>7} {row['tok_per_s']:>9.2f}"
        )
    print("-" * 78)
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge LoRA -> GGUF tren Kaggle va chay lai 4 de frozen. "
            "Chi dung khi base cua checkpoint la Qwen3-4B-Instruct-2507."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Thu muc checkpoint LoRA tren Kaggle.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Thu muc xuat GGUF.")
    parser.add_argument("--quant", default="q4_k_m", help="Kieu quantize (q4_k_m, q5_k_m, q6_k, f16, ...).")
    parser.add_argument("--skip-frozen", action="store_true", help="Bo qua buoc chay 4 de frozen.")
    parser.add_argument(
        "--notebook",
        default=None,
        help="Duong dan nckh-math.ipynb (mac dinh: /kaggle/input/... hoac reference/).",
    )
    parser.add_argument(
        "--frozen-out",
        default=DEFAULT_FROZEN_OUT,
        help="File JSONL ket qua frozen.",
    )
    parser.add_argument(
        "--llama-cpp",
        default=None,
        help="Thu muc llama.cpp da clone (tranh clone lai).",
    )
    parser.add_argument(
        "--quantize-bin",
        default=None,
        help="Duong dan binary llama-quantize (neu da build san).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_frozen_out(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.parent.is_dir():
        return path
    print(f"[CANH_BAO] Khong ghi duoc {path} (thieu thu muc cha) -> ghi vao thu muc hien tai.")
    return Path.cwd() / path.name


def main(argv: Iterable[str] | None = None) -> int:
    _force_utf8()
    args = parse_args(argv)

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    assets: FrozenAssets | None = None
    if not args.skip_frozen:
        try:
            assets = load_frozen_assets(args.notebook)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print("=" * 78)
            print("[LOI] Khong nap duoc 4 de frozen tu notebook:")
            print(f"  {exc}")
            print("  Goi y: --skip-frozen neu chi muon xuat GGUF.")
            print("=" * 78)
            return 2
        print(f"[INFO] Nap {len(assets.prompts)} de frozen tu {assets.notebook_path}")

    print("=" * 78)
    print("EXPORT GGUF - math-tutor-cli (CNTT-2026-35)")
    print("=" * 78)
    print(f"Adapter : {args.adapter}")
    print(f"Outdir  : {outdir}")
    print(f"Quant   : {args.quant}")

    model, tokenizer = load_model(args.adapter)
    commands, merged_from_export = export_gguf(
        model,
        tokenizer,
        outdir,
        args.quant,
        llama_cpp=args.llama_cpp,
        quantize_bin_path=args.quantize_bin,
    )

    print("-" * 78)
    print("[KET QUA] File GGUF trong outdir:")
    produced = _print_files(outdir)
    if not produced:
        print("  (khong thay file .gguf nao - kiem tra log phia tren)")
        print("[LOI] Khong tao duoc file .gguf. Xem log convert/quantize o tren.")
        return 2

    if assets is not None:
        merged_dir = merged_from_export
        if merged_dir is None or not (merged_dir / "config.json").is_file():
            merged_dir = ensure_merged_dir(model, tokenizer, outdir)
        del model, tokenizer
        _free_memory()
        model_eval, tokenizer_eval = load_merged_for_inference(merged_dir)
        run_frozen_benchmark(model_eval, tokenizer_eval, assets, _resolve_frozen_out(args.frozen_out))

    f16_name = f"{OUTPUT_STEM}-F16.gguf"
    quant_name = f"{OUTPUT_STEM}-{args.quant.upper()}.gguf"
    print("=" * 78)
    print("[LENH DA CHAY / LENH TUONG DUONG]")
    for command in commands:
        print(f"  {command}")
    print("  # Neu can chay lai bang tay:")
    print(
        f"  python convert_hf_to_gguf.py {outdir / 'merged_16bit'} "
        f"--outfile {outdir / f16_name} --outtype f16"
    )
    print(
        f"  llama-quantize {outdir / f16_name} {outdir / quant_name} {args.quant.upper()}"
    )
    print("[XONG] Tai file .gguf ve may va dat vao models/ (models/ nam trong .gitignore).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
