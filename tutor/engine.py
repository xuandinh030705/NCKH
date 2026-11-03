"""Engine suy luận: llama.cpp (GGUF) hoặc mock (cho test harness).

- Dựng prompt bằng jinja2 từ ``tokenizer.chat_template`` trong metadata GGUF
  (không gõ tay ChatML; Qwen3-4B-2507 và Qwen3.5 khác nhau).
- Đo TTFT, tok/s; tự giảm ``n_gpu_layers`` khi OOM.
- ``backend: mock`` trả lời theo kịch bản, không cần model/GPU.
"""
from __future__ import annotations

import glob
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .hardware import Profile, choose_profile, detect, read_gguf_meta


class EngineError(RuntimeError):
    """Lỗi nạp/sinh có thông điệp tiếng Việt cho người dùng."""


@dataclass
class GenResult:
    text: str
    prompt_tokens: int = 0
    gen_tokens: int = 0
    ttft_s: float = 0.0
    total_s: float = 0.0
    tok_per_s: float = 0.0


# --------------------------------------------------------------- nạp DLL Windows
def _bootstrap_dll_paths() -> None:
    """Thêm DLL CUDA (pip nvidia-*) và llama_cpp/lib vào đường dẫn tìm kiếm."""
    if os.name != "nt":
        return
    site = Path(sys.prefix) / "Lib" / "site-packages"
    candidates: list[Path] = [site / "llama_cpp" / "lib"]
    candidates += [Path(p) for p in glob.glob(str(site / "nvidia" / "*" / "bin"))]
    for d in candidates:
        if not d.is_dir():
            continue
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(d))  # type: ignore[attr-defined]
        except Exception:
            pass


_LLAMA_LOG_CB: Any = None


def _silence_llama_logs(llama_cpp: Any) -> None:
    """Tắt log INFO/DEBUG của llama.cpp (giữ màn hình terminal sạch)."""
    global _LLAMA_LOG_CB
    try:
        import ctypes

        if not hasattr(llama_cpp, "llama_log_set"):
            return
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
        # Giữ tham chiếu toàn cục: callback bị GC sẽ làm llama.cpp truy cập vùng nhớ đã giải phóng
        _LLAMA_LOG_CB = callback_type(lambda level, text, user: None)
        llama_cpp.llama_log_set(_LLAMA_LOG_CB, None)
    except Exception:
        pass


# ------------------------------------------------------------------ mock backend
class MockBackend:
    """Trả lời theo kịch bản để test harness/judge, không cần model."""

    def __init__(self, behavior: str = "auto") -> None:
        self.behavior = (behavior or "auto").lower()

    # -- kịch bản lỗi cố ý (dùng cho test guard/judge)
    _BAD = {
        "violating": "Thầy khen em làm đúng rồi! Em hãy làm cẩn thận nhé?",
        "vague": "Em làm gì tiếp theo? Em có thể làm như thế nào?",
        "english": "Okay, let's solve the equation: x = 5. The student is right.",
        "think": "thinking about it...</think>Chào em! Trong bài x * 3 - 4 = 11, nhóm (x * 3) đóng vai trò là số gì?",
        "leak": "Đáp số là 5 nhé! Em chỉ cần ghi 5.",
    }

    def generate(self, messages: list[dict], *, mode: str = "1", lang: str = "vi", **_: Any) -> GenResult:
        if self.behavior in self._BAD:
            text = self._BAD[self.behavior]
        else:
            text = self._good(messages, mode)
        n = max(1, len(text) // 4)
        return GenResult(text=text, prompt_tokens=len(str(messages)) // 4, gen_tokens=n,
                         ttft_s=0.01, total_s=0.05, tok_per_s=float(n) / 0.05)

    # -- kịch bản "tốt"
    def _good(self, messages: list[dict], mode: str) -> str:
        from .mathcheck import arith_claims, fmt_num, solve

        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        problem = user_msgs[0] if user_msgs else ""
        last = user_msgs[-1] if user_msgs else ""
        sol = solve(problem)

        if mode == "3" and sol is not None:
            lines = ["Bài giải:"]
            for st in sol.steps:
                lines.append(f"- {st.desc}")
                lines.append(f"  {st.expr} = {fmt_num(st.value)}")
            ans = sol.answer
            txt = " và ".join(fmt_num(v) for v in ans) if isinstance(ans, tuple) else fmt_num(ans)
            if sol.unit:
                txt = f"{txt} {sol.unit}"
            lines.append(f"Đáp số: {txt}")
            return "\n".join(lines)

        if mode == "2" and sol is not None and sol.parts:
            return (f"Ta vẽ sơ đồ đoạn thẳng: số lớn {fmt_num(sol.parts['big'])} phần, "
                    f"số bé {fmt_num(sol.parts['small'])} phần, tổng {fmt_num(sol.parts['total_parts'])} phần. "
                    f"Em hãy tính một phần bằng bao nhiêu nhé?")

        if sol is not None and sol.steps:
            claims = arith_claims(last)
            if claims and claims[-1].ok and len(user_msgs) > 1:
                step = sol.steps[min(len(user_msgs) - 2, len(sol.steps) - 1)]
                # Không nêu kết quả của bước kế tiếp: đó là đáp số của bước,
                # học sinh phải tự tính (judge B5 kiểm tra điều này).
                return (f"Đúng rồi em! Em hãy tính tiếp {step.expr} giúp thầy nhé?")
            if claims and not claims[-1].ok and len(user_msgs) > 1:
                return (f"Chưa đúng em nhé. Em hãy tính lại {claims[-1].expr.split('=')[0].strip()} "
                        f"một lần nữa nhé?")
            first = sol.steps[0]
            return (f"Chào em! Với bài này, trước tiên ta xét phép tính {first.expr}. "
                    f"Em hãy tính giúp thầy {first.expr} bằng bao nhiêu nhé?")

        return ("Chào em! Em hãy đọc kỹ đề và cho thầy biết bài toán cho những số nào, "
                "cần tìm gì nhé?")


# ----------------------------------------------------------------------- engine
class Engine:
    def __init__(self, cfg: Config, *, model_path: str | None = None) -> None:
        self.cfg = cfg
        self.model_path_override = model_path
        self.backend = str(cfg.get("engine.backend", "llama")).lower()
        self.llm: Any = None
        self.profile: Profile | None = None
        self.meta: dict[str, Any] = {}
        self.load_s: float = 0.0
        self._template: Any = None
        self._mock = MockBackend(str(cfg.get("engine.mock_behavior", "auto")))
        self.last_result: GenResult | None = None

    # ------------------------------------------------------------------ nạp
    def load(self) -> None:
        if self.backend == "mock":
            return
        if self.llm is not None:
            return
        _bootstrap_dll_paths()
        try:
            import llama_cpp
            from llama_cpp import Llama  # noqa: F401
        except Exception as exc:  # pragma: no cover - phụ thuộc máy
            raise EngineError(
                "Không nạp được llama-cpp-python. Cài lại bằng:\n"
                "  pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cu124\n"
                f"Chi tiết: {exc}"
            ) from exc
        if not bool(self.cfg.get("engine.verbose", False)):
            _silence_llama_logs(llama_cpp)

        path = self.cfg.model_path(self.model_path_override)
        if not path.exists():
            raise EngineError(
                f"Không thấy file model: {path}\n"
                "Tải bằng: python scripts/download_model.py\n"
                "hoặc đặt file .gguf vào thư mục models/ (xem models/README.md)."
            )

        self.meta = read_gguf_meta(path)
        hw = detect()
        prof = choose_profile(hw, self.meta, self.cfg)
        retries = int(self.cfg.get("model.load_retries", 3))
        layers = prof.n_gpu_layers
        last_exc: Exception | None = None
        t0 = time.time()
        for attempt in range(max(1, retries)):
            try:
                self.llm = Llama(
                    model_path=str(path),
                    n_ctx=prof.n_ctx,
                    n_batch=int(self.cfg.get("model.n_batch", 512)),
                    n_ubatch=int(self.cfg.get("model.n_ubatch", 256)),
                    n_gpu_layers=layers,
                    n_threads=prof.n_threads,
                    flash_attn=prof.flash_attn,
                    type_k=_ggml_type(prof.kv_quant),
                    type_v=_ggml_type(prof.kv_quant),
                    use_mmap=bool(self.cfg.get("model.use_mmap", True)),
                    use_mlock=bool(self.cfg.get("model.use_mlock", False)),
                    seed=int(self.cfg.get("model.seed", 42)),
                    verbose=bool(self.cfg.get("engine.verbose", False)),
                )
                self.profile = Profile(layers, prof.n_ctx, prof.kv_quant, prof.flash_attn, prof.n_threads)
                break
            except Exception as exc:  # OOM hoặc lỗi nạp
                last_exc = exc
                msg = str(exc).lower()
                if layers <= 0 or not any(k in msg for k in ("memory", "alloc", "cuda", "vram")):
                    break
                layers = max(0, layers // 2) if layers > 0 else 0
                if layers == prof.n_gpu_layers:  # tránh lặp vô hạn
                    layers = 0
        if self.llm is None:
            raise EngineError(
                "Không nạp được model (có thể hết VRAM). Thử:\n"
                "  - đóng ứng dụng dùng GPU,\n"
                "  - sửa config.yaml: model.n_gpu_layers: 0 (chạy CPU),\n"
                f"  - hoặc dùng model nhỏ hơn (Qwen3-1.7B).\nChi tiết: {last_exc}"
            )
        self.load_s = time.time() - t0
        self._template = _load_template(self.llm)
        if bool(self.cfg.get("engine.warmup", True)):
            try:
                self.generate([{"role": "user", "content": "Chào thầy"}], lang="vi", mode="1",
                              max_new_tokens=1)
            except Exception:
                pass

    def close(self) -> None:
        self.llm = None

    # --------------------------------------------------------------- sinh chữ
    def generate(
        self,
        messages: list[dict],
        *,
        lang: str = "vi",
        mode: str = "1",
        stream_cb: Callable[[str], None] | None = None,
        **override: Any,
    ) -> GenResult:
        if self.backend == "mock":
            res = self._mock.generate(messages, mode=mode, lang=lang, **override)
            self.last_result = res
            if stream_cb:
                stream_cb(res.text)
            return res

        self.load()
        assert self.llm is not None
        prompt = self._render(messages)
        decode = self.cfg.section("decode")
        max_tokens = int(override.get("max_new_tokens", decode.get("max_new_tokens_chat", 300)))
        temperature = float(override.get("temperature", decode.get("temperature", 0.25)))
        top_p = float(override.get("top_p", decode.get("top_p", 0.85)))
        rep = float(override.get("repetition_penalty", decode.get("repetition_penalty", 1.15)))
        stop = list(override.get("stop") or decode.get("stop") or ["<|im_end|>", "<|endoftext|>"])

        prompt_tokens = len(self.llm.tokenize(prompt.encode("utf-8"), add_bos=False))
        t0 = time.time()
        ttft: float | None = None
        chunks: list[str] = []
        try:
            for out in self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=rep,
                stop=stop,
                stream=True,
            ):
                piece = out["choices"][0].get("text", "")
                if not piece:
                    continue
                if ttft is None:
                    ttft = time.time() - t0
                chunks.append(piece)
                if stream_cb is not None:
                    stream_cb(piece)
        except Exception as exc:
            msg = str(exc).lower()
            if any(k in msg for k in ("memory", "alloc", "cuda")):
                raise EngineError(
                    "Hết bộ nhớ khi sinh câu trả lời. Giảm model.n_ctx hoặc đặt "
                    "model.n_gpu_layers: 0 trong config.yaml rồi chạy lại."
                ) from exc
            raise
        total = time.time() - t0
        text = "".join(chunks)
        gen_tokens = len(self.llm.tokenize(text.encode("utf-8"), add_bos=False)) if text else 0
        res = GenResult(
            text=text,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            ttft_s=ttft or total,
            total_s=total,
            tok_per_s=(gen_tokens / total) if total > 0 else 0.0,
        )
        self.last_result = res
        return res

    # -------------------------------------------------------------- template
    def _render(self, messages: list[dict]) -> str:
        if self._template is None:
            self._template = _load_template(self.llm)
        try:
            return self._template.render(
                messages=messages, add_generation_prompt=True, enable_thinking=False,
                bos_token="", eos_token="",
            )
        except Exception:
            # template không hỗ trợ enable_thinking (Qwen3-4B-Instruct-2507)
            return self._template.render(
                messages=messages, add_generation_prompt=True, bos_token="", eos_token="",
            )


def _ggml_type(kv_quant: str) -> int:
    """Mã kiểu GGML: f16=1, q8_0=8. Không hỗ trợ thì quay về f16."""
    return 8 if str(kv_quant).lower().startswith("q8") else 1


def _load_template(llm: Any) -> Any:
    from jinja2 import Template

    meta = getattr(llm, "metadata", {}) or {}
    tpl = meta.get("tokenizer.chat_template")
    if not tpl:
        raise EngineError(
            "GGUF thiếu tokenizer.chat_template — không dựng được prompt đúng chuẩn Qwen3. "
            "Hãy dùng file GGUF chính thức của unsloth/Qwen3-4B-Instruct-2507-GGUF."
        )
    return Template(tpl)


__all__ = ["Engine", "EngineError", "GenResult", "MockBackend"]
