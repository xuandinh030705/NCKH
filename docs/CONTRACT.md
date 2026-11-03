# HỢP ĐỒNG GIAO DIỆN — math-tutor-cli

Đề tài CNTT-2026-35. Nguồn sự thật duy nhất: `reference/nckh-math.ipynb`.
Mọi prompt / tham số / đề test **phải trích bằng code** (`json` + `ast`) từ notebook, không gõ lại tay.

## 0. Bất biến (không ai được vi phạm, kể cả Lead)

1. `tutor/prompts.py::BASE_PROMPTS` == đúng nguyên văn `SYSTEM_MODES` của cell 6 notebook.
2. `tutor/guard.py::clean_model_output` == logic cell 3 (xoá `<think>`, thẻ điều khiển, cắt mở đầu tiếng Anh).
3. Ba chế độ giữ nguyên tên (khoá dict) và thứ tự 1/2/3.
4. Tham số mặc định: `temperature=0.25, top_p=0.85, repetition_penalty=1.15`, `max_new_tokens=300` (chat, đa lượt) / `400` (chạy lẻ), `enable_thinking=False`.
5. Đổi chế độ ⇒ reset phiên. Lịch sử đa lượt, giữ 6 lượt gần nhất.
6. Không Streamlit/Gradio. Không torch/transformers/unsloth/bitsandbytes ở runtime máy người dùng.
7. Không nới ngưỡng, không sửa judge/guard để "cho đạt".

## 1. Cấu trúc

```
config.yaml            # mọi tham số lớn; KHÔNG hard-code trong code
config.best.yaml       # do scripts/autotune.py sinh (nếu chạy được)
requirements.txt       # runtime máy user (llama-cpp-python, pyyaml, jinja2, sympy, psutil, nvidia-ml-py)
requirements-kaggle.txt# chỉ dùng trên Kaggle (unsloth, torch, ...)
tutor/                 # thư viện
eval/                  # harness đánh giá (KHÔNG import tutor.guard)
scripts/               # entrypoint vận hành
tests/                 # pytest, không phụ thuộc mạng/GPU
data/                  # frozen_regression.jsonl, mock_scenarios
models/                # .gguf (git-ignore)
reference/             # notebook gốc
docs/                  # CONTRACT, TUNING_LOG, PROMPT_CHANGELOG
```

## 2. Chữ ký module (bắt buộc)

### tutor/config.py
```python
class Config:
    @classmethod
    def load(cls, path: str | Path) -> "Config"
    def get(self, dotted: str, default=None)          # "model.n_ctx"
    def section(self, name: str) -> dict
    root: Path                                        # thư mục chứa config.yaml
    def abort(self, msg: str) -> NoReturn             # in lỗi tiếng Việt, exit code 2
```

### tutor/engine.py
```python
@dataclass
class GenResult:
    text: str; prompt_tokens: int; gen_tokens: int
    ttft_s: float; total_s: float; tok_per_s: float

class Engine:
    def __init__(self, cfg: Config, *, model_path: str | None = None)
    def generate(self, messages: list[dict], *, lang: str, mode: str,
                 stream_cb: Callable[[str], None] | None = None,
                 **override) -> GenResult
    def load(self) -> None      # nạp model, tự giảm n_gpu_layers khi OOM (<= cfg load_retries)
    def close(self) -> None
```
- `backend: llama | mock`. Mock trả lời theo kịch bản, dùng cho test harness ([MOCK]).
- Prompt dựng bằng `jinja2` từ `tokenizer.chat_template` trong metadata GGUF; `enable_thinking=False` nếu template nhận. Không gõ tay ChatML.
- `stop = ["<|im_end|>", "<|endoftext|>"]` + eos token; `create_completion(..., stream=True)` để đo TTFT.
- Lỗi nạp: thử lại giảm `n_gpu_layers`, hết lượt mới raise `EngineError` (message tiếng Việt).

### tutor/hardware.py
```python
@dataclass class HwInfo: vram_free_gb; vram_total_gb; cpu_phys; ram_gb; gpu_name
@dataclass class Profile: n_gpu_layers; n_ctx; kv_quant; flash_attn; n_threads
def detect() -> HwInfo
def choose_profile(hw: HwInfo, model_meta: dict, cfg=None) -> Profile
def estimate_kv_gb_per_1k(model_meta: dict, kv_quant: str = "f16") -> float
```

### tutor/prompts.py
```python
BASE_PROMPTS: dict[str, str]      # "1","2","3" -> nguyên văn notebook
MODE_NAMES: dict[str, str]        # "1" -> "1. Gia sư gợi mở từng bước (Socratic cụ thể)"
APPENDIX: dict[tuple[str,str], str]
def build_system(mode: str, lang: str = "vi", cfg=None) -> str   # base + "\n\n" + appendix
def mode_key(name_or_key: str) -> str
```

### tutor/lang.py
```python
def detect(text: str) -> str     # "vi" | "en"; tiếng Việt không dấu -> "vi"
def is_english(text: str) -> bool
```

### tutor/guard.py
```python
def clean_model_output(raw: str) -> str
@dataclass class Violation: rule: str; detail: str
def validate(reply: str, ctx: dict, cfg=None) -> list[Violation]
def fallback(ctx: dict, cfg=None) -> str
```
`ctx` = `{mode, lang, turn_index, problem, numbers, student_text, solution, revealed_answer}`.

### tutor/session.py
```python
@dataclass class TurnResult: reply; raw; violations; retries; gen: GenResult
class TutorSession:
    def __init__(self, engine, mode: str = "1", cfg=None)
    def send(self, text: str) -> TurnResult
    def reset(self) -> None
    def set_mode(self, mode: str) -> None     # reset phiên
    @property history, stats
```

### tutor/mathcheck.py
```python
@dataclass class Step: desc; expr; value
@dataclass class Solution: kind; x; answer; unit; steps; parts
@dataclass class Claim: expr; stated; correct
def parse_problem(text: str) -> Problem | None
def solve(text_or_problem, **kw) -> Solution | None
def arith_claims(text: str) -> list[Claim]
def numbers_in(text: str) -> list[float]
def hidden_hint(problem: str, student_text: str, cfg=None) -> str   # "" nếu tắt
```

### tutor/problemgen.py
```python
@dataclass class Item: id; grade; topic; text; lang; answer; unit; steps; parts; difficulty
def generate(n: int, seed: int, grades=None, topics=None, style="mixed") -> list[Item]
```

### eval/
```python
# student_sim.py
class StudentSim:
    def __init__(self, item, persona: str, seed: int)
    def next_message(self, turn: int, bot_reply: str | None) -> str | None   # None = kết thúc
# judge.py  (rule-based, KHÔNG import tutor.guard)
def judge_turn(...) -> dict; def judge_dialogue(...) -> dict
# runner.py
def run(config: str, n: int, seed: int, mode: str, backend: str) -> Path   # eval/runs/<ts>/
def summarize(run_dir: Path) -> dict
```

### scripts/
| file | mục đích |
|---|---|
| `inspect_checkpoint_kaggle.py` | chạy trên Kaggle: in base_model_name_or_path, r, alpha, target_modules, hidden_size, num_hidden_layers, file/dung lượng |
| `export_gguf_kaggle.py` | merge LoRA → GGUF Q4_K_M (chỉ khi base khớp), chạy lại 4 đề frozen |
| `download_model.py` | `--repo --file` tải GGUF bằng huggingface_hub, kiểm tra dung lượng |
| `autotest.py` | chạy eval thật (hoặc mock), sinh `eval/runs/<ts>/`, in summary |
| `autotune.py` | dò lưới tham số engine + giải mã, ghi `config.best.yaml` (`--apply` mới ghi đè) |
| `benchmark.py` | đo tok/s, TTFT, nạp model; xuất JSON |

## 3. Ranh giới file (subagent chỉ sửa file của mình)

| Agent | File |
|---|---|
| S1 | `scripts/inspect_checkpoint_kaggle.py`, `scripts/export_gguf_kaggle.py`, `scripts/download_model.py`, `models/README.md` |
| S2 | `tutor/engine.py`, `tutor/hardware.py`, `tutor/config.py`, `config.yaml`, `requirements*.txt` |
| S3 | `tutor/prompts.py`, `tutor/lang.py`, `tutor/guard.py`, `tutor/session.py` |
| S4 | `tutor/mathcheck.py`, `tutor/problemgen.py`, `data/frozen_regression.jsonl` |
| S5 | `tutor/cli.py`, `tutor/__main__.py`, `README.md`, `RUN_ON_YOUR_MACHINE.md` |
| S6 | `eval/*`, `scripts/autotest.py`, `scripts/autotune.py`, `scripts/benchmark.py`, `tests/*`, `docs/TUNING_LOG.md` |

Sửa hợp đồng chỉ qua Lead. `tutor/__init__.py`, `tutor/notebook_source.py`, `pytest.ini`, `.gitignore` do Lead giữ.

## 4. Quy ước dữ liệu

- `Item.text_lang`: mỗi item có `text` (đúng `lang`), và `variants: {"vi":..., "vi_nodiac":..., "en":...}`.
- Đáp số luôn hữu hạn, dương; `answer` là `float`, `unit` là chuỗi ("" nếu không có).
- `data/frozen_regression.jsonl`: mỗi dòng `{"id","kind","grade","topic","text","lang","answer","unit","steps","dialogue":[...],"source":"notebook"}`.
- `eval/runs/<timestamp>/results.jsonl`: 1 dòng / hội thoại: `{id, mode, persona, item_id, transcript:[...], violations_before_guard, violations_after_guard, timings, retries, labels}`.
- Nhãn chứng cứ trong báo cáo: `[ĐO THẬT trên máy này]`, `[MOCK]`, `[CHƯA ĐO]`.

## 5. Cổng chất lượng

Đọc từ `eval/thresholds.yaml`. Đổi ngưỡng phải ghi lý do vào `docs/TUNING_LOG.md`.
`pytest` phải xanh, đặc biệt `tests/test_parity.py` và `tests/test_eval_selfcheck.py`.
