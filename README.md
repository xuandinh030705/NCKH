# math-tutor-cli — Gia sư Toán tiểu học chạy terminal

Chatbot dạy Toán tiểu học (lớp 2–5) chạy **hoàn toàn trên máy**, dùng `llama-cpp-python`
+ model GGUF. Nội dung sư phạm (3 chế độ, system prompt, tham số sinh, hội thoại mẫu,
4 đề kiểm tra) được **trích trực tiếp** từ notebook gốc `reference/nckh-math.ipynb`
bằng `json` + `ast`, không chép tay.

- Máy đích: GPU 4–6 GB VRAM (đã đo trên RTX 4050 Laptop 6 GB), có fallback CPU-only.
- Không dùng Streamlit/Gradio; không cần `torch`/`transformers` để chạy CLI.
- Có bộ tự đánh giá: sinh đề → mô phỏng học sinh → judge độc lập → cổng chất lượng.

---

## 1. Yêu cầu

| Thành phần | Tối thiểu | Ghi chú |
|---|---|---|
| Python | 3.10–3.12 | Đã kiểm thử với 3.12.14 |
| RAM | 8 GB | 16 GB+ thoải mái |
| VRAM | 0 GB (CPU) hoặc 4 GB+ | 4B Q4_K_M chạy toàn bộ trên GPU 6 GB |
| Ổ đĩa | ~3 GB | Trọng số GGUF chiếm phần lớn |
| HĐH | Windows 10/11, Linux, macOS | Hướng dẫn dưới đây viết cho Windows PowerShell |

## 2. Hướng dẫn nhanh cho máy mới (Quick Start)

### Bước 1: Tải project từ GitHub (nhánh `COMPLETE`)
```bash
git clone -b COMPLETE https://github.com/xuandinh030705/NCKH.git
cd NCKH
```

### Bước 2: Cài đặt và Tải Model (Tự động 1 bước)

**Trên Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```
> Script sẽ tự động: tạo `.venv`, cài đặt thư viện cần thiết, **tự động tải model GGUF** vào thư mục `models/` và chạy kiểm thử tự động.
> *(Nếu máy không có GPU NVIDIA hoặc muốn chạy hoàn toàn bằng CPU, thêm cờ: `-Cpu`)*

**Trên Linux / macOS:**
```bash
chmod +x setup.sh run.sh
./setup.sh
```

---

## 3. Cài đặt thủ công (Nếu không dùng script tự động)

Nếu bạn muốn tự thiết lập từng bước:

```powershell
# 1) Tạo môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Cài llama-cpp-python bản CUDA 12.4 (cho máy có GPU NVIDIA)
python -m pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36-cu124/llama_cpp_python-0.3.36-py3-none-win_amd64.whl
python -m pip install nvidia-cuda-runtime-cu12==12.4.127 nvidia-cublas-cu12==12.4.5.8

# Máy chỉ CPU: chạy thay thế bằng lệnh sau
# python -m pip install llama-cpp-python==0.3.36

# 3) Cài các thư viện còn lại
python -m pip install -r requirements.txt

# 4) BẮT BUỘC: Tải model trước khi chạy
python scripts\download_model.py
```
> Mặc định tải `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (~2,33 GiB) vào thư mục `models/`.
> Bạn cũng có thể tải thủ công file này từ Hugging Face và đặt vào thư mục `models/`.

---

## 4. Chạy chatbot

**Cách 1 (Khuyên dùng):**
```powershell
.\run.ps1
```
*(Script đã tự động kiểm tra model, cấu hình UTF-8 tiếng Việt và kích hoạt môi trường)*

**Cách 2:**
```powershell
$env:PYTHONIOENCODING = "utf-8"
python -m tutor
```


Trong phiên chat:

| Lệnh | Việc |
|---|---|
| gõ đề bài | bắt đầu phiên mới với đề đó |
| `/mode 1` `/mode 2` `/mode 3` | đổi chế độ (reset phiên) |
| `/reset` | làm mới phiên, giữ chế độ |
| `/stats` | số lượt, retry, fallback, tốc độ |
| `/demo` | chạy lại hội thoại mẫu + 4 đề trong notebook |
| `/help` | trợ giúp |
| `/exit` | thoát |

Chạy một câu rồi thoát (tiện test nhanh):

```powershell
python -m tutor --once "Tìm x, biết: x + 18 = 45" --mode 3
```

Các cờ hữu ích:

```powershell
python -m tutor --mock                 # không cần model/GPU, dùng backend giả lập
python -m tutor --mode 2               # vào thẳng chế độ 2
python -m tutor --stream               # in streaming thô (bỏ qua kiểm duyệt)
python -m tutor --model "D:\models\khac.gguf"
python -m tutor --verbose              # in chi tiết lỗi
```

3 chế độ (nguyên văn từ notebook):

1. **Gia sư gợi mở từng bước** — hỏi gợi ý cụ thể, gắn số liệu, không lộ đáp số.
2. **Hướng dẫn trực quan** — sơ đồ đoạn thẳng, nêu số phần bằng nhau.
3. **Bài giải mẫu chuẩn Bộ GD&ĐT** — lời giải đầy đủ, kết thúc bằng dòng `Đáp số:`.

## 5. Chạy kiểm thử

```powershell
python -m pytest -q
```

Kỳ vọng: **97 passed**. Trong đó:

- `tests/test_parity.py` thực thi trực tiếp cell 3 và cell 6 của notebook rồi so khớp
  prompt/ tham số — đây là hàng rào chống "lệch notebook".
- `tests/test_eval_selfcheck.py` kiểm tra judge/cổng chất lượng có bắt đúng lỗi giả lập.

## 6. Tự đánh giá (sinh đề → mô phỏng → chấm)

```powershell
# Không cần GPU: kiểm tra đường ống chạy đúng
python scripts\autotest.py --backend mock --n 12 --mode all

# Đo thật (mỗi chế độ 100 hội thoại; mất khoảng 10–20 phút tuỳ máy)
python scripts\autotest.py --backend llama --n 100 --mode all

# Chỉ một chế độ, ít hội thoại để thử
python scripts\autotest.py --backend llama --n 20 --mode 1

# Thoát mã 1 nếu không đạt cổng (dùng cho CI)
python scripts\autotest.py --backend llama --n 100 --mode 1 --strict
```

Kết quả ghi vào `eval/runs/<thời-gian>_<mode>_<backend>/`:

- `results.jsonl` — từng hội thoại, từng lượt, nhãn B1–B9, vi phạm, thời gian.
- `summary.json` — tỉ lệ sạch, fallback, retry, TTFT, tok/s, trạng thái cổng.
- `REPORT.md` — bản đọc nhanh.

Ngưỡng nằm ở `eval/thresholds.yaml` (không được nới). Backend `mock` **luôn** bị bỏ
qua cổng và được ghi nhãn `[MOCK]`.

## 7. Đo tốc độ

```powershell
python scripts\benchmark.py --backend llama --reps 3
```

In load time, TTFT, tok/s cho từng rep và ghi JSON nếu truyền `--out`.
Số liệu chỉ có giá trị khi chạy trên máy thật; không suy đoán từ mock.

## 8. Tinh chỉnh có kiểm soát

```powershell
# Chỉ thử, không ghi vào config
python scripts\autotune.py --backend llama --n 12 --max-configs 5

# Ghi cấu hình thắng vào config.yaml (chỉ khi thật sự tốt hơn)
python scripts\autotune.py --backend llama --n 12 --max-configs 5 --apply
```

`autotune` **không đụng** phần `decode` (temperature/top_p/repetition_penalty) để giữ
nguyên hành vi notebook; chỉ thử tham số nạp model và appendix prompt. Mọi lần thử
ghi vào `docs/TUNING_LOG.md`, thay đổi prompt ghi vào `docs/PROMPT_CHANGELOG.md`.

## 9. Cấu trúc

```
config.yaml              # mọi tham số (decode khớp notebook)
tutor/
  notebook_source.py     # đọc notebook bằng json+ast
  prompts.py             # prompt gốc + appendix
  engine.py              # llama.cpp / mock, TTFT, tok/s, tự hạ GPU layer khi OOM
  session.py             # đa lượt, retry, fallback, cắt lịch sử
  guard.py               # kiểm duyệt + câu mẫu dự phòng
  mathcheck.py           # đọc đề, giải, kiểm tra phép tính
  problemgen.py          # sinh đề theo dạng
  hardware.py            # dò VRAM/RAM, chọn profile
  cli.py                 # REPL terminal
eval/                    # judge độc lập, runner, mô phỏng học sinh, ngưỡng
scripts/                 # autotest, autotune, benchmark, tải model, xuất GGUF
tests/                   # parity + selfcheck + unit
data/frozen_regression.jsonl   # 4 đề + hội thoại mẫu lấy từ notebook
docs/                    # hợp đồng, nhật ký sửa lỗi
```

## 10. Sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `DLL load failed` khi import llama_cpp | Cài lại bước 3 (CUDA runtime + cuBLAS). Code đã tự thêm đường dẫn DLL trong `engine._bootstrap_dll_paths`. |
| Hết VRAM / `CUDA out of memory` | Giảm `model.n_ctx` (2048 → 1536), hoặc `n_gpu_layers: 0` để chạy CPU. Engine cũng tự giảm layer và thử lại. |
| Tiếng Việt lỗi font trong PowerShell | `chcp 65001` và `$env:PYTHONIOENCODING="utf-8"`. |
| Không tìm thấy model | Chạy `python scripts\download_model.py`, hoặc `--model <đường dẫn>`. |
| Muốn thử UI/logic mà không có GPU | Thêm `--mock`. |
| Đang chạy autotest mà mở thêm CLI | Đừng: hai tiến trình cùng nạp model sẽ tranh VRAM. Chờ tiến trình trước xong. |

## 11. Trạng thái bằng chứng

- `[ĐO THẬT trên máy này]` — RTX 4050 Laptop 6 GB: load 2,9 s; TTFT 0,03–0,26 s;
  42–54 tok/s; 10/10 lượt judge ok trên 5 ca đóng băng của notebook.
- `[MOCK]` — 12/12 hội thoại sạch mỗi chế độ; dùng để kiểm tra đường ống, **không**
  dùng để kết luận chất lượng.
- `[CHƯA ĐO]` — A/B với Qwen3.5-4B và các cấu hình chưa chạy thật.

Chi tiết sửa lỗi bộ chấm: `docs/EVAL_FIXES.md`.
