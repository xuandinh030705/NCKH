# Chạy trên máy này (Windows, RTX 4050 Laptop 6 GB)

Môi trường đã cài sẵn: `.venv` + `llama-cpp-python` CUDA + model
`models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`. Dưới đây là các lệnh copy–paste.

## 0. Mở PowerShell đúng thư mục

```powershell
cd D:\AGON\math-tutor-cli
$env:PYTHONIOENCODING = "utf-8"
```

Không cần `Activate.ps1` nếu gọi trực tiếp `.venv\Scripts\python.exe`.
Nếu muốn gõ `python` cho gọn:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 1. Kiểm tra nhanh trong 30 giây (không cần GPU)

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m tutor --mock --once "Tìm x, biết: x + 18 = 45" --mode 3
```

Kỳ vọng: `97 passed`; câu trả lời có `Đáp số: 27` và dòng `[MOCK]`-style timing.

## 2. Chat thật

```powershell
.venv\Scripts\python.exe -m tutor
```

Lần đầu nạp model mất ~3–11 s (tuỳ cache), sau đó mỗi lượt ~0,3–1 s.
Gõ đề bài, dùng `/mode 1|2|3`, `/reset`, `/stats`, `/demo`, `/exit`.

Chạy đúng một câu:

```powershell
.venv\Scripts\python.exe -m tutor --once "Một ô tô đi 120 km trong 2 giờ 30 phút. Tính vận tốc." --mode 3
```

## 3. Chạy bộ tự đánh giá

```powershell
# Kiểm tra đường ống, không tốn GPU
.venv\Scripts\python.exe scripts\autotest.py --backend mock --n 12 --mode all

# Đo thật: 100 hội thoại mỗi chế độ (10–20 phút)
.venv\Scripts\python.exe scripts\autotest.py --backend llama --n 100 --mode all
```

Kết quả ở `eval\runs\...`; mở `REPORT.md` để đọc nhanh.

> Chỉ chạy **một** tiến trình dùng model tại một thời điểm. Đang autotest mà mở
> thêm `python -m tutor` sẽ tranh VRAM và có thể OOM.

## 4. Đo tốc độ chính thức

```powershell
.venv\Scripts\python.exe scripts\benchmark.py --backend llama --reps 3
```

Ghi lại số load / TTFT / tok/s vào báo cáo. Đây là số `[ĐO THẬT trên máy này]`;
không dùng số của mock.

## 5. Nếu muốn thử cấu hình khác

```powershell
# thử 5 cấu hình, chưa ghi vào config
.venv\Scripts\python.exe scripts\autotune.py --backend llama --n 12 --max-configs 5

# chỉ ghi khi thắng rõ ràng
.venv\Scripts\python.exe scripts\autotune.py --backend llama --n 12 --max-configs 5 --apply
```

## 6. Sự cố nhanh

| Lỗi | Xử lý |
|---|---|
| `CUDA out of memory` | Sửa `config.yaml`: `n_ctx: 1536`, hoặc `n_gpu_layers: 0` (CPU). |
| `DLL load failed` | Cài lại `nvidia-cuda-runtime-cu12==12.4.127` và `nvidia-cublas-cu12==12.4.5.8`. |
| Chữ Việt thành ô vuông | `chcp 65001` rồi đặt `$env:PYTHONIOENCODING="utf-8"`. |
| Chậm bất thường (<12 tok/s) | Kiểm tra đang cắm sạc, tắt ứng dụng khác dùng GPU; chạy lại benchmark. |

## 7. Tệp nên đọc tiếp

- `README.md` — hướng dẫn đầy đủ, cài trên máy khác.
- `docs/CONTRACT.md` — hợp đồng giao diện và quy ước.
- `docs/EVAL_FIXES.md` — các lỗi bộ chấm đã sửa và cách kiểm chứng.
- `models/README.md` — chọn/tải model, merge LoRA trên Kaggle.
