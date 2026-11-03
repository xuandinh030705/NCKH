# models/ — trọng số GGUF (KHÔNG commit)

Thư mục này chứa file `.gguf` chạy bằng `llama-cpp-python`. Toàn bộ `models/` nằm trong
`.gitignore`: **không commit trọng số**, chỉ commit tài liệu này. Máy mới clone repo phải
tự tải file (mục 2) rồi đặt đúng tên vào đây.

## 1. Chọn file nào?

| Trường hợp | File | Dung lượng | Ghi chú |
|---|---|---|---|
| **Mặc định (khuyến nghị)** | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | ~2,5 GB | Cân bằng chất lượng / tốc độ; chạy tốt với GPU 4–6 GB hoặc CPU khá. Đây là file `config.yaml` trỏ tới. |
| GPU 6 GB, muốn chất lượng cao hơn | `Qwen3-4B-Instruct-2507-Q5_K_M.gguf` | ~2,9 GB | Cần chừa thêm VRAM cho KV cache; giảm `n_ctx` nếu OOM. |
| GPU 6 GB+, ưu tiên chất lượng | `Qwen3-4B-Instruct-2507-Q6_K.gguf` | ~3,3 GB | Thường phải hạ `n_gpu_layers` hoặc `n_ctx` trên card 6 GB. |
| Máy yếu: ≤ 3 GB VRAM hoặc chỉ CPU | `Qwen3-1.7B-Instruct-Q4_K_M.gguf` | ~1,1 GB | Chấp nhận giảm chất lượng sư phạm; dùng khi 4B không nổi. |

Dung lượng là xấp xỉ theo GB thập phân (1 GB = 10^9 byte). File Q4_K_M hiện có trong
thư mục này là 2 381 MB (~2,33 GiB).

Chọn file khác thì sửa `model.path` trong `config.yaml` (hoặc truyền `--model` cho CLI),
**không sửa code**. Tên file không cần khớp tuyệt đối, chỉ cần đường dẫn đúng.

## 2. Tải model

Dùng `scripts/download_model.py` (cần mạng, dùng `huggingface_hub`):

```bash
# Mặc định: Qwen3-4B-Instruct-2507 Q4_K_M
python scripts/download_model.py \
  --repo unsloth/Qwen3-4B-Instruct-2507-GGUF \
  --file Qwen3-4B-Instruct-2507-Q4_K_M.gguf

# Bản nhẹ cho máy yếu
python scripts/download_model.py \
  --repo unsloth/Qwen3-1.7B-GGUF \
  --file Qwen3-1.7B-Instruct-Q4_K_M.gguf
```

Script tải về `models/`, kiểm tra dung lượng tối thiểu và cảnh báo nếu file thiếu/hỏng.
Nếu mạng chập chờn, tải thủ công bằng trình duyệt từ trang Hugging Face của repo rồi
đặt file vào `models/` với đúng tên như trên.

Kiểm tra nhanh sau khi tải:

```bash
python -c "from pathlib import Path; p=Path('models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf'); print(p.exists(), round(p.stat().st_size/2**30, 2), 'GiB')"
```

## 3. Merge LoRA trên Kaggle rồi mang `.gguf` về

Chỉ làm khi **bước 0 xác nhận base khớp** (xem mục 4). Trình tự:

1. Upload/Add Data checkpoint LoRA lên Kaggle (dataset chứa `checkpoint-30`).
2. Soi checkpoint trước, không cần GPU:

   ```bash
   !python scripts/inspect_checkpoint_kaggle.py \
     --path /kaggle/input/models/aegold/math-nckh/transformers/default/1/checkpoint-30
   ```

   Đọc dòng cuối `KET_LUAN: ...` và khối `JSON_SUMMARY: {...}` (dán lại cho nhóm).
3. Nếu `KET_LUAN: BASE_KHOP -> duoc phep merge LoRA` thì merge + xuất GGUF:

   ```bash
   !python scripts/export_gguf_kaggle.py \
     --adapter /kaggle/input/models/aegold/math-nckh/transformers/default/1/checkpoint-30 \
     --outdir /kaggle/working/gguf --quant q4_k_m
   ```

   Script tự chạy lại 4 đề frozen và ghi `/kaggle/working/frozen_outputs.jsonl`
   (bỏ qua bằng `--skip-frozen` nếu chỉ cần file GGUF).
4. Tải file `.gguf` trong `/kaggle/working/gguf/` về máy (nút Download ở tab Output),
   đổi tên thành `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (hoặc giữ tên và sửa `config.yaml`)
   rồi đặt vào `models/`.
5. Chạy `python -m tutor` (hoặc `scripts/benchmark.py`) để xác nhận model nạp được và
   đo tok/s trên máy thật.

Lưu ý: merge LoRA cần base **đúng phiên bản** đã dùng khi fine-tune. Nếu Kaggle không
nạp được base (thiếu quyền/tên repo đổi), dừng lại và báo Lead, không tự thay base khác.

## 4. CẢNH BÁO: base của `checkpoint-30` chưa chắc chắn

Checkpoint hiện có **nhiều khả năng được fine-tune trên Qwen3-8B**, không phải
Qwen3-4B-Instruct-2507. LoRA chỉ là delta so với base: **không thể tái sử dụng LoRA
giữa hai base khác kích thước/nhánh** — merge sẽ lỗi shape hoặc cho ra model rác.

Vì vậy:

- Luôn chạy `scripts/inspect_checkpoint_kaggle.py` trước khi merge.
- `BASE_LA_8B` ⇒ **không merge**: dùng GGUF gốc Qwen3-8B (nếu máy đủ VRAM) hoặc
  Qwen3-4B + prompt, và ghi rõ trong báo cáo là "chưa dùng được LoRA".
- `BASE_KHAC` / `BASE_KHONG_XAC_DINH` ⇒ coi như không merge, báo Lead.
- Không sửa prompt/ngưỡng để bù cho việc thiếu LoRA; mọi thay đổi phải ghi
  `docs/PROMPT_CHANGELOG.md` / `docs/TUNING_LOG.md`.

## 5. Vì sao không commit?

- File GGUF 1–3 GB vượt giới hạn Git/GitHub, làm repo phình và clone chậm.
- Trọng số có thể thay đổi độc lập với code; tải theo mục 2 là đủ tái lập.
- `models/` chỉ giữ `README.md`; mọi file khác bị `.gitignore` chặn.
