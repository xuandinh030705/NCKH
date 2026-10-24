# Nhật ký tinh chỉnh (autotune)

Nguyên tắc:

- **Không đụng `decode.*`** (temperature 0.25, top_p 0.85, repetition_penalty 1.15,
  max_new_tokens 300/400) — đây là hành vi của notebook, `tests/test_parity.py` canh.
- Chỉ thử tham số nạp model (`n_ctx`, `n_batch`, `flash_attn`, `kv_quant`,
  `n_gpu_layers`) và appendix prompt.
- Chỉ `--apply` khi cấu hình mới thắng **rõ** trên cùng số hội thoại/seed; nếu hoà,
  giữ nguyên bản gốc.
- Mỗi dòng ghi: ngày, máy, cấu hình, N, seed, kết quả, quyết định.

## Bảng cấu hình đang dùng (baseline)

| Ngày | Máy | Cấu hình | N / seed | Kết quả | Quyết định |
|---|---|---|---|---|---|
| 2026-10-24 | RTX 4050 Laptop 6 GB | `n_ctx=2048`, `n_batch=512`, `n_ubatch=256`, `flash_attn=auto`, `kv_quant=auto`, `n_gpu_layers=auto`, appendix bật | smoke 5 ca đóng băng | load 2,9 s; TTFT 0,03–0,26 s; 42–54 tok/s; 10/10 lượt judge ok | giữ nguyên |

## Các vòng autotune

_(chưa chạy vòng nào trên máy thật tại thời điểm ghi; sẽ bổ sung sau baseline N=100)_

Mẫu ghi sau khi chạy:

| Vòng | Cấu hình thử | N / seed | clean_rate | B-label thấp nhất | fallback | tok/s | Quyết định |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

## Lưu ý khi đọc số

- Backend `mock` chỉ để kiểm tra đường ống; **không** dùng để chọn cấu hình.
- So sánh phải cùng `--n`, cùng `--seed`, cùng máy và cùng trạng thái sạc/GPU.
- Nếu chênh lệch nằm trong nhiễu (ví dụ 1–2 hội thoại), coi là hoà → không đổi.
