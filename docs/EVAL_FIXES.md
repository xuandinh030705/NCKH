# Sửa lỗi bộ chấm (eval) — không đụng nội dung notebook

Mọi thay đổi dưới đây chỉ nằm trong **bộ đọc phép tính của eval/guard**.
Nội dung notebook (`reference/nckh-math.ipynb`), system prompt, tham số decode và
hành vi 3 chế độ **không đổi**. `tests/test_parity.py` vẫn chạy trực tiếp cell 3 và
cell 6 của notebook và so khớp nguyên văn.

## 2026-10-24 — chuỗi nhiều phép tính bị cắt

**Hiện tượng.** Câu trả lời dạng `500 x 10 : 100 = 50` bị chấm là sai (B9), dù
phép tính đúng. Một hội thoại mode 3 trong lần chạy mock bị đánh "không sạch"
oan (11/12 thay vì 12/12).

**Nguyên nhân.** Bộ tách phép tính chỉ nhận đúng một toán tử `a op b = c`, nên
với chuỗi 3 số nó lấy phần đuôi `10 : 100 = 50` → 0,1 ≠ 50. Ngoài ra `x` ASCII
(mô hình hay viết) không nằm trong tập ký tự toán học, còn `×` (U+00D7) thì có.

**Cách sửa.**

- `eval/judge.py::arith_claims`: đổi `x`/`X` **nằm giữa hai chữ số** thành `*`
  trước khi quét; bỏ ký hiệu LaTeX `\( \) \[ \] $`; bỏ dấu `:` của câu dẫn ở đầu
  biểu thức; cho phép số âm ở vế phải. `_arith_ok` chỉ chấm các phép tính đọc
  được giá trị (chuỗi lạ không bị coi là sai).
- `tutor/mathcheck.py::arith_claims`: `_CLAIM_RE` bắt trọn chuỗi
  `số (op số)+ = số`, tính bằng `ast` an toàn (`_eval_arith`), không dùng `eval`
  trần. `x` đứng một mình vẫn là ẩn số nên `x * 3 = 12` không bị coi là phép tính.

**Kiểm chứng.**

- `tests/test_mathcheck.py::test_arith_claims_chuoi_nhieu_phep_tinh`
- `tests/test_eval_selfcheck.py::test_judge_arithmetic_chuoi_nhieu_phep_tinh`
  (gồm cả ca LaTeX `\( 9 + 15 = 24 \)` và ca sai thật `500 x 10 : 100 = 60`)
- `pytest -q`: 97 passed.
- Mock lại 3 chế độ, n=12: cả 3 chế độ đạt 12/12 hội thoại sạch.
- Chạy thật Qwen3-4B trên 5 ca đóng băng của notebook: 10/10 lượt judge ok
  (`eval/runs/real_smoke_20261024T144614Z.txt`).

## Ghi chú về chế độ 3

Prompt chế độ 3 trong notebook chỉ yêu cầu "lời giải hoàn chỉnh theo chuẩn tiểu
học…", **không** cấm từ `phương trình`/`hai vế` (khác chế độ 1). Vì vậy judge chỉ
áp B2 cho chế độ 1–2, đúng như notebook. Mô hình có thể dùng `gọi x` ở chế độ 3;
đó là hành vi của prompt gốc, không phải lỗi parity.
