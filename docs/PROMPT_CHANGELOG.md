# Nhật ký thay đổi prompt

## Quy tắc bất biến

- Prompt gốc của 3 chế độ lấy **nguyên văn** từ `reference/nckh-math.ipynb`
  (`SYSTEM_MODES`, cell 6). `tests/test_parity.py` thực thi lại cell 6 và so khớp.
- Mọi thay đổi chỉ được **nối thêm appendix** ở cuối prompt, không sửa/xoá câu chữ gốc.
- `build_system(mode, lang)` trả `BASE_PROMPTS[mode] + "\n\n" + appendix`.
  Kiểm tra: `runtime.startswith(base) == True` cho cả 3 chế độ.
- Bật/tắt appendix bằng `config.yaml` → `prompts.append_language_rule`,
  `prompts.append_rule_reminder`, `prompts.few_shot`.

## Appendix hiện có (`tutor/prompts.py`)

| Phần | Nội dung | Lý do | Áp dụng |
|---|---|---|---|
| `_VI_LANGUAGE_RULE` | Bắt buộc 100% tiếng Việt, không suy nghĩ/trả lời bằng tiếng Anh | Notebook có ghi chú chống "english thought" ở cell trước; model 4B đôi khi lẫn tiếng Anh | Cả 3 chế độ, khi `lang=vi` |
| `_EN_LANGUAGE_RULE` | Bắt buộc 100% tiếng Anh | Phục vụ `--lang en` của eval, không ảnh hưởng `vi` | Cả 3 chế độ, khi `lang=en` |
| `_VI_RULE_REMINDER` | Nhắc lại: không dùng `phương trình/hai vế/chuyển vế/đặt biến`; tối đa 1 câu hỏi và phải chứa số từ đề; không khen ở lượt 1 | Ba quy tắc này đã có trong prompt gốc nhưng model 4B hay quên; nhắc lại không thêm luật mới | Chế độ 1 |
| `_MODE2_VI` | Nêu số phần bằng nhau, tổng/hiệu ứng với bao nhiêu phần | Prompt gốc yêu cầu mô tả sơ đồ; nhắc để judge B7 có căn cứ | Chế độ 2 |
| `_MODE3_VI` | Kết thúc bằng đúng một dòng `Đáp số: ...` kèm đơn vị | Prompt gốc yêu cầu "đáp số"; nhắc định dạng để guard/judge đọc được | Chế độ 3 |
| `few_shot` | Các dòng `Ví dụ:` **lấy nguyên văn từ prompt gốc** (không gõ lại) | Tăng độ bám số liệu khi đo thấy cần; mặc định TẮT | Chế độ 1, khi bật |

## Vì sao không sửa prompt gốc

Sửa câu chữ gốc sẽ làm mất khả năng đối chiếu với notebook và có thể thay đổi hành vi
đã đo. Nếu cần cải thiện, thêm appendix rồi đo bằng `scripts/autotest.py`; chỉ giữ
appendix mới nếu số liệu tốt hơn rõ ràng (ghi vào `docs/TUNING_LOG.md`).

## Lịch sử

- 2026-10-24: tách `BASE_PROMPTS` (nguyên văn notebook) khỏi `APPENDIX`; thêm kiểm tra
  `startswith` trong quy trình xác minh. Không đổi nội dung gốc.
- 2026-10-24: sửa bộ chấm phép tính (không liên quan prompt) — xem `docs/EVAL_FIXES.md`.
