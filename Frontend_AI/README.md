# 🎓 Frontend Gia sư Toán Tiểu học — AI Socratic

Giao diện Web tương tác trực quan cho mô hình **Gia sư Toán Tiểu học AI (phương pháp Socratic)**, được xây dựng bằng **Gradio Blocks** theo phong cách phẳng, tối giản (Sharp Minimalist - 0 Border Radius), hỗ trợ Light/Dark mode và quản lý lịch sử hội thoại liên tục.

---

## ✨ Tính năng nổi bật

1. **Giao diện tối giản (Sharp / Zero Border Radius):**
   - Loại bỏ 100% bo góc (`border-radius: 0px`), đường nét sắc sảo, hiện đại tương tự ChatGPT/Claude.
   - Hỗ trợ chuyển đổi mượt mà giữa **Giao diện sáng (Light)** và **Giao diện tối (Dark)**, tự động lưu lựa chọn qua `localStorage`.
   - Bỏ các bảng thông số kỹ thuật rườm rà, tập trung 100% vào trải nghiệm trò chuyện sư phạm giữa gia sư và học sinh.

2. **Quản lý lịch sử hội thoại (Chat History):**
   - Tự động lưu các phiên trò chuyện vào `chat_history.json`.
   - Cho phép học sinh chọn xem lại hoặc tiếp tục bất kỳ buổi học nào trong danh sách.
   - Hỗ trợ xoá lịch sử và bắt đầu cuộc trò chuyện mới chỉ với 1 click.

3. **Tự động nhận diện khối lớp (Lớp 1 → 5):**
   - Tự động phân loại đề bài toán thuộc phạm vi chương trình Lớp 1, 2, 3, 4 hay 5 để đưa ra ngôn từ và phương pháp sư phạm phù hợp tâm lý lứa tuổi.
   - Giới hạn nghiêm ngặt trong phạm vi Toán tiểu học (từ chối giải theo phương trình cấp 2, không đặt ẩn $x, y$, không dùng số âm, không dùng chuyển vế đổi dấu).

4. **3 Chế độ sư phạm linh hoạt:**
   - **Chế độ 1 — Gợi mở từng bước (Socratic):** Đặt câu hỏi định hướng, không tiết lộ ngay đáp án để kích thích tư duy của học sinh.
   - **Chế độ 2 — Hướng dẫn phương pháp (Sơ đồ đoạn thẳng):** Hướng dẫn phân tích tỉ số, vẽ sơ đồ đoạn thẳng thực tế (không giải ra số thập phân vô lý như 4.8 viên bi).
   - **Chế độ 3 — Lời giải chi tiết (Mẫu):** Trình bày hoàn chỉnh từng bước giải toán tiểu học kèm câu lời giải, phép tính và dòng `Đáp số: [kết quả] [đơn vị]`.

5. **Bộ kiểm tra số học học sinh:**
   - Tự động phát hiện các phép tính của học sinh bằng thuật toán Python để ngăn AI khen nhầm kết quả sai (ví dụ học sinh tính $45 - 18 = 30$, AI sẽ lập tức nhắc học sinh tính lại cẩn thận).

---

## 📂 Cấu trúc thư mục `Frontend_AI`

```text
Frontend_AI/
├── app.py                 # Mã nguồn giao diện chính (Gradio Blocks, CSS, Event Handlers)
├── adapter.py             # Cầu nối (Adapter) giữa Giao diện và Backend Engine (Llama.cpp / Mock)
├── history_manager.py     # Module lưu trữ và quản lý lịch sử trò chuyện (JSON)
├── __init__.py            # Khởi tạo package Python
├── run_ui.bat             # File khởi chạy nhanh trên Windows (Click đúp chuột là chạy)
├── run_ui.ps1             # Script khởi chạy qua PowerShell (hỗ trợ tham số -Mock, -Port)
├── requirements.txt       # Danh sách thư viện Python cần thiết
└── README.md              # Tài liệu hướng dẫn sử dụng
```

---

## 🚀 Hướng dẫn khởi chạy

### Cách 1: Khởi chạy nhanh bằng file `.bat` (Windows)
Click đúp chuột vào file:
```cmd
run_ui.bat
```
Ứng dụng sẽ tự động dò tìm môi trường Python/`.venv` và mở trình duyệt tại: `http://localhost:7860`.

---

### Cách 2: Khởi chạy bằng PowerShell
Mở PowerShell trong thư mục `Frontend_AI` hoặc thư mục dự án:

- **Chạy cùng mô hình AI thực tế (GPU/CUDA):**
  ```powershell
  .\run_ui.ps1
  ```

- **Chạy chế độ thử nghiệm nhanh (Mock Mode - Không cần tải model):**
  ```powershell
  .\run_ui.ps1 -Mock
  ```

- **Chạy trên cổng khác (Ví dụ 8080):**
  ```powershell
  .\run_ui.ps1 -Port 8080
  ```

---

### Cách 3: Chạy trực tiếp bằng lệnh Python
```bash
python app.py
```
Hoặc chạy chế độ Mock:
```bash
python app.py --mock
```

---

## ⚙️ Yêu cầu môi trường
- Python >= 3.10
- Cài đặt thư viện:
  ```bash
  pip install -r requirements.txt
  ```
- Nếu chạy backend thực tế với model GGUF: Cần `llama-cpp-python` (hỗ trợ CUDA nếu có card rời NVIDIA).

