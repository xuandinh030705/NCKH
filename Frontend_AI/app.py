"""app.py – Giao diện tối giản (Zero Border Radius, Lịch sử trò chuyện, Tự nhận diện Lớp 1-5).

- Bỏ hoàn toàn bảng chỉ số NCKH (giao diện gọn nhẹ như ChatGPT/Claude).
- Sửa triệt để lỗi hội thoại đa lượt ('list' has no attribute 'strip').
- Tự động nhận diện bài toán khối Lớp 1 đến 5 và tối ưu trong phạm vi tiểu học.
- 100% Zero Border Radius (góc vuông phẳng, sắc nét).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURR_DIR.parent
for p in [str(CURR_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import gradio as gr

try:
    from adapter import TutorAdapter, MODE_CHOICES, _extract_mode_key
    from history_manager import HistoryManager, extract_clean_text
except ImportError:
    from Frontend_AI.adapter import TutorAdapter, MODE_CHOICES, _extract_mode_key
    from Frontend_AI.history_manager import HistoryManager, extract_clean_text

# =========================================================================
# CSS TỐI GIẢN - KHÔNG BORDER RADIUS (SHARP / ZERO BORDER RADIUS)
# =========================================================================
SHARP_MINIMAL_CSS = """
/* ===== LOẠI BỎ TOÀN BỘ BORDER-RADIUS (0px) ===== */
*, *::before, *::after,
button, input, textarea, select, a, div, span, p, label, form,
.gradio-container, .gradio-container *,
.message, .bubble-wrap, .message-bubble-border,
.user, .bot, .assistant,
.card, .panel, .box,
#chatbot-window, #sidebar-col, #main-chat-col,
#input-box-container, #chat-input, #btn-send, #btn-new-chat,
.gr-button, .gr-box, .gr-input, .gr-panel, .gr-form, .gr-block,
.wrap, .prose {
    border-radius: 0px !important;
    -webkit-border-radius: 0px !important;
    -moz-border-radius: 0px !important;
}

/* ===== BIẾN MÀU THEME (LIGHT / DARK) ===== */
:root {
    --bg-page: #ffffff;
    --bg-surface: #ffffff;
    --bg-sidebar: #f8fafc;
    --bg-input: #ffffff;
    --border-color: #e2e8f0;
    --border-dark: #0f172a;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #94a3b8;
    --accent-color: #0f172a;
}

.dark, body.dark, html.dark, .dark .gradio-container {
    --bg-page: #09090b;
    --bg-surface: #18181b;
    --bg-sidebar: #121215;
    --bg-input: #18181b;
    --border-color: #27272a;
    --border-dark: #3f3f46;
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --accent-color: #f4f4f5;
}

body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    background-color: var(--bg-page) !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    padding: 0 !important;
}

footer { display: none !important; }

/* ===== SIDEBAR TRÁI ===== */
#sidebar-col {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-color) !important;
    padding: 16px 14px !important;
    min-height: 100vh !important;
}

#brand-header {
    padding-bottom: 12px;
    border-bottom: 2px solid var(--border-dark);
    margin-bottom: 14px;
}
#brand-header h2 {
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin: 0 !important;
    color: var(--text-primary) !important;
}
#brand-header p {
    font-size: 0.78rem !important;
    color: var(--text-secondary) !important;
    margin: 3px 0 0 0 !important;
}

/* NÚT CUỘC TRÒ CHUYỆN MỚI */
#btn-new-chat {
    background-color: var(--text-primary) !important;
    color: var(--bg-page) !important;
    border: 1px solid var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 10px !important;
    width: 100% !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
}
#btn-new-chat:hover { opacity: 0.9; }

.sidebar-section-title {
    font-size: 0.72rem !important;
    font-weight: 800 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin: 16px 0 6px 0 !important;
}

/* KHU VỰC CHAT CHÍNH */
#main-chat-col {
    padding: 12px 24px 20px 24px !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* THANH TRẠNG THÁI TRÊN CÙNG */
#top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0 12px 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 12px;
}
.top-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--text-primary);
}
.top-badge {
    font-size: 0.75rem;
    font-weight: 600;
    background-color: var(--bg-sidebar);
    color: var(--text-secondary);
    padding: 3px 8px;
    border: 1px solid var(--border-color);
}

/* CỬA SỔ CHATBOT */
#chatbot-window {
    background: transparent !important;
    border: 1px solid var(--border-color) !important;
    min-height: 560px !important;
}

/* KHUNG NHẬP LIỆU PHẲNG */
#input-box-container {
    background-color: var(--bg-input) !important;
    border: 1.5px solid var(--border-dark) !important;
    padding: 4px 8px 4px 12px !important;
}

#chat-input textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 0.95rem !important;
    color: var(--text-primary) !important;
    padding: 8px 0 !important;
    resize: none !important;
}
#chat-input textarea:focus {
    outline: none !important;
    box-shadow: none !important;
}

#btn-send {
    background: var(--text-primary) !important;
    color: var(--bg-page) !important;
    border: none !important;
    width: 38px !important;
    height: 38px !important;
    min-width: 38px !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
#btn-send:hover { opacity: 0.85; }

/* NÚT THEME TOGGLE */
#btn-theme-toggle {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-color) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 8px !important;
    margin-top: 14px !important;
    width: 100% !important;
    cursor: pointer !important;
}
#btn-theme-toggle:hover {
    color: var(--text-primary) !important;
    border-color: var(--text-primary) !important;
}

.disclaimer-text {
    text-align: center;
    font-size: 0.74rem;
    color: var(--text-muted);
    margin-top: 8px;
}
"""

THEME_TOGGLE_JS = """
() => {
    const isDark = document.documentElement.classList.toggle('dark');
    document.body.classList.toggle('dark', isDark);
    document.querySelectorAll('.gradio-container').forEach(el => el.classList.toggle('dark', isDark));
    try {
        localStorage.setItem('math_tutor_theme', isDark ? 'dark' : 'light');
    } catch(e) {}
    return isDark ? '☀️ Giao diện sáng' : '🌙 Giao diện tối';
}
"""

INIT_THEME_JS = """
() => {
    try {
        const saved = localStorage.getItem('math_tutor_theme');
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (saved === 'dark' || (!saved && prefersDark)) {
            document.documentElement.classList.add('dark');
            document.body.classList.add('dark');
            document.querySelectorAll('.gradio-container').forEach(el => el.classList.add('dark'));
            return '☀️ Giao diện sáng';
        }
    } catch(e) {}
    return '🌙 Giao diện tối';
}
"""


# =========================================================================
# XÂY DỰNG GIAO DIỆN CHÍNH
# =========================================================================
def build_ui(adapter: TutorAdapter) -> gr.Blocks:
    hw = adapter.hw_info()
    backend_name = "Llama.cpp (GPU)" if adapter.engine.backend == "llama" else "Mock Engine"

    with gr.Blocks(title="Gia sư Toán Tiểu học — AI Socratic") as demo:

        current_session_id = gr.State("")

        with gr.Row(equal_height=False):

            # -------------------------------------------------------------
            # THANH BÊN (SIDEBAR) TRÁI (260px)
            # -------------------------------------------------------------
            with gr.Column(scale=3, min_width=250, elem_id="sidebar-col"):

                # Thương hiệu
                gr.HTML(
                    """
                    <div id="brand-header">
                        <h2>Gia sư Toán AI</h2>
                        <p>Toán Tiểu học Lớp 1 → 5 • Phương pháp Socratic</p>
                    </div>
                    """
                )

                # Nút Cuộc trò chuyện mới
                btn_new_chat = gr.Button(
                    "➕ Cuộc trò chuyện mới",
                    elem_id="btn-new-chat",
                )

                # Chế độ sư phạm
                gr.HTML('<div class="sidebar-section-title">Phương pháp sư phạm</div>')
                mode_dropdown = gr.Dropdown(
                    choices=MODE_CHOICES,
                    value=MODE_CHOICES[0],
                    show_label=False,
                    interactive=True,
                )

                # Lịch sử trò chuyện
                gr.HTML('<div class="sidebar-section-title">Lịch sử trò chuyện</div>')
                history_dropdown = gr.Dropdown(
                    choices=HistoryManager.get_choices(),
                    value=None,
                    show_label=False,
                    interactive=True,
                    allow_custom_value=True,
                )

                btn_clear_history = gr.Button(
                    "🗑️ Xoá toàn bộ lịch sử",
                    size="sm",
                    variant="secondary",
                )

                # Accordion: Bảng nháp
                with gr.Accordion("📝 Bảng nháp tính toán", open=False):
                    scratchpad = gr.Textbox(
                        placeholder="Nháp nhanh phép tính ở đây...\nVí dụ: 45 - 18 = 27",
                        lines=3,
                        show_label=False,
                    )

                # Nút đổi Theme
                btn_theme = gr.Button(
                    "🌙 Giao diện tối",
                    elem_id="btn-theme-toggle",
                )

            # -------------------------------------------------------------
            # KHU VỰC CHAT CHÍNH Ở GIỮA
            # -------------------------------------------------------------
            with gr.Column(scale=9, min_width=450, elem_id="main-chat-col"):

                # Thanh Top-bar nhỏ gọn
                gr.HTML(
                    f"""
                    <div id="top-bar">
                        <span class="top-title">💡 Trợ lý Gia sư Toán Tiểu học (Lớp 1 — 5)</span>
                        <span class="top-badge">{backend_name} • {hw['gpu']}</span>
                    </div>
                    """
                )

                # Khung Chatbot chính
                chatbot = gr.Chatbot(
                    value=[],
                    elem_id="chatbot-window",
                    height=580,
                    show_label=False,
                    placeholder=(
                        "### 📚 Chào em! Thầy là Gia sư Toán Tiểu học.\n"
                        "- Em hãy nhập bất kỳ bài toán nào từ **Lớp 1 đến Lớp 5** vào ô bên dưới.\n"
                        "- Thầy sẽ **tự động nhận diện khối lớp và dạng toán** để hướng dẫn em theo phương pháp gợi mở từng bước.\n"
                        "- *Lưu ý: Thầy chỉ chuyên hỗ trợ Toán Tiểu học (Lớp 1–5), không giải toán cấp 2/cấp 3 nhé!*"
                    ),
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$",   "display": False},
                    ],
                )

                # Hộp nhập liệu góc vuông phẳng
                with gr.Row(elem_id="input-box-container"):
                    user_input = gr.Textbox(
                        placeholder="Nhập bài toán tiểu học hoặc câu trả lời tiếp theo của em...",
                        show_label=False,
                        lines=1,
                        max_lines=5,
                        elem_id="chat-input",
                        scale=10,
                        autofocus=True,
                    )
                    btn_send = gr.Button(
                        "↑",
                        elem_id="btn-send",
                        scale=1,
                    )

                gr.HTML(
                    '<p class="disclaimer-text">'
                    'Gia sư AI nhận diện bài toán từ Lớp 1 đến Lớp 5 và tối ưu trong phạm vi Toán tiểu học. '
                    'Hãy luôn kiểm tra lại các bước tính toán nhé!'
                    '</p>'
                )

        # =================================================================
        # LOGIC XỬ LÝ SỰ KIỆN (AN TOÀN ĐA LƯỢT)
        # =================================================================
        def on_send(user_text: str, history: list, mode: str, sess_id: str):
            clean_text = extract_clean_text(user_text)
            if not clean_text:
                yield history or [], "", sess_id, gr.update()
                return

            new_mode = _extract_mode_key(mode)
            if new_mode != adapter.current_mode:
                adapter.reset(mode=mode)

            latest_history = history or []
            for hist_updated in adapter.send_stream(clean_text, history):
                latest_history = hist_updated
                yield hist_updated, "", sess_id, gr.update()

            # Tự động lưu cuộc trò chuyện vào Lịch sử
            new_sess_id = HistoryManager.save_session(
                session_id=sess_id,
                messages=latest_history,
                mode=new_mode,
            )
            updated_choices = HistoryManager.get_choices()
            yield (
                latest_history,
                "",
                new_sess_id,
                gr.update(choices=updated_choices, value=new_sess_id),
            )

        btn_send.click(
            on_send,
            inputs=[user_input, chatbot, mode_dropdown, current_session_id],
            outputs=[chatbot, user_input, current_session_id, history_dropdown],
        )

        user_input.submit(
            on_send,
            inputs=[user_input, chatbot, mode_dropdown, current_session_id],
            outputs=[chatbot, user_input, current_session_id, history_dropdown],
        )

        # Nút Cuộc trò chuyện mới
        def on_new_chat(mode: str):
            adapter.reset(mode=mode)
            return [], "", None

        btn_new_chat.click(
            on_new_chat,
            inputs=[mode_dropdown],
            outputs=[chatbot, current_session_id, history_dropdown],
        )

        # Khi người dùng chọn xem lại một phiên chat từ Lịch sử
        def on_select_history(sess_id: str):
            if not sess_id:
                return gr.update(), "", gr.update()
            session_data = HistoryManager.get_session(sess_id)
            if not session_data:
                return gr.update(), "", gr.update()

            messages = session_data.get("messages", [])
            saved_mode = session_data.get("mode", "1")
            adapter.restore_session(messages, mode=saved_mode)

            matched_mode = next((m for m in MODE_CHOICES if m.startswith(saved_mode)), MODE_CHOICES[0])
            return messages, sess_id, matched_mode

        history_dropdown.change(
            on_select_history,
            inputs=[history_dropdown],
            outputs=[chatbot, current_session_id, mode_dropdown],
        )

        # Xoá toàn bộ lịch sử
        def on_clear_history():
            HistoryManager.clear_all()
            return gr.update(choices=[], value=None), "", []

        btn_clear_history.click(
            on_clear_history,
            inputs=[],
            outputs=[history_dropdown, current_session_id, chatbot],
        )

        # Khi đổi phương pháp sư phạm
        def on_mode_change(mode: str, history: list):
            if not history:
                return history
            adapter.reset(mode=mode)
            return []

        mode_dropdown.change(
            on_mode_change,
            inputs=[mode_dropdown, chatbot],
            outputs=[chatbot],
        )

        # Chuyển đổi Theme Light/Dark
        btn_theme.click(
            fn=None,
            js=THEME_TOGGLE_JS,
            inputs=[],
            outputs=[btn_theme],
        )

        # Áp dụng theme lưu trong localStorage
        demo.load(
            fn=None,
            js=INIT_THEME_JS,
            inputs=[],
            outputs=[btn_theme],
        )

    return demo


# =========================================================================
# KHỞI CHẠY CHÍNH
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Gia sư Toán Socratic - UI tối giản chuẩn")
    parser.add_argument("--mock",   action="store_true", help="Chạy mock nếu không muốn nạp model GGUF")
    parser.add_argument("--config", default=None,        help="Đường dẫn config.yaml")
    parser.add_argument("--share",  action="store_true", help="Tạo link public internet")
    parser.add_argument("--port",   type=int, default=7860, help="Port (mặc định 7860)")
    args = parser.parse_args()

    print("=" * 64)
    print(" GIA SU TOAN TIEU HOC - ZERO BORDER RADIUS & LICH SU CHAT")
    print("=" * 64)
    print("Dang khoi dong backend model...", flush=True)

    try:
        adapter = TutorAdapter.from_config(config_path=args.config, mock=args.mock)
    except Exception as exc:
        print(f"\n[LOI ENGINE] {exc}")
        print("Tu dong chuyen sang MockEngine...")
        adapter = TutorAdapter.from_config(config_path=args.config, mock=True)

    print("Backend san sang! Dang mo giao dien Gradio...", flush=True)
    demo = build_ui(adapter)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        show_error=True,
        css=SHARP_MINIMAL_CSS,
    )


if __name__ == "__main__":
    main()
