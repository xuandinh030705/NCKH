"""history_manager.py – Quản lý lịch sử hội thoại cho Gia sư Toán Tiểu học.

Lưu trữ danh sách các phiên trò chuyện vào file JSON cục bộ (chat_history.json).
Xử lý an toàn mọi định dạng tin nhắn của Gradio 6 (str, list, dict, TextMessage).
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

CURR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURR_DIR.parent

# Ưu tiên lưu tại thư mục gốc dự án nếu có, nếu chạy độc lập thì lưu ngay tại thư mục hiện tại
HISTORY_FILE = (PROJECT_ROOT / "chat_history.json") if (PROJECT_ROOT / "config.yaml").exists() else (CURR_DIR / "chat_history.json")


def extract_clean_text(content: Any) -> str:
    """Trích xuất chuỗi văn bản thuần tuý từ bất kỳ cấu trúc dữ liệu nào của Gradio."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(str(p["text"]))
            elif hasattr(p, "text"):
                parts.append(str(p.text))
            else:
                parts.append(str(p))
        return " ".join(parts).strip()
    if isinstance(content, dict) and "text" in content:
        return str(content["text"]).strip()
    if hasattr(content, "text"):
        return str(content.text).strip()
    return str(content or "").strip()


class HistoryManager:
    """Quản lý các phiên hội thoại được lưu."""

    @staticmethod
    def load_all() -> list[dict[str, Any]]:
        """Đọc toàn bộ lịch sử từ file JSON."""
        if not HISTORY_FILE.exists():
            return []
        try:
            with HISTORY_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
        return []

    @staticmethod
    def save_all(sessions: list[dict[str, Any]]) -> None:
        """Ghi danh sách phiên vào file JSON."""
        try:
            with HISTORY_FILE.open("w", encoding="utf-8") as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @classmethod
    def save_session(
        cls,
        session_id: str,
        messages: list[dict[str, Any]],
        mode: str = "1",
        title: str | None = None,
    ) -> str:
        """Lưu hoặc cập nhật một phiên chat (chuẩn hoá văn bản an toàn)."""
        if not messages:
            return session_id

        # Chuẩn hoá toàn bộ tin nhắn thành string thuần
        cleaned_messages = []
        for m in messages:
            role = str(m.get("role", "user"))
            content = extract_clean_text(m.get("content", ""))
            if content:
                cleaned_messages.append({"role": role, "content": content})

        if not cleaned_messages:
            return session_id

        sessions = cls.load_all()

        # Tạo tiêu đề từ tin nhắn đầu tiên của user nếu chưa có
        if not title:
            first_user = ""
            for m in cleaned_messages:
                if m.get("role") == "user":
                    first_user = m["content"]
                    break

            if first_user:
                title = first_user[:32].strip() + ("..." if len(first_user) > 32 else "")
            else:
                title = f"Bài toán {time.strftime('%H:%M %d/%m')}"

        existing = next((s for s in sessions if s.get("id") == session_id), None)
        timestamp = time.strftime("%H:%M %d/%m/%Y")

        if existing:
            existing["messages"] = cleaned_messages
            existing["mode"] = mode
            existing["title"] = title
            existing["updated_at"] = timestamp
        else:
            session_id = session_id or str(uuid.uuid4())[:8]
            sessions.insert(0, {
                "id": session_id,
                "title": title,
                "mode": mode,
                "created_at": timestamp,
                "updated_at": timestamp,
                "messages": cleaned_messages,
            })

        # Giữ tối đa 30 phiên gần nhất
        sessions = sessions[:30]
        cls.save_all(sessions)
        return session_id

    @classmethod
    def get_session(cls, session_id: str) -> dict[str, Any] | None:
        """Lấy chi tiết một phiên chat theo id."""
        if not session_id:
            return None
        sessions = cls.load_all()
        return next((s for s in sessions if s.get("id") == session_id), None)

    @classmethod
    def delete_session(cls, session_id: str) -> list[dict[str, Any]]:
        """Xoá một phiên chat."""
        sessions = cls.load_all()
        sessions = [s for s in sessions if s.get("id") != session_id]
        cls.save_all(sessions)
        return sessions

    @classmethod
    def clear_all(cls) -> None:
        """Xoá toàn bộ lịch sử."""
        if HISTORY_FILE.exists():
            try:
                HISTORY_FILE.unlink()
            except Exception:
                cls.save_all([])

    @classmethod
    def get_choices(cls) -> list[tuple[str, str]]:
        """Trả về danh sách (tiêu đề, session_id) cho Dropdown."""
        sessions = cls.load_all()
        choices = []
        for s in sessions:
            label = f"💬 {s.get('title', 'Bài toán')} ({s.get('updated_at', '')[:10]})"
            choices.append((label, s["id"]))
        return choices

