"""adapter.py – Bridge giữa TutorSession, Engine và Gradio.

- Tự động nhận diện bài toán các lớp 1 đến 5 và tối ưu trong phạm vi toán tiểu học.
- Chuẩn hoá Chế độ 2: Hướng dẫn sơ đồ đoạn thẳng chuẩn mực, không giải tắt ra số thập phân vô lý (như 4.8 viên bi).
- Chuẩn hoá an toàn dữ liệu tin nhắn (loại bỏ triệt để lỗi 'list' has no attribute 'strip').
- Chống tràn ngữ cảnh (context window 3072) khi hỏi nhiều bài toán liên tiếp.
- Tự động kiểm tra tính toán học sinh (chống việc AI khen phép tính sai như 45 - 18 = 30).
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Generator, Any

CURR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURR_DIR.parent
for p in [str(CURR_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tutor.config import Config
from tutor.engine import Engine, EngineError, _bootstrap_dll_paths
from tutor.mathcheck import arith_claims
from tutor.prompts import DEFAULT_MODE
from tutor.session import TutorSession, TurnResult

try:
    from history_manager import extract_clean_text
except ImportError:
    from Frontend_AI.history_manager import extract_clean_text


# =========================================================================
# LỜI NHẮC PHẠM VI TOÁN TIỂU HỌC & CHẾ ĐỘ 2, 3 CHUẨN MỰC
# =========================================================================
ELEMENTARY_MATH_SYSTEM_PROMPT = """

[QUY TẮC BẮT BUỘC: PHẠM VI TOÁN TIỂU HỌC & NHẬN DIỆN KHỐI LỚP 1–5]:
1. TỰ ĐỘNG NHẬN DIỆN KHỐI LỚP VÀ ĐIỀU CHỈNH SƯ PHẠM:
   - Lớp 1: Số đếm, so sánh, cộng trừ trong phạm vi 100, bài toán 1 phép tính đơn giản.
   - Lớp 2: Cộng trừ có nhớ, bảng nhân chia 2 và 5, tìm thành phần chưa biết (số hạng, số bị trừ, thừa số).
   - Lớp 3: Bảng nhân chia 2 đến 9, số đến 100.000, 4 phép tính, chu vi và diện tích hình vuông/chữ nhật, bài toán 2 bước tính.
   - Lớp 4: Số tự nhiên nhiều chữ số, 4 phép tính nâng cao, các bài toán kinh điển: Tìm hai số khi biết Tổng và Hiệu, Tổng và Tỉ số, Hiệu và Tỉ số, Trung bình cộng, Phân số và phép tính phân số.
   - Lớp 5: Phân số thập phân, Hỗn số, Số thập phân và phép tính số thập phân, Tỉ số phần trăm, Toán chuyển động đều (vận tốc v, quãng đường s, thời gian t), Diện tích hình thang, hình tròn, Diện tích và thể tích hình hộp chữ nhật/hình lập phương.
   => Hãy tự động nhận diện bài toán thuộc khối lớp nào để xưng hô thân mật (thầy/em), đưa ra câu hỏi gợi mở với ngôn từ và độ khó phù hợp chính xác lứa tuổi học sinh lớp đó.

2. TUYỆT ĐỐI KHÔNG VƯỢT QUÁ PHẠM VI TOÁN TIỂU HỌC:
   - CHỈ sử dụng phương pháp toán tiểu học: suy luận số học, vẽ sơ đồ đoạn thẳng, rút về đơn vị, phương pháp thử chọn trực quan.
   - TUYỆT ĐỐI CẤM sử dụng kiến thức cấp 2 và cấp 3: không dùng 'phương trình', không dùng 'chuyển vế đổi dấu', không dùng 'đặt ẩn x, y', không dùng số âm, căn bậc hai, hàm số, đạo hàm.
   - NẾU BÀI TOÁN NẰM NGOÀI PHẠM VI TIỂU HỌC: Hãy từ chối một cách lịch sự, nhã nhặn: 'Bài toán này thuộc chương trình cấp 2/cấp 3 (vượt quá phạm vi Toán tiểu học lớp 1–5). Thầy là gia sư chuyên về Toán tiểu học, em hãy gửi bài toán tiểu học để thầy trò mình cùng giải nhé!'

3. KIỂM TRA PHÉP TÍNH CỦA HỌC SINH (CHỐNG PHÊ DUYỆT SAI):
   - Khi học sinh nêu một phép tính hoặc kết quả (ví dụ 45 - 18 = 30):
     Em PHẢI kiểm tra lại thật kỹ tính đúng sai!
     + Nếu học sinh tính SAI (ví dụ 45 - 18 = 27 chứ không phải 30):
       TUYỆT ĐỐI KHÔNG khen đúng, KHÔNG công nhận kết quả sai!
       Hãy nhẹ nhàng chỉ ra: 'Phép tính của em chưa chính xác đâu. Em hãy tính lại thật cẩn thận xem 45 - 18 bằng bao nhiêu nhé?'
     + Nếu học sinh tính ĐÚNG:
       Em mới được khen và hướng dẫn tiếp bước tiếp theo.

4. QUY TẮC RIÊNG CHO CHẾ ĐỘ 2 (HƯỚNG DẪN PHƯƠNG PHÁP & SƠ ĐỒ ĐOẠN THẲNG):
   - MỤC TIÊU: Hướng dẫn học sinh cách phân tích tỉ lệ và vẽ sơ đồ đoạn thẳng, KHÔNG GIẢI RA ĐÁP SỐ CUỐI CÙNG!
   - NỘI DUNG:
     + Bước 1: Xác định đại lượng bé nhất quy ước là 1 phần (1 đoạn thẳng).
     + Bước 2: Xác định các đại lượng khác gồm mấy phần bằng nhau tương ứng.
     + Bước 3: Gắn Tổng (hoặc Hiệu) vào sơ đồ, tính tổng số phần bằng nhau.
     + Bước 4 (Kết thúc): Đặt câu hỏi để học sinh tự tính giá trị 1 phần.
   - CHÚ Ý TÍNH THỰC TẾ:
     + Các vật thể rời rạc (viên bi, cái kẹo, con vật, con người) BẮT BUỘC là số tự nhiên nguyên vẹn.
     + Nếu tổng không chia hết cho tổng số phần (như 48 không chia hết cho 10), hãy dừng lại và nhận xét ngay: 'Tổng 48 viên bi không chia hết cho 10 phần bằng nhau vì số viên bi phải là số tự nhiên nguyên vẹn. Em hãy kiểm tra lại đề bài xem có phải tổng số bi là 40, 50 hay 60 viên không nhé!' - TUYỆT ĐỐI KHÔNG chia ra số thập phân (như 4.8 viên bi) và KHÔNG giải phương trình x.

5. QUY TẮC RIÊNG CHO CHẾ ĐỘ 3 (LỜI GIẢI CHI TIẾT):
   - MỤC TIÊU: Trình bày bài giải mẫu hoàn chỉnh, rõ ràng, chuẩn mực theo phương pháp Toán tiểu học từ đầu đến cuối.
   - TRÌNH BÀY:
     + Nêu rõ từng câu lời giải và phép tính kèm đơn vị trong ngoặc đơn.
     + Đối với bài toán chuyển động:
       * Thời gian xe chạy từ điểm xuất phát đến điểm đích: [Giờ đến] - [Giờ khởi hành].
       * Thời gian thực tế xe di chuyển: [Thời gian chạy] - [Thời gian nghỉ] (nếu có).
       * Vận tốc = Quãng đường chia cho thời gian thực tế di chuyển.
     + BẮT BUỘC kết thúc toàn bộ bài giải bằng dòng: 'Đáp số: [kết quả] [đơn vị]'.
     + Tuyệt đối không dừng giữa chừng, không nhắc đến các con số lạ không có trong đề bài.
"""

MODE_CHOICES: list[str] = [
    "1 – Gợi mở từng bước (Socratic)",
    "2 – Hướng dẫn phương pháp (Sơ đồ)",
    "3 – Lời giải chi tiết (Mẫu)",
]


def _extract_mode_key(choice: str) -> str:
    return str(choice).strip()[:1] or "1"


def is_new_problem_statement(text: str) -> bool:
    """Nhận diện xem học sinh có đưa ra bài toán mới hoàn toàn không."""
    low = text.lower().strip()
    if any(k in low for k in ("em tính", "con tính", "em ra", "kết quả là", "có đúng không", "thầy ơi", "đáp án")):
        return False
    keywords = [
        "tìm x", "hỏi ", "bao nhiêu", "một ô tô", "một người",
        "một thửa ruộng", "một cửa hàng", "một đàn", "biết rằng", "đề bài:"
    ]
    if any(k in low for k in keywords) and len(low) > 12:
        return True
    if re.search(r"\bx\s*[\+\-\*\/×:÷]\s*\d+\s*=", low) or re.search(r"\d+\s*[\+\-\*\/×:÷]\s*x\s*=", low):
        return True
    return False


# =========================================================================
# LỚP ĐIỀU HỢP CHO GRADIO FRONTEND
# =========================================================================
class TutorAdapter:
    """Wrapper cho Engine + TutorSession kết nối trực tiếp với giao diện."""

    def __init__(self, cfg: Config, engine: Engine) -> None:
        self.cfg = cfg
        self.engine = engine
        self._session: TutorSession | None = None
        self.current_mode: str = DEFAULT_MODE

    @classmethod
    def from_config(cls, config_path: str | None = None, mock: bool = False) -> "TutorAdapter":
        """Khởi tạo Engine thực (llama.cpp) hoặc mock với n_ctx mở rộng."""
        _bootstrap_dll_paths()
        cfg = Config.load(config_path)

        model_file = cfg.resolve(str(cfg.get("model.path", "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")))
        overrides: dict[str, Any] = {
            "model.n_ctx": 3072,               # Mở rộng context window lên 3072 tránh tràn
            "session.max_history_turns": 4,    # Giữ tối đa 4 lượt gần nhất
            "math.hidden_hint": False,         # Tắt gợi ý toán ẩn heuristic để model tự giải chính xác
        }

        if mock:
            print("[ENGINE] Khoi chay o che do MOCK theo yeu cau.")
            overrides["engine.backend"] = "mock"
        elif not model_file.exists():
            print(f"[ENGINE] Khong tim thay file model {model_file}. Tu dong fallback sang MockEngine.")
            overrides["engine.backend"] = "mock"
        else:
            print(f"[ENGINE] Tim thay model: {model_file}. Khoi dong LlamaCppEngine (CUDA/CPU)...")
            overrides["engine.backend"] = "llama"

        cfg = cfg.with_overrides(overrides)
        engine = Engine(cfg)
        engine.load()
        n_ctx_str = engine.profile.n_ctx if engine.profile else "N/A"
        print(f"[ENGINE] San sang! Backend = {engine.backend} | n_ctx = {n_ctx_str}")
        return cls(cfg, engine)

    def _ensure_session(self) -> TutorSession:
        if self._session is None:
            self.reset()
        return self._session  # type: ignore[return-value]

    def reset(self, mode: str | None = None) -> None:
        """Làm mới phiên học."""
        if mode is not None:
            self.current_mode = _extract_mode_key(mode)
        if self._session is None:
            self._session = TutorSession(self.engine, mode=self.current_mode, cfg=self.cfg)
        else:
            self._session.set_mode(self.current_mode)

    def restore_session(self, messages: list[dict[str, Any]], mode: str = "1") -> None:
        """Khôi phục lại phiên chat từ lịch sử."""
        self.reset(mode=mode)
        session = self._ensure_session()
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = extract_clean_text(msg.get("content", ""))
            if role in ("user", "assistant") and content:
                session.history.append({"role": role, "content": content})
                if role == "user" and not session.problem:
                    session.problem = content
        session.turn_index = len([m for m in messages if str(m.get("role")) == "user"])

    def send_stream(
        self,
        user_text: str,
        history: list[dict[str, Any]] | None,
    ) -> Generator[list[dict[str, str]], None, None]:
        """Gửi tin nhắn và yield từng token dạng chuỗi sạch (không có đối tượng lồng nhau)."""
        session = self._ensure_session()
        tokens: list[str] = []
        done_flag: list[bool] = [False]
        result_holder: list[TurnResult | None] = [None]
        error_holder: list[str] = [""]

        clean_user_text = extract_clean_text(user_text)

        # 1. Phát hiện nếu học sinh đưa bài toán mới hoàn toàn trong cùng cuộc trò chuyện
        if is_new_problem_statement(clean_user_text):
            session.problem = clean_user_text
            session.turn_index = 0
            if session.history:
                session.history = [session.history[0]]

        # 2. Chuẩn hoá danh sách tin nhắn giao diện
        clean_messages: list[dict[str, str]] = []
        for item in (history or []):
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = extract_clean_text(item.get("content", ""))
                if content:
                    clean_messages.append({"role": role, "content": content})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                u_text = extract_clean_text(item[0])
                a_text = extract_clean_text(item[1])
                if u_text:
                    clean_messages.append({"role": "user", "content": u_text})
                if a_text:
                    clean_messages.append({"role": "assistant", "content": a_text})

        # 3. Cắt bớt lịch sử phiên nếu quá dài để không bao giờ vượt context window
        if len(session.history) > 5:
            session.history = [session.history[0]] + session.history[-4:]

        clean_messages.append({"role": "user", "content": clean_user_text})
        clean_messages.append({"role": "assistant", "content": "..."})
        yield clean_messages

        def _run() -> None:
            try:
                # Token rộng rãi để giải hoàn chỉnh mà không bị cắt cụt giữa chừng
                if self.current_mode == "3":
                    max_tokens = 650
                elif self.current_mode == "2":
                    max_tokens = 500
                else:
                    max_tokens = 350
                res = session.send(
                    clean_user_text,
                    stream_cb=lambda s: tokens.append(s),
                    raw=False,
                    max_tokens=max_tokens,
                )
                result_holder[0] = res
            except EngineError as exc:
                error_holder[0] = f"⚠️ Lỗi backend engine: {exc}"
            except Exception as exc:
                error_holder[0] = f"⚠️ Lỗi xử lý: {exc}"
            finally:
                done_flag[0] = True

        t = Thread(target=_run, daemon=True)
        t.start()

        # Streaming từng từ ra màn hình
        partial = ""
        while not done_flag[0] or tokens:
            while tokens:
                partial += tokens.pop(0)
                clean_messages[-1]["content"] = partial
                yield clean_messages
            time.sleep(0.01)

        if error_holder[0]:
            clean_messages[-1]["content"] = error_holder[0]
            yield clean_messages
            return

        result = result_holder[0]
        if result is not None:
            clean_messages[-1]["content"] = result.reply

        yield clean_messages

    def hw_info(self) -> dict:
        try:
            hw = self.engine.hw
            return {
                "gpu": hw.gpu_name if hw.gpu_name else "CPU",
                "ram": f"{hw.ram_gb:.0f} GB",
            }
        except Exception:
            return {"gpu": "CPU", "ram": "—"}

    def close(self) -> None:
        try:
            self.engine.close()
        except Exception:
            pass
