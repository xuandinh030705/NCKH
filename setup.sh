#!/usr/bin/env bash
# math-tutor-cli setup for Linux / macOS
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== math-tutor-cli setup (Linux/macOS) ==="

# 1. Check Python
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        if "$cmd" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" 2>/dev/null; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[LỖI] Cần Python 3.10 trở lên!"
    exit 1
fi
echo "[setup] Dùng Python: $($PYTHON_CMD --version)"

# 2. Virtual environment
if [ ! -d ".venv" ]; then
    echo "[setup] Đang tạo môi trường ảo .venv ..."
    "$PYTHON_CMD" -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip --quiet

# 3. Dependencies
echo "[setup] Cài đặt phụ thuộc ..."
pip install llama-cpp-python==0.3.36
pip install -r requirements.txt

# 4. Download Model
echo "[setup] Tải model Qwen3-4B-Instruct-2507-Q4_K_M.gguf ..."
python scripts/download_model.py

# 5. Tests
echo "[setup] Chạy kiểm thử pytest ..."
pip install pytest --quiet
pytest -q

echo "=== Xong! Chạy chatbot bằng: ./run.sh ==="
