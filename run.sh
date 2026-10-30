#!/usr/bin/env bash
# math-tutor-cli runner for Linux / macOS
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

if [ ! -f "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" ] && [ "$1" != "--mock" ]; then
    echo "[THÔNG BÁO] Chưa có model trong models/."
    echo "Đang tải model (lần đầu tiên)..."
    python scripts/download_model.py
fi

python -m tutor "$@"
