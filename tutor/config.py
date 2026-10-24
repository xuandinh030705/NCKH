"""Nạp cấu hình YAML + truy cập theo đường dẫn điểm."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, NoReturn

try:  # PyYAML là phụ thuộc bắt buộc, nhưng vẫn cho lỗi dễ hiểu
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    print("Thiếu PyYAML. Chạy: pip install -r requirements.txt", file=sys.stderr)
    raise

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:
    """Cấu hình bất biến theo quy ước: đọc bằng cfg.get('a.b.c')."""

    def __init__(self, raw: dict[str, Any], root: Path, source: Path | None = None) -> None:
        self._raw = raw
        self.root = root
        self.source = source or (root / "config.yaml")

    # ------------------------------------------------------------------ nạp
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            path = os.environ.get("MATH_TUTOR_CONFIG") or (PROJECT_ROOT / "config.yaml")
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Không thấy file cấu hình: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config.yaml phải là mapping: {p}")
        return cls(raw, p.parent, p)

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Bản sao có ghi đè theo đường dẫn điểm (dùng cho autotune)."""
        raw = copy.deepcopy(self._raw)
        for dotted, value in overrides.items():
            node = raw
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return Config(raw, self.root, self.source)

    # --------------------------------------------------------------- truy cập
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._raw
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def section(self, name: str) -> dict[str, Any]:
        value = self._raw.get(name, {})
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._raw)

    # ------------------------------------------------------------- tiện ích
    def resolve(self, path: str | Path) -> Path:
        """Đường dẫn tương đối -> tuyệt đối theo thư mục chứa config.yaml."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self.root / p)

    def model_path(self, override: str | None = None) -> Path:
        """Model chính nếu tồn tại, ngược lại thử danh sách dự phòng."""
        candidates: list[Path] = []
        if override:
            candidates.append(self.resolve(override))
        primary = self.get("model.path")
        if primary:
            candidates.append(self.resolve(primary))
        if self.get("model.auto_fallback", True):
            for item in self.get("model.fallback_models", []) or []:
                candidates.append(self.resolve(item))
        for c in candidates:
            if c.exists():
                return c
        return candidates[0] if candidates else self.resolve("models/model.gguf")

    @staticmethod
    def abort(msg: str, code: int = 2) -> NoReturn:
        """In lỗi tiếng Việt gọn gàng rồi thoát (không traceback)."""
        print(f"LỖI: {msg}", file=sys.stderr)
        raise SystemExit(code)


def load_config(path: str | Path | None = None) -> Config:
    """Hàm tiện dụng cho script/CLI."""
    return Config.load(path)


__all__ = ["Config", "PROJECT_ROOT", "load_config"]
