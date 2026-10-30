"""Sinh ``tutor/_nb_snapshot.py`` và ``data/frozen_regression.jsonl`` TỪ notebook.

Không gõ lại bất kỳ prompt/đề bài nào: mọi chuỗi lấy qua ``tutor.notebook_source``
(json + ast), đáp số/bước giải lấy qua ``tutor.mathcheck.solve``.

Cách dùng::

    python scripts/extract_notebook.py            # ghi file
    python scripts/extract_notebook.py --check    # chỉ kiểm tra, exit 1 nếu lệch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tutor.mathcheck import solve  # noqa: E402
from tutor.notebook_source import DEFAULT_NOTEBOOK, load  # noqa: E402

SNAPSHOT = ROOT / "tutor" / "_nb_snapshot.py"
FROZEN = ROOT / "data" / "frozen_regression.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_snapshot() -> str:
    nb = load()
    sha = _sha256(DEFAULT_NOTEBOOK)
    base_prompts = {key: nb.system_modes[name] for key, name in nb.mode_names.items()}
    lines = [
        '"""BẢN CHỤP DỰ PHÒNG — AUTO-GENERATED, KHÔNG SỬA TAY.',
        "",
        "Sinh bằng: python scripts/extract_notebook.py",
        f"Nguồn: {DEFAULT_NOTEBOOK.name} (sha256={sha})",
        "Chỉ dùng khi thiếu notebook gốc; tests/test_parity.py luôn so với notebook thật.",
        '"""',
        "from __future__ import annotations",
        "",
        f'SOURCE = "reference/{DEFAULT_NOTEBOOK.name}"',
        f"SOURCE_SHA256 = {sha!r}",
        "",
        f"MODE_NAMES: dict[str, str] = {nb.mode_names!r}",
        "",
        f"BASE_PROMPTS: dict[str, str] = {base_prompts!r}",
        "",
        f"CONCRETE_SOCRATIC_PROMPT: str = {nb.concrete_socratic_prompt!r}",
        "",
        f"DEMO_DIALOGUE: tuple[str, ...] = {tuple(nb.demo_dialogue)!r}",
        "",
        f"ENGLISH_THOUGHT_STARTERS: tuple[str, ...] = {tuple(nb.english_thought_starters)!r}",
        "",
        f"DECODE: dict[str, float] = {nb.decode!r}",
        f"MAX_NEW_TOKENS_CHAT: int = {int(nb.max_new_tokens_chat)}",
        f"MAX_NEW_TOKENS_SINGLE: int = {int(nb.max_new_tokens_single)}",
        "",
    ]
    return "\n".join(lines)


def _grade_number(label: str) -> int:
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    return int(digits) if digits else 0


def build_frozen() -> str:
    nb = load()
    rows: list[dict] = []

    for idx, test in enumerate(nb.elementary_tests, start=1):
        text = test["prompt"]
        sol = solve(text)
        if sol is None or sol.answer is None:
            raise SystemExit(f"Không giải được đề notebook: {text!r}")
        answer = list(sol.answer) if isinstance(sol.answer, tuple) else float(sol.answer)
        rows.append(
            {
                "id": f"nb-{idx:02d}",
                "kind": sol.kind,
                "grade": _grade_number(test.get("grade", "")),
                "topic": test.get("topic", ""),
                "text": text,
                "lang": "vi",
                "answer": answer,
                "unit": sol.unit,
                "steps": [
                    {"desc": st.desc, "expr": st.expr, "value": st.value}
                    for st in sol.steps
                ],
                "parts": sol.parts,
                "dialogue": [],
                "source": "notebook",
            }
        )

    demo_text = nb.default_user_text or (nb.demo_dialogue[0] if nb.demo_dialogue else "")
    sol = solve(demo_text)
    if sol is None or sol.answer is None:
        raise SystemExit(f"Không giải được đề demo: {demo_text!r}")
    rows.append(
        {
            "id": "nb-demo",
            "kind": sol.kind,
            "grade": 0,
            "topic": "Hội thoại mẫu MỤC 3",
            "text": demo_text,
            "lang": "vi",
            "answer": float(sol.answer),
            "unit": sol.unit,
            "steps": [
                {"desc": st.desc, "expr": st.expr, "value": st.value}
                for st in sol.steps
            ],
            "parts": sol.parts,
            "dialogue": list(nb.demo_dialogue),
            "source": "notebook",
        }
    )
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trích notebook -> snapshot + frozen data")
    parser.add_argument("--check", action="store_true", help="chỉ kiểm tra, không ghi")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    targets = {
        SNAPSHOT: build_snapshot(),
        FROZEN: build_frozen(),
    }

    if args.check:
        bad = [str(p.relative_to(ROOT)) for p, content in targets.items()
               if not p.exists() or p.read_text(encoding="utf-8") != content]
        if bad:
            print("LỆCH so với notebook: " + ", ".join(bad))
            print("Chạy: python scripts/extract_notebook.py")
            return 1
        if not args.quiet:
            print("OK: snapshot + frozen data khớp notebook.")
        return 0

    for path, content in targets.items():
        _write(path, content)
        if not args.quiet:
            print(f"Đã ghi {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
