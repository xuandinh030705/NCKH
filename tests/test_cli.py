"""Kiểm tra CLI (không cần model/GPU)."""
from __future__ import annotations

from tutor import cli


def test_parser_defaults() -> None:
    args = cli.build_parser().parse_args([])
    assert args.mode == "1"
    assert args.mock is False
    assert args.once is None


def test_once_with_mock(capsys) -> None:
    code = cli.main(["--mock", "--once", "Tìm x, biết: x + 18 = 45"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Chào em" in out
    assert "tok/s" in out


def test_mode2_with_mock(capsys) -> None:
    code = cli.main(["--mock", "--mode", "2", "--once", "Tìm x, biết: x + 18 = 45"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.strip()


def test_invalid_mode(capsys) -> None:
    code = cli.main(["--mock", "--mode", "9", "--once", "Tìm x, biết: x + 18 = 45"])
    out = capsys.readouterr().out
    assert code == 2
    assert "LỖI" in out


def test_missing_config(capsys) -> None:
    code = cli.main(["--mock", "--config", "khong-ton-tai.yaml", "--once", "x"])
    out = capsys.readouterr().out
    assert code == 2
    assert "LỖI" in out
