@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
title Gia su Toan Tieu hoc AI - Gradio Frontend
cd /d "%~dp0"

echo ============================================================
echo   GIA SU TOAN TIEU HOC (AI SOCRATIC) - FRONTEND
echo   Giao dien Gradio toi gian, 0 Border-Radius, Lich su Chat
echo ============================================================
echo.

set PYCMD=..\.venv\Scripts\python.exe
if not exist "%PYCMD%" (
    set PYCMD=.venv\Scripts\python.exe
)
if not exist "%PYCMD%" (
    echo [CANH BAO] Khong tim thay .venv, dung python he thong...
    set PYCMD=python
)

echo Dang khoi dong ung dung tai http://localhost:7860 ...
echo.

%PYCMD% app.py %*
pause

