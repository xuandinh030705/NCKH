@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
if exist .venv\Scripts\python.exe (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)
%PY% -m tutor %*
