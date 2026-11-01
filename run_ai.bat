@echo off
title VEYRA AI Assistant
cd /d "D:\Projects\veyra"
set "PATH=C:\Users\shreyash\AppData\Local\Python\bin;%PATH%"

echo Starting VEYRA AI...
python app.py
if %errorlevel% neq 0 (
    echo.
    echo [Fallback] Trying direct Python 3.14 runtime...
    "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" app.py
)
pause
