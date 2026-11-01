@echo off
title VEYRA ChatGPT-4 Web Studio
cd /d "D:\Projects\veyra"
set "PATH=C:\Users\shreyash\AppData\Local\Python\bin;%PATH%"

echo ===================================================
echo     Launching VEYRA ChatGPT-4 Web Studio
echo ===================================================
echo Starting server and opening your browser...
echo.

python -m veyra.web_ui --port 7860
if %errorlevel% neq 0 (
    echo.
    echo [Fallback] Trying Python 3.14 runtime...
    "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" -m veyra.web_ui --port 7860
)
pause
