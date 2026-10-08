@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Сначала установите программу: запустите setup\install.bat
  echo (нужен Python 3.12 и интернет; подробности в README).
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "tiktok\ui.py"
