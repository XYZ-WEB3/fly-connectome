@echo off
chcp 65001 >nul
rem Перетащи видеофайл на этот значок: муха посмотрит ролик, готовое видео появится в папке «готово».
rem Режим «зрение и слух» (без распознавания речи). Все настройки — в окне «Муха-рендер.bat» или в tiktok\react.py --help.
cd /d "%~dp0"
if "%~1"=="" (
  echo Перетащи видеофайл на этот значок мышкой.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -W ignore tiktok\react.py "%~1" --format streamer --flyeye on --meaning off --lock cli
echo.
echo Готовое видео — в папке «готово».
pause
