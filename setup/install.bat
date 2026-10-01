@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
echo.
echo  === Установка: муха с мозгом из коннектома FlyWire ===
echo.

rem --- какой Python: лучше 3.12 через лаунчер py, иначе просто python
set "PY="
py -3.12 --version >nul 2>nul && set "PY=py -3.12"
if not defined PY python --version >nul 2>nul && set "PY=python"
if not defined PY (
  echo Не найден Python. Поставь Python 3.12 с python.org
  echo и при установке отметь галочку "Add python.exe to PATH". Потом запусти этот файл ещё раз.
  pause
  exit /b 1
)
echo Python:
%PY% --version

rem --- своё окружение в папке .venv (ничего не ставится в систему)
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo [1/5] Создаю окружение .venv ...
  %PY% -m venv .venv || goto :err
)
set "VPY=.venv\Scripts\python.exe"

echo.
echo [2/5] Обновляю pip ...
"%VPY%" -m pip install --upgrade pip || goto :err

echo.
echo [3/5] Ставлю PyTorch с поддержкой видеокарты NVIDIA (около 3 ГБ, это долго) ...
"%VPY%" -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 || goto :err

echo.
echo [4/5] Ставлю остальные пакеты ...
"%VPY%" -m pip install -r requirements.txt || goto :err

echo.
echo [5/5] Скачиваю данные коннектома FlyWire (около 135 МБ) ...
"%VPY%" setup\get_data.py || goto :err

echo.
echo Проверка:
"%VPY%" setup\check.py
echo.
echo Готово. Дальше: перетащи видео на "Муха-ролик.bat" или открой "Муха-рендер.bat".
pause
exit /b 0

:err
echo.
echo Установка остановилась с ошибкой. Что делать — docs\install.md, раздел "Если что-то не так".
pause
exit /b 1
