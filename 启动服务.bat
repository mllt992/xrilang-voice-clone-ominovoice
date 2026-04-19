@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

if not defined ENV_NAME set "ENV_NAME=omnivoice"
if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=8000"
set "RELOAD_FLAG="
set "CUDA_VISIBLE_DEVICES=0"
set "TF32=1"
set "PYTHONUTF8=1"

if /I "%DEV_RELOAD%"=="1" (
    set "RELOAD_FLAG=--reload"
)

echo ========================================
echo   OmniVoice local launcher
echo ========================================
echo.

where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] conda was not found in PATH.
    echo         Please install Anaconda or Miniconda first.
    pause
    exit /b 1
)

for /f "usebackq delims=" %%i in (`conda info --base 2^>nul`) do set "CONDA_BASE=%%i"
if not defined CONDA_BASE (
    echo [ERROR] Failed to locate the conda base directory.
    pause
    exit /b 1
)

call "%CONDA_BASE%\condabin\conda.bat" activate "%ENV_NAME%"
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env "%ENV_NAME%".
    echo         Please make sure the env already exists.
    pause
    exit /b 1
)

echo [INFO] Project: %CD%
echo [INFO] Env: %ENV_NAME%
echo.

echo [INFO] Python executable:
python -c "import sys; print(sys.executable)"
if errorlevel 1 (
    echo [ERROR] Python is not available after env activation.
    pause
    exit /b 1
)

echo.
echo [INFO] Checking runtime packages...
python -c "import sys, fastapi, soundfile, torch; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
if errorlevel 1 (
    echo.
    echo [ERROR] Missing runtime dependencies in env "%ENV_NAME%".
    echo         Run:
    echo           conda activate %ENV_NAME%
    echo           pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not defined HF_TOKEN (
    echo.
    echo [INFO] HF_TOKEN is not set in the current shell.
    echo        This is OK if config.py already contains a valid token.
)

echo.
echo [INFO] Starting server at http://%HOST%:%PORT%
if defined RELOAD_FLAG (
    echo [INFO] Dev reload is enabled.
)
echo [INFO] Press Ctrl+C to stop.
echo.

netstat -ano -p tcp | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [ERROR] Port %PORT% is already in use on %HOST%.
    echo         Try another port, for example:
    echo           set PORT=8001 ^&^& ??????.bat
    echo.
    pause
    exit /b 1
)

python -m uvicorn api.main:app --host %HOST% --port %PORT% %RELOAD_FLAG%
set "APP_EXIT=%ERRORLEVEL%"

if not "%APP_EXIT%"=="0" (
    echo.
    echo [ERROR] Server exited with code %APP_EXIT%.
)

pause
