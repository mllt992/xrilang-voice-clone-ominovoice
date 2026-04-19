@echo off
chcp 65001 >nul
echo ========================================
echo   OmniVoice 语音克隆服务启动器
echo ========================================
echo.

cd /d "%~dp0"

REM 查找 conda 环境路径（自适应）
where conda >nul 2>&1
if %errorlevel%==0 (
    REM 获取当前 conda 环境的路径
    for /f "delims=" %%i in ('conda env list --json 2^>nul ^| findstr /i "omnivoice"') do (
        echo 已激活 conda 环境
    )
) else (
    REM 如果没有 conda，尝试常见路径
    if exist "D:\SotfwareData\MyAnconda\envs\omnivoice\python.exe" (
        set PATH=D:\SotfwareData\MyAnconda\envs\omnivoice;D:\SotfwareData\MyAnconda\envs\omnivoice\Scripts;D:\SotfwareData\MyAnconda\envs\omnivoice\Library\bin;D:\SotfwareData\MyAnconda\envs\omnivoice\DLLs;%PATH%
    )
)

REM 检查并复制缺失的 DLL（如 liblzma.dll）
if exist "D:\SotfwareData\MyAnconda\envs\omnivoice\Library\bin\liblzma.dll" (
    if not exist "D:\SotfwareData\MyAnconda\envs\omnivoice\DLLs\liblzma.dll" (
        copy /Y "D:\SotfwareData\MyAnconda\envs\omnivoice\Library\bin\liblzma.dll" "D:\SotfwareData\MyAnconda\envs\omnivoice\DLLs\liblzma.dll" >nul
    )
)

REM 设置 HuggingFace Token（从环境变量读取，或直接填写）
if not defined HF_TOKEN (
    echo 警告: HF_TOKEN 未设置，下载可能较慢
    echo 请运行前设置: set HF_TOKEN=hf_xxx
    echo.
)

echo 正在启动服务...
echo 请在浏览器中打开: http://localhost:8000
echo 按 Ctrl+C 停止服务
echo.

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
pause
