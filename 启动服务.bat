@echo off
chcp 65001 >nul
echo ========================================
echo   OmniVoice 语音克隆服务启动器
echo ========================================
echo.

cd /d "%~dp0"

REM 设置 conda 环境路径（如果需要）
if exist "D:\SotfwareData\MyAnconda\envs\omnivoice\python.exe" (
    set PATH=D:\SotfwareData\MyAnconda\envs\omnivoice;D:\SotfwareData\MyAnconda\envs\omnivoice\Scripts;D:\SotfwareData\MyAnconda\envs\omnivoice\Library\bin;D:\SotfwareData\MyAnconda\envs\omnivoice\DLLs;%PATH%
    echo 已配置 conda 环境路径
    echo.
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
