@echo off
chcp 65001 >nul
echo ========================================
echo   OmniVoice 语音克隆服务启动器
echo ========================================
echo.

cd /d "%~dp0"

REM 尝试激活 conda 环境（如果 conda 可用）
where conda >nul 2>&1
if %errorlevel%==0 (
    call conda activate omnivoice
) else (
    echo 警告: 未找到 conda，请确保已安装 omnivoice 依赖
    echo.
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
