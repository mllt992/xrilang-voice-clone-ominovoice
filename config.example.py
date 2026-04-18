# -*- coding: utf-8 -*-
"""
配置文件示例
复制此文件为 config.py 并填入你的配置
"""
import os
import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 模型配置
MODEL_NAME = "k2-fsa/OmniVoice"

# HuggingFace Token（必需，用于下载模型）
# 请从 https://huggingface.co/settings/tokens 获取
HF_TOKEN = "hf_your_token_here"

# 音色文件目录
VOICES_DIR = PROJECT_ROOT / "voices"

# 合成音频输出目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _find_ffmpeg() -> list:
    """自动查找 ffmpeg 路径"""
    candidates = []

    # 1. 先检查系统 PATH 中是否有 ffmpeg
    ffmpeg_in_path = shutil.which("ffmpeg")
    if ffmpeg_in_path:
        candidates.append(Path(ffmpeg_in_path))

    # 2. 常见的 Windows 安装位置
    if os.name == 'nt':
        # 用户本地 WinGet
        winget_path = Path(os.environ.get("LOCALAPPDATA", ""))
        if winget_path:
            candidates.extend(winget_path.glob("Microsoft/WinGet/Packages/*ffmpeg*/ffmpeg-*/bin/ffmpeg.exe"))

        # MSYS2 / Git Bash / WSL
        for base in [Path("C:/msys64"), Path("C:/Program Files/Git"), Path("C:/Program Files")]:
            if base.exists():
                candidates.extend(base.glob("**/ffmpeg.exe"))

    # 3. 常见的 Unix 位置
    for prefix in ["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"]:
        p = Path(prefix) / "ffmpeg"
        if p.exists():
            candidates.append(p)

    # 去重并返回存在的
    seen = set()
    result = []
    for p in candidates:
        p_str = str(p.resolve())
        if p.exists() and p_str not in seen:
            seen.add(p_str)
            result.append(p)

    return result


# ffmpeg 路径（自动查找）
FFMPEG_CANDIDATES = _find_ffmpeg()

if FFMPEG_CANDIDATES:
    ffmpeg_path = FFMPEG_CANDIDATES[0]
else:
    ffmpeg_path = None

# 确保目录存在
VOICES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
