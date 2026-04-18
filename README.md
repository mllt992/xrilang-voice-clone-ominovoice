# OmniVoice 语音克隆服务 MVP

基于 OmniVoice 的语音克隆服务，提供音色克隆和语音合成功能。

## 功能特性

- 🎙️ **音色克隆**：上传音频文件，克隆音色
- 🔊 **语音合成**：使用克隆的音色合成任意文本
- 📋 **音色管理**：查看和管理已克隆的音色
- 📁 **历史记录**：查看和管理合成音频

## 项目结构

```
xrilang-voice-clone-ominovoice/
├── config.example.py   # 配置示例（复制为 config.py）
├── config.py           # 本地配置文件（不提交）
├── core/              # 核心库
│   └── __init__.py   # 语音克隆核心功能
├── api/               # FastAPI 服务
│   └── main.py       # API 路由
├── static/            # 前端静态文件
│   └── index.html    # Web 界面
├── voices/            # 音色文件目录 (.pt)
├── outputs/           # 合成音频输出目录
├── requirements.txt   # Python 依赖
├── 启动服务.bat        # Windows 启动脚本
└── README.md
```

## 快速开始

### 环境要求

- Python 3.10+
- CUDA GPU (推荐，CPU 也可运行但较慢)
- ffmpeg (自动检测，如未找到请安装)

### 1. 配置

复制配置示例文件并填入你的 Token：

```bash
cp config.example.py config.py
```

然后编辑 `config.py`，填入你的 HuggingFace Token：

```python
HF_TOKEN = "hf_your_token_here"  # 第 18 行
```

获取 Token: https://huggingface.co/settings/tokens

### 2. 安装依赖

```bash
pip install omnivoice fastapi uvicorn python-multipart pydub soundfile
```

### 3. 安装 ffmpeg

```bash
# Windows
winget install ffmpeg

# Mac
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

### 4. 启动服务

```bash
# Windows
启动服务.bat

# 或手动启动
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 5. 访问服务

打开浏览器访问: http://localhost:8000

## API 接口

### 音色克隆
```
POST /api/voice/clone
Form Data:
  - voice_name: 音色名称
  - ref_audio: 参考音频文件
  - ref_text: (可选) 参考音频文本
```

### 音色列表
```
GET /api/voice/list
```

### 语音合成
```
POST /api/synthesize
Form Data:
  - text: 要合成的文本
  - voice_id: 音色 ID
  - language: 语言 (Chinese/English/Japanese)
  - speed: 语速 (0.5-2.0)
```

### 合成音频列表
```
GET /api/output/list
```

## 使用示例

### Python 调用

```python
import sys
sys.path.insert(0, 'path/to/project')

from core import clone_voice, synthesize, list_voices

# 克隆音色
result = clone_voice(
    ref_audio="参考音频.mp3",
    voice_name="my_voice",
    ref_text=None  # 自动转录
)
print(result)

# 合成语音
result = synthesize(
    text="要合成的文本内容",
    voice_id="my_voice",
    language="Chinese",
    speed=1.0
)
print(result)

# 查看音色列表
voices = list_voices()
print(voices)
```

## 技术栈

- **模型**: OmniVoice (k2-fsa/OmniVoice)
- **后端**: FastAPI + Uvicorn
- **前端**: HTML5 + CSS3 + JavaScript
- **音频处理**: pydub, soundfile
