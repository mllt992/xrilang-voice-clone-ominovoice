# OmniVoice 语音克隆服务

基于 [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) 的本地 Web 服务，提供音色克隆、Voice Design、Auto Voice 和语音合成能力。

## 当前增强点

- 支持 `音色克隆`、`Voice Design`、`Auto Voice`
- 支持 `已克隆音色 + instruct` 组合使用
- 支持同名音色 `强制重建`
- 增加了上传大小限制、文件名校验和输出路径安全校验
- 优化了输出文件命名，避免并发下文件名冲突
- Web UI 去掉了动态 HTML 注入风险点
- 启动脚本默认使用 `conda` 环境 `omnivoice`

## 项目结构

```text
xrilang-voice-clone-ominovoice/
├── api/
│   └── main.py
├── core/
│   ├── __init__.py
│   ├── service_utils.py
│   └── voice_clone_prompt.py
├── static/
│   └── index.html
├── voices/
├── outputs/
├── config.example.py
├── requirements.txt
├── 启动服务.bat
└── README.md
```

## 环境要求

- 已存在 `conda` 环境：`omnivoice`
- Python 3.10+
- CUDA GPU 推荐
- `ffmpeg`

## 快速开始

### 1. 切换到 `omnivoice` 环境

```bash
conda activate omnivoice
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果 `torch` 尚未安装，请按你的 CUDA 版本先安装 PyTorch，再执行上面的命令。

### 3. 配置

复制配置文件：

```bash
copy config.example.py config.py
```

然后编辑 `config.py`：

```python
HF_TOKEN = "hf_xxx"
```

如果你已经在系统环境变量里设置了 `HF_TOKEN`，服务会优先使用环境变量。

### 4. 启动

Windows 下直接双击：

```text
启动服务.bat
```

或命令行启动：

```bash
conda run -n omnivoice python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

启动后访问：

```text
http://127.0.0.1:8000
```

## 使用说明

### 音色克隆

- 音色名称只允许字母、数字、下划线、短横线
- 参考音频建议 3-10 秒
- 如果已存在同名音色，不勾选“强制重建”时会直接复用缓存

### 语音合成

支持 3 种模式：

1. `Auto Voice`
   不选音色，也不填 `instruct`
2. `Voice Design`
   不选音色，只填 `instruct`
3. `Voice Clone + instruct`
   选中已有音色，同时填写 `instruct`

官方 OmniVoice 对中文方言场景建议结合 `ref_audio + instruct` 使用，本项目已支持这一组合。

## API

### 健康检查

```http
GET /api/health
```

### 克隆音色

```http
POST /api/voice/clone
Content-Type: multipart/form-data
```

字段：

- `voice_name`: 音色名称
- `ref_audio`: 参考音频
- `ref_text`: 可选，参考音频文本
- `rebuild`: 可选，是否强制重建

### 音色列表

```http
GET /api/voice/list
```

### 合成语音

```http
POST /api/synthesize
Content-Type: multipart/form-data
```

字段：

- `text`: 待合成文本
- `voice_id`: 可选，留空时可走 Auto Voice 或纯 Voice Design
- `language`: 语言
- `speed`: 语速
- `duration`: 固定时长
- `num_step`: 推理步数
- `guidance_scale`: 引导强度
- `t_shift`: 时间偏移
- `layer_penalty_factor`: 层惩罚
- `position_temperature`: 位置温度
- `class_temperature`: 类别温度
- `denoise`: 去噪
- `preprocess_prompt`: 预处理参考音频
- `postprocess_output`: 后处理输出
- `instruct`: 可选，Voice Design 指令

### 合成历史

```http
GET /api/output/list
```

### 播放/下载输出

```http
GET /api/output/{filename}
```

## 配置项

`config.example.py` 中新增了这些可选项：

- `MAX_UPLOAD_MB`
- `MAX_TEXT_LENGTH`
- `DEFAULT_AUDIO_CHUNK_DURATION`
- `DEFAULT_AUDIO_CHUNK_THRESHOLD`
- `ALLOWED_AUDIO_EXTENSIONS`

## 注意事项

- `voices/` 和 `outputs/` 默认是本地缓存目录，不要直接暴露给公网下载
- 如果要对外提供服务，建议再加鉴权、限流和异步任务队列
- 当前仓库更适合作为本地或内网推理服务，而不是公网开放平台
