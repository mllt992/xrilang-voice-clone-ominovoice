# -*- coding: utf-8 -*-
"""
FastAPI 服务
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
from typing import Optional

import config as app_config
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core import clone_voice, synthesize, list_voices, list_outputs
from core.service_utils import clean_optional_text, resolve_file_in_dir
from config import PROJECT_ROOT

MAX_UPLOAD_MB = float(getattr(app_config, "MAX_UPLOAD_MB", 20))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
ALLOWED_AUDIO_EXTENSIONS = tuple(
    getattr(
        app_config,
        "ALLOWED_AUDIO_EXTENSIONS",
        (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"),
    )
)


def _result_response(result: dict, success_status: int = 200) -> JSONResponse:
    payload = dict(result)
    status_code = int(payload.pop("status_code", success_status))
    return JSONResponse(status_code=status_code, content=payload)


def _validate_audio_suffix(upload: UploadFile) -> str:
    filename = clean_optional_text(upload.filename) or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        supported = ", ".join(ALLOWED_AUDIO_EXTENSIONS)
        raise HTTPException(status_code=400, detail=f"参考音频格式不支持，请使用: {supported}")
    return suffix

# 创建 FastAPI 应用
app = FastAPI(title="OmniVoice 语音克隆服务", version="1.0.0")

# 挂载静态文件目录
static_dir = PROJECT_ROOT / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    """返回前端页面"""
    return FileResponse(str(static_dir / "index.html"))


@app.get("/api/health")
async def api_health():
    return {
        "success": True,
        "service": "omnivoice-web",
        "max_upload_mb": MAX_UPLOAD_MB,
    }


# ============ 音色克隆接口 ============

@app.post("/api/voice/clone")
async def api_clone_voice(
    voice_name: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
    rebuild: bool = Form(False),
):
    """
    克隆音色接口

    - voice_name: 音色名称（用于标识）
    - ref_audio: 参考音频文件（支持 mp3, wav, m4a 等）
    - ref_text: 参考音频文本（可选）
    """
    suffix = _validate_audio_suffix(ref_audio)
    content = await ref_audio.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"参考音频过大，请控制在 {MAX_UPLOAD_MB:g} MB 以内。")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await run_in_threadpool(
            clone_voice,
            ref_audio=tmp_path,
            voice_name=voice_name,
            ref_text=ref_text,
            rebuild=rebuild,
        )
        return _result_response(result)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        await ref_audio.close()


@app.get("/api/voice/list")
async def api_list_voices():
    """音色列表接口"""
    voices = await run_in_threadpool(list_voices)
    return {
        "success": True,
        "voices": voices,
        "total": len(voices),
    }


# ============ 语音合成接口 ============

@app.post("/api/synthesize")
async def api_synthesize(
    text: str = Form(...),
    voice_id: Optional[str] = Form(None),
    language: str = Form("Chinese"),
    # Speed & Duration
    speed: float = Form(1.0),
    duration: Optional[float] = Form(None),
    # Quality
    num_step: int = Form(16),
    guidance_scale: float = Form(2.0),
    # Advanced
    t_shift: float = Form(0.1),
    layer_penalty_factor: float = Form(5.0),
    position_temperature: float = Form(5.0),
    class_temperature: float = Form(0.0),
    # Options
    denoise: bool = Form(True),
    preprocess_prompt: bool = Form(True),
    postprocess_output: bool = Form(True),
    audio_chunk_duration: float = Form(15.0),
    audio_chunk_threshold: float = Form(30.0),
    # Voice Design
    instruct: Optional[str] = Form(None),
):
    """
    语音合成接口

    - text: 要合成的文本
    - voice_id: 音色 ID
    - language: 语言
    - speed: 语速 (0.5-2.0)
    - duration: 固定时长（秒）
    - num_step: 扩散步数 (4-64)
    - guidance_scale: 引导强度 (0.0-5.0)
    - t_shift: 时间偏移 (0.0-1.0)
    - layer_penalty_factor: 层惩罚 (0.0-10.0)
    - position_temperature: 位置温度 (0.0-10.0)
    - class_temperature: 类别温度 (0.0-5.0)
    - denoise: 是否去噪
    - preprocess_prompt: 是否预处理
    - postprocess_output: 是否后处理
    - instruct: Voice Design 指令
    """
    result = await run_in_threadpool(
        synthesize,
        text=text,
        voice_id=voice_id,
        language=language,
        speed=speed,
        duration=duration,
        num_step=num_step,
        guidance_scale=guidance_scale,
        t_shift=t_shift,
        layer_penalty_factor=layer_penalty_factor,
        position_temperature=position_temperature,
        class_temperature=class_temperature,
        denoise=denoise,
        preprocess_prompt=preprocess_prompt,
        postprocess_output=postprocess_output,
        audio_chunk_duration=audio_chunk_duration,
        audio_chunk_threshold=audio_chunk_threshold,
        instruct=instruct,
    )
    return _result_response(result)


@app.get("/api/output/list")
async def api_list_outputs():
    """合成音频列表接口"""
    outputs = await run_in_threadpool(list_outputs)
    return {
        "success": True,
        "outputs": outputs,
        "total": len(outputs),
    }


@app.get("/api/output/{filename}")
async def api_get_output(filename: str):
    """下载合成音频"""
    from config import OUTPUT_DIR

    try:
        file_path = resolve_file_in_dir(OUTPUT_DIR, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(file_path))


# ============ 启动 ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
