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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core import clone_voice, synthesize, list_voices, list_outputs
from config import PROJECT_ROOT

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


# ============ 音色克隆接口 ============

@app.post("/api/voice/clone")
async def api_clone_voice(
    voice_name: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
):
    """
    克隆音色接口

    - voice_name: 音色名称（用于标识）
    - ref_audio: 参考音频文件（支持 mp3, wav, m4a 等）
    - ref_text: 参考音频文本（可选）
    """
    # 保存上传的音频文件到临时目录
    suffix = Path(ref_audio.filename).suffix if ref_audio.filename else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await ref_audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 调用克隆函数
        result = clone_voice(
            ref_audio=tmp_path,
            voice_name=voice_name,
            ref_text=ref_text,
        )
        return result
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/api/voice/list")
async def api_list_voices():
    """音色列表接口"""
    voices = list_voices()
    return {
        "success": True,
        "voices": voices,
        "total": len(voices),
    }


# ============ 语音合成接口 ============

@app.post("/api/synthesize")
async def api_synthesize(
    text: str = Form(...),
    voice_id: str = Form(...),
    language: str = Form("Chinese"),
    # Speed & Duration
    speed: float = Form(1.0),
    duration: Optional[float] = Form(None),
    # Quality
    num_step: int = Form(32),
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
    result = synthesize(
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
        instruct=instruct,
    )
    return result


@app.get("/api/output/list")
async def api_list_outputs():
    """合成音频列表接口"""
    outputs = list_outputs()
    return {
        "success": True,
        "outputs": outputs,
        "total": len(outputs),
    }


@app.get("/api/output/{filename}")
async def api_get_output(filename: str):
    """下载合成音频"""
    from config import OUTPUT_DIR
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(file_path)


# ============ 启动 ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
