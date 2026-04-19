# -*- coding: utf-8 -*-
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as app_config
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import OUTPUT_DIR, PROJECT_ROOT
from core import analyze_reference_audio, clone_voice, list_outputs, list_voices, synthesize
from core.service_utils import clean_optional_text, resolve_file_in_dir


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


async def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = _validate_audio_suffix(upload)
    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"参考音频过大，请控制在 {MAX_UPLOAD_MB:g} MB 以内。")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return tmp.name


app = FastAPI(title="OmniVoice 语音克隆服务", version="1.1.0")

static_dir = PROJECT_ROOT / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/api/health")
async def api_health():
    return {
        "success": True,
        "service": "omnivoice-web",
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.post("/api/voice/analyze")
async def api_analyze_reference_audio(ref_audio: UploadFile = File(...)):
    tmp_path = await _save_upload_to_temp(ref_audio)

    try:
        report = await run_in_threadpool(analyze_reference_audio, tmp_path)
        return {
            "success": True,
            "quality_report": report,
            "message": report["summary"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"参考音频分析失败: {str(exc)}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        await ref_audio.close()


@app.post("/api/voice/clone")
async def api_clone_voice(
    voice_name: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
    rebuild: bool = Form(False),
):
    tmp_path = await _save_upload_to_temp(ref_audio)

    try:
        quality_report = None
        try:
            quality_report = await run_in_threadpool(analyze_reference_audio, tmp_path)
        except Exception:
            quality_report = None

        result = await run_in_threadpool(
            clone_voice,
            ref_audio=tmp_path,
            voice_name=voice_name,
            ref_text=ref_text,
            rebuild=rebuild,
        )
        if quality_report is not None:
            result["quality_report"] = quality_report
        return _result_response(result)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        await ref_audio.close()


@app.get("/api/voice/list")
async def api_list_voices():
    voices = await run_in_threadpool(list_voices)
    return {
        "success": True,
        "voices": voices,
        "total": len(voices),
    }


@app.post("/api/synthesize")
async def api_synthesize(
    text: str = Form(...),
    voice_id: Optional[str] = Form(None),
    language: str = Form("Chinese"),
    speed: float = Form(1.0),
    duration: Optional[float] = Form(None),
    num_step: int = Form(32),
    guidance_scale: float = Form(2.0),
    t_shift: float = Form(0.1),
    layer_penalty_factor: float = Form(5.0),
    position_temperature: float = Form(5.0),
    class_temperature: float = Form(0.0),
    denoise: bool = Form(True),
    preprocess_prompt: bool = Form(True),
    postprocess_output: bool = Form(True),
    audio_chunk_duration: float = Form(15.0),
    audio_chunk_threshold: float = Form(30.0),
    auto_prosody: bool = Form(True),
    auto_prosody_debug: bool = Form(False),
    instruct: Optional[str] = Form(None),
):
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
        auto_prosody=auto_prosody,
        auto_prosody_debug=auto_prosody_debug,
        instruct=instruct,
    )
    return _result_response(result)


@app.get("/api/output/list")
async def api_list_outputs():
    outputs = await run_in_threadpool(list_outputs)
    return {
        "success": True,
        "outputs": outputs,
        "total": len(outputs),
    }


@app.get("/api/output/{filename}")
async def api_get_output(filename: str):
    try:
        file_path = resolve_file_in_dir(OUTPUT_DIR, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在。")
    return FileResponse(str(file_path))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
