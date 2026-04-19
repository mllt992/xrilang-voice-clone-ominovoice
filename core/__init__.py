# -*- coding: utf-8 -*-
"""
语音克隆核心库
依赖: pip install omnivoice (从 pip 安装即可，VoiceClonePrompt 使用本地兼容版本)
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch

import config as app_config


def _load_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or getattr(app_config, "HF_TOKEN", None)
    if not token:
        return None

    token = token.strip()
    if not token or token == "hf_your_token_here":
        return None

    return token


HF_TOKEN = _load_hf_token()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
else:
    logging.warning("HF_TOKEN not set. Model download may fail or be slow.")

from omnivoice import OmniVoice

# VoiceClonePrompt 使用本地兼容版本（pip 版本未导出）
from core.voice_clone_prompt import VoiceClonePrompt
from core.service_utils import (
    build_output_filename,
    clean_optional_text,
    resolve_file_in_dir,
    validate_voice_name,
)

from config import (
    MODEL_NAME,
    VOICES_DIR,
    OUTPUT_DIR,
    FFMPEG_CANDIDATES,
)

# 全局模型实例（单例模式）
_model_instance: Optional[OmniVoice] = None
_model_lock = threading.RLock()

MAX_TEXT_LENGTH = int(getattr(app_config, "MAX_TEXT_LENGTH", 3000))
DEFAULT_AUDIO_CHUNK_DURATION = float(getattr(app_config, "DEFAULT_AUDIO_CHUNK_DURATION", 15.0))
DEFAULT_AUDIO_CHUNK_THRESHOLD = float(getattr(app_config, "DEFAULT_AUDIO_CHUNK_THRESHOLD", 30.0))


def _ensure_ffmpeg() -> None:
    """设置 ffmpeg 到 PATH"""
    for ffmpeg_path in FFMPEG_CANDIDATES:
        if ffmpeg_path.exists():
            os.environ["PATH"] = str(ffmpeg_path.parent) + os.pathsep + os.environ.get("PATH", "")
            logging.info("Using ffmpeg: %s", ffmpeg_path)
            return
    logging.warning("ffmpeg not found in configured paths.")


def _ensure_tf32() -> None:
    """Enable TF32 for faster matrix operations on Ampere+ GPUs."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("TF32 enabled for CUDA acceleration")


def _get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_model_dtype(device: str) -> torch.dtype:
    return torch.float16 if device != "cpu" else torch.float32


def get_model() -> OmniVoice:
    """
    获取全局模型实例（单例模式）
    """
    global _model_instance

    if _model_instance is not None:
        return _model_instance

    with _model_lock:
        if _model_instance is not None:
            return _model_instance

        _ensure_tf32()
        _ensure_ffmpeg()

        device = _get_best_device()
        dtype = _get_model_dtype(device)

        # device_map="auto" 会自动在多设备间分配模型层
        device_map_strategy = "auto" if device == "cuda" else device

        logging.info("Loading OmniVoice on %s with dtype=%s, device_map=%s", device, dtype, device_map_strategy)
        _model_instance = OmniVoice.from_pretrained(
            MODEL_NAME,
            device_map=device_map_strategy,
            dtype=dtype,
            load_asr=True,
            low_cpu_mem_usage=True,
        )
        logging.info("Model loaded successfully!")

        return _model_instance


def clone_voice(
    ref_audio: str | Path,
    voice_name: str,
    ref_text: Optional[str] = None,
    rebuild: bool = False,
) -> dict:
    """
    从参考音频克隆音色

    Args:
        ref_audio: 参考音频文件路径
        voice_name: 音色名称（用于保存 .pt 文件）
        ref_text: 参考音频文本（None = 自动转录）
        rebuild: 是否强制重建

    Returns:
        {
            "success": bool,
            "voice_id": str,
            "voice_name": str,
            "pt_path": str,
            "message": str
        }
    """
    try:
        voice_name = validate_voice_name(voice_name)
    except ValueError as exc:
        return {
            "success": False,
            "status_code": 400,
            "voice_id": clean_optional_text(voice_name) or "",
            "voice_name": clean_optional_text(voice_name) or "",
            "pt_path": "",
            "message": str(exc),
        }

    ref_audio = Path(ref_audio)
    ref_text = clean_optional_text(ref_text)

    if not ref_audio.exists():
        return {
            "success": False,
            "status_code": 400,
            "voice_id": voice_name,
            "voice_name": voice_name,
            "pt_path": "",
            "message": f"参考音频文件不存在: {ref_audio}",
        }

    pt_path = VOICES_DIR / f"{voice_name}.pt"

    try:
        if pt_path.exists() and not rebuild:
            logging.info("Loading cached voice prompt: %s", pt_path)
            VoiceClonePrompt.load(pt_path)
            return {
                "success": True,
                "status_code": 200,
                "cached": True,
                "voice_id": voice_name,
                "voice_name": voice_name,
                "pt_path": str(pt_path),
                "message": "音色已存在，已复用缓存。勾选“强制重建”可重新编码参考音频。",
            }

        with _model_lock:
            model = get_model()
            logging.info("Creating voice prompt from: %s", ref_audio)
            raw_prompt = model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),
                ref_text=ref_text,
                preprocess_prompt=True,
            )
            # 转换为本地 VoiceClonePrompt（支持 save 方法）
            prompt = VoiceClonePrompt(
                ref_audio_tokens=raw_prompt.ref_audio_tokens,
                ref_text=raw_prompt.ref_text,
                ref_rms=raw_prompt.ref_rms,
            )
            prompt.save(pt_path)
            logging.info("Voice prompt saved to: %s", pt_path)

        return {
            "success": True,
            "status_code": 200,
            "cached": False,
            "voice_id": voice_name,
            "voice_name": voice_name,
            "pt_path": str(pt_path),
            "message": "音色克隆成功",
        }
    except Exception as e:
        logging.error("Voice clone failed: %s", str(e))
        return {
            "success": False,
            "status_code": 500,
            "voice_id": voice_name,
            "voice_name": voice_name,
            "pt_path": str(pt_path),
            "message": f"音色克隆失败: {str(e)}",
        }


def synthesize(
    text: str,
    voice_id: Optional[str] = None,
    language: str = "Chinese",
    output_filename: Optional[str] = None,
    # Speed & Duration
    speed: float = 1.0,
    duration: Optional[float] = None,
    # Quality
    num_step: int = 16,
    guidance_scale: float = 2.0,
    # Advanced
    t_shift: float = 0.1,
    layer_penalty_factor: float = 5.0,
    position_temperature: float = 5.0,
    class_temperature: float = 0.0,
    # Options
    denoise: bool = True,
    preprocess_prompt: bool = True,
    postprocess_output: bool = True,
    audio_chunk_duration: float = DEFAULT_AUDIO_CHUNK_DURATION,
    audio_chunk_threshold: float = DEFAULT_AUDIO_CHUNK_THRESHOLD,
    # Voice Design (can be combined with voice clone prompt)
    instruct: Optional[str] = None,
) -> dict:
    """
    使用指定音色合成语音

    Args:
        text: 要合成的文本
        voice_id: 音色 ID（可选，对应 voices/ 目录下的 .pt 文件）
        language: 语言
        output_filename: 输出文件名（None = 自动生成带时间戳的文件名）
        speed: 语速 (0.5-2.0, 1.0=正常, >1加速, <1减速)
        duration: 固定时长（秒），优先级高于 speed
        num_step: 扩散步数 (4-64, 越大质量越好越慢)
        guidance_scale: 引导强度 (0.0-5.0, 越高越符合描述)
        t_shift: 时间偏移 (0.0-1.0, 影响语速变化感)
        layer_penalty_factor: 层惩罚 (0.0-10.0, 影响声音层次感)
        position_temperature: 位置温度 (0.0-10.0, 越高越随机)
        class_temperature: 类别温度 (0.0-5.0, 越高越随机)
        denoise: 是否去噪
        preprocess_prompt: 是否预处理参考音频
        postprocess_output: 是否后处理输出
        audio_chunk_duration: 长文本分段目标时长
        audio_chunk_threshold: 长文本启用分段的阈值
        instruct: Voice Design 指令（可与 voice clone prompt 组合）

    Returns:
        {
            "success": bool,
            "text": str,
            "voice_id": str,
            "audio_path": str,
            "sample_rate": int,
            "message": str
        }
    """
    from omnivoice import OmniVoiceGenerationConfig

    text = clean_optional_text(text)
    instruct = clean_optional_text(instruct)
    language = clean_optional_text(language) or "Chinese"

    if not text:
        return {
            "success": False,
            "status_code": 400,
            "text": "",
            "voice_id": clean_optional_text(voice_id) or "",
            "audio_path": "",
            "sample_rate": 0,
            "message": "待合成文本不能为空。",
        }

    if len(text) > MAX_TEXT_LENGTH:
        return {
            "success": False,
            "status_code": 400,
            "text": text,
            "voice_id": clean_optional_text(voice_id) or "",
            "audio_path": "",
            "sample_rate": 0,
            "message": f"待合成文本过长，请控制在 {MAX_TEXT_LENGTH} 个字符以内。",
        }

    normalized_voice_id: Optional[str] = None
    pt_path: Optional[Path] = None
    if clean_optional_text(voice_id):
        try:
            normalized_voice_id = validate_voice_name(voice_id or "")
        except ValueError as exc:
            return {
                "success": False,
                "status_code": 400,
                "text": text,
                "voice_id": clean_optional_text(voice_id) or "",
                "audio_path": "",
                "sample_rate": 0,
                "message": str(exc),
            }

        pt_path = VOICES_DIR / f"{normalized_voice_id}.pt"
        if not pt_path.exists():
            return {
                "success": False,
                "status_code": 404,
                "text": text,
                "voice_id": normalized_voice_id,
                "audio_path": "",
                "sample_rate": 0,
                "message": f"音色文件不存在: {normalized_voice_id}.pt",
            }

    if output_filename is None:
        audio_path = OUTPUT_DIR / build_output_filename(normalized_voice_id)
    else:
        try:
            audio_path = resolve_file_in_dir(OUTPUT_DIR, output_filename)
        except ValueError as exc:
            return {
                "success": False,
                "status_code": 400,
                "text": text,
                "voice_id": normalized_voice_id or "",
                "audio_path": "",
                "sample_rate": 0,
                "message": str(exc),
            }

    try:
        config_kwargs = dict(
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
        )

        try:
            gen_config = OmniVoiceGenerationConfig(**config_kwargs)
        except TypeError:
            config_kwargs.pop("audio_chunk_duration", None)
            config_kwargs.pop("audio_chunk_threshold", None)
            gen_config = OmniVoiceGenerationConfig(**config_kwargs)

        kw = dict(
            text=text,
            language=language,
            generation_config=gen_config,
        )

        if speed != 1.0:
            kw["speed"] = speed
        if duration is not None:
            kw["duration"] = duration

        voice_mode = "auto_voice"
        if normalized_voice_id:
            voice_prompt = VoiceClonePrompt.load(pt_path)
            kw["voice_clone_prompt"] = voice_prompt
            voice_mode = "voice_clone"
        if instruct:
            kw["instruct"] = instruct
            voice_mode = "voice_clone+instruct" if normalized_voice_id else "voice_design"

        audio_path.parent.mkdir(parents=True, exist_ok=True)
        with _model_lock:
            model = get_model()
            logging.info("Generating audio with mode=%s: %s", voice_mode, text[:50] + "..." if len(text) > 50 else text)
            audios = model.generate(**kw)
        sf.write(audio_path, audios[0], model.sampling_rate)

        return {
            "success": True,
            "status_code": 200,
            "mode": voice_mode,
            "text": text,
            "voice_id": normalized_voice_id or "",
            "audio_path": str(audio_path),
            "sample_rate": model.sampling_rate,
            "message": "语音合成成功",
        }
    except Exception as e:
        logging.error("Synthesis failed: %s", str(e))
        return {
            "success": False,
            "status_code": 500,
            "text": text,
            "voice_id": normalized_voice_id or "",
            "audio_path": "",
            "sample_rate": 0,
            "message": f"语音合成失败: {str(e)}",
        }


def list_voices() -> list:
    """
    列出所有可用的音色

    Returns:
        [
            {
                "voice_id": str,
                "voice_name": str,
                "pt_path": str,
                "created_time": str
            }
        ]
    """
    voices = []
    for pt_file in VOICES_DIR.glob("*.pt"):
        voice_id = pt_file.stem
        stat = pt_file.stat()
        voices.append({
            "voice_id": voice_id,
            "voice_name": voice_id,
            "pt_path": str(pt_file),
            "created_time": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
            "sort_key": stat.st_ctime,
        })
    return [
        {k: v for k, v in voice.items() if k != "sort_key"}
        for voice in sorted(voices, key=lambda item: item["sort_key"], reverse=True)
    ]


def list_outputs() -> list:
    """
    列出所有合成的音频文件

    Returns:
        [
            {
                "filename": str,
                "file_path": str,
                "size": int,
                "created_time": str
            }
        ]
    """
    outputs = []
    for wav_file in OUTPUT_DIR.glob("*.wav"):
        stat = wav_file.stat()
        outputs.append({
            "filename": wav_file.name,
            "file_path": str(wav_file),
            "size": stat.st_size,
            "created_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "sort_key": stat.st_mtime,
        })
    return [
        {k: v for k, v in output.items() if k != "sort_key"}
        for output in sorted(outputs, key=lambda item: item["sort_key"], reverse=True)
    ]
