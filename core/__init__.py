# -*- coding: utf-8 -*-
"""
语音克隆核心库
依赖: pip install omnivoice
"""
from __future__ import annotations

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch

# 先导入 config 获取 HF_TOKEN
from config import HF_TOKEN

# 设置 HF_TOKEN（必须在导入 omnivoice 之前）
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
else:
    logging.warning("HF_TOKEN not set in config.py. Download may be slow or fail.")

from omnivoice import OmniVoice, VoiceClonePrompt

from config import (
    MODEL_NAME,
    VOICES_DIR,
    OUTPUT_DIR,
    FFMPEG_CANDIDATES,
)

# 全局模型实例（单例模式）
_model_instance: Optional[OmniVoice] = None


def _ensure_ffmpeg() -> None:
    """设置 ffmpeg 到 PATH"""
    for ffmpeg_path in FFMPEG_CANDIDATES:
        if ffmpeg_path.exists():
            os.environ["PATH"] = str(ffmpeg_path.parent) + os.pathsep + os.environ.get("PATH", "")
            logging.info("Using ffmpeg: %s", ffmpeg_path)
            return
    logging.warning("ffmpeg not found in configured paths.")


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

    _ensure_ffmpeg()

    device = _get_best_device()
    dtype = _get_model_dtype(device)

    logging.info("Loading OmniVoice on %s with dtype=%s", device, dtype)
    _model_instance = OmniVoice.from_pretrained(
        MODEL_NAME,
        device_map=device,
        dtype=dtype,
        load_asr=True,
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
    ref_audio = Path(ref_audio)

    if not ref_audio.exists():
        return {
            "success": False,
            "voice_id": voice_name,
            "voice_name": voice_name,
            "pt_path": "",
            "message": f"参考音频文件不存在: {ref_audio}",
        }

    pt_path = VOICES_DIR / f"{voice_name}.pt"
    model = get_model()

    try:
        if pt_path.exists() and not rebuild:
            logging.info("Loading cached voice prompt: %s", pt_path)
            prompt = VoiceClonePrompt.load(pt_path)
        else:
            logging.info("Creating voice prompt from: %s", ref_audio)
            prompt = model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),
                ref_text=ref_text,
                preprocess_prompt=True,
            )
            prompt.save(pt_path)
            logging.info("Voice prompt saved to: %s", pt_path)

        return {
            "success": True,
            "voice_id": voice_name,
            "voice_name": voice_name,
            "pt_path": str(pt_path),
            "message": "音色克隆成功",
        }
    except Exception as e:
        logging.error("Voice clone failed: %s", str(e))
        return {
            "success": False,
            "voice_id": voice_name,
            "voice_name": voice_name,
            "pt_path": str(pt_path),
            "message": f"音色克隆失败: {str(e)}",
        }


def synthesize(
    text: str,
    voice_id: str,
    language: str = "Chinese",
    output_filename: Optional[str] = None,
    speed: float = 1.0,
    num_step: int = 32,
    guidance_scale: float = 2.0,
) -> dict:
    """
    使用指定音色合成语音

    Args:
        text: 要合成的文本
        voice_id: 音色 ID（对应 voices/ 目录下的 .pt 文件）
        language: 语言
        output_filename: 输出文件名（None = 自动生成带时间戳的文件名）
        speed: 语速
        num_step: 扩散步数
        guidance_scale: 引导 scale

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
    pt_path = VOICES_DIR / f"{voice_id}.pt"

    if not pt_path.exists():
        return {
            "success": False,
            "text": text,
            "voice_id": voice_id,
            "audio_path": "",
            "sample_rate": 0,
            "message": f"音色文件不存在: {voice_id}.pt",
        }

    try:
        voice_prompt = VoiceClonePrompt.load(pt_path)
        model = get_model()

        logging.info("Generating audio: %s", text[:50] + "..." if len(text) > 50 else text)
        audios = model.generate(
            text=text,
            language=language,
            voice_clone_prompt=voice_prompt,
            speed=speed,
            num_step=num_step,
            guidance_scale=guidance_scale,
        )

        # 生成输出文件名
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"synth_{voice_id}_{timestamp}.wav"

        audio_path = OUTPUT_DIR / output_filename
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, audios[0], model.sampling_rate)

        return {
            "success": True,
            "text": text,
            "voice_id": voice_id,
            "audio_path": str(audio_path),
            "sample_rate": model.sampling_rate,
            "message": "语音合成成功",
        }
    except Exception as e:
        logging.error("Synthesis failed: %s", str(e))
        return {
            "success": False,
            "text": text,
            "voice_id": voice_id,
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
        })
    return voices


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
            "created_time": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return sorted(outputs, key=lambda x: x["created_time"], reverse=True)
