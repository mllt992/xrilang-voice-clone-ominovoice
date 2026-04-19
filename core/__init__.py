# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
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

from core.audio_quality import analyze_reference_audio
from core.expressive_text import build_auto_prosody_plan
from core.service_utils import (
    build_output_filename,
    clean_optional_text,
    resolve_file_in_dir,
    validate_voice_name,
)
from core.voice_clone_prompt import VoiceClonePrompt
from config import FFMPEG_CANDIDATES, MODEL_NAME, OUTPUT_DIR, SUPPORTED_MODELS, VOICES_DIR


# 模型实例缓存
_model_instances: dict[str, object] = {}
_model_lock = threading.RLock()

MAX_TEXT_LENGTH = int(getattr(app_config, "MAX_TEXT_LENGTH", 3000))
DEFAULT_AUDIO_CHUNK_DURATION = float(getattr(app_config, "DEFAULT_AUDIO_CHUNK_DURATION", 15.0))
DEFAULT_AUDIO_CHUNK_THRESHOLD = float(getattr(app_config, "DEFAULT_AUDIO_CHUNK_THRESHOLD", 30.0))
DEFAULT_AUTO_PROSODY = bool(getattr(app_config, "DEFAULT_AUTO_PROSODY", True))


def _ensure_ffmpeg() -> None:
    for ffmpeg_path in FFMPEG_CANDIDATES:
        if ffmpeg_path.exists():
            os.environ["PATH"] = str(ffmpeg_path.parent) + os.pathsep + os.environ.get("PATH", "")
            logging.info("Using ffmpeg: %s", ffmpeg_path)
            return
    logging.warning("ffmpeg not found in configured paths.")


def _ensure_tf32() -> None:
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


def get_model(model_name: Optional[str] = None) -> object:
    """获取指定名称的模型实例"""
    if model_name is None:
        model_name = MODEL_NAME

    if model_name in _model_instances:
        return _model_instances[model_name]

    with _model_lock:
        if model_name in _model_instances:
            return _model_instances[model_name]

        _ensure_tf32()
        _ensure_ffmpeg()

        model_config = SUPPORTED_MODELS.get(model_name)
        if model_config is None:
            raise ValueError(f"Unsupported model: {model_name}")

        device = _get_best_device()
        dtype = _get_model_dtype(device)
        device_map_strategy = "auto" if device == "cuda" else device

        model_class = model_config["class"]
        model_module = model_config["module"]
        local_path = model_config.get("local_path")

        if model_module == "omnivoice":
            logging.info(
                "Loading OmniVoice on %s with dtype=%s, device_map=%s",
                device,
                dtype,
                device_map_strategy,
            )
            model = OmniVoice.from_pretrained(
                local_path or model_name,
                device_map=device_map_strategy,
                dtype=dtype,
                load_asr=True,
                low_cpu_mem_usage=True,
            )
        elif model_module == "voxcpm":
            logging.info(
                "Loading VoxCPM2 on %s with dtype=%s",
                device,
                dtype,
            )
            import voxcpm
            VoxCPM = getattr(voxcpm, model_class)
            model = VoxCPM.from_pretrained(
                local_path or model_name,
                load_denoiser=False,
            )
        else:
            raise ValueError(f"Unknown model module: {model_module}")

        _model_instances[model_name] = model
        logging.info("Model %s loaded successfully", model_name)
        return model


def get_model_info(model_name: Optional[str] = None) -> dict:
    """获取模型信息"""
    if model_name is None:
        model_name = MODEL_NAME
    config = SUPPORTED_MODELS.get(model_name, {})
    return {
        "model_name": model_name,
        "name": config.get("name", model_name),
        "description": config.get("description", ""),
        "is_loaded": model_name in _model_instances,
    }


def _build_generation_config(
    *,
    num_step: int,
    guidance_scale: float,
    t_shift: float,
    layer_penalty_factor: float,
    position_temperature: float,
    class_temperature: float,
    denoise: bool,
    preprocess_prompt: bool,
    postprocess_output: bool,
    audio_chunk_duration: float,
    audio_chunk_threshold: float,
):
    from omnivoice import OmniVoiceGenerationConfig

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
        return OmniVoiceGenerationConfig(**config_kwargs)
    except TypeError:
        config_kwargs.pop("audio_chunk_duration", None)
        config_kwargs.pop("audio_chunk_threshold", None)
        return OmniVoiceGenerationConfig(**config_kwargs)


def _build_generation_kwargs(
    *,
    text: str,
    language: str,
    generation_config,
    speed: float,
    duration: Optional[float],
    voice_prompt: Optional[VoiceClonePrompt],
    instruct: Optional[str],
) -> dict:
    kwargs = {
        "text": text,
        "language": language,
        "generation_config": generation_config,
    }

    if speed != 1.0:
        kwargs["speed"] = speed
    if duration is not None:
        kwargs["duration"] = duration
    if voice_prompt is not None:
        kwargs["voice_clone_prompt"] = voice_prompt
    if instruct:
        kwargs["instruct"] = instruct

    return kwargs


def _generate_audio_array(
    model: OmniVoice,
    *,
    text: str,
    language: str,
    generation_config,
    speed: float,
    duration: Optional[float],
    voice_prompt: Optional[VoiceClonePrompt],
    instruct: Optional[str],
) -> np.ndarray:
    kwargs = _build_generation_kwargs(
        text=text,
        language=language,
        generation_config=generation_config,
        speed=speed,
        duration=duration,
        voice_prompt=voice_prompt,
        instruct=instruct,
    )
    audios = model.generate(**kwargs)
    return np.asarray(audios[0], dtype=np.float32)


def _generate_audio_array_voxcpm(
    model,
    *,
    text: str,
    reference_wav_path: Optional[str],
    instruct: Optional[str],
    cfg_value: float,
    inference_timesteps: int,
) -> np.ndarray:
    """VoxCPM2 的音频生成函数"""
    # 构建文本（包含 instruct 指令）
    if instruct:
        full_text = f"({instruct}){text}"
    else:
        full_text = text

    gen_kwargs = {
        "text": full_text,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
    }

    if reference_wav_path:
        gen_kwargs["reference_wav_path"] = reference_wav_path

    audio = model.generate(**gen_kwargs)
    # VoxCPM2 generate() 直接返回音频数组
    return np.asarray(audio, dtype=np.float32)


def _light_trim_audio_edges(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -48.0,
    padding_ms: int = 20,
) -> np.ndarray:
    if audio.size <= 1:
        return audio

    amplitude = np.abs(audio)
    smoothing_window = max(1, int(sample_rate * 0.01))
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window, dtype=np.float32) / smoothing_window
        amplitude = np.convolve(amplitude, kernel, mode="same")

    threshold = float(10 ** (threshold_db / 20.0))
    active = np.flatnonzero(amplitude >= threshold)
    if active.size == 0:
        return audio

    padding = int(sample_rate * padding_ms / 1000.0)
    start = max(0, int(active[0]) - padding)
    end = min(audio.size, int(active[-1]) + padding + 1)
    return np.asarray(audio[start:end], dtype=np.float32)


def _append_silence(chunks: list[np.ndarray], sample_rate: int, pause_ms: int) -> None:
    if pause_ms <= 0:
        return

    silence_length = max(1, int(sample_rate * pause_ms / 1000.0))
    chunks.append(np.zeros(silence_length, dtype=np.float32))


def _resolve_voice_mode(has_voice_prompt: bool, instruct: Optional[str], auto_prosody_used: bool) -> str:
    voice_mode = "auto_voice"
    if has_voice_prompt:
        voice_mode = "voice_clone"
    if instruct:
        voice_mode = "voice_clone+instruct" if has_voice_prompt else "voice_design"
    if auto_prosody_used:
        voice_mode = f"{voice_mode}+auto_prosody"
    return voice_mode


def clone_voice(
    ref_audio: str | Path,
    voice_name: str,
    ref_text: Optional[str] = None,
    rebuild: bool = False,
    model_name: Optional[str] = None,
) -> dict:
    if model_name is None:
        model_name = MODEL_NAME

    model_config = SUPPORTED_MODELS.get(model_name, {})
    model_module = model_config.get("module")

    # VoxCPM2 不需要单独的音色克隆步骤，直接在 generate 中使用参考音频
    if model_module == "voxcpm":
        return {
            "success": False,
            "status_code": 400,
            "voice_id": clean_optional_text(voice_name) or "",
            "voice_name": clean_optional_text(voice_name) or "",
            "pt_path": "",
            "message": "VoxCPM2 不需要预先克隆音色，请直接在合成时选择参考音频。",
        }

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
            model = get_model(model_name)
            logging.info("Creating voice prompt from: %s", ref_audio)
            raw_prompt = model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),
                ref_text=ref_text,
                preprocess_prompt=True,
            )
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
    except Exception as exc:
        logging.error("Voice clone failed: %s", str(exc))
        return {
            "success": False,
            "status_code": 500,
            "voice_id": voice_name,
            "voice_name": voice_name,
            "pt_path": str(pt_path),
            "message": f"音色克隆失败: {str(exc)}",
        }


def synthesize(
    text: str,
    voice_id: Optional[str] = None,
    language: str = "Chinese",
    output_filename: Optional[str] = None,
    speed: float = 1.0,
    duration: Optional[float] = None,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    t_shift: float = 0.1,
    layer_penalty_factor: float = 5.0,
    position_temperature: float = 5.0,
    class_temperature: float = 0.0,
    denoise: bool = True,
    preprocess_prompt: bool = True,
    postprocess_output: bool = True,
    audio_chunk_duration: float = DEFAULT_AUDIO_CHUNK_DURATION,
    audio_chunk_threshold: float = DEFAULT_AUDIO_CHUNK_THRESHOLD,
    auto_prosody: bool = DEFAULT_AUTO_PROSODY,
    auto_prosody_debug: bool = False,
    instruct: Optional[str] = None,
    model_name: Optional[str] = None,
    reference_wav_path: Optional[str] = None,
) -> dict:
    if model_name is None:
        model_name = MODEL_NAME

    model_config = SUPPORTED_MODELS.get(model_name, {})
    model_module = model_config.get("module")

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
        gen_config = _build_generation_config(
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
        pause_safe_gen_config = None

        voice_prompt: Optional[VoiceClonePrompt] = None
        if normalized_voice_id:
            voice_prompt = VoiceClonePrompt.load(pt_path)

        audio_path.parent.mkdir(parents=True, exist_ok=True)

        auto_prosody_used = False
        auto_prosody_reason = ""
        auto_prosody_plan = None

        with _model_lock:
            model = get_model(model_name)

            # VoxCPM2 使用不同的生成逻辑
            if model_module == "voxcpm":
                # VoxCPM2 简化处理：不支持 auto_prosody 和 voice_prompt
                if auto_prosody:
                    auto_prosody_reason = "VoxCPM2 不支持自动韵律功能"
                    auto_prosody_used = False

                # 获取参考音频路径（优先使用 reference_wav_path，否则从 voice_id 获取）
                ref_wav = reference_wav_path
                if not ref_wav and normalized_voice_id:
                    # OmniVoice 音色文件不适用于 VoxCPM2
                    auto_prosody_reason = "VoxCPM2 不支持 OmniVoice 音色文件，请使用参考音频"
                    ref_wav = None

                audio = _generate_audio_array_voxcpm(
                    model,
                    text=text,
                    reference_wav_path=ref_wav,
                    instruct=instruct,
                    cfg_value=guidance_scale,
                    inference_timesteps=num_step,
                )
                sample_rate = model.tts_model.sample_rate
                voice_mode = "voice_clone" if ref_wav else "voice_design"

                audio_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(audio_path, audio, sample_rate)

                message = "语音合成成功"
                if auto_prosody_reason:
                    message += f"（自动韵律未生效：{auto_prosody_reason}）"

                return {
                    "success": True,
                    "status_code": 200,
                    "mode": voice_mode,
                    "text": text,
                    "voice_id": normalized_voice_id or "",
                    "audio_path": str(audio_path),
                    "sample_rate": sample_rate,
                    "message": message,
                    "auto_prosody": auto_prosody,
                    "auto_prosody_used": False,
                    "auto_prosody_reason": auto_prosody_reason,
                }

            if auto_prosody and duration is None:
                auto_prosody_plan = build_auto_prosody_plan(
                    text=text,
                    language=language,
                    base_speed=speed,
                    base_instruct=instruct,
                )
                segments = auto_prosody_plan.segments
                if auto_prosody_plan.preserve_pauses and postprocess_output:
                    pause_safe_gen_config = _build_generation_config(
                        num_step=num_step,
                        guidance_scale=guidance_scale,
                        t_shift=t_shift,
                        layer_penalty_factor=layer_penalty_factor,
                        position_temperature=position_temperature,
                        class_temperature=class_temperature,
                        denoise=denoise,
                        preprocess_prompt=preprocess_prompt,
                        postprocess_output=False,
                        audio_chunk_duration=audio_chunk_duration,
                        audio_chunk_threshold=audio_chunk_threshold,
                    )
                needs_segment_render = (
                    len(segments) > 1
                    or any(segment.injected_tag for segment in segments)
                    or any(abs(segment.speed - speed) >= 0.02 for segment in segments)
                    or any((segment.instruct_hint or "") != (instruct or "") for segment in segments)
                    or any(segment.preserve_silence for segment in segments)
                )

                if needs_segment_render:
                    audio_chunks: list[np.ndarray] = []
                    for index, segment in enumerate(segments):
                        logging.info(
                            "Auto prosody segment %s/%s style=%s speed=%.3f text=%s",
                            index + 1,
                            len(segments),
                            segment.style,
                            segment.speed,
                            segment.render_text[:60] + "..." if len(segment.render_text) > 60 else segment.render_text,
                        )
                        segment_config = (
                            pause_safe_gen_config
                            if pause_safe_gen_config is not None and segment.preserve_silence
                            else gen_config
                        )
                        segment_audio = _generate_audio_array(
                            model,
                            text=segment.render_text,
                            language=language,
                            generation_config=segment_config,
                            speed=segment.speed,
                            duration=None,
                            voice_prompt=voice_prompt,
                            instruct=segment.instruct_hint,
                        )
                        if pause_safe_gen_config is not None and segment.preserve_silence:
                            segment_audio = _light_trim_audio_edges(segment_audio, model.sampling_rate)

                        audio_chunks.append(segment_audio)
                        if index < len(segments) - 1:
                            _append_silence(audio_chunks, model.sampling_rate, segment.pause_ms)

                    audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(1, dtype=np.float32)
                    auto_prosody_used = True
                else:
                    auto_prosody_reason = "文本不需要短语级韵律调整。"
                    audio = _generate_audio_array(
                        model,
                        text=auto_prosody_plan.normalized_text,
                        language=language,
                        generation_config=gen_config,
                        speed=speed,
                        duration=None,
                        voice_prompt=voice_prompt,
                        instruct=instruct,
                    )
            else:
                if auto_prosody and duration is not None:
                    auto_prosody_reason = "固定时长模式下不会启用自动韵律。"
                audio = _generate_audio_array(
                    model,
                    text=text,
                    language=language,
                    generation_config=gen_config,
                    speed=speed,
                    duration=duration,
                    voice_prompt=voice_prompt,
                    instruct=instruct,
                )

        voice_mode = _resolve_voice_mode(
            has_voice_prompt=voice_prompt is not None,
            instruct=instruct,
            auto_prosody_used=auto_prosody_used,
        )
        message = "语音合成成功"
        if auto_prosody_used and auto_prosody_plan is not None:
            message += f"（已启用自动韵律，{len(auto_prosody_plan.segments)} 段短语规划）"
        elif auto_prosody_reason:
            message += f"（自动韵律未生效：{auto_prosody_reason}）"

        sf.write(audio_path, audio, model.sampling_rate)

        response = {
            "success": True,
            "status_code": 200,
            "mode": voice_mode,
            "text": text,
            "voice_id": normalized_voice_id or "",
            "audio_path": str(audio_path),
            "sample_rate": model.sampling_rate,
            "message": message,
            "auto_prosody": auto_prosody,
            "auto_prosody_used": auto_prosody_used,
            "auto_prosody_reason": auto_prosody_reason,
        }
        if auto_prosody_debug and auto_prosody_plan is not None:
            response["auto_prosody_plan"] = auto_prosody_plan.to_dict()

        return response
    except Exception as exc:
        logging.error("Synthesis failed: %s", str(exc))
        return {
            "success": False,
            "status_code": 500,
            "text": text,
            "voice_id": normalized_voice_id or "",
            "audio_path": "",
            "sample_rate": 0,
            "message": f"语音合成失败: {str(exc)}",
        }


def list_voices() -> list:
    voices = []
    for pt_file in VOICES_DIR.glob("*.pt"):
        stat = pt_file.stat()
        voices.append(
            {
                "voice_id": pt_file.stem,
                "voice_name": pt_file.stem,
                "pt_path": str(pt_file),
                "created_time": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                "sort_key": stat.st_ctime,
            }
        )

    return [
        {key: value for key, value in voice.items() if key != "sort_key"}
        for voice in sorted(voices, key=lambda item: item["sort_key"], reverse=True)
    ]


def list_outputs() -> list:
    outputs = []
    for wav_file in OUTPUT_DIR.glob("*.wav"):
        stat = wav_file.stat()
        outputs.append(
            {
                "filename": wav_file.name,
                "file_path": str(wav_file),
                "size": stat.st_size,
                "created_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "sort_key": stat.st_mtime,
            }
        )

    return [
        {key: value for key, value in output.items() if key != "sort_key"}
        for output in sorted(outputs, key=lambda item: item["sort_key"], reverse=True)
    ]
