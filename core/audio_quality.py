from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from config import FFMPEG_CANDIDATES


def _ensure_audio_backend() -> None:
    if getattr(AudioSegment, "converter", None):
        converter = Path(str(AudioSegment.converter))
        if converter.exists():
            return

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
        return

    for candidate in FFMPEG_CANDIDATES:
        if candidate.exists():
            AudioSegment.converter = str(candidate)
            ffprobe = candidate.with_name("ffprobe.exe" if candidate.suffix.lower() == ".exe" else "ffprobe")
            if ffprobe.exists():
                AudioSegment.ffprobe = str(ffprobe)
            return


def _to_db(value: float, floor: float = -96.0) -> float:
    if value <= 0:
        return floor
    return max(floor, 20.0 * math.log10(value))


def _to_float_samples(segment: AudioSegment) -> np.ndarray:
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if samples.size == 0:
        return samples

    if segment.sample_width == 1 and float(np.min(samples)) >= 0.0:
        samples = samples - 128.0

    scale = float(1 << (8 * segment.sample_width - 1))
    return np.clip(samples / scale, -1.0, 1.0)


def _window_rms(samples: np.ndarray, sample_rate: int, window_ms: int = 50) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)

    window_size = max(1, int(sample_rate * window_ms / 1000))
    usable = samples.size // window_size
    if usable <= 0:
        return np.asarray([float(np.sqrt(np.mean(np.square(samples)) + 1e-12))], dtype=np.float32)

    trimmed = samples[: usable * window_size]
    windows = trimmed.reshape(usable, window_size)
    return np.sqrt(np.mean(np.square(windows), axis=1) + 1e-12)


def _leading_trailing_silence_ms(window_db: np.ndarray, threshold_db: float) -> tuple[int, int]:
    if window_db.size == 0:
        return 0, 0

    non_silent = np.where(window_db >= threshold_db)[0]
    if non_silent.size == 0:
        total_ms = int(window_db.size * 50)
        return total_ms, total_ms

    leading = int(non_silent[0] * 50)
    trailing = int((window_db.size - non_silent[-1] - 1) * 50)
    return leading, trailing


def _append_alert(
    alerts: list[dict],
    recommendations: list[str],
    *,
    severity: str,
    title: str,
    detail: str,
    recommendation: str | None = None,
) -> None:
    alerts.append(
        {
            "severity": severity,
            "title": title,
            "detail": detail,
        }
    )
    if recommendation and recommendation not in recommendations:
        recommendations.append(recommendation)


def analyze_reference_audio(audio_path: str | Path) -> dict:
    _ensure_audio_backend()

    path = Path(audio_path)
    segment = AudioSegment.from_file(path)

    original_channels = int(segment.channels)
    sample_rate = int(segment.frame_rate)
    duration_sec = float(len(segment)) / 1000.0

    mono_segment = segment.set_channels(1)
    samples = _to_float_samples(mono_segment)
    if samples.size == 0:
        raise ValueError("参考音频为空，无法分析。")

    abs_samples = np.abs(samples)
    peak = float(np.max(abs_samples))
    rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-12))
    peak_dbfs = _to_db(peak)
    rms_dbfs = _to_db(rms)
    clipping_ratio = float(np.mean(abs_samples >= 0.995))

    window_rms = _window_rms(samples, sample_rate=sample_rate)
    window_db = np.asarray([_to_db(float(value)) for value in window_rms], dtype=np.float32)
    silence_threshold_db = max(-50.0, min(-32.0, rms_dbfs - 18.0))
    silence_ratio = float(np.mean(window_db < silence_threshold_db)) if window_db.size else 0.0
    leading_silence_ms, trailing_silence_ms = _leading_trailing_silence_ms(window_db, silence_threshold_db)
    crest_factor_db = peak_dbfs - rms_dbfs

    score = 100.0
    alerts: list[dict] = []
    recommendations: list[str] = []

    if duration_sec < 3.0:
        penalty = min(32.0, (3.0 - duration_sec) * 14.0)
        score -= penalty
        _append_alert(
            alerts,
            recommendations,
            severity="critical" if duration_sec < 2.0 else "warning",
            title="参考音频偏短",
            detail=f"当前约 {duration_sec:.1f} 秒，建议准备 3 到 10 秒的连续语音。",
            recommendation="补一段更完整、更连贯的参考语音，优先选择 6 到 12 秒的清晰人声。",
        )
    elif duration_sec > 12.0:
        penalty = min(18.0, (duration_sec - 12.0) * 2.6)
        score -= penalty
        _append_alert(
            alerts,
            recommendations,
            severity="warning",
            title="参考音频偏长",
            detail=f"当前约 {duration_sec:.1f} 秒，过长样本容易混入多段情绪和无效停顿。",
            recommendation="裁掉前后空白和无关内容，保留 3 到 10 秒主体人声。",
        )

    if sample_rate < 16000:
        score -= 15.0
        _append_alert(
            alerts,
            recommendations,
            severity="critical",
            title="采样率偏低",
            detail=f"当前采样率为 {sample_rate} Hz，建议至少 16000 Hz，最好 22050 Hz 以上。",
            recommendation="尽量使用原始录音或更高质量导出的音频文件。",
        )
    elif sample_rate < 22050:
        score -= 6.0
        _append_alert(
            alerts,
            recommendations,
            severity="warning",
            title="采样率一般",
            detail=f"当前采样率为 {sample_rate} Hz，可用，但更高采样率通常更稳。",
        )

    if rms_dbfs < -27.0:
        penalty = min(18.0, (-27.0 - rms_dbfs) * 2.2)
        score -= penalty
        _append_alert(
            alerts,
            recommendations,
            severity="warning",
            title="整体音量偏低",
            detail=f"平均响度约 {rms_dbfs:.1f} dBFS，录音偏小会削弱说话人特征。",
            recommendation="提高录音输入音量，避免距离麦克风过远。",
        )

    if peak_dbfs < -6.0:
        penalty = min(10.0, (-6.0 - peak_dbfs) * 1.5)
        score -= penalty
        _append_alert(
            alerts,
            recommendations,
            severity="warning",
            title="峰值偏低",
            detail=f"峰值约 {peak_dbfs:.1f} dBFS，声音偏远或能量不足。",
        )

    if clipping_ratio > 0.002:
        penalty = min(28.0, clipping_ratio * 4000.0)
        score -= penalty
        _append_alert(
            alerts,
            recommendations,
            severity="critical",
            title="存在削波失真",
            detail=f"约 {clipping_ratio * 100:.2f}% 的采样点接近满幅，可能已经爆音。",
            recommendation="重新录制，避免输入增益过高和后期过度压限。",
        )

    if silence_ratio > 0.35:
        penalty = min(18.0, (silence_ratio - 0.35) * 50.0)
        score -= penalty
        _append_alert(
            alerts,
            recommendations,
            severity="warning",
            title="静音占比偏高",
            detail=f"静音窗口约占 {silence_ratio * 100:.1f}%，样本里可能有较多空白或停顿。",
            recommendation="裁掉长停顿、无关片段和环境空白，只保留主体语音。",
        )

    if leading_silence_ms > 700:
        penalty = min(12.0, (leading_silence_ms - 700) / 120.0)
        score -= penalty
        _append_alert(
            alerts,
            recommendations,
            severity="warning",
            title="开头静音偏长",
            detail=f"开头静音约 {leading_silence_ms} ms。",
            recommendation="裁掉开头空白，让说话人更快进入有效语音。",
        )

    if trailing_silence_ms > 900:
        penalty = min(10.0, (trailing_silence_ms - 900) / 140.0)
        score -= penalty
        _append_alert(
            alerts,
            recommendations,
            severity="warning",
            title="结尾静音偏长",
            detail=f"结尾静音约 {trailing_silence_ms} ms。",
        )

    if crest_factor_db < 5.0:
        penalty = min(14.0, (5.0 - crest_factor_db) * 2.8)
        score -= penalty
        _append_alert(
            alerts,
            recommendations,
            severity="warning",
            title="动态起伏偏弱",
            detail=f"峰均差约 {crest_factor_db:.1f} dB，音频可能存在较重压缩、底噪或混响。",
            recommendation="优先选择干声、近讲、少混响的录音，不要直接用带 BGM 的成品音频。",
        )

    if original_channels > 1:
        _append_alert(
            alerts,
            recommendations,
            severity="info",
            title="检测到多声道音频",
            detail=f"当前为 {original_channels} 声道，系统会自动下混到单声道进行分析和克隆。",
        )

    score = max(0.0, min(100.0, score))
    score_int = int(round(score))

    if score_int >= 88:
        grade = "excellent"
        grade_label = "优秀"
        summary = "参考音频质量较好，适合直接克隆。"
    elif score_int >= 72:
        grade = "good"
        grade_label = "可用"
        summary = "参考音频整体可用，但仍有一些可优化项。"
    elif score_int >= 56:
        grade = "fair"
        grade_label = "有风险"
        summary = "参考音频可以尝试，但克隆稳定性和情感表现可能受影响。"
    else:
        grade = "risky"
        grade_label = "风险较高"
        summary = "参考音频风险较高，建议先处理音频再克隆。"

    should_block = score_int < 45 or any(alert["severity"] == "critical" for alert in alerts)
    should_warn = should_block or score_int < 72 or any(alert["severity"] == "warning" for alert in alerts)

    return {
        "score": score_int,
        "grade": grade,
        "grade_label": grade_label,
        "summary": summary,
        "should_warn": should_warn,
        "should_block": should_block,
        "metrics": {
            "duration_sec": round(duration_sec, 2),
            "sample_rate": sample_rate,
            "channels": original_channels,
            "peak_dbfs": round(peak_dbfs, 2),
            "rms_dbfs": round(rms_dbfs, 2),
            "silence_ratio": round(silence_ratio, 4),
            "leading_silence_ms": int(leading_silence_ms),
            "trailing_silence_ms": int(trailing_silence_ms),
            "clipping_ratio": round(clipping_ratio, 6),
            "crest_factor_db": round(crest_factor_db, 2),
        },
        "alerts": alerts,
        "recommendations": recommendations,
    }
