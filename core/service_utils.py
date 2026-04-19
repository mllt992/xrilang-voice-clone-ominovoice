from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional


VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def clean_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    value = value.strip()
    return value or None


def validate_voice_name(name: str) -> str:
    normalized = clean_optional_text(name)
    if not normalized:
        raise ValueError("音色名称不能为空。")

    if not VOICE_NAME_RE.fullmatch(normalized):
        raise ValueError("音色名称仅支持 1-64 位字母、数字、下划线和短横线。")

    return normalized


def resolve_file_in_dir(base_dir: Path, filename: str) -> Path:
    normalized_name = clean_optional_text(filename)
    if not normalized_name:
        raise ValueError("文件名不能为空。")

    if Path(normalized_name).name != normalized_name:
        raise ValueError("文件名不合法。")

    base_path = base_dir.resolve()
    target_path = (base_path / normalized_name).resolve()
    if target_path.parent != base_path:
        raise ValueError("文件路径不合法。")

    return target_path


def build_output_filename(voice_id: Optional[str]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    label = voice_id or "auto"
    safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", label)[:32] or "auto"
    return f"synth_{safe_label}_{stamp}.wav"
