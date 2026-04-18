# -*- coding: utf-8 -*-
"""
VoiceClonePrompt 类
从 omnivoice 包复制，仅用于兼容 pip 版本
"""
from dataclasses import dataclass
from typing import Any, Union
import os

import torch


@dataclass
class VoiceClonePrompt:
    ref_audio_tokens: torch.Tensor  # (C, T)
    ref_text: str
    ref_rms: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prompt into CPU tensors for reuse across runs."""
        return {
            "ref_audio_tokens": self.ref_audio_tokens.detach().cpu(),
            "ref_text": self.ref_text,
            "ref_rms": float(self.ref_rms),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceClonePrompt":
        """Rebuild a prompt saved by :meth:`to_dict`."""
        ref_audio_tokens = data["ref_audio_tokens"]
        if not isinstance(ref_audio_tokens, torch.Tensor):
            ref_audio_tokens = torch.as_tensor(ref_audio_tokens)

        return cls(
            ref_audio_tokens=ref_audio_tokens,
            ref_text=str(data["ref_text"]),
            ref_rms=float(data["ref_rms"]),
        )

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Persist the prompt so the reference audio only needs encoding once."""
        save_path = os.fspath(path)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(self.to_dict(), save_path)

    @classmethod
    def load(
        cls,
        path: Union[str, os.PathLike],
        map_location: Union[str, torch.device] = "cpu",
    ) -> "VoiceClonePrompt":
        """Load a prompt saved by :meth:`save`."""
        data = torch.load(os.fspath(path), map_location=map_location)
        if not isinstance(data, dict):
            raise TypeError(
                "Voice clone prompt file is invalid. Expected a dict payload "
                "created by VoiceClonePrompt.save()."
            )
        return cls.from_dict(data)
