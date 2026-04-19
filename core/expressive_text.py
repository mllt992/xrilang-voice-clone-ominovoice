from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional


NON_VERBAL_TAG_RE = re.compile(r"\[[^\[\]]+\]")
CJK_CHAR_RE = re.compile(r"[\u3400-\u9fff]")
QUESTION_RE = re.compile(r"[?？]|(?:吗|么|嘛|呢)\s*$")
EXCITED_RE = re.compile(r"[!！]|(?:太棒了|太好了|真棒|真好|居然|竟然|好耶|哇|哇塞)")
LAUGHTER_RE = re.compile(r"(?:\[laughter\]|哈哈|呵呵|嘿嘿|笑出声)", re.IGNORECASE)
SIGH_RE = re.compile(r"(?:\[sigh\]|唉|哎|唉呀|哎呀|唉哟|……|\.{3,})", re.IGNORECASE)
WHISPER_RE = re.compile(r"(?:耳语|悄声|小声|轻声|低声|压低声音|whisper)", re.IGNORECASE)
BOUNDARY_CHARS = {"。", "！", "？", "!", "?", "；", ";", "…", "\n"}
SOFT_BOUNDARY_CHARS = {"，", ",", "、", "：", ":"}


@dataclass(frozen=True)
class ProsodySegment:
    text: str
    render_text: str
    speed: float
    pause_ms: int
    style: str
    injected_tag: Optional[str] = None
    instruct_hint: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AutoProsodyPlan:
    normalized_text: str
    segments: list[ProsodySegment]

    def to_dict(self) -> dict:
        return {
            "normalized_text": self.normalized_text,
            "segments": [segment.to_dict() for segment in self.segments],
        }


def build_auto_prosody_plan(
    text: str,
    language: str,
    base_speed: float = 1.0,
    base_instruct: Optional[str] = None,
) -> AutoProsodyPlan:
    normalized_text = normalize_tts_text(text, language)
    raw_segments = split_text_into_segments(normalized_text, language)
    segments: list[ProsodySegment] = []

    for raw_segment in raw_segments:
        segment = raw_segment.strip()
        if not segment:
            continue

        style = detect_segment_style(segment)
        render_text, injected_tag = maybe_inject_non_verbal_tag(segment, style)
        speed = compute_segment_speed(segment, style, base_speed)
        pause_ms = compute_pause_ms(segment, style)
        instruct_hint = merge_instruct(base_instruct, style)

        segments.append(
            ProsodySegment(
                text=segment,
                render_text=render_text,
                speed=speed,
                pause_ms=pause_ms,
                style=style,
                injected_tag=injected_tag,
                instruct_hint=instruct_hint,
            )
        )

    if not segments:
        segments.append(
            ProsodySegment(
                text=normalized_text,
                render_text=normalized_text,
                speed=clamp_speed(base_speed),
                pause_ms=0,
                style="neutral",
                instruct_hint=clean_instruct(base_instruct),
            )
        )

    return AutoProsodyPlan(normalized_text=normalized_text, segments=segments)


def normalize_tts_text(text: str, language: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u3000", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"\.{3,}", "……", normalized)
    normalized = re.sub(r"…{3,}", "……", normalized)

    if is_cjk_language(language):
        normalized = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", normalized)
        normalized = re.sub(r"\s*([，。！？；：、])\s*", r"\1", normalized)
        normalized = re.sub(r"([。！？；：、，])(?=[\u3400-\u9fff])", r"\1", normalized)

    return normalized.strip()


def split_text_into_segments(text: str, language: str) -> list[str]:
    if not text:
        return []

    max_length = 48 if is_cjk_language(language) else 140
    segments: list[str] = []
    buffer: list[str] = []

    for index, char in enumerate(text):
        buffer.append(char)
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if char in BOUNDARY_CHARS and next_char not in BOUNDARY_CHARS:
            segment = "".join(buffer).strip()
            if segment:
                segments.extend(split_long_segment(segment, max_length=max_length))
            buffer = []

    if buffer:
        segment = "".join(buffer).strip()
        if segment:
            segments.extend(split_long_segment(segment, max_length=max_length))

    return [segment for segment in segments if segment]


def split_long_segment(segment: str, max_length: int) -> list[str]:
    if len(segment) <= max_length:
        return [segment]

    pieces: list[str] = []
    buffer: list[str] = []

    for char in segment:
        buffer.append(char)
        if char in SOFT_BOUNDARY_CHARS and len(buffer) >= max(10, int(max_length * 0.45)):
            piece = "".join(buffer).strip()
            if piece:
                pieces.append(piece)
            buffer = []

    if buffer:
        piece = "".join(buffer).strip()
        if piece:
            pieces.append(piece)

    return pieces or [segment]


def detect_segment_style(segment: str) -> str:
    if WHISPER_RE.search(segment):
        return "whisper"
    if LAUGHTER_RE.search(segment):
        return "laughter"
    if SIGH_RE.search(segment):
        return "sigh"
    if QUESTION_RE.search(segment):
        return "question"
    if EXCITED_RE.search(segment):
        return "excited"
    return "neutral"


def maybe_inject_non_verbal_tag(segment: str, style: str) -> tuple[str, Optional[str]]:
    if NON_VERBAL_TAG_RE.search(segment):
        return segment, None

    injected_tag: Optional[str] = None
    stripped = strip_terminal_punctuation(segment)

    if style == "laughter":
        injected_tag = "[laughter]"
    elif style == "sigh":
        injected_tag = "[sigh]"
    elif style == "question" and len(stripped) <= 18:
        injected_tag = "[question-en]"
    elif style == "excited" and len(stripped) <= 14:
        injected_tag = "[surprise-ah]"

    if not injected_tag:
        return segment, None

    return f"{injected_tag} {segment}".strip(), injected_tag


def compute_segment_speed(segment: str, style: str, base_speed: float) -> float:
    delta = 0.0

    if style == "whisper":
        delta -= 0.08
    elif style == "sigh":
        delta -= 0.08
    elif style == "question":
        delta -= 0.03
    elif style == "excited":
        delta += 0.04
    elif style == "laughter":
        delta += 0.03

    if len(segment) >= 36:
        delta -= 0.04
    elif len(segment) <= 8 and style in {"question", "excited"}:
        delta += 0.02

    if segment.endswith(("……", "…")):
        delta -= 0.03

    return clamp_speed(base_speed + delta)


def compute_pause_ms(segment: str, style: str) -> int:
    if style == "whisper":
        return 260
    if segment.endswith(("？", "?")):
        return 280
    if segment.endswith(("！", "!")):
        return 240
    if segment.endswith(("……", "…")):
        return 320
    if segment.endswith(("，", ",", "、", "：", ":")):
        return 140
    return 220


def merge_instruct(base_instruct: Optional[str], style: str) -> Optional[str]:
    instruct = clean_instruct(base_instruct)
    if style != "whisper":
        return instruct

    if not instruct:
        return "whisper"
    if "whisper" in instruct.lower():
        return instruct
    return f"{instruct}, whisper"


def clean_instruct(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def strip_terminal_punctuation(text: str) -> str:
    return text.rstrip("。！？!?；;，,、：:… ")


def clamp_speed(value: float) -> float:
    return max(0.78, min(1.18, round(value, 3)))


def is_cjk_language(language: str) -> bool:
    normalized = language.strip().lower()
    return normalized in {"chinese", "mandarin", "zh", "zh-cn", "zh-hans"} or bool(CJK_CHAR_RE.search(normalized))
