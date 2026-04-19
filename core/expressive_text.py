from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional


NON_VERBAL_TAG_RE = re.compile(r"\[[^\[\]]+\]")
CJK_CHAR_RE = re.compile(r"[\u3400-\u9fff]")
QUESTION_RE = re.compile(r"[?？]|(?:吗|么|嘛|呢)\s*$")
EXCITED_RE = re.compile(r"[!！]|(?:太棒了|太好了|真棒|真好|好耶|哇|哇塞|太赞了)")
SURPRISED_RE = re.compile(r"(?:居然|竟然|原来|不会吧|天哪|没想到|真没想到|突然)")
LAUGHTER_RE = re.compile(r"(?:\[laughter\]|哈哈|呵呵|嘿嘿|笑出声)", re.IGNORECASE)
SIGH_RE = re.compile(r"(?:\[sigh\]|唉|哎|唉呀|哎呀|唉哟|……|\.{3,})", re.IGNORECASE)
WHISPER_RE = re.compile(r"(?:耳语|悄声|小声|轻声|低声|压低声音|whisper)", re.IGNORECASE)
URGENT_RE = re.compile(r"(?:快点|赶紧|立刻|马上|别动|小心|不好了|糟了|hurry|quickly|right now)", re.IGNORECASE)
GENTLE_RE = re.compile(r"(?:轻轻地|慢慢地|别怕|没事了|放心|calmly|gently|slowly)", re.IGNORECASE)
CONTRAST_RE = re.compile(r"^(?:但是|可是|不过|然而|只是|却|but\b|however\b|yet\b)", re.IGNORECASE)
PREFIX_CUE_RE = re.compile(
    r"^\s*(?:但是|可是|不过|然而|结果|原来|突然|终于|其实|反而|偏偏|所以|然后|快点|赶紧|立刻|马上|别怕|没事了|"
    r"but\b|however\b|yet\b|then\b|so\b|actually\b|suddenly\b|hurry\b)",
    re.IGNORECASE,
)
PITCH_HINT_RE = re.compile(
    r"(?:very low pitch|low pitch|moderate pitch|high pitch|very high pitch)",
    re.IGNORECASE,
)
STYLE_HINT_RE = re.compile(r"(?:whisper)", re.IGNORECASE)

HARD_BOUNDARY_CHARS = {"。", "！", "？", "!", "?", "；", ";", "…", "\n"}
SOFT_BOUNDARY_CHARS = {"，", ",", "、", "：", ":"}
CONNECTOR_TOKENS = (
    "但是",
    "可是",
    "不过",
    "然而",
    "只是",
    "却",
    "结果",
    "原来",
    "终于",
    "突然",
    "其实",
    "反而",
    "偏偏",
    "所以",
    "然后",
)
EN_CONNECTOR_RE = re.compile(r"^(?:but|however|yet|then|so|actually|suddenly)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SegmentDraft:
    text: str
    boundary: str


@dataclass(frozen=True)
class ProsodySegment:
    text: str
    render_text: str
    speed: float
    pause_ms: int
    style: str
    boundary: str
    injected_tag: Optional[str] = None
    instruct_hint: Optional[str] = None
    preserve_silence: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AutoProsodyPlan:
    normalized_text: str
    segments: list[ProsodySegment]
    segmentation_mode: str = "phrase_v2"
    preserve_pauses: bool = False

    def to_dict(self) -> dict:
        return {
            "normalized_text": self.normalized_text,
            "segmentation_mode": self.segmentation_mode,
            "preserve_pauses": self.preserve_pauses,
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
    preserve_pauses = False

    for raw_segment in raw_segments:
        segment = raw_segment.text.strip()
        if not segment:
            continue

        style = detect_segment_style(segment, raw_segment.boundary)
        render_text, injected_tag = maybe_inject_non_verbal_tag(segment, style)
        speed = compute_segment_speed(segment, style, raw_segment.boundary, base_speed)
        pause_ms = compute_pause_ms(segment, style, raw_segment.boundary)
        instruct_hint = merge_instruct(base_instruct, style)
        preserve_silence = should_preserve_silence(
            segment,
            style=style,
            boundary=raw_segment.boundary,
            injected_tag=injected_tag,
        )
        preserve_pauses = preserve_pauses or preserve_silence

        segments.append(
            ProsodySegment(
                text=segment,
                render_text=render_text,
                speed=speed,
                pause_ms=pause_ms,
                style=style,
                boundary=raw_segment.boundary,
                injected_tag=injected_tag,
                instruct_hint=instruct_hint,
                preserve_silence=preserve_silence,
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
                boundary="tail",
                instruct_hint=clean_instruct(base_instruct),
                preserve_silence=False,
            )
        )

    return AutoProsodyPlan(
        normalized_text=normalized_text,
        segments=segments,
        preserve_pauses=preserve_pauses,
    )


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


def split_text_into_segments(text: str, language: str) -> list[SegmentDraft]:
    if not text:
        return []

    max_length = 48 if is_cjk_language(language) else 140
    segments: list[SegmentDraft] = []

    for chunk in split_by_hard_boundaries(text):
        segments.extend(
            split_phrase_segment(
                chunk.text,
                final_boundary=chunk.boundary,
                language=language,
                max_length=max_length,
            )
        )

    return [segment for segment in segments if segment.text]


def split_by_hard_boundaries(text: str) -> list[SegmentDraft]:
    segments: list[SegmentDraft] = []
    buffer: list[str] = []
    index = 0

    while index < len(text):
        char = text[index]
        buffer.append(char)

        if char in HARD_BOUNDARY_CHARS:
            boundary_chars = [char]
            index += 1
            while index < len(text) and text[index] in HARD_BOUNDARY_CHARS:
                buffer.append(text[index])
                boundary_chars.append(text[index])
                index += 1

            piece = "".join(buffer).strip()
            if piece:
                segments.append(SegmentDraft(piece, classify_boundary(boundary_chars)))
            buffer = []
            continue

        index += 1

    if buffer:
        piece = "".join(buffer).strip()
        if piece:
            segments.append(SegmentDraft(piece, "tail"))

    return segments


def split_phrase_segment(
    text: str,
    *,
    final_boundary: str,
    language: str,
    max_length: int,
) -> list[SegmentDraft]:
    if not text:
        return []

    min_phrase_length = 8 if is_cjk_language(language) else 24
    segments: list[SegmentDraft] = []
    buffer: list[str] = []
    index = 0

    while index < len(text):
        connector = match_connector(text, index, language)
        if connector and buffer:
            left = "".join(buffer).strip()
            right = text[index:].strip()
            if (
                text_length(left) >= min_phrase_length
                and text_length(right) >= min_phrase_length
                and not left.endswith(tuple(SOFT_BOUNDARY_CHARS))
            ):
                segments.append(SegmentDraft(left, "connector"))
                buffer = []
                continue

        char = text[index]
        buffer.append(char)
        current_text = "".join(buffer)
        remaining_text = text[index + 1 :]

        if char in SOFT_BOUNDARY_CHARS:
            if should_split_on_soft_boundary(current_text, remaining_text, max_length=max_length, language=language):
                piece = current_text.strip()
                if piece:
                    segments.append(SegmentDraft(piece, "minor"))
                buffer = []
        elif text_length(current_text) >= max_length:
            piece = current_text.strip()
            if piece:
                segments.append(SegmentDraft(piece, "length"))
            buffer = []

        index += 1

    tail = "".join(buffer).strip()
    if tail:
        segments.append(SegmentDraft(tail, final_boundary))

    return merge_short_segments(segments, min_phrase_length=min_phrase_length)


def merge_short_segments(segments: list[SegmentDraft], min_phrase_length: int) -> list[SegmentDraft]:
    if not segments:
        return []

    short_threshold = max(4, min_phrase_length // 2)
    merged: list[SegmentDraft] = []

    for segment in segments:
        if not merged:
            merged.append(segment)
            continue

        previous = merged[-1]
        if text_length(previous.text) < short_threshold and previous.boundary in {"minor", "connector", "length"}:
            merged[-1] = SegmentDraft(previous.text + segment.text, segment.boundary)
            continue

        merged.append(segment)

    if len(merged) >= 2 and text_length(merged[-1].text) < short_threshold:
        tail = merged.pop()
        previous = merged.pop()
        merged.append(SegmentDraft(previous.text + tail.text, tail.boundary))

    return merged


def detect_segment_style(segment: str, boundary: str) -> str:
    if WHISPER_RE.search(segment):
        return "whisper"
    if LAUGHTER_RE.search(segment):
        return "laughter"
    if SIGH_RE.search(segment):
        return "sigh"
    if URGENT_RE.search(segment):
        return "urgent"
    if QUESTION_RE.search(segment) or boundary == "question":
        return "question"
    if SURPRISED_RE.search(segment):
        return "surprised"
    if EXCITED_RE.search(segment) or boundary == "exclaim":
        return "excited"
    if GENTLE_RE.search(segment):
        return "gentle"
    if CONTRAST_RE.search(segment):
        return "contrast"
    return "neutral"


def maybe_inject_non_verbal_tag(segment: str, style: str) -> tuple[str, Optional[str]]:
    if NON_VERBAL_TAG_RE.search(segment):
        return segment, None

    injected_tag: Optional[str] = None
    stripped = strip_terminal_punctuation(segment)

    if style == "laughter":
        injected_tag = "[laughter]"
    elif style == "sigh" and len(stripped) <= 18:
        injected_tag = "[sigh]"
    elif style == "question" and len(stripped) <= 16:
        injected_tag = "[question-en]"
    elif style in {"surprised", "excited"} and len(stripped) <= 14:
        injected_tag = "[surprise-ah]"

    if not injected_tag:
        return segment, None

    return f"{injected_tag} {segment}".strip(), injected_tag


def compute_segment_speed(segment: str, style: str, boundary: str, base_speed: float) -> float:
    delta_map = {
        "whisper": -0.10,
        "laughter": 0.03,
        "sigh": -0.10,
        "urgent": 0.08,
        "question": -0.04,
        "surprised": 0.02,
        "excited": 0.05,
        "gentle": -0.07,
        "contrast": -0.03,
    }
    delta = delta_map.get(style, 0.0)

    if boundary == "ellipsis":
        delta -= 0.05
    elif boundary == "newline":
        delta -= 0.03
    elif boundary == "length":
        delta -= 0.04
    elif boundary == "connector":
        delta -= 0.01

    if len(segment) >= 36:
        delta -= 0.04
    elif len(segment) <= 8 and style in {"question", "surprised", "excited", "urgent"}:
        delta += 0.02

    if starts_with_connector(segment):
        delta -= 0.02

    if segment.endswith(("……", "…")):
        delta -= 0.03

    return clamp_speed(base_speed + delta)


def compute_pause_ms(segment: str, style: str, boundary: str) -> int:
    pause_map = {
        "tail": 0,
        "terminal": 220,
        "question": 290,
        "exclaim": 240,
        "ellipsis": 340,
        "newline": 380,
        "minor": 150,
        "connector": 180,
        "length": 140,
    }
    pause_ms = pause_map.get(boundary, 220)

    if style in {"whisper", "sigh", "gentle"}:
        pause_ms += 40
    elif style == "contrast":
        pause_ms += 20
    elif style == "urgent":
        pause_ms -= 30

    if segment.endswith(("……", "…")):
        pause_ms = max(pause_ms, 340)

    return max(90, min(420, pause_ms))


def merge_instruct(base_instruct: Optional[str], style: str) -> Optional[str]:
    instruct = clean_instruct(base_instruct)

    if style == "whisper":
        return append_instruct_attribute(instruct, "whisper", category="style")
    if style in {"surprised", "excited", "urgent"}:
        return append_instruct_attribute(instruct, "high pitch", category="pitch")
    if style in {"sigh", "gentle"}:
        return append_instruct_attribute(instruct, "low pitch", category="pitch")
    return instruct


def append_instruct_attribute(base_instruct: Optional[str], value: str, *, category: str) -> Optional[str]:
    instruct = clean_instruct(base_instruct)
    if not instruct:
        return value

    if category == "pitch" and PITCH_HINT_RE.search(instruct):
        return instruct
    if category == "style" and STYLE_HINT_RE.search(instruct):
        return instruct

    tokens = [token.strip() for token in instruct.split(",") if token.strip()]
    if any(token.lower() == value.lower() for token in tokens):
        return instruct

    tokens.append(value)
    return ", ".join(tokens)


def should_preserve_silence(
    segment: str,
    *,
    style: str,
    boundary: str,
    injected_tag: Optional[str],
) -> bool:
    if injected_tag is not None:
        return True
    if boundary in {"ellipsis", "newline"}:
        return True
    if style in {"whisper", "sigh", "laughter"}:
        return True
    return segment.endswith(("……", "…"))


def should_split_on_soft_boundary(
    current_text: str,
    remaining_text: str,
    *,
    max_length: int,
    language: str,
) -> bool:
    min_phrase_length = 8 if is_cjk_language(language) else 24
    current_length = text_length(current_text)
    remaining_length = text_length(remaining_text)

    if current_length >= min_phrase_length:
        return True
    if remaining_length >= int(max_length * 0.6):
        return True
    return starts_with_prefix_cue(remaining_text)


def classify_boundary(boundary_chars: list[str]) -> str:
    if "\n" in boundary_chars:
        return "newline"
    if any(char in {"?", "？"} for char in boundary_chars):
        return "question"
    if any(char in {"!", "！"} for char in boundary_chars):
        return "exclaim"
    if "…" in boundary_chars:
        return "ellipsis"
    return "terminal"


def match_connector(text: str, index: int, language: str) -> Optional[str]:
    remaining = text[index:]

    if is_cjk_language(language):
        for token in CONNECTOR_TOKENS:
            if remaining.startswith(token):
                return token
        return None

    match = EN_CONNECTOR_RE.match(remaining)
    if match:
        return match.group(0)
    return None


def starts_with_connector(text: str) -> bool:
    candidate = text.lstrip()
    return any(candidate.startswith(token) for token in CONNECTOR_TOKENS) or bool(EN_CONNECTOR_RE.match(candidate))


def starts_with_prefix_cue(text: str) -> bool:
    return bool(PREFIX_CUE_RE.match(text))


def clean_instruct(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def strip_terminal_punctuation(text: str) -> str:
    return text.rstrip("。！？!?；;，,、：:… ")


def clamp_speed(value: float) -> float:
    return max(0.78, min(1.18, round(value, 3)))


def text_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def is_cjk_language(language: str) -> bool:
    normalized = language.strip().lower()
    return normalized in {"chinese", "mandarin", "zh", "zh-cn", "zh-hans"} or bool(CJK_CHAR_RE.search(normalized))
