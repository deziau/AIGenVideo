"""Claude as 'creative director': looks at keyframes of the reference video plus the
user's request and writes a production-ready prompt for the video model."""
from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass

import anthropic
import cv2
import numpy as np

from . import config

PREVIEW_LOOKS = ["cinematic", "cartoon", "comic", "oil_painting", "watercolor", "pencil_sketch", "noir", "vibrant"]

SYSTEM_PROMPT = """You are the creative director of an AI video studio.
You receive keyframes sampled in order from a short reference video, and a request from the user
describing the video they want. A video-to-video model will transform the reference clip using the
prompt you write: it keeps the reference's motion, timing and composition and changes whatever the
prompt describes (subjects, setting, style, lighting, weather, time of day, camera look).

Write a prompt that makes the result look spectacular while staying faithful to the user's request:
- Describe the target scene concretely: subjects, wardrobe/materials, environment, lighting, colour
  palette, atmosphere, lens and film look, and the motion that should be preserved from the reference.
- Refer to things visible in the reference by their position or role so the model maps them correctly.
- Use present-tense visual language only; no camera-instruction jargon the model cannot see, no
  mentions of 'the reference video' or 'the user'.
- If the request is vague, make bold, tasteful creative choices that fit it.
- Keep `generation_prompt` under 900 characters.
- If the request asks for something unsafe or depicts a real private person in a harmful way, steer
  it to a safe interpretation and explain that in `notes`."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "A short, catchy title for the video."},
        "reference_summary": {"type": "string", "description": "One or two sentences on what happens in the reference."},
        "generation_prompt": {"type": "string", "description": "The prompt for the video-to-video model."},
        "preview_look": {"type": "string", "enum": PREVIEW_LOOKS,
                         "description": "Closest local filter, used only when no generative model is configured."},
        "notes": {"type": "string", "description": "Brief notes for the user on creative choices made."},
    },
    "required": ["title", "reference_summary", "generation_prompt", "preview_look", "notes"],
    "additionalProperties": False,
}


@dataclass
class DirectorPlan:
    title: str
    reference_summary: str
    generation_prompt: str
    preview_look: str
    notes: str
    source: str  # "claude" or "fallback"

    def to_dict(self) -> dict:
        return asdict(self)


def _jpeg_b64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Could not encode keyframe")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def fallback_plan(user_request: str) -> DirectorPlan:
    text = user_request.strip() or "A cinematic reimagining of this scene"
    lowered = text.lower()
    look = next((k for k in PREVIEW_LOOKS if k.replace("_", " ") in lowered), "cinematic")
    return DirectorPlan(
        title=text[:60],
        reference_summary="(Claude not configured - reference was not analyzed.)",
        generation_prompt=f"{text}. Cinematic lighting, rich detail, smooth natural motion, high production value.",
        preview_look=look,
        notes="Set ANTHROPIC_API_KEY to have Claude study the reference video and write a richer prompt.",
        source="fallback",
    )


def plan_video(frames: list[np.ndarray], user_request: str, client: anthropic.Anthropic | None = None) -> DirectorPlan:
    if client is None:
        if not config.anthropic_enabled():
            return fallback_plan(user_request)
        client = anthropic.Anthropic()

    content: list[dict] = []
    for i, frame in enumerate(frames):
        content.append({"type": "text", "text": f"Keyframe {i + 1} of {len(frames)}"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpeg_b64(frame)}})
    content.append({"type": "text", "text": f"<user_request>\n{user_request.strip()}\n</user_request>"})

    response = client.beta.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason == "refusal":
        raise ValueError("Claude declined this request. Try rephrasing what you want the video to show.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude's response was cut off; please try again.")

    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    return DirectorPlan(**data, source="claude")
