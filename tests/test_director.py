import json
from types import SimpleNamespace

import numpy as np

from app.prompt_director import OUTPUT_SCHEMA, fallback_plan, plan_video


class FakeMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.calls = payload, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason=self.stop_reason,
                               content=[SimpleNamespace(type="text", text=json.dumps(self.payload))])


def fake_client(payload, stop_reason="end_turn"):
    msgs = FakeMessages(payload, stop_reason)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


PLAN = {"title": "Neon Run", "reference_summary": "A person jogs.", "generation_prompt": "Cyberpunk jogger",
        "preview_look": "cinematic", "notes": "Went bold."}


def test_plan_video_sends_frames_and_request():
    client, msgs = fake_client(PLAN)
    frames = [np.zeros((20, 30, 3), np.uint8)] * 3
    plan = plan_video(frames, "make it cyberpunk", client=client)
    assert plan.generation_prompt == "Cyberpunk jogger" and plan.source == "claude"
    call = msgs.calls[0]
    blocks = call["messages"][0]["content"]
    assert sum(b["type"] == "image" for b in blocks) == 3
    assert "make it cyberpunk" in blocks[-1]["text"]
    assert call["output_config"]["format"]["schema"] is OUTPUT_SCHEMA
    assert call["fallbacks"] == "default"


def test_refusal_is_reported():
    client, _ = fake_client(PLAN, stop_reason="refusal")
    try:
        plan_video([np.zeros((8, 8, 3), np.uint8)], "x", client=client)
    except ValueError as e:
        assert "declined" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_fallback_plan_picks_look_from_text():
    assert fallback_plan("make it a watercolor dream").preview_look == "watercolor"
    assert fallback_plan("space opera").preview_look == "cinematic"


def test_fallback_prompt_has_single_period():
    assert ".." not in fallback_plan("Cyberpunk neon night.").generation_prompt
