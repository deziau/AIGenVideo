"""Video generation backends.

RunwayGenerator  - real generative video-to-video (Runway gen4_aleph) that follows the prompt.
local_preview    - no API needed; applies a look to the reference so the flow can be tried offline.
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Callable

import cv2
import httpx
import numpy as np

from . import config
from .stylize import TemporalSmoother, apply_style
from .video_io import fit_size, iter_frames, probe, write_video

# Output ratios gen4_aleph accepts, as (width, height).
RUNWAY_RATIOS = [(1280, 720), (720, 1280), (1104, 832), (832, 1104), (960, 960), (1584, 672), (848, 480), (640, 480)]
# Data URIs above this size are rejected by the API; longer clips must be hosted and passed by URL.
RUNWAY_DATA_URI_LIMIT = 16 * 1024 * 1024


def nearest_ratio(width: int, height: int) -> str:
    target = width / height
    w, h = min(RUNWAY_RATIOS, key=lambda r: abs(np.log((r[0] / r[1]) / target)))
    return f"{w}:{h}"


class RunwayGenerator:
    def __init__(self, api_key: str | None = None, base_url: str = config.RUNWAY_API_BASE, poll_interval: float = 5.0):
        self.api_key = api_key or os.environ["RUNWAYML_API_SECRET"]
        self.poll_interval = poll_interval
        self.http = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(120.0, connect=15.0),
            headers={"Authorization": f"Bearer {self.api_key}", "X-Runway-Version": config.RUNWAY_API_VERSION},
        )
        # Separate client without the API key: outputs are served from a CDN.
        self.download_http = httpx.Client(timeout=120.0, follow_redirects=True)

    def generate(self, clip: Path, prompt: str, out_path: Path,
                 on_status: Callable[[str], None] = lambda s: None, timeout_s: float = 900) -> Path:
        raw = clip.read_bytes()
        if len(raw) > RUNWAY_DATA_URI_LIMIT:
            raise ValueError("Reference clip is too large to upload inline; shorten it or lower AIGV_MAX_SIDE.")
        info = probe(clip)
        body = {
            "model": "gen4_aleph",
            "videoUri": "data:video/mp4;base64," + base64.b64encode(raw).decode("ascii"),
            "promptText": prompt[:1000],
            "ratio": nearest_ratio(info.width, info.height),
        }
        resp = self.http.post("/video_to_video", json=body)
        self._raise_for_status(resp)
        task_id = resp.json()["id"]
        on_status("Queued with Runway")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            resp = self.http.get(f"/tasks/{task_id}")
            self._raise_for_status(resp)
            task = resp.json()
            status = task.get("status")
            if status == "SUCCEEDED":
                on_status("Downloading result")
                return self._download(task["output"][0], out_path)
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"Runway task {status.lower()}: {task.get('failure') or task.get('failureCode') or 'unknown error'}")
            progress = task.get("progress")
            on_status(f"Rendering on Runway ({status.lower()}{f', {int(progress * 100)}%' if progress else ''})")
        raise TimeoutError("Runway did not finish in time")

    def _download(self, url: str, out_path: Path) -> Path:
        with self.download_http.stream("GET", url) as r:
            r.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return out_path

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.is_error:
            try:
                detail = resp.json().get("error") or resp.text
            except ValueError:
                detail = resp.text
            raise RuntimeError(f"Runway API error {resp.status_code}: {str(detail)[:300]}")


# --- Local preview looks -----------------------------------------------------------

def _vignette(h: int, w: int, amount: float) -> np.ndarray:
    kx = cv2.getGaussianKernel(w, w * 0.6)
    ky = cv2.getGaussianKernel(h, h * 0.6)
    mask = ky @ kx.T
    mask = mask / mask.max()
    return (1 - amount) + amount * mask


def cinematic(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32) / 255.0
    f = f * f * (3 - 2 * f)  # gentle S-curve for contrast
    lum = f.mean(axis=2, keepdims=True)
    teal = np.array([0.55, 0.45, 0.20], np.float32)   # BGR, pushed into shadows
    orange = np.array([0.25, 0.55, 0.85], np.float32)  # BGR, pushed into highlights
    f = f * 0.8 + 0.2 * ((1 - lum) * teal * f * 2 + lum * orange * f * 2)
    f *= _vignette(*img.shape[:2], 0.35)[..., None]
    out = np.clip(f * 255, 0, 255).astype(np.uint8)
    bar = int(img.shape[0] * 0.06)  # letterbox
    out[:bar] = 0
    out[-bar:] = 0
    return out


def noir(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray).astype(np.float32)
    gray *= _vignette(*img.shape[:2], 0.5)
    return cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def vibrant(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.45, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.08, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


GRADES = {"cinematic": cinematic, "noir": noir, "vibrant": vibrant}


def local_preview(src: Path, look: str, out_path: Path, on_progress: Callable[[float], None] = lambda p: None) -> Path:
    info = probe(src)
    size = fit_size(info.width, info.height, config.MAX_SIDE)
    smooth = TemporalSmoother(0.3)
    fn = GRADES.get(look) or (lambda f: apply_style(f, look, 0.6))
    total = max(info.frame_count, 1)
    frames = (smooth(fn(f)) for f in iter_frames(src, size))
    return write_video(frames, out_path, size, info.fps, audio_from=src,
                       on_frame=lambda i: on_progress(min(i / total, 1.0)))
