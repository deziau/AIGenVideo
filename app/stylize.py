"""Frame-by-frame artistic filters: cartoon, comic, oil painting, watercolor, pencil sketch.

All filters are deterministic per frame (no k-means or random seeds), which keeps
flicker low; `TemporalSmoother` blends consecutive outputs to calm it further.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

STYLES = {
    "cartoon": "Cartoon - flat colours with clean ink outlines",
    "comic": "Comic book - bold ink, posterized shading, halftone dots",
    "oil_painting": "Oil painting - thick, blended brush strokes",
    "watercolor": "Watercolor - soft washes on textured paper",
    "pencil_sketch": "Pencil sketch - graphite on paper",
}


def _smooth_colors(img: np.ndarray, passes: int) -> np.ndarray:
    # Bilateral filtering at half resolution is much faster and looks the same.
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    for _ in range(passes):
        small = cv2.bilateralFilter(small, 9, 50, 7)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _posterize(img: np.ndarray, levels: int, saturation: float) -> np.ndarray:
    """Band lightness into flat tones (in LAB, so hues stay true) and boost colour."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[..., 0] / 255.0
    lab[..., 0] = (np.floor(lightness * levels) + 0.5) / levels * 255
    lab[..., 1:] = (lab[..., 1:] - 128) * saturation + 128
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return cv2.medianBlur(out, 5)


def _ink_lines(img: np.ndarray, thickness: int, min_area: int) -> np.ndarray:
    """Boolean mask of clean outlines, with tiny specks removed."""
    gray = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7, 50, 50)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 110)
    edges = cv2.dilate(edges, np.ones((thickness, thickness), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def cartoon(img: np.ndarray, strength: float) -> np.ndarray:
    color = _smooth_colors(img, 3 + int(strength * 4))
    out = _posterize(color, int(round(9 - 4 * strength)), 1.15 + 0.25 * strength)
    lines = _ink_lines(img, 2, 25)
    out[lines] = (out[lines] * 0.15).astype(np.uint8)
    return out


def comic(img: np.ndarray, strength: float) -> np.ndarray:
    color = _smooth_colors(img, 4 + int(strength * 3))
    out = _posterize(color, 4, 1.4 + 0.3 * strength)
    h, w = img.shape[:2]
    # Halftone dots in the shadows.
    period = 6
    yy, xx = np.mgrid[0:h, 0:w]
    dots = ((xx % period - period / 2) ** 2 + (yy % period - period / 2) ** 2) < (period / 2.6) ** 2
    shade = dots & (cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) < 120)
    out[shade] = (out[shade] * (0.45 + 0.25 * (1 - strength))).astype(np.uint8)
    lines = _ink_lines(img, 3, 40)
    out[lines] = 0
    return out


def _kuwahara(img: np.ndarray, radius: int) -> np.ndarray:
    """Kuwahara filter: each pixel takes the mean of its least-varied neighbouring quadrant,
    which flattens texture into brush-like patches while keeping edges crisp."""
    h, w = img.shape[:2]
    k = radius + 1
    mean = cv2.blur(img.astype(np.float32), (k, k), borderType=cv2.BORDER_REFLECT)
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    m1 = cv2.blur(lum, (k, k), borderType=cv2.BORDER_REFLECT)
    var = cv2.blur(lum * lum, (k, k), borderType=cv2.BORDER_REFLECT) - m1 * m1
    pad, o = radius, radius // 2
    mean_p = cv2.copyMakeBorder(mean, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    var_p = cv2.copyMakeBorder(var, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    best, best_var = None, None
    for dy, dx in ((-o, -o), (-o, o), (o, -o), (o, o)):
        m = mean_p[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
        v = var_p[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
        if best is None:
            best, best_var = m.copy(), v.copy()
        else:
            sel = v < best_var
            best[sel] = m[sel]
            best_var[sel] = v[sel]
    return np.clip(best, 0, 255).astype(np.uint8)


def oil_painting(img: np.ndarray, strength: float) -> np.ndarray:
    out = _kuwahara(cv2.bilateralFilter(img, 5, 40, 40), 3 + int(round(strength * 5)))
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.12, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # Light canvas weave.
    return np.clip(out.astype(np.float32) * _canvas(*img.shape[:2])[..., None], 0, 255).astype(np.uint8)


_paper_cache: dict[tuple[int, int], np.ndarray] = {}


def _paper(h: int, w: int) -> np.ndarray:
    """Static paper grain; generated once per size so it does not flicker between frames."""
    key = (h, w)
    if key not in _paper_cache:
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 1, (h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), 1.2)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        _paper_cache[key] = 0.93 + 0.07 * noise
    return _paper_cache[key]


_canvas_cache: dict[tuple[int, int], np.ndarray] = {}


def _canvas(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key not in _canvas_cache:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        weave = np.sin(xx * 1.9) * np.sin(yy * 1.9)
        _canvas_cache[key] = 0.97 + 0.03 * weave * _paper(h, w)
    return _canvas_cache[key]


def watercolor(img: np.ndarray, strength: float) -> np.ndarray:
    out = cv2.stylization(img, sigma_s=30 + 90 * strength, sigma_r=0.3 + 0.3 * strength)
    # Pigment bleeding: lighten and soften.
    out = cv2.addWeighted(out, 0.85, np.full_like(out, 255), 0.15, 0)
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 60, 140)
    edges = cv2.GaussianBlur(edges, (3, 3), 0).astype(np.float32) / 255.0
    out = out.astype(np.float32) * (1 - 0.25 * edges[..., None])
    out *= _paper(*img.shape[:2])[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def pencil_sketch(img: np.ndarray, strength: float) -> np.ndarray:
    gray, colored = cv2.pencilSketch(img, sigma_s=40 + 30 * strength, sigma_r=0.08, shade_factor=0.03 + 0.04 * strength)
    sketch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    sketch *= _paper(*img.shape[:2])[..., None]
    return np.clip(sketch, 0, 255).astype(np.uint8)


FILTERS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "cartoon": cartoon,
    "comic": comic,
    "oil_painting": oil_painting,
    "watercolor": watercolor,
    "pencil_sketch": pencil_sketch,
}


def apply_style(img: np.ndarray, style: str, strength: float = 0.6) -> np.ndarray:
    if style not in FILTERS:
        raise ValueError(f"Unknown style '{style}'. Choose from: {', '.join(FILTERS)}")
    return FILTERS[style](img, float(np.clip(strength, 0.0, 1.0)))


class TemporalSmoother:
    """Exponential moving average across frames; resets on hard cuts."""

    def __init__(self, amount: float):
        self.amount = float(np.clip(amount, 0.0, 0.9))
        self.prev: np.ndarray | None = None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self.amount <= 0:
            return frame
        cur = frame.astype(np.float32)
        if self.prev is not None and np.mean(np.abs(cur - self.prev)) < 40:
            cur = self.amount * self.prev + (1 - self.amount) * cur
        self.prev = cur
        return cur.astype(np.uint8)
