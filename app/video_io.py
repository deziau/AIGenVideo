"""Reading frames with OpenCV and writing browser-playable H.264 MP4s with ffmpeg."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    info = VideoInfo(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS) or 24.0,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    if info.width <= 0 or info.height <= 0:
        raise ValueError("Video has no readable frames")
    return info


def fit_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Scale down to max_side and round to even dimensions (required by yuv420p)."""
    scale = min(1.0, max_side / max(width, height))
    w = max(2, int(round(width * scale / 2)) * 2)
    h = max(2, int(round(height * scale / 2)) * 2)
    return w, h


def iter_frames(path: Path, size: tuple[int, int] | None = None, max_frames: int | None = None) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    try:
        n = 0
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if size and (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            yield frame
            n += 1
    finally:
        cap.release()


def sample_frames(path: Path, count: int, max_side: int = 768) -> list[np.ndarray]:
    """Evenly spaced keyframes, used to show the reference video to Claude."""
    info = probe(path)
    size = fit_size(info.width, info.height, max_side)
    total = max(info.frame_count, 1)
    wanted = sorted({int(i * (total - 1) / max(count - 1, 1)) for i in range(count)})
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        for idx in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
    return frames


def write_video(
    frames: Iterator[np.ndarray],
    out_path: Path,
    size: tuple[int, int],
    fps: float,
    audio_from: Path | None = None,
    on_frame: Callable[[int], None] | None = None,
) -> Path:
    """Pipe BGR frames into ffmpeg; copies the audio track from `audio_from` when present."""
    w, h = size
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-"]
    if audio_from:
        cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i, frame in enumerate(frames):
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if on_frame:
                on_frame(i + 1)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {err.strip()[:500]}")
    return out_path


def trim_clip(src: Path, out_path: Path, seconds: float, max_side: int) -> Path:
    """Short, downscaled H.264 copy of the reference, sized for upload to the generator."""
    info = probe(src)
    w, h = fit_size(info.width, info.height, max_side)
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-t", f"{seconds:.2f}",
           "-vf", f"scale={w}:{h}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(out_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr.strip()[:500]}")
    return out_path
