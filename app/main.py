from __future__ import annotations

import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from . import config
from .generators import RunwayGenerator, local_preview
from .jobs import Job, JobStore
from .prompt_director import plan_video
from .stylize import STYLES, TemporalSmoother, apply_style
from .video_io import fit_size, iter_frames, probe, sample_frames, trim_clip, write_video

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}

app = FastAPI(title="AIGenVideo")
jobs = JobStore()


def _save_upload(upload: UploadFile) -> Path:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(ALLOWED_EXT))}")
    dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with dest.open("wb") as f:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB")
            f.write(chunk)
    try:
        probe(dest)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(exc))
    return dest


def _cleanup_after(work, *paths: Path):
    """Wrap a job so its uploaded inputs are deleted once it finishes."""
    def run(job: Job) -> dict:
        try:
            return work(job)
        finally:
            for p in paths:
                p.unlink(missing_ok=True)
            (config.UPLOAD_DIR / f"clip_{job.id}.mp4").unlink(missing_ok=True)
    return run


def _output_url(path: Path) -> str:
    return f"/media/{path.name}"


@app.get("/api/config")
def get_config() -> dict:
    return {
        "claude": config.anthropic_enabled(),
        "runway": config.runway_enabled(),
        "styles": [{"id": k, "label": v} for k, v in STYLES.items()],
        "reference_seconds": config.REFERENCE_SECONDS,
    }


@app.post("/api/generate")
def generate(video: UploadFile = File(...), prompt: str = Form(...)) -> dict:
    if not prompt.strip():
        raise HTTPException(400, "Describe the video you want")
    if len(prompt) > 4000:
        raise HTTPException(400, "Prompt is too long (max 4000 characters)")
    src = _save_upload(video)

    def work(job: Job) -> dict:
        job.update(message="Claude is studying your reference video", progress=0.05)
        plan = plan_video(sample_frames(src, 8), prompt)
        result = {"plan": plan.to_dict()}
        out = config.OUTPUT_DIR / f"gen_{job.id}.mp4"

        if config.runway_enabled():
            job.update(message="Preparing reference clip", progress=0.15)
            clip = trim_clip(src, config.UPLOAD_DIR / f"clip_{job.id}.mp4", config.REFERENCE_SECONDS, 1280)
            RunwayGenerator().generate(clip, plan.generation_prompt, out,
                                       on_status=lambda s: job.update(message=s, progress=max(job.progress, 0.3)))
            result["engine"] = "runway"
        else:
            job.update(message=f"Rendering local preview ({plan.preview_look.replace('_', ' ')})", progress=0.2)
            local_preview(src, plan.preview_look, out, on_progress=lambda p: job.update(progress=0.2 + 0.8 * p))
            result["engine"] = "local_preview"
        result["video_url"] = _output_url(out)
        return result

    return jobs.submit("generate", _cleanup_after(work, src)).to_dict()


@app.post("/api/stylize")
def stylize(
    video: UploadFile = File(...),
    style: str = Form("cartoon"),
    strength: float = Form(0.6),
    smoothing: float = Form(0.35),
) -> dict:
    if style not in STYLES:
        raise HTTPException(400, f"Unknown style '{style}'")
    if not 0 <= strength <= 1 or not 0 <= smoothing <= 0.9:
        raise HTTPException(400, "strength must be 0-1 and smoothing 0-0.9")
    src = _save_upload(video)

    def work(job: Job) -> dict:
        info = probe(src)
        size = fit_size(info.width, info.height, config.MAX_SIDE)
        total = max(info.frame_count, 1)
        smooth = TemporalSmoother(smoothing)
        out = config.OUTPUT_DIR / f"{style}_{job.id}.mp4"
        job.update(message=f"Painting frames ({STYLES[style].split(' - ')[0]})")
        frames = (smooth(apply_style(f, style, strength)) for f in iter_frames(src, size))
        write_video(frames, out, size, info.fps, audio_from=src,
                    on_frame=lambda i: job.update(progress=min(i / total, 0.99), message=f"Frame {i} of {total}"))
        return {"video_url": _output_url(out), "style": style}

    return jobs.submit("stylize", _cleanup_after(work, src)).to_dict()


@app.post("/api/stylize/preview")
def stylize_preview(video: UploadFile = File(...), style: str = Form("cartoon"), strength: float = Form(0.6)) -> Response:
    """Single stylized frame so users can compare looks before rendering the whole video."""
    if style not in STYLES:
        raise HTTPException(400, f"Unknown style '{style}'")
    src = _save_upload(video)
    frames = sample_frames(src, 3, max_side=config.MAX_SIDE)
    src.unlink(missing_ok=True)
    if not frames:
        raise HTTPException(400, "Could not read frames from video")
    ok, buf = cv2.imencode(".jpg", apply_style(frames[len(frames) // 2], style, strength), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


app.mount("/media", StaticFiles(directory=config.OUTPUT_DIR), name="media")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
