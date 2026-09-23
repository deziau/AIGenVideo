# AIGenVideo Studio

A web app with two modes:

1. **Create from reference.** Upload a reference video and describe the video you want.
   Claude looks at keyframes from the reference and your request, then writes a detailed prompt
   for a video model. The video model (Runway `gen4_aleph`, video-to-video) renders a new
   video that keeps the reference's motion and framing and changes what your prompt describes.
2. **Cartoonize / Paint.** Upload any video and turn it into a **cartoon**, **comic book**,
   **oil painting**, **watercolor** or **pencil sketch**. This runs locally with OpenCV and needs
   no API keys. It keeps the original audio and has a flicker-smoothing control. You can preview
   a single frame before rendering the whole video.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # then fill in the keys you have
set -a && source .env && set +a
uvicorn app.main:app --reload
```

Open http://localhost:8000.

| Key | What it enables | Without it |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Claude studies the reference and writes the prompt | Your text is used as the prompt |
| `RUNWAYML_API_SECRET` | Generative video-to-video rendering | A local "preview" render applies the closest look (cinematic, noir, cartoon…) to the reference |

The Cartoonize / Paint mode always works, even without keys. ffmpeg is bundled through `imageio-ffmpeg`, so you don't need to install it separately.

## How it works

```
reference.mp4 ──► 8 keyframes ──► Claude (vision + structured output) ──► title, summary, prompt
      │                                                                          │
      └──► first N seconds, ≤1280px ─────────────────────────────► Runway gen4_aleph ──► result.mp4
```

- `app/prompt_director.py`: Claude call. It uses a JSON-schema response, adaptive thinking and
  server-side refusal fallbacks.
- `app/generators.py`: Runway client (submit a task, poll it, download the result) and the offline preview looks.
- `app/stylize.py`: the art filters: LAB posterization with cleaned-up ink lines (cartoon/comic),
  a Kuwahara filter (oil), edge-preserving stylization on paper grain (watercolor), and pencil sketch.
  Every filter is deterministic per frame, and an optional EMA smoother reduces flicker.
- `app/video_io.py`: OpenCV frame reading and H.264/AAC encoding through ffmpeg.
- `app/main.py`: the FastAPI endpoints. Long jobs run on a thread pool, and the UI polls `/api/jobs/{id}`.
- `static/`: a dependency-free frontend with light and dark themes that works on mobile.

## Configuration

| Env var | Default | |
| --- | --- | --- |
| `AIGV_CLAUDE_MODEL` | `claude-opus-5` | Model that writes the prompt |
| `AIGV_REFERENCE_SECONDS` | `5` | Seconds of the reference sent to Runway |
| `AIGV_MAX_SIDE` | `1280` | Max output dimension for local processing |
| `AIGV_MAX_UPLOAD_MB` | `200` | Upload size limit |
| `AIGV_DATA_DIR` | `./data` | Where uploads and outputs are stored |

## Tests

```bash
pytest
```

The tests generate a synthetic clip with ffmpeg. They mock both Claude and Runway, so they need no keys or network.

## Notes and limits

- Runway receives the reference clip as an inline data URI, which works for short clips.
  For long or high-resolution references, host the file and pass its URL instead.
- Job state lives in memory. For multi-user or production use, move it to a real queue
  (Redis/RQ, Celery) and put outputs in object storage.
- Only upload videos you have the rights to use.
