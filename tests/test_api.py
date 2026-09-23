import time

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def wait(job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.2)
    raise TimeoutError


def upload(path):
    return {"video": ("clip.mp4", path.read_bytes(), "video/mp4")}


def test_config_and_index():
    cfg = client.get("/api/config").json()
    assert cfg["claude"] is False and cfg["runway"] is False
    assert {s["id"] for s in cfg["styles"]} >= {"cartoon", "oil_painting"}
    assert "AIGenVideo" in client.get("/").text


def test_stylize_end_to_end(sample_video):
    r = client.post("/api/stylize", files=upload(sample_video), data={"style": "cartoon", "strength": "0.7"})
    assert r.status_code == 200
    job = wait(r.json()["id"])
    assert job["status"] == "done", job
    video = client.get(job["result"]["video_url"])
    assert video.status_code == 200 and len(video.content) > 1000


def test_stylize_preview(sample_video):
    r = client.post("/api/stylize/preview", files=upload(sample_video), data={"style": "watercolor"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_generate_local_preview_without_keys(sample_video):
    r = client.post("/api/generate", files=upload(sample_video), data={"prompt": "moody film noir detective scene"})
    job = wait(r.json()["id"])
    assert job["status"] == "done", job
    assert job["result"]["engine"] == "local_preview"
    assert job["result"]["plan"]["preview_look"] == "noir"


def test_validation(sample_video):
    assert client.post("/api/stylize", files=upload(sample_video), data={"style": "bogus"}).status_code == 400
    bad = {"video": ("notes.txt", b"hello", "text/plain")}
    assert client.post("/api/stylize", files=bad, data={"style": "cartoon"}).status_code == 400
    fake = {"video": ("x.mp4", b"not a video", "video/mp4")}
    assert client.post("/api/stylize", files=fake, data={"style": "cartoon"}).status_code == 400
    assert client.get("/api/jobs/missing").status_code == 404
