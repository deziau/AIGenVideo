import json

import httpx

from app.generators import RunwayGenerator, nearest_ratio


def test_nearest_ratio():
    assert nearest_ratio(1920, 1080) == "1280:720"
    assert nearest_ratio(1080, 1920) == "720:1280"
    assert nearest_ratio(500, 500) == "960:960"


def test_generate_flow(sample_video, tmp_path):
    seen = {"polls": 0}

    def api(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/video_to_video"):
            body = json.loads(request.content)
            seen["body"] = body
            seen["auth"] = request.headers["authorization"]
            return httpx.Response(200, json={"id": "task-1"})
        seen["polls"] += 1
        if seen["polls"] < 2:
            return httpx.Response(200, json={"status": "RUNNING", "progress": 0.5})
        return httpx.Response(200, json={"status": "SUCCEEDED", "output": ["https://cdn.example/out.mp4"]})

    def cdn(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, content=b"MP4DATA")

    gen = RunwayGenerator(api_key="k", poll_interval=0)
    gen.http = httpx.Client(base_url="https://api.test/v1", transport=httpx.MockTransport(api),
                            headers=dict(gen.http.headers))
    gen.download_http = httpx.Client(transport=httpx.MockTransport(cdn))
    statuses = []
    out = gen.generate(sample_video, "a prompt", tmp_path / "o.mp4", on_status=statuses.append)

    assert out.read_bytes() == b"MP4DATA"
    assert seen["body"]["model"] == "gen4_aleph"
    assert seen["body"]["videoUri"].startswith("data:video/mp4;base64,")
    assert seen["body"]["ratio"] == "1280:720"
    assert seen["auth"] == "Bearer k"
    assert any("50%" in s for s in statuses)


def test_failed_task_raises(sample_video, tmp_path):
    def api(request):
        if request.method == "POST":
            return httpx.Response(200, json={"id": "t"})
        return httpx.Response(200, json={"status": "FAILED", "failure": "bad input"})

    gen = RunwayGenerator(api_key="k", poll_interval=0)
    gen.http = httpx.Client(base_url="https://api.test/v1", transport=httpx.MockTransport(api))
    try:
        gen.generate(sample_video, "p", tmp_path / "o.mp4")
    except RuntimeError as e:
        assert "bad input" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
