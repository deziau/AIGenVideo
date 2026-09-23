import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Isolate data and make sure tests never call real APIs.
os.environ["AIGV_DATA_DIR"] = tempfile.mkdtemp(prefix="aigv-test-")
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
os.environ.pop("RUNWAYML_API_SECRET", None)

from app.video_io import FFMPEG  # noqa: E402
import subprocess  # noqa: E402


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """2-second 320x180 clip with a moving shape and a sine-wave audio track."""
    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=12:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out),
    ], check=True)
    return out


@pytest.fixture
def frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.zeros((120, 160, 3), np.uint8)
    img[20:100, 30:130] = (40, 160, 220)
    img += rng.integers(0, 30, img.shape, dtype=np.uint8)
    return img
