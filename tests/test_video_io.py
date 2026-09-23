import subprocess

from app.video_io import FFMPEG, fit_size, iter_frames, probe, sample_frames, trim_clip, write_video


def test_fit_size_even_and_bounded():
    assert fit_size(1920, 1080, 1280) == (1280, 720)
    assert fit_size(333, 201, 1280) == (332, 200)
    w, h = fit_size(1081, 1921, 640)
    assert max(w, h) <= 640 and w % 2 == 0 and h % 2 == 0


def test_probe_and_sample(sample_video):
    info = probe(sample_video)
    assert (info.width, info.height) == (320, 180)
    assert 20 <= info.frame_count <= 25
    assert len(sample_frames(sample_video, 5)) == 5


def test_write_video_keeps_audio(sample_video, tmp_path):
    out = tmp_path / "o.mp4"
    frames = iter_frames(sample_video, (320, 180))
    write_video(frames, out, (320, 180), 12, audio_from=sample_video)
    streams = subprocess.run([FFMPEG, "-i", str(out)], capture_output=True, text=True).stderr
    assert "Video: h264" in streams and "Audio: aac" in streams


def test_trim_clip(sample_video, tmp_path):
    out = trim_clip(sample_video, tmp_path / "c.mp4", 1.0, 160)
    info = probe(out)
    assert info.width == 160 and info.duration <= 1.2
