import numpy as np
import pytest

from app.stylize import FILTERS, STYLES, TemporalSmoother, apply_style


def test_every_style_is_exposed():
    assert set(FILTERS) == set(STYLES)


@pytest.mark.parametrize("style", list(FILTERS))
@pytest.mark.parametrize("strength", [0.0, 0.6, 1.0])
def test_style_keeps_shape_and_changes_image(style, strength, frame):
    out = apply_style(frame, style, strength)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert np.mean(np.abs(out.astype(int) - frame.astype(int))) > 1


def test_styles_are_deterministic(frame):
    for style in FILTERS:
        assert np.array_equal(apply_style(frame, style), apply_style(frame, style))


def test_unknown_style_rejected(frame):
    with pytest.raises(ValueError):
        apply_style(frame, "nope")


def test_smoother_blends_but_resets_on_cut():
    s = TemporalSmoother(0.5)
    a = np.full((4, 4, 3), 100, np.uint8)
    s(a)
    assert s(np.full((4, 4, 3), 120, np.uint8))[0, 0, 0] == 110
    assert s(np.full((4, 4, 3), 250, np.uint8))[0, 0, 0] == 250  # hard cut
