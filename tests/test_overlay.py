"""Overlay rendering tests."""

import numpy as np
import pytest

from face_presence.overlay import OverlayRenderer, format_duration
from face_presence.tracking import Track


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00"),
        (5, "00:05"),
        (65, "01:05"),
        (3665, "01:01:05"),
        (-10, "00:00"),
    ],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_draw_adds_boxes_and_hud_without_changing_the_shape() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    original = frame.copy()
    track = Track(
        track_id=7,
        box=(10.0, 20.0, 80.0, 90.0),
        last_seen=0.0,
        landmarks=np.array(
            [[20.0, 30.0], [40.0, 50.0], [60.0, 70.0], [80.0, 90.0], [90.0, 100.0]],
            dtype=np.float32,
        ),
        person_id=1,
        label="Person_001",
        timer_s=65.0,
    )
    renderer = OverlayRenderer("Test", show_fps=True, show_landmarks=True)

    result = renderer.draw(frame, [track], 24.5, 0, "dshow", 2)

    assert result.shape == (240, 320, 3)
    assert result.dtype == np.uint8
    assert not np.array_equal(result, original)


def test_draw_handles_an_empty_scene() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    renderer = OverlayRenderer("Test", show_fps=False, show_landmarks=False)

    result = renderer.draw(frame, [], 0.0, 1, "avfoundation", 4)

    assert result.shape == (240, 320, 3)
    assert int(result.sum()) > 0
