"""Face detection wrapper tests."""

from pathlib import Path

import numpy as np
import pytest

from face_presence.config import DetectionSettings
from face_presence.detection import (
    Detection,
    DetectionError,
    FaceDetector,
    scale_face_rows,
    scaled_size,
)


def make_row(
    x: float = 10.0,
    y: float = 20.0,
    width: float = 50.0,
    height: float = 60.0,
    score: float = 0.9,
) -> np.ndarray:
    return np.array(
        [
            x,
            y,
            width,
            height,
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            90.0,
            100.0,
            score,
        ],
        dtype=np.float32,
    )


def test_scaled_size_preserves_aspect_ratio() -> None:
    assert scaled_size(640, 480, 320) == (320, 240)
    assert scaled_size(1280, 720, 320) == (320, 180)
    assert scaled_size(160, 120, 320) == (160, 120)


def test_scaled_size_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        scaled_size(0, 480, 320)
    with pytest.raises(ValueError, match="input_width"):
        scaled_size(640, 480, 0)


def test_scale_face_rows_scales_box_and_landmarks() -> None:
    rows = np.stack([make_row()])

    scaled = scale_face_rows(rows, scale_x=2.0, scale_y=0.5)

    assert scaled[0, 0] == pytest.approx(20.0)
    assert scaled[0, 1] == pytest.approx(10.0)
    assert scaled[0, 2] == pytest.approx(100.0)
    assert scaled[0, 3] == pytest.approx(30.0)
    assert scaled[0, 4] == pytest.approx(20.0)
    assert scaled[0, 5] == pytest.approx(10.0)
    assert scaled[0, 14] == pytest.approx(0.9)


def test_detection_exposes_box_landmarks_and_score() -> None:
    detection = Detection(make_row())

    assert detection.box == (10, 20, 50, 60)
    assert detection.score == pytest.approx(0.9)
    assert detection.landmarks.shape == (5, 2)
    assert detection.landmarks[0].tolist() == pytest.approx([10.0, 20.0])


def test_detection_rejects_incomplete_rows() -> None:
    with pytest.raises(ValueError, match="15 values"):
        Detection(np.zeros(14, dtype=np.float32))


def test_face_detector_requires_a_model(tmp_path: Path) -> None:
    with pytest.raises(DetectionError, match="not found"):
        FaceDetector(DetectionSettings(), tmp_path / "missing.onnx")
