"""YuNet face detection wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from face_presence.config import DetectionSettings


class DetectionError(RuntimeError):
    """Raised when the detector cannot be loaded or run."""


@dataclass(frozen=True)
class Detection:
    """A face detection expressed in the coordinates of the source frame."""

    face_row: np.ndarray

    def __post_init__(self) -> None:
        row = np.asarray(self.face_row, dtype=np.float32)
        if row.shape != (15,):
            raise ValueError("face_row must contain exactly 15 values")
        object.__setattr__(self, "face_row", row)

    @property
    def x(self) -> int:
        return int(round(float(self.face_row[0])))

    @property
    def y(self) -> int:
        return int(round(float(self.face_row[1])))

    @property
    def width(self) -> int:
        return int(round(float(self.face_row[2])))

    @property
    def height(self) -> int:
        return int(round(float(self.face_row[3])))

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    @property
    def landmarks(self) -> np.ndarray:
        return self.face_row[4:14].reshape(5, 2)

    @property
    def score(self) -> float:
        return float(self.face_row[14])


def scaled_size(width: int, height: int, input_width: int) -> tuple[int, int]:
    """Return the detection size that preserves the frame aspect ratio."""
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    if input_width <= 0:
        raise ValueError("input_width must be positive")

    target_width = min(input_width, width)
    scale = target_width / width
    target_height = max(1, round(height * scale))
    return target_width, target_height


def scale_face_rows(
    faces: np.ndarray, scale_x: float, scale_y: float
) -> np.ndarray:
    """Scale detector output rows from the detection frame back to the source frame."""
    scaled = np.asarray(faces, dtype=np.float32).copy()
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    scaled[:, 2] *= scale_x
    scaled[:, 3] *= scale_y
    scaled[:, 4:14:2] *= scale_x
    scaled[:, 5:14:2] *= scale_y
    return scaled


class FaceDetector:
    """Thin wrapper around OpenCV's YuNet detector."""

    def __init__(self, settings: DetectionSettings, model_path: Path) -> None:
        if not model_path.is_file():
            raise DetectionError(
                f"Face detector model not found: {model_path}. "
                "Run 'python -m face_presence setup' first."
            )

        self._input_width = settings.input_width
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(model_path),
                "",
                (settings.input_width, settings.input_width),
                settings.score_threshold,
                settings.nms_threshold,
                50,
            )
        except cv2.error as exc:
            raise DetectionError(f"Could not load the face detector: {exc}") from exc

        self._last_size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        if width == 0 or height == 0:
            return []

        target_width, target_height = scaled_size(width, height, self._input_width)
        if self._last_size != (target_width, target_height):
            self._detector.setInputSize((target_width, target_height))
            self._last_size = (target_width, target_height)

        resized = cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
        try:
            _, faces = self._detector.detect(resized)
        except cv2.error as exc:
            raise DetectionError(f"Face detection failed: {exc}") from exc

        if faces is None or len(faces) == 0:
            return []

        rows = scale_face_rows(
            faces,
            scale_x=width / target_width,
            scale_y=height / target_height,
        )
        return [Detection(row) for row in rows]
