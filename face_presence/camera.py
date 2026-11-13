"""Camera discovery and capture helpers."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np

from face_presence.config import CameraSettings

_BACKENDS: dict[str, int] = {
    "default": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "v4l2": cv2.CAP_V4L2,
}


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened."""


@dataclass(frozen=True)
class CameraInfo:
    index: int
    backend: str
    width: int
    height: int


@dataclass
class Camera:
    capture: cv2.VideoCapture
    index: int
    backend: str
    width: int
    height: int

    def read(self) -> np.ndarray | None:
        """Read the next frame, or return None when the device stops delivering frames."""
        try:
            ok, frame = self.capture.read()
        except cv2.error:
            return None
        return frame if ok and frame is not None and frame.size else None

    def release(self) -> None:
        self.capture.release()


def backend_candidates(
    backend: str = "auto", platform: str | None = None
) -> list[tuple[int, str]]:
    """Return the camera backends to try, in order, for the current platform."""
    platform = platform or sys.platform
    backend = backend.lower()

    if backend != "auto":
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown camera backend: {backend}")
        return [(_BACKENDS[backend], backend)]

    if platform.startswith("win"):
        names = ["dshow", "msmf", "default"]
    elif platform == "darwin":
        names = ["avfoundation", "default"]
    else:
        names = ["v4l2", "default"]

    candidates: list[tuple[int, str]] = []
    for name in names:
        candidate = (_BACKENDS[name], name)
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def open_camera(settings: CameraSettings, retries: int = 3) -> Camera:
    """Open the configured camera, trying the available backends and a few retries."""
    errors: list[str] = []

    for attempt in range(1, retries + 1):
        for backend_id, backend_name in backend_candidates(settings.backend):
            capture = cv2.VideoCapture(settings.index, backend_id)
            if not capture.isOpened():
                capture.release()
                errors.append(f"{backend_name}: could not open device {settings.index}")
                continue

            _configure(capture, settings.width, settings.height, settings.fps)
            frame = _read_frame(capture)
            if frame is None:
                capture.release()
                errors.append(f"{backend_name}: no frames from device {settings.index}")
                continue

            height, width = frame.shape[:2]
            return Camera(capture, settings.index, backend_name, width, height)

        if attempt < retries:
            time.sleep(0.3 * attempt)

    detail = "; ".join(errors[-6:]) or "no backends available"
    raise CameraError(
        f"Could not open camera {settings.index}. {detail}. "
        "Run 'python -m face_presence cameras' to list available devices."
    )


def probe_cameras(
    max_index: int = 5,
    width: int = 640,
    height: int = 480,
    backend: str = "auto",
) -> list[CameraInfo]:
    """Scan camera indexes and return the devices that deliver at least one frame."""
    if max_index < 0:
        raise ValueError("max_index must be zero or greater")

    previous_level = _silence_opencv_logging()
    found: list[CameraInfo] = []

    try:
        with _suppress_native_stderr():
            for index in range(max_index + 1):
                for backend_id, backend_name in backend_candidates(backend):
                    capture = cv2.VideoCapture(index, backend_id)
                    if not capture.isOpened():
                        capture.release()
                        continue

                    _configure(capture, width, height, 30)
                    frame = _read_frame(capture, attempts=3)
                    capture.release()

                    if frame is None:
                        continue

                    found.append(
                        CameraInfo(index, backend_name, frame.shape[1], frame.shape[0])
                    )
                    break
    finally:
        if previous_level is not None:
            cv2.utils.logging.setLogLevel(previous_level)

    return found


def _configure(capture: cv2.VideoCapture, width: int, height: int, fps: int) -> None:
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def _read_frame(capture: cv2.VideoCapture, attempts: int = 5) -> np.ndarray | None:
    for _ in range(attempts):
        try:
            ok, frame = capture.read()
        except cv2.error:
            return None
        if ok and frame is not None and frame.size:
            return frame
        time.sleep(0.05)
    return None


def _silence_opencv_logging() -> int | None:
    try:
        previous = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        return None
    return int(previous)


@contextmanager
def _suppress_native_stderr() -> Iterator[None]:
    """Redirect the process stderr to the null device while probing devices."""
    try:
        stderr_fd = sys.stderr.fileno()
        saved_fd = os.dup(stderr_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
    except (AttributeError, OSError, ValueError):
        yield
        return

    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull_fd)
