"""Live camera preview pipeline."""

from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import suppress
from dataclasses import replace

import cv2

from face_presence.camera import open_camera
from face_presence.config import Settings
from face_presence.detection import FaceDetector
from face_presence.identity import IdentityManager
from face_presence.model_assets import SFACE, YUNET
from face_presence.overlay import OverlayRenderer
from face_presence.recognition import FaceRecognizer, Gallery
from face_presence.storage import Database
from face_presence.tracking import IoUTracker

logger = logging.getLogger(__name__)


class FpsMeter:
    """Estimate the processing rate over a rolling window of frame timestamps."""

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=max(2, window))

    def tick(self, now: float) -> float:
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


def next_detection_interval(
    current: int, base: int, fps: float, max_interval: int = 4
) -> int:
    """Adjust the detection interval to keep the preview responsive."""
    if fps <= 0.0:
        return current
    if fps < 10.0 and current < max_interval:
        return current + 1
    if fps > 24.0 and current > base:
        return current - 1
    return current


def run_app(settings: Settings, camera_index: int | None = None) -> int:
    """Open the configured camera and run the live preview until the user quits."""
    detector = FaceDetector(
        settings.detection,
        settings.models.directory / YUNET.filename,
    )
    recognizer = FaceRecognizer(settings.models.directory / SFACE.filename)
    database = Database(settings.storage.db_path)

    try:
        gallery = Gallery(database.load_persons())
        identity = IdentityManager(recognizer, gallery, database, settings.recognition)

        camera_settings = settings.camera
        if camera_index is not None:
            camera_settings = replace(camera_settings, index=camera_index)

        camera = open_camera(camera_settings)
        tracker = IoUTracker()
        overlay = OverlayRenderer(
            settings.ui.window_title,
            show_fps=settings.ui.show_fps,
            show_landmarks=settings.ui.show_landmarks,
        )
        fps_meter = FpsMeter()

        base_interval = settings.detection.interval_frames
        interval = base_interval
        max_interval = max(4, base_interval)
        frame_number = 0
        consecutive_failures = 0
        last_adjustment = time.monotonic()

        logger.info(
            "Camera %d opened with %s at %dx%d",
            camera.index,
            camera.backend,
            camera.width,
            camera.height,
        )
        logger.info("Known people loaded: %d", gallery.count)
        logger.info("Press q or Esc to quit.")

        try:
            while True:
                frame = camera.read()
                now = time.monotonic()

                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 60:
                        logger.error("The camera stopped delivering frames.")
                        return 1
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0
                frame_number += 1

                if (frame_number - 1) % interval == 0:
                    detections = detector.detect(frame)
                    tracks = tracker.update(detections, now)
                    identity.update(frame, tracks, now)
                else:
                    tracks = tracker.active(now)

                fps = fps_meter.tick(now)
                overlay.draw(
                    frame,
                    tracks,
                    fps,
                    camera_settings.index,
                    camera.backend,
                    interval,
                )

                try:
                    cv2.imshow(overlay.window_title, frame)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as exc:
                    logger.error("Could not render the preview window: %s", exc)
                    return 1

                if key in (27, ord("q"), ord("Q")):
                    break
                if frame_number > 10 and not _window_is_visible(overlay.window_title):
                    break

                if settings.detection.adaptive and now - last_adjustment >= 2.0:
                    interval = next_detection_interval(
                        interval, base_interval, fps, max_interval
                    )
                    last_adjustment = now
        except KeyboardInterrupt:
            logger.info("Interrupted.")
        finally:
            camera.release()
            with suppress(cv2.error):
                cv2.destroyAllWindows()
    finally:
        database.close()

    return 0


def _window_is_visible(title: str) -> bool:
    try:
        return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return True
