"""Live overlay rendering."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from face_presence.tracking import Track

_TRACK_COLOR = (0, 200, 255)
_LABEL_TEXT_COLOR = (20, 20, 20)
_HUD_TEXT_COLOR = (240, 240, 240)
_HUD_PANEL_COLOR = (20, 20, 20)
_HUD_SCALE = 0.45
_LABEL_SCALE = 0.5
_TEXT_THICKNESS = 1


def format_duration(seconds: float) -> str:
    """Format a duration as MM:SS, or HH:MM:SS for sessions over an hour."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class OverlayRenderer:
    """Draw face boxes, labels and a compact status panel on a frame."""

    def __init__(
        self,
        window_title: str,
        show_fps: bool = True,
        show_landmarks: bool = False,
    ) -> None:
        self.window_title = window_title
        self.show_fps = show_fps
        self.show_landmarks = show_landmarks

    def draw(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        fps: float,
        camera_index: int,
        backend: str,
        interval_frames: int,
    ) -> np.ndarray:
        for track in tracks:
            self._draw_track(frame, track)

        self._draw_hud(frame, fps, len(tracks), camera_index, backend, interval_frames)
        self._draw_hint(frame)
        return frame

    def _draw_track(self, frame: np.ndarray, track: Track) -> None:
        height, width = frame.shape[:2]
        x, y, box_width, box_height = (int(round(value)) for value in track.box)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        x2 = max(0, min(x + box_width, width - 1))
        y2 = max(0, min(y + box_height, height - 1))

        cv2.rectangle(frame, (x, y), (x2, y2), _TRACK_COLOR, 2, cv2.LINE_AA)
        self._draw_label(frame, track, x, y)

        if self.show_landmarks and track.landmarks is not None:
            for point in track.landmarks:
                center = (int(round(float(point[0]))), int(round(float(point[1]))))
                cv2.circle(frame, center, 2, _TRACK_COLOR, -1, cv2.LINE_AA)

    def _draw_label(self, frame: np.ndarray, track: Track, x: int, y: int) -> None:
        label = track.display_label
        if track.timer_s is not None:
            label = f"{label}  {format_duration(track.timer_s)}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(
            label, font, _LABEL_SCALE, _TEXT_THICKNESS
        )
        top = max(0, y - text_height - 10)
        bottom = min(frame.shape[0] - 1, top + text_height + 10)
        right = min(frame.shape[1] - 1, x + text_width + 12)

        cv2.rectangle(frame, (x, top), (right, bottom), _TRACK_COLOR, -1)
        cv2.putText(
            frame,
            label,
            (x + 6, bottom - 5),
            font,
            _LABEL_SCALE,
            _LABEL_TEXT_COLOR,
            _TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        face_count: int,
        camera_index: int,
        backend: str,
        interval_frames: int,
    ) -> None:
        lines: list[str] = []
        if self.show_fps:
            lines.append(f"FPS {fps:4.1f}")
        lines.append(f"Faces {face_count}")
        lines.append(f"Camera {camera_index} | {backend} | {frame.shape[1]}x{frame.shape[0]}")
        lines.append(f"Detection every {interval_frames} frame(s)")

        font = cv2.FONT_HERSHEY_SIMPLEX
        sizes = [
            cv2.getTextSize(line, font, _HUD_SCALE, _TEXT_THICKNESS)[0] for line in lines
        ]
        panel_width = max(width for width, _ in sizes) + 20
        panel_height = sum(height for _, height in sizes) + 10 * (len(lines) + 1)

        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), _HUD_PANEL_COLOR, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0.0, frame)

        y = 18
        for line, (_, text_height) in zip(lines, sizes, strict=True):
            y += text_height
            cv2.putText(
                frame,
                line,
                (18, y),
                font,
                _HUD_SCALE,
                _HUD_TEXT_COLOR,
                _TEXT_THICKNESS,
                cv2.LINE_AA,
            )
            y += 6

    def _draw_hint(self, frame: np.ndarray) -> None:
        cv2.putText(
            frame,
            "q / ESC: quit",
            (12, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            _HUD_TEXT_COLOR,
            _TEXT_THICKNESS,
            cv2.LINE_AA,
        )
