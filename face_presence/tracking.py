"""Lightweight IoU tracker for face detections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from face_presence.detection import Detection

Box = tuple[float, float, float, float]


@dataclass
class Track:
    """A face followed across consecutive detection passes."""

    track_id: int
    box: Box
    last_seen: float
    hits: int = 1
    score: float = 0.0
    face_row: np.ndarray | None = None
    landmarks: np.ndarray | None = None
    person_id: int | None = None
    label: str | None = None
    timer_s: float | None = None

    @property
    def display_label(self) -> str:
        return self.label or f"Face {self.track_id:02d}"


def iou(box_a: Box, box_b: Box) -> float:
    """Return the intersection over union of two boxes in x, y, width, height form."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def greedy_match(
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Match tracks to detections by descending IoU, using each item at most once."""
    candidates: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            score = iou(track.box, detection.box)
            if score >= iou_threshold:
                candidates.append((score, track_index, detection_index))

    candidates.sort(reverse=True)

    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for score, track_index, detection_index in candidates:
        if track_index in matched_tracks or detection_index in matched_detections:
            continue
        matched_tracks.add(track_index)
        matched_detections.add(detection_index)
        matches.append((track_index, detection_index, score))

    return matches


class IoUTracker:
    """Assign stable track identifiers to detections across frames."""

    def __init__(self, iou_threshold: float = 0.3, max_age_s: float = 0.75) -> None:
        self._iou_threshold = iou_threshold
        self._max_age_s = max_age_s
        self._tracks: list[Track] = []
        self._next_id = 1

    def active(self, now: float) -> list[Track]:
        """Return tracks that have been seen recently, dropping expired ones."""
        self._tracks = [
            track for track in self._tracks if now - track.last_seen <= self._max_age_s
        ]
        return list(self._tracks)

    def update(self, detections: Sequence[Detection], now: float) -> list[Track]:
        """Update the tracker with a new detection pass and return active tracks."""
        active = self.active(now)
        matches = greedy_match(active, detections, self._iou_threshold)

        matched_detections: set[int] = set()
        for track_index, detection_index, _ in matches:
            track = active[track_index]
            detection = detections[detection_index]
            track.box = detection.box
            track.last_seen = now
            track.hits += 1
            track.score = detection.score
            track.face_row = detection.face_row
            track.landmarks = detection.landmarks
            matched_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            active.append(
                Track(
                    track_id=self._next_id,
                    box=detection.box,
                    last_seen=now,
                    score=detection.score,
                    face_row=detection.face_row,
                    landmarks=detection.landmarks,
                )
            )
            self._next_id += 1

        self._tracks = active
        return list(active)
