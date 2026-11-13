"""IoU tracker tests."""

import numpy as np
import pytest

from face_presence.detection import Detection
from face_presence.tracking import IoUTracker, Track, iou


def make_detection(
    x: float, y: float, width: float, height: float, score: float = 0.9
) -> Detection:
    row = np.array(
        [
            x,
            y,
            width,
            height,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            score,
        ],
        dtype=np.float32,
    )
    return Detection(row)


def test_iou_of_identical_boxes_is_one() -> None:
    assert iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero() -> None:
    assert iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 10.0, 10.0)) == 0.0


def test_iou_of_partial_overlap() -> None:
    assert iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 10.0, 10.0)) == pytest.approx(
        1.0 / 3.0
    )


def test_tracker_keeps_identity_for_overlapping_detections() -> None:
    tracker = IoUTracker(iou_threshold=0.3, max_age_s=1.0)

    first = tracker.update([make_detection(10, 10, 50, 50)], now=0.0)
    track_id = first[0].track_id

    second = tracker.update([make_detection(12, 12, 50, 50)], now=0.1)

    assert len(second) == 1
    assert second[0].track_id == track_id
    assert second[0].hits == 2


def test_tracker_creates_a_new_identity_for_a_distant_face() -> None:
    tracker = IoUTracker(iou_threshold=0.3, max_age_s=1.0)
    first = tracker.update([make_detection(10, 10, 50, 50)], now=0.0)

    tracks = tracker.update([make_detection(300, 300, 40, 40)], now=0.1)

    assert len(tracks) == 2
    assert {track.track_id for track in tracks} != {first[0].track_id}


def test_tracker_expires_unseen_tracks() -> None:
    tracker = IoUTracker(iou_threshold=0.3, max_age_s=1.0)
    tracker.update([make_detection(10, 10, 50, 50)], now=0.0)

    assert len(tracker.active(0.5)) == 1
    assert tracker.active(1.1) == []


def test_tracker_matches_multiple_faces_by_position() -> None:
    tracker = IoUTracker(iou_threshold=0.3, max_age_s=1.0)
    first = tracker.update(
        [
            make_detection(10, 10, 40, 40),
            make_detection(200, 10, 40, 40),
        ],
        now=0.0,
    )
    ids_by_x = {round(track.box[0]): track.track_id for track in first}

    second = tracker.update(
        [
            make_detection(202, 12, 40, 40),
            make_detection(12, 12, 40, 40),
        ],
        now=0.1,
    )

    assert len(second) == 2
    second_ids_by_x = {round(track.box[0]): track.track_id for track in second}
    assert second_ids_by_x[202] == ids_by_x[200]
    assert second_ids_by_x[12] == ids_by_x[10]


def test_track_exposes_a_display_label_and_landmarks() -> None:
    track = Track(
        track_id=3,
        box=(10.0, 20.0, 30.0, 40.0),
        last_seen=0.0,
        landmarks=np.zeros((5, 2), dtype=np.float32),
    )

    assert track.display_label == "Face 03"
    assert track.landmarks is not None
