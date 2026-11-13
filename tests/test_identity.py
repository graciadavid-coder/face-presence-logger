"""Identity confirmation and enrollment tests."""

import numpy as np

from face_presence.config import RecognitionSettings
from face_presence.identity import IdentityManager
from face_presence.recognition import Gallery
from face_presence.storage import PersonRecord
from face_presence.tracking import Track


def make_vector(index: int = 0) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector


def make_track(
    hits: int = 2, size: float = 120.0, person_id: int | None = None
) -> Track:
    face_row = np.array(
        [
            0.0,
            0.0,
            size,
            size,
            10.0,
            10.0,
            20.0,
            20.0,
            30.0,
            30.0,
            40.0,
            40.0,
            50.0,
            50.0,
            0.95,
        ],
        dtype=np.float32,
    )
    return Track(
        track_id=1,
        box=(0.0, 0.0, size, size),
        last_seen=0.0,
        hits=hits,
        score=0.95,
        face_row=face_row,
        person_id=person_id,
    )


class FakeRecognizer:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.calls = 0

    def embedding(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self.vector


class FakeDatabase:
    def __init__(self) -> None:
        self.records: list[PersonRecord] = []

    def add_person(self, embedding: np.ndarray) -> PersonRecord:
        person_id = len(self.records) + 1
        record = PersonRecord(
            person_id,
            f"Person_{person_id:03d}",
            embedding,
            "2026-01-01T00:00:00Z",
        )
        self.records.append(record)
        return record


def default_settings() -> RecognitionSettings:
    return RecognitionSettings(
        match_threshold=0.38,
        match_margin=0.05,
        enroll_threshold=0.36,
        enroll_margin=0.03,
        confirm_votes=2,
        enroll_samples=3,
        min_face_px=80,
        interval_s=0.5,
    )


def test_unseen_face_is_enrolled_after_three_samples() -> None:
    vector = make_vector()
    recognizer = FakeRecognizer(vector)
    database = FakeDatabase()
    gallery = Gallery([])
    manager = IdentityManager(recognizer, gallery, database, default_settings())
    track = make_track()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    manager.update(frame, [track], now=0.0)
    assert track.person_id is None
    assert database.records == []

    manager.update(frame, [track], now=0.1)
    assert recognizer.calls == 1

    manager.update(frame, [track], now=0.6)
    manager.update(frame, [track], now=1.2)

    assert track.person_id == 1
    assert track.label == "Person_001"
    assert gallery.count == 1
    assert len(database.records) == 1


def test_known_face_is_confirmed_after_two_votes() -> None:
    vector = make_vector(1)
    person = PersonRecord(1, "Person_001", vector, "2026-01-01T00:00:00Z")
    recognizer = FakeRecognizer(vector)
    database = FakeDatabase()
    manager = IdentityManager(
        recognizer, Gallery([person]), database, default_settings()
    )
    track = make_track()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    manager.update(frame, [track], now=0.0)
    assert track.person_id is None

    manager.update(frame, [track], now=0.6)

    assert track.person_id == 1
    assert track.label == "Person_001"
    assert database.records == []


def test_new_tracks_wait_for_a_second_detection() -> None:
    recognizer = FakeRecognizer(make_vector())
    manager = IdentityManager(
        recognizer, Gallery([]), FakeDatabase(), default_settings()
    )
    track = make_track(hits=1)

    manager.update(np.zeros((240, 320, 3), dtype=np.uint8), [track], now=0.0)

    assert recognizer.calls == 0


def test_small_faces_are_not_recognized() -> None:
    recognizer = FakeRecognizer(make_vector())
    manager = IdentityManager(
        recognizer, Gallery([]), FakeDatabase(), default_settings()
    )
    track = make_track(size=40.0)

    manager.update(np.zeros((240, 320, 3), dtype=np.uint8), [track], now=0.0)

    assert recognizer.calls == 0


def test_confirmed_tracks_are_skipped() -> None:
    recognizer = FakeRecognizer(make_vector())
    manager = IdentityManager(
        recognizer, Gallery([]), FakeDatabase(), default_settings()
    )
    track = make_track(person_id=1)

    manager.update(np.zeros((240, 320, 3), dtype=np.uint8), [track], now=0.0)

    assert recognizer.calls == 0


def test_borderline_face_is_reused_instead_of_enrolled() -> None:
    base = make_vector(0)
    similarity = 0.37
    borderline = np.zeros(128, dtype=np.float32)
    borderline[0] = similarity
    borderline[1] = np.sqrt(1.0 - similarity**2)
    person = PersonRecord(
        1, "Person_001", base, "2026-01-01T00:00:00Z"
    )
    recognizer = FakeRecognizer(borderline)
    database = FakeDatabase()
    manager = IdentityManager(
        recognizer,
        Gallery([person]),
        database,
        default_settings(),
    )
    track = make_track()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    manager.update(frame, [track], now=0.0)
    manager.update(frame, [track], now=0.6)
    manager.update(frame, [track], now=1.2)

    assert track.person_id == 1
    assert track.label == "Person_001"
    assert database.records == []
