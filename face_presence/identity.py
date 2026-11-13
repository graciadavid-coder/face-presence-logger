"""Per-track identity confirmation and automatic enrollment."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from face_presence.config import RecognitionSettings
from face_presence.recognition import (
    FaceRecognizer,
    Gallery,
    RecognitionError,
    average_embeddings,
)
from face_presence.storage import Database, PersonRecord
from face_presence.tracking import Track

logger = logging.getLogger(__name__)

MIN_TRACK_HITS = 2


@dataclass
class _TrackState:
    last_attempt: float = 0.0
    last_candidate: int | None = None
    votes: dict[int, int] = field(default_factory=dict)
    samples: list[np.ndarray] = field(default_factory=list)


class IdentityManager:
    """Confirm identities across frames and enroll faces that are genuinely new."""

    def __init__(
        self,
        recognizer: FaceRecognizer,
        gallery: Gallery,
        database: Database,
        settings: RecognitionSettings,
    ) -> None:
        self._recognizer = recognizer
        self._gallery = gallery
        self._database = database
        self._settings = settings
        self._states: dict[int, _TrackState] = {}

    def update(self, frame: np.ndarray, tracks: Sequence[Track], now: float) -> None:
        """Process the tracks from a detection pass and assign identities when possible."""
        active_ids = {track.track_id for track in tracks}
        self._states = {
            track_id: state
            for track_id, state in self._states.items()
            if track_id in active_ids
        }

        for track in tracks:
            if track.person_id is not None:
                continue
            if track.hits < MIN_TRACK_HITS or track.face_row is None:
                continue
            if min(track.box[2], track.box[3]) < self._settings.min_face_px:
                continue

            state = self._states.setdefault(
                track.track_id,
                _TrackState(last_attempt=now - self._settings.interval_s),
            )
            if now - state.last_attempt < self._settings.interval_s:
                continue
            state.last_attempt = now

            try:
                embedding = self._recognizer.embedding(frame, track.face_row)
            except RecognitionError as exc:
                logger.warning("Skipping a face that could not be encoded: %s", exc)
                continue

            match = self._gallery.match(
                embedding,
                self._settings.match_threshold,
                self._settings.match_margin,
            )
            if match.person is not None:
                self._register_vote(track, state, match.person)
                continue

            state.votes.clear()
            state.last_candidate = None
            state.samples.append(embedding)
            if len(state.samples) >= self._settings.enroll_samples:
                self._enroll(track, state)

    def _register_vote(
        self, track: Track, state: _TrackState, person: PersonRecord
    ) -> None:
        if state.last_candidate == person.id:
            state.votes[person.id] = state.votes.get(person.id, 0) + 1
        else:
            state.votes = {person.id: 1}
            state.last_candidate = person.id
        state.samples.clear()

        if state.votes[person.id] >= self._settings.confirm_votes:
            self._assign(track, person)
            state.votes.clear()
            logger.info("Recognized %s", person.label)

    def _enroll(self, track: Track, state: _TrackState) -> None:
        embedding = average_embeddings(state.samples)
        match = self._gallery.match(
            embedding,
            self._settings.enroll_threshold,
            self._settings.enroll_margin,
        )
        logger.debug(
            "Enrollment check: best=%s score=%.3f margin=%.3f",
            match.person.label if match.person is not None else "none",
            match.score,
            match.margin,
        )
        if match.person is not None:
            self._assign(track, match.person)
            state.samples.clear()
            logger.info("Recognized %s during enrollment", match.person.label)
            return

        person = self._database.add_person(embedding)
        self._gallery.add(person)
        self._assign(track, person)
        state.samples.clear()
        logger.info("Enrolled %s", person.label)

    @staticmethod
    def _assign(track: Track, person: PersonRecord) -> None:
        track.person_id = person.id
        track.label = person.label
