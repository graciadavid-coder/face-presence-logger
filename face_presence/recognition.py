"""SFace recognition and in-memory face gallery."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from face_presence.storage import EMBEDDING_SIZE, PersonRecord


class RecognitionError(RuntimeError):
    """Raised when the recognizer cannot be loaded or used."""


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return an L2-normalized copy of an embedding."""
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0.0 else array


def average_embeddings(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Average normalized embeddings and normalize the result."""
    if not vectors:
        raise ValueError("at least one embedding is required")
    stacked = np.stack([normalize(vector) for vector in vectors])
    return normalize(stacked.mean(axis=0))


@dataclass(frozen=True)
class MatchResult:
    person: PersonRecord | None
    score: float
    margin: float


class FaceRecognizer:
    """Thin wrapper around OpenCV's SFace recognizer."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise RecognitionError(
                f"Face recognition model not found: {model_path}. "
                "Run 'python -m face_presence setup' first."
            )
        try:
            self._recognizer = cv2.FaceRecognizerSF.create(str(model_path), "")
        except cv2.error as exc:
            raise RecognitionError(f"Could not load the face recognizer: {exc}") from exc

    def embedding(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """Align a detected face and return its normalized 128-dimensional embedding."""
        try:
            aligned = self._recognizer.alignCrop(frame, face_row)
            feature = self._recognizer.feature(aligned)
        except cv2.error as exc:
            raise RecognitionError(f"Could not compute a face embedding: {exc}") from exc

        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        if vector.size != EMBEDDING_SIZE:
            raise RecognitionError(
                f"Unexpected embedding size: {vector.size} "
                f"(expected {EMBEDDING_SIZE})"
            )
        return normalize(vector)


class Gallery:
    """Known people held in memory as a normalized embedding matrix."""

    def __init__(self, people: Sequence[PersonRecord]) -> None:
        self._people = list(people)
        if self._people:
            try:
                matrix = np.stack(
                    [normalize(person.embedding) for person in self._people]
                )
            except ValueError as exc:
                raise RecognitionError("The gallery contains an invalid embedding") from exc
            if matrix.shape[1] != EMBEDDING_SIZE:
                raise RecognitionError("The gallery contains an invalid embedding")
            self._matrix = matrix.astype(np.float32)
        else:
            self._matrix = np.empty((0, EMBEDDING_SIZE), dtype=np.float32)

    @property
    def count(self) -> int:
        return len(self._people)

    def match(
        self,
        embedding: np.ndarray,
        threshold: float,
        margin_threshold: float,
    ) -> MatchResult:
        """Return the best acceptable match for an embedding."""
        query = normalize(embedding)
        if not self._people:
            return MatchResult(None, 0.0, 1.0)

        scores = self._matrix @ query
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        second_score = (
            float(np.partition(scores, -2)[-2]) if len(scores) > 1 else 0.0
        )
        margin = best_score - second_score

        person = None
        if best_score >= threshold and margin >= margin_threshold:
            person = self._people[best_index]
        return MatchResult(person, best_score, margin)

    def add(self, person: PersonRecord) -> None:
        """Add a newly enrolled person to the in-memory gallery."""
        vector = normalize(person.embedding).reshape(1, -1)
        if vector.shape[1] != EMBEDDING_SIZE:
            raise RecognitionError("The person has an invalid embedding")
        self._people.append(person)
        if self._matrix.size == 0:
            self._matrix = vector.astype(np.float32)
        else:
            self._matrix = np.vstack([self._matrix, vector]).astype(np.float32)
