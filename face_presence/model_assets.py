"""Download and verify the ONNX models used by the face pipeline."""

from __future__ import annotations

import hashlib
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_OPENCV_ZOO_COMMIT = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
_BASE_URL = f"https://github.com/opencv/opencv_zoo/raw/{_OPENCV_ZOO_COMMIT}"

ProgressCallback = Callable[[int, int], None]


class ModelError(RuntimeError):
    """Raised when a model cannot be downloaded or verified."""


@dataclass(frozen=True)
class ModelFile:
    filename: str
    url: str
    sha256: str
    size_bytes: int


YUNET = ModelFile(
    filename="face_detection_yunet_2023mar.onnx",
    url=(
        f"{_BASE_URL}/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx"
    ),
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    size_bytes=232_589,
)

SFACE = ModelFile(
    filename="face_recognition_sface_2021dec.onnx",
    url=(
        f"{_BASE_URL}/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx"
    ),
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    size_bytes=38_696_353,
)

MODEL_FILES = (YUNET, SFACE)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str) -> bool:
    """Return True when the file exists and matches the expected digest."""
    return path.is_file() and sha256_file(path) == expected_sha256


def download_file(
    spec: ModelFile,
    destination: Path,
    *,
    timeout: float = 120.0,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download a model to a temporary file and move it into place after verification."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f"{destination.name}.part")

    try:
        _download(spec, temp_path, timeout=timeout, progress=progress)
        if temp_path.stat().st_size != spec.size_bytes:
            raise ModelError(f"Unexpected size for {spec.filename}")
        if not verify_file(temp_path, spec.sha256):
            raise ModelError(f"Checksum verification failed for {spec.filename}")
        temp_path.replace(destination)
    except (urllib.error.URLError, TimeoutError) as exc:
        temp_path.unlink(missing_ok=True)
        raise ModelError(f"Could not download {spec.filename}: {exc}") from exc
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return destination


def ensure_models(
    models_dir: Path,
    *,
    timeout: float = 120.0,
    progress: ProgressCallback | None = None,
) -> list[Path]:
    """Verify every required model, downloading any missing or corrupt file."""
    models_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for spec in MODEL_FILES:
        destination = models_dir / spec.filename
        if verify_file(destination, spec.sha256):
            logger.info("Model ready: %s", destination.name)
            paths.append(destination)
            continue

        if destination.exists():
            logger.warning("Removing invalid model file: %s", destination.name)
            destination.unlink()

        logger.info(
            "Downloading %s (%.1f MB)", spec.filename, spec.size_bytes / 1_000_000
        )
        download_file(spec, destination, timeout=timeout, progress=progress)
        if not verify_file(destination, spec.sha256):
            raise ModelError(f"Checksum verification failed for {spec.filename}")

        logger.info("Verified: %s", destination.name)
        paths.append(destination)

    return paths


def _download(
    spec: ModelFile,
    destination: Path,
    *,
    timeout: float,
    progress: ProgressCallback | None,
) -> None:
    request = urllib.request.Request(
        spec.url, headers={"User-Agent": "face-presence-logger"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length") or spec.size_bytes)
        downloaded = 0
        with destination.open("wb") as handle:
            while chunk := response.read(1024 * 256):
                handle.write(chunk)
                downloaded += len(chunk)
                if progress is not None:
                    progress(downloaded, total)
