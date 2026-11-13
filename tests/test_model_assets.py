"""Model download and verification tests."""

import hashlib
from pathlib import Path

import pytest

from face_presence import model_assets
from face_presence.model_assets import (
    ModelError,
    ModelFile,
    download_file,
    ensure_models,
    sha256_file,
    verify_file,
)


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert verify_file(path, sha256_file(path))
    assert not verify_file(path, "0" * 64)


def test_verify_file_rejects_missing_file(tmp_path: Path) -> None:
    assert not verify_file(tmp_path / "missing.onnx", "0" * 64)


def test_ensure_models_reuses_valid_files(tmp_path: Path, monkeypatch) -> None:
    content = b"model bytes"
    spec = ModelFile(
        filename="model.onnx",
        url="https://example.invalid/model.onnx",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )
    monkeypatch.setattr(model_assets, "MODEL_FILES", (spec,))
    (tmp_path / spec.filename).write_bytes(content)

    assert ensure_models(tmp_path) == [tmp_path / spec.filename]


def test_ensure_models_downloads_missing_files(tmp_path: Path, monkeypatch) -> None:
    content = b"model bytes"
    spec = ModelFile(
        filename="model.onnx",
        url="https://example.invalid/model.onnx",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )
    monkeypatch.setattr(model_assets, "MODEL_FILES", (spec,))
    calls: list[str] = []

    def fake_download(spec: ModelFile, destination: Path, **kwargs) -> Path:
        calls.append(spec.filename)
        destination.write_bytes(content)
        return destination

    monkeypatch.setattr(model_assets, "download_file", fake_download)

    result = ensure_models(tmp_path)

    assert calls == ["model.onnx"]
    assert result == [tmp_path / spec.filename]
    assert result[0].read_bytes() == content


def test_ensure_models_replaces_corrupt_files(tmp_path: Path, monkeypatch) -> None:
    content = b"model bytes"
    spec = ModelFile(
        filename="model.onnx",
        url="https://example.invalid/model.onnx",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )
    monkeypatch.setattr(model_assets, "MODEL_FILES", (spec,))
    (tmp_path / spec.filename).write_bytes(b"corrupt")

    def fake_download(spec: ModelFile, destination: Path, **kwargs) -> Path:
        destination.write_bytes(content)
        return destination

    monkeypatch.setattr(model_assets, "download_file", fake_download)

    ensure_models(tmp_path)

    assert (tmp_path / spec.filename).read_bytes() == content


def test_download_file_rejects_bad_checksum(tmp_path: Path, monkeypatch) -> None:
    content = b"good"
    spec = ModelFile(
        filename="model.onnx",
        url="https://example.invalid/model.onnx",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )

    def fake_download(spec: ModelFile, destination: Path, **kwargs) -> None:
        destination.write_bytes(b"evil")

    monkeypatch.setattr(model_assets, "_download", fake_download)

    with pytest.raises(ModelError, match="Checksum"):
        download_file(spec, tmp_path / spec.filename)

    assert not (tmp_path / spec.filename).exists()
    assert not (tmp_path / f"{spec.filename}.part").exists()
