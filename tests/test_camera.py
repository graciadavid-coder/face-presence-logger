"""Camera backend selection tests."""

import cv2
import pytest

from face_presence.camera import backend_candidates, probe_cameras


def test_windows_backend_order() -> None:
    names = [name for _, name in backend_candidates(platform="win32")]

    assert names == ["msmf", "dshow", "default"]


def test_macos_backend_order() -> None:
    names = [name for _, name in backend_candidates(platform="darwin")]

    assert names == ["avfoundation", "default"]


def test_linux_backend_order() -> None:
    names = [name for _, name in backend_candidates(platform="linux")]

    assert names == ["v4l2", "default"]


def test_explicit_backend_is_used() -> None:
    assert backend_candidates("avfoundation", platform="win32") == [
        (cv2.CAP_AVFOUNDATION, "avfoundation")
    ]


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown camera backend"):
        backend_candidates("bogus")


def test_probe_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="max_index"):
        probe_cameras(max_index=-1)
