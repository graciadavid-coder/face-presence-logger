"""Pipeline helper tests."""

import pytest

from face_presence.pipeline import FpsMeter, next_detection_interval


def test_fps_meter_returns_zero_for_the_first_frame() -> None:
    meter = FpsMeter(window=5)

    assert meter.tick(0.0) == 0.0
    assert meter.tick(0.5) == pytest.approx(2.0)
    assert meter.tick(1.0) == pytest.approx(2.0)


def test_fps_meter_handles_repeated_timestamps() -> None:
    meter = FpsMeter(window=5)

    assert meter.tick(1.0) == 0.0
    assert meter.tick(1.0) == 0.0


def test_detection_interval_increases_when_the_preview_is_slow() -> None:
    assert next_detection_interval(2, 2, 8.0) == 3
    assert next_detection_interval(3, 2, 8.0) == 4
    assert next_detection_interval(4, 2, 8.0) == 4


def test_detection_interval_recovers_when_the_preview_is_fast() -> None:
    assert next_detection_interval(4, 2, 30.0) == 3
    assert next_detection_interval(3, 2, 30.0) == 2
    assert next_detection_interval(2, 2, 30.0) == 2


def test_detection_interval_stays_stable_in_the_middle_range() -> None:
    assert next_detection_interval(3, 2, 15.0) == 3
    assert next_detection_interval(2, 2, 0.0) == 2
