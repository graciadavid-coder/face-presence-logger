"""Repository hygiene checks."""

from scripts.check_repo_hygiene import find_violations


def test_allows_documented_placeholders() -> None:
    assert find_violations(["README.md", "data/.gitkeep", "models/.gitkeep"]) == []


def test_blocks_runtime_directories_and_media() -> None:
    paths = [
        "data/presence.db",
        "models/face.onnx",
        "captures/frame.jpg",
        "sessions.csv",
        "README.md",
    ]

    assert find_violations(paths) == paths[:4]
