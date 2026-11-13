"""Command line entry point."""

from __future__ import annotations

import argparse
import logging

from face_presence import __version__
from face_presence.camera import CameraError, probe_cameras
from face_presence.config import ConfigError, Settings, load_settings
from face_presence.detection import DetectionError
from face_presence.model_assets import ModelError, ensure_models
from face_presence.pipeline import run_app

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face_presence",
        description="CPU-only face recognition and presence logging.",
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to the TOML configuration file (default: config.toml).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "setup", help="Download and verify the face models, then create runtime folders."
    )

    cameras_parser = subparsers.add_parser(
        "cameras", help="List the cameras available on this machine."
    )
    cameras_parser.add_argument(
        "--max-index",
        type=int,
        default=5,
        help="Highest camera index to probe (default: 5).",
    )

    run_parser = subparsers.add_parser(
        "run", help="Open the camera and start the live preview."
    )
    run_parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index to use instead of the value in config.toml.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        settings = load_settings(args.config)
        if args.command == "setup":
            return _run_setup(settings)
        if args.command == "cameras":
            return _run_cameras(settings, args.max_index)
        if args.command == "run":
            if args.camera is not None and args.camera < 0:
                print("--camera must be zero or greater.")
                return 2
            return run_app(settings, camera_index=args.camera)
    except (ConfigError, ModelError, CameraError, DetectionError) as exc:
        logger.error("%s", exc)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def _run_setup(settings: Settings) -> int:
    models = ensure_models(settings.models.directory)
    for path in models:
        print(f"  {path.name}  {path.stat().st_size / 1_000_000:.1f} MB")

    for directory in (
        settings.storage.db_path.parent,
        settings.storage.export_dir,
        settings.storage.captures_dir,
        settings.storage.log_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    print("Setup complete.")
    return 0


def _run_cameras(settings: Settings, max_index: int) -> int:
    if max_index < 0:
        print("--max-index must be zero or greater.")
        return 2

    print(f"Scanning camera indexes 0-{max_index}...")
    cameras = probe_cameras(
        max_index=max_index,
        width=settings.camera.width,
        height=settings.camera.height,
        backend=settings.camera.backend,
    )

    if not cameras:
        print("No cameras detected.")
        print(
            "Check that the device is connected and that this application has "
            "camera permission."
        )
        return 1

    print(f"{'Index':<6}{'Backend':<16}Resolution")
    for camera in cameras:
        print(f"{camera.index:<6}{camera.backend:<16}{camera.width}x{camera.height}")
    print()
    print("Set [camera].index in config.toml to the device you want to use.")
    return 0
