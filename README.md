# Face Presence Logger

[![tests](https://github.com/graciadavid-coder/face-presence-logger/actions/workflows/tests.yml/badge.svg)](https://github.com/graciadavid-coder/face-presence-logger/actions/workflows/tests.yml)

Face Presence Logger is a lightweight, CPU-only face recognition and presence logging application for a single camera. It is designed to detect faces in a video stream, assign a local identifier to each new person, recognize returning people, and record how long each person stays in view.

The runtime uses OpenCV's DNN module with two ONNX models from OpenCV Zoo. It does not require a GPU, a deep-learning framework, or a cloud service.

## Project status

The project is being built in milestones. This revision includes the configuration layer, verified model downloads, camera discovery, and a live preview with YuNet detection and IoU tracking. Face recognition, automatic enrollment, SQLite presence sessions, and CSV export are the next milestones. The commands documented below are functional today.

## How it works

1. **Detection.** YuNet locates faces and their five facial landmarks on a reduced copy of each frame.
2. **Tracking.** A lightweight IoU tracker follows each face between detection passes.
3. **Recognition.** SFace aligns each face and produces a 128-dimensional embedding. The embedding is compared against the local gallery using cosine similarity.
4. **Enrollment.** A face that does not match an existing person is confirmed across several frames and stored as `Person_001`, `Person_002`, and so on.
5. **Presence.** Each confirmed person opens a session with a start time, last detection time, and duration. Sessions close after a configurable absence period.
6. **Storage.** SQLite is the source of truth. Sessions can be exported to CSV for reporting.

## MVP scope

- CPU-only detection and recognition.
- Automatic enrollment with local identifiers.
- Presence sessions with entry time, last detection, and duration.
- SQLite storage and CSV export.
- Live overlay with face boxes, identifiers, and timers.
- Selection of the default camera or an external camera.
- Local-only data, with runtime artifacts excluded from version control.

## Requirements

- Python 3.11 or newer.
- A camera accessible to the operating system.
- Windows 10 or newer, or macOS 13 or newer. Linux is supported through V4L2.
- Internet access during the first model setup.

## Installation

### Windows

```powershell
git clone https://github.com/graciadavid-coder/face-presence-logger.git
cd face-presence-logger
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, allow it for the current session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### macOS

```bash
git clone https://github.com/graciadavid-coder/face-presence-logger.git
cd face-presence-logger
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first time the application opens a camera, macOS asks for camera access. Grant permission to the terminal or IDE that launches the process. The prompt does not appear again unless the permission is reset.

### Environment isolation

If your shell defines `PYTHONPATH`, clear it before running the project so packages from other environments are not imported:

```powershell
# Windows
$env:PYTHONPATH = ""
```

```bash
# macOS
unset PYTHONPATH
```

## Download the models

```bash
python -m face_presence setup
```

The command downloads YuNet and SFace from a pinned OpenCV Zoo revision, verifies their SHA-256 digests, and creates the runtime folders under `data/`. The models are not committed to the repository.

## Discover cameras

```bash
python -m face_presence cameras
```

The command probes camera indexes and prints the backend and resolution of each device it can open. Set the desired index in `config.toml`:

```toml
[camera]
index = 1
```

The backend is selected automatically: Media Foundation with a DirectShow fallback on Windows, AVFoundation on macOS, and V4L2 on Linux. A specific backend can be forced with `[camera].backend`.

## Run the live preview

```bash
python -m face_presence run
```

Override the camera index from `config.toml`:

```bash
python -m face_presence run --camera 1
```

The window draws a box and a temporary `Face NN` label for each tracked face, together with FPS, camera details, and the current detection interval. Press `q` or `Esc`, or close the window, to quit. Recognition and presence timers are not wired in yet; the preview does not create or modify the database, and frames stay in memory.

## Configuration

All settings live in `config.toml`. Relative paths are resolved from the directory that contains the file.

| Section | Key | Purpose |
| --- | --- | --- |
| `camera` | `index` | Camera device index |
| `camera` | `backend` | `auto`, `dshow`, `msmf`, `avfoundation`, `v4l2`, or `default` |
| `detection` | `input_width` | Width used for face detection |
| `detection` | `interval_frames` | Frames between detection passes |
| `recognition` | `match_threshold` | Minimum cosine similarity for a match |
| `recognition` | `match_margin` | Required lead over the second-best match |
| `presence` | `absence_timeout_s` | Seconds without a detection before a session closes |

## Project layout

```
face-presence-logger/
├─ .github/workflows/tests.yml
├─ config.toml
├─ face_presence/
│  ├─ camera.py          # camera discovery and capture
│  ├─ cli.py             # command line interface
│  ├─ config.py          # TOML loading and validation
│  ├─ detection.py       # YuNet detection wrapper
│  ├─ model_assets.py    # model download and checksum verification
│  ├─ overlay.py         # boxes, labels and status panel
│  ├─ pipeline.py        # live preview loop
│  └─ tracking.py        # IoU tracker
├─ tests/
├─ models/               # downloaded ONNX files (ignored by Git)
└─ data/                 # database, exports, captures and logs (ignored by Git)
```

## Data and privacy

- `data/` is excluded from version control. It holds the SQLite database, CSV exports, manual captures, and logs.
- The live preview processes frames in memory and does not write them to disk.
- Camera frames are not written to disk automatically.
- Face embeddings are biometric templates. Treat `data/presence.db` as sensitive and do not commit or share it.
- Deleting `data/` removes all local records and allows the application to start from a clean state.
- No data leaves the machine. There is no telemetry and no cloud dependency at runtime.

Use the application only where you have the right to process images of the people involved. Depending on your jurisdiction, you may need consent, signage, or a legal basis for biometric processing.

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest
```

The test suite covers configuration, model verification, camera backend selection, detection scaling, IoU tracking, overlay rendering, and pipeline helpers without requiring a camera or the ONNX models. GitHub Actions runs lint and tests on Linux, Windows, and macOS.

## Models

The models are downloaded from [OpenCV Zoo](https://github.com/opencv/opencv_zoo) at a pinned revision and verified against fixed SHA-256 digests. They remain subject to their original licenses; see the model cards in the OpenCV Zoo repository for details.

## License

MIT License. Copyright (c) 2026 David Gracia.
