from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / ".example_assets"

COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
COCO_IMAGE_NAME = "coco_000000039769.jpg"

SAMPLE_VIDEO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
SAMPLE_VIDEO_NAME = "vtest.avi"


def download_if_missing(url: str, destination: Path) -> None:
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name} -> {destination}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as target:
        shutil.copyfileobj(response, target)
    print(f"Downloaded {destination.name}", file=sys.stderr)


def resolve_input_path(
    input_path: Path | None,
    default_path: Path,
    default_url: str | None = None,
) -> Path:
    resolved_input_path = input_path or default_path
    if input_path is None and default_url is not None:
        download_if_missing(default_url, resolved_input_path)
    if resolved_input_path.exists():
        return resolved_input_path

    print(f"Error: input file not found: {resolved_input_path}", file=sys.stderr)
    raise SystemExit(1)
