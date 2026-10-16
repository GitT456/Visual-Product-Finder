"""Runtime settings. Everything can be overridden with environment variables (prefix VPF_)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "data" / "vpf_artifacts"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "images"


@dataclass(frozen=True)
class Settings:
    bundle_dir: Path = DEFAULT_BUNDLE_DIR      # folder with the engine files from the evaluation notebook
    image_dir: Path | None = None              # catalog images (optional, only needed for thumbnails)
    max_upload_bytes: int = 10 * 1024 * 1024   # largest accepted upload
    max_pixels: int = 60_000_000               # largest accepted image (width x height), protects memory
    min_side: int = 32                         # smallest accepted image side in pixels
    max_k: int = 20                            # largest number of results a client may request
    device: str = "auto"                       # "auto", "cpu" or "cuda"
    cors_origins: tuple[str, ...] = ()         # empty: same-origin only
    thumbnail_cache_size: int = 512            # number of thumbnails kept in memory

    @classmethod
    def from_env(cls) -> "Settings":
        image_dir_env = os.environ.get("VPF_IMAGE_DIR")
        if image_dir_env:
            image_dir: Path | None = Path(image_dir_env)
        else:
            image_dir = DEFAULT_IMAGE_DIR if DEFAULT_IMAGE_DIR.is_dir() else None

        origins = tuple(o.strip() for o in os.environ.get("VPF_CORS_ORIGINS", "").split(",") if o.strip())
        return cls(
            bundle_dir=Path(os.environ.get("VPF_BUNDLE_DIR", DEFAULT_BUNDLE_DIR)),
            image_dir=image_dir,
            max_upload_bytes=int(os.environ.get("VPF_MAX_UPLOAD_MB", "10")) * 1024 * 1024,
            device=os.environ.get("VPF_DEVICE", "auto"),
            cors_origins=origins,
        )
