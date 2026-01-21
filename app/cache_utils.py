import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ImageOps

DEFAULT_CACHE_DIR = os.getenv("CACHE_DIR", "/cache")
DEFAULT_MAX_CACHE_GB = float(os.getenv("CACHE_MAX_GB", "10"))
DEFAULT_WEBP_QUALITY = int(os.getenv("CACHE_WEBP_QUALITY", "70"))
SETTINGS_FILENAME = "cache_settings.json"

_cache_settings: Optional[Dict[str, Any]] = None


def get_cache_dir() -> Path:
    return Path(DEFAULT_CACHE_DIR)


def ensure_cache_dirs() -> Path:
    base_dir = get_cache_dir()
    (base_dir / "thumbs" / "photos").mkdir(parents=True, exist_ok=True)
    (base_dir / "thumbs" / "faces").mkdir(parents=True, exist_ok=True)
    return base_dir


def _settings_path() -> Path:
    return ensure_cache_dirs() / SETTINGS_FILENAME


def load_cache_settings() -> Dict[str, Any]:
    global _cache_settings
    if _cache_settings is not None:
        return _cache_settings

    settings = {
        "max_cache_gb": DEFAULT_MAX_CACHE_GB,
        "webp_quality": DEFAULT_WEBP_QUALITY,
    }

    settings_path = _settings_path()
    if settings_path.exists():
        try:
            with settings_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    for key in settings:
                        if key in data:
                            settings[key] = data[key]
        except (OSError, json.JSONDecodeError):
            pass

    _cache_settings = settings
    return settings


def update_cache_settings(new_settings: Dict[str, Any]) -> Dict[str, Any]:
    settings = load_cache_settings().copy()
    for key in ("max_cache_gb", "webp_quality"):
        if key in new_settings and new_settings[key] is not None:
            settings[key] = new_settings[key]

    settings_path = _settings_path()
    try:
        with settings_path.open("w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=2)
    except OSError:
        pass

    global _cache_settings
    _cache_settings = settings
    return settings


def build_cache_path(category: str, cache_key: str, size: int) -> Path:
    safe_category = "photos" if category == "photos" else "faces"
    filename = f"{cache_key}_{size}.webp"
    return ensure_cache_dirs() / "thumbs" / safe_category / filename


def touch_cache_file(path: Path) -> None:
    try:
        os.utime(path, None)
    except OSError:
        pass


def save_webp(image: Image.Image, output_path: Path, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "WEBP", quality=quality, method=6)


def create_thumbnail(image_path: str, max_size: int) -> Image.Image:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    image.thumbnail((max_size, max_size), Image.LANCZOS)
    return image


def enforce_cache_limit() -> None:
    settings = load_cache_settings()
    max_cache_bytes = int(settings["max_cache_gb"] * 1024 ** 3)
    base_dir = ensure_cache_dirs()

    total_size = 0
    files: list[tuple[float, int, Path]] = []

    for root, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename == SETTINGS_FILENAME:
                continue
            path = Path(root) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            total_size += stat.st_size
            files.append((stat.st_mtime, stat.st_size, path))

    if total_size <= max_cache_bytes:
        return

    files.sort(key=lambda item: item[0])
    for _, size, path in files:
        try:
            path.unlink()
        except OSError:
            continue
        total_size -= size
        if total_size <= max_cache_bytes:
            break
