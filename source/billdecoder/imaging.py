"""Image loading and normalisation so the model sees every bill at a consistent, readable size."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
MAX_SIDE = 2000   # larger images are shrunk (keeps payload small, text stays legible)
MIN_SIDE = 1500   # small photos/screenshots are upscaled so tiny digits are easier to read


@dataclass(frozen=True)
class BillImage:
    data: bytes
    mime_type: str = "image/jpeg"


def find_bill_image(bills_dir: Path, bill_id: str) -> Path:
    """bill_id is the file name without its extension (KESC_0002 -> KESC_0002.png)."""
    for ext in IMAGE_EXTENSIONS:
        for candidate in (bills_dir / f"{bill_id}{ext}", bills_dir / f"{bill_id}{ext.upper()}"):
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No image found for {bill_id} in {bills_dir}")


def load_bill_image(path: Path) -> BillImage:
    """Open, fix orientation, flatten to RGB, resize and re-encode deterministically."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)  # phone photos often carry a rotation flag
        img = img.convert("RGB")
        longest = max(img.size)
        if longest > MAX_SIDE:
            scale = MAX_SIDE / longest
        elif longest < MIN_SIDE:
            scale = MIN_SIDE / longest
        else:
            scale = 1.0
        if scale != 1.0:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92, optimize=True)
    return BillImage(data=buf.getvalue())
