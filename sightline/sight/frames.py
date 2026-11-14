"""Turn whatever the camera gives us into a frame Jev will accept.

Jev's image rules (Images page): PNG or JPEG, each side 280..1568 px, pixels
rotated upright (EXIF orientation is stripped, not applied). It rejects
anything outside the limits instead of resizing, so we do the resizing here.

Also home to the synthetic "sample desk" image that demo mode uses, drawn
with Pillow so the repo carries no binary files and no one's face.
"""

import io

from PIL import Image, ImageDraw, ImageOps

from sight import config


def prepare_frame(raw: bytes | Image.Image) -> Image.Image:
    """Bytes or a PIL image in, an upright RGB image within Jev's limits out."""
    img = Image.open(io.BytesIO(raw)) if isinstance(raw, bytes) else raw
    img = ImageOps.exif_transpose(img)  # rotate the PIXELS upright (phones/uploads)
    img = img.convert("RGB")            # drop alpha / palette: JPEG wants plain RGB

    # Shrink so the long side is at most SEND_LONG_SIDE (fewer tokens = cheaper).
    # thumbnail() keeps the aspect ratio and never enlarges.
    img.thumbnail((config.SEND_LONG_SIDE, config.SEND_LONG_SIDE))

    # Enlarge if a side is under the 280 px minimum (tiny test images, odd crops).
    short = min(img.size)
    if short < config.MIN_SIDE:
        scale = config.MIN_SIDE / short
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    return img


def to_jpeg(img: Image.Image) -> bytes:
    """Encode as a standard baseline JPEG (the format the docs call safest)."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=config.JPEG_QUALITY)  # baseline unless progressive=True
    return buf.getvalue()


# ── The demo scene ──────────────────────────────────────────────────────────
# Positions are fractions of the image (the same convention as Jev's Box
# answers), so demo.py can return boxes that line up with what's drawn.
SAMPLE_OBJECTS = {
    "mug":    {"left": 0.12, "top": 0.42, "right": 0.30, "bottom": 0.78},
    "laptop": {"left": 0.40, "top": 0.22, "right": 0.86, "bottom": 0.80},
    "book":   {"left": 0.05, "top": 0.82, "right": 0.38, "bottom": 0.95},
}


def make_sample_image(width: int = 960, height: int = 640) -> Image.Image:
    """Draw a cartoon desk: a mug, an open laptop, and a book."""
    img = Image.new("RGB", (width, height), (232, 222, 206))  # warm wall
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(height * 0.62), width, height], fill=(150, 108, 72))  # desk

    def px(box: dict) -> list[int]:
        return [int(box["left"] * width), int(box["top"] * height),
                int(box["right"] * width), int(box["bottom"] * height)]

    # Mug: a body plus a handle.
    l, t, r, b = px(SAMPLE_OBJECTS["mug"])
    d.rounded_rectangle([l, t, r - 30, b], radius=12, fill=(200, 60, 50))
    d.arc([r - 55, t + 40, r, b - 50], start=-90, end=90, fill=(200, 60, 50), width=14)

    # Laptop: a dark screen above a grey keyboard base.
    l, t, r, b = px(SAMPLE_OBJECTS["laptop"])
    mid = t + int((b - t) * 0.72)
    d.rectangle([l + 20, t, r - 20, mid], fill=(40, 44, 52))
    d.rectangle([l + 34, t + 14, r - 34, mid - 14], fill=(70, 130, 180))  # lit screen
    d.polygon([(l, b), (r, b), (r - 20, mid), (l + 20, mid)], fill=(170, 172, 178))

    # Book: a flat slab with page edges.
    l, t, r, b = px(SAMPLE_OBJECTS["book"])
    d.rectangle([l, t, r, b], fill=(40, 90, 60))
    d.rectangle([l + 6, t + 6, r - 6, t + 14], fill=(245, 240, 225))
    return img
