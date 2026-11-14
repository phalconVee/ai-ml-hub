"""Draw Box answers on the frame.

Jev's box edges are fractions of the image measured from the top-left
corner, so the same answer fits any image size: multiply left/right by the
width and top/bottom by the height to get pixels.
"""

from PIL import Image, ImageDraw, ImageFont

from sight.decisions import Decision

# Colour = decision status, so you can read the decision layer off the picture.
COLOURS = {"auto": (34, 197, 94), "review": (245, 158, 11), "skipped": (148, 163, 184)}


def to_pixels(box: dict, width: int, height: int) -> tuple[int, int, int, int]:
    """Fractions -> (x0, y0, x1, y1) pixels, clamped inside the image."""
    def clamp(v: float) -> float:
        return min(1.0, max(0.0, v))
    return (round(clamp(box["left"]) * width), round(clamp(box["top"]) * height),
            round(clamp(box["right"]) * width), round(clamp(box["bottom"]) * height))


def draw_boxes(image: Image.Image, decisions: list[Decision],
               show_skipped: bool = False) -> Image.Image:
    """Return a COPY of the image with one labelled rectangle per Box decision."""
    out = image.copy()
    d = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=max(14, out.width // 45))
    line = max(3, out.width // 150)   # thick enough to survive a thumbnail

    for dec in decisions:
        if dec.box is None or (dec.status == "skipped" and not show_skipped):
            continue
        colour = COLOURS[dec.status]
        x0, y0, x1, y1 = to_pixels(dec.box, out.width, out.height)
        d.rectangle([x0, y0, x1, y1], outline=colour, width=line)

        conf = "?" if dec.value is None else f"{dec.value:.2f}"
        label = f"{dec.subject} {conf}" + ("" if dec.status == "auto" else f" ({dec.status})")
        # Label on a filled tab above the box (or inside it if the box touches the top).
        tx0, ty0, tx1, ty1 = d.textbbox((0, 0), label, font=font)
        tw, th = tx1 - tx0 + 8, ty1 - ty0 + 6
        ty = y0 - th if y0 - th >= 0 else y0
        d.rectangle([x0, ty, x0 + tw, ty + th], fill=colour)
        d.text((x0 + 4, ty + 2 - ty0), label, fill=(0, 0, 0), font=font)
    return out
