"""Decoding and cleaning of uploaded images.

Everything here is independent of the model, so it can be tested without PyTorch.
"""
from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImage(ValueError):
    """The upload is not a usable image. The message is safe to show to the client."""


def load_image(data: bytes, *, max_pixels: int, min_side: int = 32) -> Image.Image:
    """Decode uploaded bytes into an RGB image.

    * Rejects empty, corrupted, tiny and huge images (the size is checked before decoding).
    * Applies the EXIF orientation, so phone photos are not rotated.
    * Puts transparent images on a white background (catalog photos have white backgrounds;
      a black background would change the embedding).
    """
    if not data:
        raise InvalidImage("The uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size                      # header only, nothing decoded yet
        if width * height > max_pixels:
            raise InvalidImage(f"The image is too large ({width}x{height} pixels).")
        image.load()
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImage("The file is not a valid image, or it is corrupted.") from exc

    image = ImageOps.exif_transpose(image)

    if min(image.size) < min_side:
        raise InvalidImage(f"The image is too small ({image.width}x{image.height}); each side needs at least {min_side} pixels.")

    has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")
