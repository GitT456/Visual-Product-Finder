import io

import pytest
from PIL import Image

from app.imaging import InvalidImage, load_image
from conftest import png_bytes

LIMITS = dict(max_pixels=1_000_000, min_side=32)


def test_valid_png_becomes_rgb():
    image = load_image(png_bytes(200), **LIMITS)
    assert image.mode == "RGB" and image.size == (64, 64) and image.getpixel((0, 0)) == (200, 0, 0)


def test_transparent_png_gets_a_white_background():
    buffer = io.BytesIO()
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(buffer, "PNG")      # fully transparent
    assert load_image(buffer.getvalue(), **LIMITS).getpixel((5, 5)) == (255, 255, 255)


def test_exif_orientation_is_applied():
    image = Image.new("RGB", (80, 40), (10, 20, 30))
    exif = Image.Exif()
    exif[0x0112] = 6                                                    # "rotate 90 degrees"
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif)
    assert load_image(buffer.getvalue(), **LIMITS).size == (40, 80)


@pytest.mark.parametrize("data", [b"", b"this is not an image", b"\x89PNG\r\n\x1a\n" + b"broken"])
def test_garbage_is_rejected(data):
    with pytest.raises(InvalidImage):
        load_image(data, **LIMITS)


def test_too_small_and_too_large_are_rejected():
    with pytest.raises(InvalidImage, match="too small"):
        load_image(png_bytes(size=(20, 20)), **LIMITS)
    with pytest.raises(InvalidImage, match="too large"):
        load_image(png_bytes(size=(2000, 2000)), max_pixels=1_000_000, min_side=32)
