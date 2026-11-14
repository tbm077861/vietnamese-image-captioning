import torch
from PIL import Image

from vic.data.annotations import make_record
from vic.data.dataset import ImageStore
from vic.data.image import load_image, preprocess_image, to_rgb


def _half_red_image():
    image = Image.new("RGB", (40, 20), "white")
    image.paste((255, 0, 0), (0, 0, 20, 20))  # red on the left half
    return image


def test_preprocess_image_shape_and_exif_orientation():
    image = _half_red_image()
    tensor = preprocess_image(image, 32)
    assert tensor.shape == (3, 32, 32) and tensor.dtype == torch.float32
    assert tensor[1, 16, 2] < tensor[1, 16, 30]  # green channel: the red side is darker

    image.getexif()[0x0112] = 3  # EXIF orientation: stored rotated by 180 degrees
    flipped = preprocess_image(image, 32)
    assert flipped[1, 16, 2] > flipped[1, 16, 30]


def test_transparent_areas_are_shown_on_white():
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))  # fully transparent black
    image.paste((255, 0, 0, 255), (0, 0, 4, 8))  # opaque red on the left half
    rgb = to_rgb(image)
    assert rgb.mode == "RGB"
    assert rgb.getpixel((1, 4)) == (255, 0, 0) and rgb.getpixel((6, 4)) == (255, 255, 255)
    assert to_rgb(Image.new("L", (4, 4), 128)).getpixel((0, 0)) == (128, 128, 128)


def test_non_rgb_and_truncated_files_load(tmp_path):
    palette = tmp_path / "palette.png"
    _half_red_image().convert("P").save(palette)
    assert load_image(palette).mode == "RGB"

    cut = tmp_path / "cut.jpg"
    Image.effect_noise((96, 96), 64).convert("RGB").save(cut, quality=95)
    cut.write_bytes(cut.read_bytes()[:-40])
    assert load_image(cut).size == (96, 96)

    records = [make_record("p", "palette.png", "train", ["x"])]
    store = ImageStore(tmp_path, records, image_size=16)
    assert store["p"].shape == (3, 16, 16)
