"""Shared fixtures: a small synthetic catalog bundle and a fake embedder (no PyTorch needed)."""
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N, DIM = 60, 16
TWIN_ROWS = (10, 11)           # two products with an identical image (identical embedding)


def make_bundle(folder: Path, n: int = N, dim: int = DIM) -> Path:
    rng = np.random.RandomState(0)
    E = rng.rand(n, dim).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    E[TWIN_ROWS[1]] = E[TWIN_ROWS[0]]

    article = ["Tshirts", "Casual Shoes", "Watches"]
    master = ["Apparel", "Footwear", "Accessories"]
    colours = ["Blue", "Black", "White", "Red"]
    products = pd.DataFrame({
        "id": 1000 + np.arange(n),
        "gender": ["Men" if i % 2 == 0 else "Women" for i in range(n)],
        "masterCategory": [master[i % 3] for i in range(n)],
        "subCategory": [["Topwear", "Shoes", "Watches"][i % 3] for i in range(n)],
        "articleType": [article[i % 3] for i in range(n)],
        "baseColour": [colours[i % 4] if i != 7 else np.nan for i in range(n)],
        "season": ["Summer"] * n,
        "year": [2012.0 if i != 5 else np.nan for i in range(n)],
        "usage": ["Casual"] * n,
        "productDisplayName": [f"Product {i}" for i in range(n)],
        "image": [f"{1000 + i}.jpg" for i in range(n)],
        "embedding_row": np.arange(n),
    })

    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "catalog_embeddings.npy", E)
    products.to_csv(folder / "products.csv", index=False)
    config = {
        "architecture": "test", "weights": "ResNet50_Weights.IMAGENET1K_V2", "embedding_dim": dim,
        "normalization": "L2", "similarity": "cosine", "mixed_precision_fp16": False,
        "preprocessing": {"mode": "crop", "resize_size": 232, "crop_size": 224, "mean": [0.485, 0.456, 0.406],
                          "std": [0.229, 0.224, 0.225], "interpolation": "bilinear"},
        "catalog": {"size": n, "dtype": "float32", "md5": hashlib.md5(E.tobytes()).hexdigest()},
    }
    (folder / "model_config.json").write_text(json.dumps(config))
    (folder / "search_settings.json").write_text(json.dumps(
        {"k": 5, "candidates": 20, "deduplicate_results": True, "duplicate_threshold": 0.999}))
    return folder


class FakeEmbedder:
    """Maps an image to a catalog vector: the red value of the first pixel is the catalog row."""
    description = "fake embedder"

    def __init__(self, engine):
        self.engine = engine

    def embed(self, image):
        return self.engine.vector(image.getpixel((0, 0))[0] % self.engine.size)


def png_bytes(red: int = 0, size=(64, 64), mode="RGB") -> bytes:
    color = (red, 0, 0) if mode == "RGB" else (red, 0, 0, 255)
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def bundle_dir(tmp_path):
    return make_bundle(tmp_path / "bundle")


@pytest.fixture
def engine(bundle_dir):
    from app.engine import SearchEngine
    return SearchEngine.load(bundle_dir)


@pytest.fixture
def image_dir(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for i in range(6):
        Image.new("RGB", (400, 500), (200, 100, 50)).save(folder / f"{1000 + i}.jpg", "JPEG")
    return folder
