"""Checks that the API produces the same embeddings as the notebooks did for the catalog.

For random catalog products, the product image is sent through the API's own pipeline
(decode -> preprocess -> ResNet50) and compared with the stored embedding of that product.
Cosine similarity should be about 1.0. This is the test that proves that uploaded photos
are processed exactly like the catalog images (and that rows and products are aligned).

Usage:
    python scripts/verify_against_catalog.py --images data/images --n 50
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEFAULT_BUNDLE_DIR, Settings          # noqa: E402
from app.embedder import Embedder                            # noqa: E402
from app.engine import SearchEngine                          # noqa: E402
from app.imaging import load_image                           # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_DIR, help="folder with the engine files")
    parser.add_argument("--images", type=Path, required=True, help="folder with the catalog images")
    parser.add_argument("--n", type=int, default=50, help="number of random products to check")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--threshold", type=float, default=0.999, help="minimum acceptable cosine similarity")
    args = parser.parse_args()

    settings = Settings()
    engine = SearchEngine.load(args.bundle)
    embedder = Embedder(engine.config, device=args.device)
    print(f"Catalog: {engine.size} products | model: {embedder.description}")

    rng = np.random.RandomState(0)
    rows = rng.choice(engine.size, size=min(args.n, engine.size), replace=False)

    similarities, own_first, missing = [], 0, 0
    for row in rows:
        path = args.images / engine.image_filename(int(row))
        if not path.is_file():
            missing += 1
            continue
        image = load_image(path.read_bytes(), max_pixels=settings.max_pixels, min_side=settings.min_side)
        vector = embedder.embed(image)
        similarities.append(float(vector @ engine.E[row]))
        top = engine.search(vector, k=1, dedupe=False)[0]
        own_first += int(top.row == row or top.score > engine.dup_threshold)

    if not similarities:
        print("No images found. Check --images.")
        return 2

    sims = np.array(similarities)
    print(f"Checked {len(sims)} products ({missing} image files missing)")
    print(f"Cosine similarity to the stored embedding: min {sims.min():.5f} | mean {sims.mean():.5f}")
    print(f"Best match is the product itself (or an identical twin): {own_first}/{len(sims)}")

    ok = sims.min() >= args.threshold and own_first == len(sims)
    print("RESULT:", "OK, the API processes images like the catalog" if ok else "PROBLEM, see the numbers above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
