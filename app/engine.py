"""The search engine: loads and verifies the catalog bundle, and searches it.

Only NumPy and pandas are used here (no PyTorch), so the engine is cheap to test and to run.

The bundle is produced by the evaluation notebook:
    catalog_embeddings.npy  (N, dim) float32, L2-normalized; row i describes row i of products.csv
    products.csv            product metadata, same row order, column `embedding_row`
    model_config.json       how the embeddings were made (+ checksum)
    search_settings.json    default k, candidate count and the duplicate rule
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

FILTER_COLUMNS = {
    "gender": "gender",
    "master_category": "masterCategory",
    "sub_category": "subCategory",
    "article_type": "articleType",
    "base_colour": "baseColour",
}

RECORD_COLUMNS = {
    "id": "id",
    "name": "productDisplayName",
    "article_type": "articleType",
    "master_category": "masterCategory",
    "sub_category": "subCategory",
    "gender": "gender",
    "base_colour": "baseColour",
    "season": "season",
    "year": "year",
    "usage": "usage",
}


class BundleError(RuntimeError):
    """The catalog bundle is missing, inconsistent or damaged."""


class UnknownFilterValue(ValueError):
    """A filter value does not exist in the catalog."""


class Hit(NamedTuple):
    row: int        # row in catalog_embeddings.npy and products.csv
    score: float    # cosine similarity


class SearchEngine:
    def __init__(self, embeddings: np.ndarray, products: pd.DataFrame, config: dict, settings: dict | None = None):
        settings = settings or {}
        E = np.ascontiguousarray(embeddings, dtype=np.float32)

        if E.ndim != 2:
            raise BundleError("The embedding matrix must be 2-dimensional.")
        expected_dim = config.get("embedding_dim")
        if expected_dim and E.shape[1] != expected_dim:
            raise BundleError(f"Embedding size {E.shape[1]} does not match model_config.json ({expected_dim}).")
        if len(products) != len(E):
            raise BundleError(f"{len(E)} embeddings but {len(products)} products: the files do not belong together.")
        if not np.isfinite(E).all():
            raise BundleError("The embeddings contain NaN or infinite values.")
        norms = np.linalg.norm(E, axis=1)
        if norms.min() < 0.99 or norms.max() > 1.01:
            raise BundleError("The embeddings are not L2-normalized (or contain empty rows).")
        if "id" not in products.columns or not products["id"].is_unique:
            raise BundleError("products.csv needs a unique `id` column.")
        if "embedding_row" in products.columns and not (products["embedding_row"].to_numpy() == np.arange(len(E))).all():
            raise BundleError("Row order mismatch: `embedding_row` does not match the position in the file.")

        self.E = E
        self.products = products.reset_index(drop=True)
        self.config = config
        self.settings = settings

        self.k_default = int(settings.get("k", 5))
        self.candidates = int(settings.get("candidates", 20))
        self.dedupe_default = bool(settings.get("deduplicate_results", True))
        self.dup_threshold = float(settings.get("duplicate_threshold", 0.999))

        ids = self.products["id"].to_numpy()
        self.id_to_row = {int(i): row for row, i in enumerate(ids)}

        self._cols = {col: self.products[col].to_numpy(dtype=object)
                      for col in RECORD_COLUMNS.values() if col in self.products.columns}
        self._filter_arrays: dict[str, np.ndarray] = {}
        self._filter_values: dict[str, list[str]] = {}
        for key, col in FILTER_COLUMNS.items():
            if col in self.products.columns:
                text = self.products[col].fillna("").astype(str).str.strip()
                self._filter_arrays[key] = text.str.lower().to_numpy()
                self._filter_values[key] = sorted(v for v in text.unique() if v)

    # ----- loading -------------------------------------------------------------------------

    @classmethod
    def load(cls, bundle_dir: str | Path) -> "SearchEngine":
        folder = Path(bundle_dir)
        required = ["catalog_embeddings.npy", "products.csv", "model_config.json"]
        missing = [name for name in required if not (folder / name).is_file()]
        if missing:
            raise BundleError(f"Missing file(s) in {folder}: {', '.join(missing)}. "
                              "Copy the `vpf_artifacts` folder from the evaluation notebook output there (see README).")

        config = json.loads((folder / "model_config.json").read_text(encoding="utf-8"))
        settings_path = folder / "search_settings.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else {}

        embeddings = np.load(folder / "catalog_embeddings.npy")
        if embeddings.dtype != np.float32:
            raise BundleError(f"Embeddings must be float32, found {embeddings.dtype}.")

        expected_md5 = config.get("catalog", {}).get("md5")
        if expected_md5:
            actual = hashlib.md5(np.ascontiguousarray(embeddings)).hexdigest()
            if actual != expected_md5:
                raise BundleError("Checksum mismatch: catalog_embeddings.npy does not belong to this model_config.json.")

        products = pd.read_csv(folder / "products.csv")
        return cls(embeddings, products, config, settings)

    # ----- information ---------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.E)

    @property
    def dim(self) -> int:
        return self.E.shape[1]

    def filter_values(self) -> dict[str, list[str]]:
        return self._filter_values

    def row_of(self, product_id: int) -> int | None:
        return self.id_to_row.get(int(product_id))

    def vector(self, row: int) -> np.ndarray:
        return self.E[row].copy()

    def record(self, row: int) -> dict:
        """Metadata of one product with plain Python values (None instead of NaN)."""
        out: dict = {}
        for key, col in RECORD_COLUMNS.items():
            value = self._cols[col][row] if col in self._cols else None
            if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
                out[key] = None
            elif key in ("id", "year"):
                out[key] = int(value)
            else:
                out[key] = str(value)
        return out

    def image_filename(self, row: int) -> str:
        if "image" in self.products.columns:
            return str(self.products["image"].iat[row])
        return f"{int(self.products['id'].iat[row])}.jpg"

    # ----- search --------------------------------------------------------------------------

    def _prepare_query(self, query) -> np.ndarray:
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"The query vector must have shape ({self.dim},), got {q.shape}.")
        if not np.isfinite(q).all():
            raise ValueError("The query vector contains NaN or infinite values.")
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            raise ValueError("The query vector is empty.")
        return q / norm

    def _filter_mask(self, filters: dict[str, str] | None) -> np.ndarray | None:
        if not filters:
            return None
        mask = np.ones(self.size, dtype=bool)
        for key, value in filters.items():
            if value is None or str(value).strip() == "":
                continue
            if key not in self._filter_arrays:
                raise UnknownFilterValue(f"Unknown filter '{key}'.")
            wanted = str(value).strip().lower()
            if wanted not in {v.lower() for v in self._filter_values[key]}:
                raise UnknownFilterValue(f"No product has {key} = '{value}'. See /filters for the available values.")
            mask &= self._filter_arrays[key] == wanted
        return mask

    def validate_filters(self, filters: dict[str, str] | None) -> None:
        """Raise UnknownFilterValue early (before any expensive work) if a filter value does not exist."""
        self._filter_mask(filters)

    def search(self, query, k: int | None = None, *, filters: dict[str, str] | None = None,
               dedupe: bool | None = None, exclude_row: int | None = None,
               exclude_twins: bool = False) -> list[Hit]:
        """Return the k most similar products, best first.

        filters        keep only products with these attribute values (case-insensitive)
        dedupe         show only one product among practically identical images (default: from settings)
        exclude_row    leave this catalog row out (used when the query is itself a catalog product)
        exclude_twins  also leave out products whose image is practically identical to the query
        """
        k = self.k_default if k is None else int(k)
        if k < 1:
            raise ValueError("k must be at least 1.")
        dedupe = self.dedupe_default if dedupe is None else dedupe

        q = self._prepare_query(query)
        scores = self.E @ q                                   # cosine similarity: all vectors have length 1

        mask = self._filter_mask(filters)
        if mask is not None:
            scores[~mask] = -np.inf
        if exclude_row is not None:
            scores[exclude_row] = -np.inf
        if exclude_twins:
            scores[scores > self.dup_threshold] = -np.inf

        # More candidates than k, so that skipped duplicates can be replaced.
        pool = min(self.size, max(self.candidates, 4 * k))
        part = np.argpartition(-scores, pool - 1)[:pool]
        order = part[np.argsort(-scores[part])]
        order = order[np.isfinite(scores[order])]

        if dedupe and len(order) > 1:
            gram = self.E[order] @ self.E[order].T
            kept: list[int] = []
            for i in range(len(order)):
                if not kept or (gram[i, kept] <= self.dup_threshold).all():
                    kept.append(i)
                if len(kept) == k:
                    break
            order = order[kept]
        else:
            order = order[:k]

        return [Hit(row=int(r), score=float(scores[r])) for r in order]
