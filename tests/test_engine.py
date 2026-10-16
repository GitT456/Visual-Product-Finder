import json

import numpy as np
import pandas as pd
import pytest

from app.engine import BundleError, SearchEngine, UnknownFilterValue
from conftest import DIM, N, TWIN_ROWS, make_bundle


def test_loads_and_reports_size(engine):
    assert engine.size == N and engine.dim == DIM
    assert engine.row_of(1005) == 5 and engine.row_of(99999) is None


def test_query_with_a_catalog_vector_finds_itself_first(engine):
    hits = engine.search(engine.vector(30), k=5)
    assert hits[0].row == 30 and hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_scores_are_sorted_and_k_is_respected(engine):
    for k in (1, 3, 5, 12):
        hits = engine.search(engine.vector(3), k=k)
        assert len(hits) == k
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)


def test_search_matches_brute_force(engine):
    q = engine.vector(20)
    hits = engine.search(q, k=8, dedupe=False, exclude_row=20)
    expected = np.argsort(-(engine.E @ q))
    expected = [r for r in expected if r != 20][:8]
    assert [h.row for h in hits] == expected


def test_exclude_row_removes_the_query_product(engine):
    rows = [h.row for h in engine.search(engine.vector(30), k=10, exclude_row=30)]
    assert 30 not in rows


def test_dedupe_collapses_identical_images(engine):
    a, b = TWIN_ROWS
    with_twins = [h.row for h in engine.search(engine.vector(a), k=5, dedupe=False)]
    deduped = [h.row for h in engine.search(engine.vector(a), k=5, dedupe=True)]
    assert a in with_twins and b in with_twins
    assert (a in deduped) != (b in deduped)         # exactly one of the two twins remains
    assert len(deduped) == 5                        # and the freed slot is filled by another product


def test_exclude_twins_removes_identical_products(engine):
    a, b = TWIN_ROWS
    rows = [h.row for h in engine.search(engine.vector(a), k=10, exclude_row=a, exclude_twins=True)]
    assert a not in rows and b not in rows


def test_filters_are_applied_and_case_insensitive(engine):
    hits = engine.search(engine.vector(0), k=10, filters={"article_type": "watches", "gender": "MEN"})
    assert hits
    for h in hits:
        rec = engine.record(h.row)
        assert rec["article_type"] == "Watches" and rec["gender"] == "Men"


def test_filter_with_few_matches_returns_fewer_results(engine):
    hits = engine.search(engine.vector(0), k=20, filters={"article_type": "Watches", "base_colour": "Red"})
    assert 0 < len(hits) < 20


def test_unknown_filter_value_raises(engine):
    with pytest.raises(UnknownFilterValue):
        engine.search(engine.vector(0), filters={"article_type": "Spaceships"})


def test_invalid_queries_are_rejected(engine):
    with pytest.raises(ValueError):
        engine.search(np.zeros(DIM))
    with pytest.raises(ValueError):
        engine.search(np.ones(DIM + 1))
    with pytest.raises(ValueError):
        engine.search(np.full(DIM, np.nan))
    with pytest.raises(ValueError):
        engine.search(engine.vector(0), k=0)


def test_k_larger_than_catalog_is_safe(engine):
    assert len(engine.search(engine.vector(0), k=1000, dedupe=False)) == N


def test_record_converts_missing_values_to_none(engine):
    assert engine.record(7)["base_colour"] is None
    assert engine.record(5)["year"] is None
    assert engine.record(0)["year"] == 2012 and isinstance(engine.record(0)["id"], int)


def test_bundle_checks(tmp_path):
    # missing file
    folder = make_bundle(tmp_path / "a")
    (folder / "products.csv").unlink()
    with pytest.raises(BundleError, match="Missing"):
        SearchEngine.load(folder)

    # wrong checksum (embeddings changed after the config was written)
    folder = make_bundle(tmp_path / "b")
    E = np.load(folder / "catalog_embeddings.npy")
    E[0] = E[1]
    np.save(folder / "catalog_embeddings.npy", E)
    with pytest.raises(BundleError, match="Checksum"):
        SearchEngine.load(folder)

    # rows no longer aligned
    folder = make_bundle(tmp_path / "c")
    products = pd.read_csv(folder / "products.csv")
    products = products.iloc[::-1]
    products.to_csv(folder / "products.csv", index=False)
    with pytest.raises(BundleError, match="Row order"):
        SearchEngine.load(folder)

    # different number of products
    folder = make_bundle(tmp_path / "d")
    pd.read_csv(folder / "products.csv").iloc[:-1].to_csv(folder / "products.csv", index=False)
    with pytest.raises(BundleError, match="do not belong"):
        SearchEngine.load(folder)

    # wrong embedding size
    folder = make_bundle(tmp_path / "e")
    cfg = json.loads((folder / "model_config.json").read_text())
    cfg["embedding_dim"] = 2048
    (folder / "model_config.json").write_text(json.dumps(cfg))
    with pytest.raises(BundleError, match="Embedding size"):
        SearchEngine.load(folder)
