import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.engine import SearchEngine
from app.main import create_app
from conftest import FakeEmbedder, N, TWIN_ROWS, png_bytes


def build_client(bundle_dir, **settings_kwargs):
    engine = SearchEngine.load(bundle_dir)
    settings = Settings(bundle_dir=bundle_dir, **settings_kwargs)
    app = create_app(settings, engine=engine, embedder=FakeEmbedder(engine))
    return TestClient(app), engine


@pytest.fixture
def client(bundle_dir):
    test_client, _ = build_client(bundle_dir)
    return test_client


def upload(client, red, filename="photo.png", content_type="image/png", params=None, data=None):
    return client.post("/search", params=params or {},
                       files={"file": (filename, data if data is not None else png_bytes(red), content_type)})


# ---- service ----------------------------------------------------------------------------

def test_health_and_info(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["catalog_size"] == N
    info = client.get("/info").json()
    assert info["catalog"]["size"] == N and info["search"]["default_k"] == 5
    assert info["limits"]["max_k"] == 20


def test_filters_endpoint(client):
    values = client.get("/filters").json()
    assert "Watches" in values["article_type"] and set(values["gender"]) == {"Men", "Women"}


def test_website_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "text/html" in response.headers["content-type"]


# ---- search by image --------------------------------------------------------------------

def test_search_returns_the_matching_product_first(client):
    response = upload(client, red=30)
    assert response.status_code == 200
    body = response.json()
    assert body["k"] == 5 and len(body["results"]) == 5
    first = body["results"][0]
    assert first["id"] == 1030 and first["rank"] == 1 and first["score"] == pytest.approx(1.0, abs=1e-4)
    assert first["image_url"] == "/images/1030" and first["name"] == "Product 30"
    assert [r["rank"] for r in body["results"]] == [1, 2, 3, 4, 5]
    assert set(body["timings"]) == {"decode_ms", "embed_ms", "search_ms", "total_ms"}
    assert body["query"]["width"] == 64 and body["query"]["filename"] == "photo.png"
    assert "x-process-time-ms" in response.headers


def test_k_parameter(client):
    assert len(upload(client, 3, params={"k": 2}).json()["results"]) == 2
    assert upload(client, 3, params={"k": 0}).status_code == 422
    assert upload(client, 3, params={"k": 21}).status_code == 422


def test_duplicate_results_are_collapsed_but_can_be_kept(client):
    a, b = TWIN_ROWS
    ids = lambda r: [x["id"] for x in r.json()["results"]]                     # noqa: E731
    deduped = ids(upload(client, a, params={"k": 5}))
    raw = ids(upload(client, a, params={"k": 5, "dedupe": "false"}))
    assert (1000 + a in deduped) != (1000 + b in deduped)
    assert 1000 + a in raw and 1000 + b in raw


def test_filters_in_search(client):
    results = upload(client, 0, params={"k": 10, "article_type": "Watches", "gender": "Men"}).json()["results"]
    assert results and all(r["article_type"] == "Watches" and r["gender"] == "Men" for r in results)
    bad = upload(client, 0, params={"article_type": "Spaceships"})
    assert bad.status_code == 400 and "Spaceships" in bad.json()["detail"]


def test_invalid_uploads(client):
    assert upload(client, 0, data=b"not an image", filename="a.jpg", content_type="image/jpeg").status_code == 400
    assert upload(client, 0, data=b"", content_type="image/png").status_code == 400
    assert upload(client, 0, data=b"hello", filename="a.txt", content_type="text/plain").status_code == 415
    assert client.post("/search").status_code == 422                             # no file at all


def test_octet_stream_is_accepted_if_the_content_is_an_image(client):
    assert upload(client, 5, content_type="application/octet-stream").status_code == 200


def test_transparent_png_is_accepted(client):
    assert upload(client, 5, data=png_bytes(5, mode="RGBA")).status_code == 200


def test_upload_size_limit(bundle_dir):
    small_client, _ = build_client(bundle_dir, max_upload_bytes=100)
    response = upload(small_client, 1)
    assert response.status_code == 413


# ---- catalog ----------------------------------------------------------------------------

def test_product_lookup(client):
    body = client.get("/products/1012").json()
    assert body["id"] == 1012 and body["article_type"] == "Tshirts"
    assert client.get("/products/42").status_code == 404
    assert client.get("/products/abc").status_code == 422


def test_similar_products_exclude_the_product_and_its_twin(client):
    a, b = TWIN_ROWS
    body = client.get(f"/products/{1000 + a}/similar", params={"k": 10}).json()
    ids = [r["id"] for r in body["results"]]
    assert body["product"]["id"] == 1000 + a
    assert 1000 + a not in ids and 1000 + b not in ids and len(ids) == 10
    assert client.get("/products/42/similar").status_code == 404
    assert client.get("/products/1000/similar", params={"article_type": "Nope"}).status_code == 400


# ---- images -----------------------------------------------------------------------------

def test_images_without_a_configured_folder(client):
    assert client.get("/images/1000").status_code == 404


def test_images_and_thumbnails(bundle_dir, image_dir):
    from io import BytesIO
    from PIL import Image
    image_client, _ = build_client(bundle_dir, image_dir=image_dir)

    original = image_client.get("/images/1001")
    assert original.status_code == 200 and original.headers["content-type"] == "image/jpeg"
    assert Image.open(BytesIO(original.content)).size == (400, 500)

    thumb = image_client.get("/images/1001", params={"size": 100})
    assert thumb.status_code == 200 and max(Image.open(BytesIO(thumb.content)).size) == 100
    assert "cache-control" in thumb.headers

    assert image_client.get("/images/1030").status_code == 404                  # file does not exist
    assert image_client.get("/images/777").status_code == 404                   # unknown product
    assert image_client.get("/images/1001", params={"size": 5000}).status_code == 422


def test_image_paths_cannot_escape_the_image_folder(bundle_dir, image_dir):
    import pandas as pd
    products = pd.read_csv(bundle_dir / "products.csv")
    products.loc[0, "image"] = "../secret.jpg"
    products.to_csv(bundle_dir / "products.csv", index=False)
    (image_dir.parent / "secret.jpg").write_bytes(b"secret")
    image_client, _ = build_client(bundle_dir, image_dir=image_dir)
    assert image_client.get("/images/1000").status_code == 404


def test_cors_is_off_by_default_and_configurable(bundle_dir):
    default_client, _ = build_client(bundle_dir)
    assert "access-control-allow-origin" not in default_client.get("/health", headers={"Origin": "http://x.test"}).headers
    cors_client, _ = build_client(bundle_dir, cors_origins=("http://x.test",))
    response = cors_client.get("/health", headers={"Origin": "http://x.test"})
    assert response.headers["access-control-allow-origin"] == "http://x.test"


# ---- documentation and startup -----------------------------------------------------------

def test_openapi_schema_and_docs_are_generated(client):
    schema = client.get("/openapi.json").json()
    assert {"/search", "/products/{product_id}/similar", "/images/{product_id}", "/health"} <= set(schema["paths"])
    assert client.get("/docs").status_code == 200


def test_startup_with_injected_components(bundle_dir):
    engine = SearchEngine.load(bundle_dir)
    app = create_app(Settings(bundle_dir=bundle_dir), engine=engine, embedder=FakeEmbedder(engine))
    with TestClient(app) as running:                      # runs the lifespan (startup and shutdown)
        assert running.get("/health").status_code == 200


def test_service_unavailable_until_the_engine_is_loaded(bundle_dir):
    app = create_app(Settings(bundle_dir=bundle_dir))     # nothing loaded, lifespan not started
    assert TestClient(app).get("/health").status_code == 503
