"""FastAPI application: visual search over the product catalog.

Run:  uvicorn app.main:app --host 127.0.0.1 --port 8000
Docs: http://127.0.0.1:8000/docs

NOTE: this module intentionally does not use `from __future__ import annotations`.
The endpoints are defined inside `create_app` and refer to its local `settings`,
which FastAPI can only resolve if annotations are evaluated immediately.
"""
import io
import logging
import mimetypes
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import __version__
from .config import PROJECT_ROOT, Settings
from .engine import SearchEngine, UnknownFilterValue
from .imaging import InvalidImage, load_image
from .schemas import (FiltersResponse, HealthResponse, HitOut, InfoResponse, ProductOut, QueryInfo,
                      SearchResponse, SimilarResponse, Timings)

logger = logging.getLogger("vpf")

DESCRIPTION = """
Upload a photo of a fashion product and get visually similar products from the catalog.

**How it works:** the image is converted into a 2048-number embedding with a pretrained ResNet50
(same preprocessing as the catalog), and compared with every catalog embedding by cosine similarity.

Scores are **cosine similarities**: compare them only between the results of one query.
"""

TAGS = [
    {"name": "Search", "description": "Find similar products."},
    {"name": "Catalog", "description": "Look up catalog products and their images."},
    {"name": "Service", "description": "Health and configuration."},
]

ALLOWED_TYPE_HINT = "Upload an image file (JPEG, PNG or WebP)."


def _product(engine: SearchEngine, row: int) -> ProductOut:
    record = engine.record(row)
    return ProductOut(image_url=f"/images/{record['id']}", **record)


def _hit(engine: SearchEngine, hit, rank: int) -> HitOut:
    record = engine.record(hit.row)
    return HitOut(rank=rank, score=round(hit.score, 6), image_url=f"/images/{record['id']}", **record)


def _make_thumbnail(path: str, size: int) -> bytes:
    with Image.open(path) as image:
        image.draft("RGB", (size * 2, size * 2))       # lets the JPEG decoder skip work for big files
        image = image.convert("RGB")
        image.thumbnail((size, size))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=85)
        return buffer.getvalue()


def create_app(settings: Settings | None = None, engine: SearchEngine | None = None, embedder: Any = None) -> FastAPI:
    """Build the application. `engine` and `embedder` can be injected (this is how the tests work)."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is None:
            app.state.engine = SearchEngine.load(settings.bundle_dir)
            logger.info("Catalog loaded: %d products, %d dimensions", app.state.engine.size, app.state.engine.dim)
        if app.state.embedder is None:
            from .embedder import Embedder
            app.state.embedder = Embedder(app.state.engine.config, device=settings.device)
        warmup = getattr(app.state.embedder, "warmup", None)
        if callable(warmup):
            warmup()
        yield

    app = FastAPI(title="Visual Product Finder API", version=__version__, description=DESCRIPTION,
                  openapi_tags=TAGS, lifespan=lifespan)
    app.state.engine = engine
    app.state.embedder = embedder

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-ms"] = f"{elapsed:.1f}"
        logger.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, elapsed)
        return response

    # ----- dependencies --------------------------------------------------------------------

    def get_engine(request: Request) -> SearchEngine:
        found = request.app.state.engine
        if found is None:
            raise HTTPException(503, "The search engine is not loaded yet.")
        return found

    def get_embedder(request: Request):
        found = request.app.state.embedder
        if found is None:
            raise HTTPException(503, "The image model is not loaded yet.")
        return found

    def filters_dep(
        gender: Annotated[str | None, Query(description="Only products for this gender, e.g. Men or Women")] = None,
        master_category: Annotated[str | None, Query(description="e.g. Apparel, Footwear, Accessories")] = None,
        sub_category: Annotated[str | None, Query(description="e.g. Topwear, Watches")] = None,
        article_type: Annotated[str | None, Query(description="e.g. Tshirts, Casual Shoes")] = None,
        base_colour: Annotated[str | None, Query(description="e.g. Blue, Black")] = None,
    ) -> dict[str, str]:
        raw = {"gender": gender, "master_category": master_category, "sub_category": sub_category,
               "article_type": article_type, "base_colour": base_colour}
        return {key: value.strip() for key, value in raw.items() if value and value.strip()}

    EngineDep = Annotated[SearchEngine, Depends(get_engine)]
    FiltersDep = Annotated[dict, Depends(filters_dep)]
    KParam = Annotated[int | None, Query(ge=1, le=settings.max_k, description="Number of results (default from the catalog settings)")]
    DedupeParam = Annotated[bool | None, Query(description="Show only one product among practically identical images")]

    def checked_filters(engine_: SearchEngine, filters: dict) -> dict:
        try:
            engine_.validate_filters(filters)
        except UnknownFilterValue as exc:
            raise HTTPException(400, str(exc)) from exc
        return filters

    # ----- search --------------------------------------------------------------------------

    @app.post("/search", response_model=SearchResponse, tags=["Search"],
              summary="Find products similar to an uploaded image",
              responses={400: {"description": "Not a valid image, or an unknown filter value"},
                         413: {"description": "File too large"},
                         415: {"description": "Not an image file"}})
    def search_by_image(
        file: Annotated[UploadFile, File(description="Photo of a product (JPEG, PNG or WebP)")],
        engine_: EngineDep,
        embedder_: Annotated[Any, Depends(get_embedder)],
        filters: FiltersDep,
        k: KParam = None,
        dedupe: DedupeParam = None,
    ):
        content_type = (file.content_type or "").lower()
        if content_type and not content_type.startswith("image/") and content_type != "application/octet-stream":
            raise HTTPException(415, f"Unsupported file type '{content_type}'. {ALLOWED_TYPE_HINT}")

        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"The file is larger than {settings.max_upload_bytes / 1024 / 1024:.0f} MB.")

        checked_filters(engine_, filters)           # fail fast, before any expensive work
        t0 = time.perf_counter()
        try:
            image = load_image(data, max_pixels=settings.max_pixels, min_side=settings.min_side)
        except InvalidImage as exc:
            raise HTTPException(400, str(exc)) from exc
        t1 = time.perf_counter()
        vector = embedder_.embed(image)
        t2 = time.perf_counter()
        hits = engine_.search(vector, k, filters=filters, dedupe=dedupe)
        t3 = time.perf_counter()

        return SearchResponse(
            query=QueryInfo(filename=file.filename, content_type=file.content_type, bytes=len(data),
                            width=image.width, height=image.height),
            k=k or engine_.k_default,
            results=[_hit(engine_, hit, rank) for rank, hit in enumerate(hits, start=1)],
            timings=Timings(decode_ms=round((t1 - t0) * 1000, 1), embed_ms=round((t2 - t1) * 1000, 1),
                            search_ms=round((t3 - t2) * 1000, 1), total_ms=round((t3 - t0) * 1000, 1)),
        )

    # ----- catalog -------------------------------------------------------------------------

    @app.get("/products/{product_id}", response_model=ProductOut, tags=["Catalog"], summary="Product details")
    def get_product(product_id: int, engine_: EngineDep):
        row = engine_.row_of(product_id)
        if row is None:
            raise HTTPException(404, f"Product {product_id} not found.")
        return _product(engine_, row)

    @app.get("/products/{product_id}/similar", response_model=SimilarResponse, tags=["Search"],
             summary="Products similar to a catalog product")
    def similar_products(product_id: int, engine_: EngineDep, filters: FiltersDep, k: KParam = None, dedupe: DedupeParam = None):
        """The product itself and products with a practically identical image are left out."""
        row = engine_.row_of(product_id)
        if row is None:
            raise HTTPException(404, f"Product {product_id} not found.")
        checked_filters(engine_, filters)
        t0 = time.perf_counter()
        hits = engine_.search(engine_.vector(row), k, filters=filters, dedupe=dedupe, exclude_row=row, exclude_twins=True)
        elapsed = (time.perf_counter() - t0) * 1000
        return SimilarResponse(product=_product(engine_, row), k=k or engine_.k_default,
                               results=[_hit(engine_, hit, rank) for rank, hit in enumerate(hits, start=1)],
                               search_ms=round(elapsed, 1))

    thumbnails = lru_cache(maxsize=settings.thumbnail_cache_size)(_make_thumbnail)

    @app.get("/images/{product_id}", tags=["Catalog"], summary="Product image (optionally as a thumbnail)",
             responses={200: {"content": {"image/jpeg": {}}}, 404: {"description": "Unknown product or image not available"}})
    def product_image(
        product_id: int,
        engine_: EngineDep,
        size: Annotated[int, Query(ge=0, le=1600, description="Longest side in pixels. 0 returns the original file.")] = 0,
    ):
        if settings.image_dir is None:
            raise HTTPException(404, "Product images are not configured. Set VPF_IMAGE_DIR (see README).")
        row = engine_.row_of(product_id)
        if row is None:
            raise HTTPException(404, f"Product {product_id} not found.")

        base = settings.image_dir.resolve()
        path = (base / engine_.image_filename(row)).resolve()
        if base not in path.parents or not path.is_file():      # never leave the image folder
            raise HTTPException(404, f"No image file for product {product_id}.")

        headers = {"Cache-Control": "public, max-age=86400"}
        if size == 0:
            return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "image/jpeg", headers=headers)
        try:
            return Response(thumbnails(str(path), size), media_type="image/jpeg", headers=headers)
        except OSError as exc:
            raise HTTPException(404, f"The image of product {product_id} could not be read.") from exc

    @app.get("/filters", response_model=FiltersResponse, tags=["Catalog"], summary="Values that can be used as filters")
    def list_filters(engine_: EngineDep):
        return FiltersResponse(**engine_.filter_values())

    # ----- service -------------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["Service"], summary="Is the service running?")
    def health(request: Request, engine_: EngineDep):
        model = getattr(request.app.state.embedder, "description", "unknown")
        return HealthResponse(status="ok", version=__version__, catalog_size=engine_.size,
                              embedding_dim=engine_.dim, model=model)

    @app.get("/info", response_model=InfoResponse, tags=["Service"], summary="Model, catalog and search configuration")
    def info(request: Request, engine_: EngineDep):
        cfg = engine_.config
        return InfoResponse(
            version=__version__,
            model={"architecture": cfg.get("architecture"), "weights": cfg.get("weights"),
                   "embedding_dim": engine_.dim, "mixed_precision_fp16": cfg.get("mixed_precision_fp16"),
                   "preprocessing": cfg.get("preprocessing"),
                   "running_on": getattr(request.app.state.embedder, "description", None)},
            catalog={"size": engine_.size, "dtype": "float32", "md5": cfg.get("catalog", {}).get("md5")},
            search={"default_k": engine_.k_default, "candidates": engine_.candidates,
                    "deduplicate_results": engine_.dedupe_default, "duplicate_threshold": engine_.dup_threshold,
                    "similarity": cfg.get("similarity", "cosine")},
            limits={"max_k": settings.max_k, "max_upload_mb": round(settings.max_upload_bytes / 1024 / 1024, 1),
                    "max_pixels": settings.max_pixels, "min_side": settings.min_side},
        )

    # ----- website -------------------------------------------------------------------------

    static_dir = PROJECT_ROOT / "static"
    if (static_dir / "index.html").is_file():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", include_in_schema=False)
        def website():
            return FileResponse(static_dir / "index.html")

    return app


app = create_app()
