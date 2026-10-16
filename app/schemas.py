"""Response models. They define the JSON the API returns and the documentation at /docs."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProductOut(BaseModel):
    id: int = Field(description="Product ID from the catalog")
    name: str | None = Field(None, description="Product display name")
    article_type: str | None = None
    master_category: str | None = None
    sub_category: str | None = None
    gender: str | None = None
    base_colour: str | None = None
    season: str | None = None
    year: int | None = None
    usage: str | None = None
    image_url: str = Field(description="Relative URL of the product image (needs the image folder, see README)")


class HitOut(ProductOut):
    rank: int = Field(description="1 is the most similar product")
    score: float = Field(description="Cosine similarity between the query and the product (higher is more similar)")


class QueryInfo(BaseModel):
    filename: str | None = None
    content_type: str | None = None
    bytes: int
    width: int
    height: int


class Timings(BaseModel):
    decode_ms: float
    embed_ms: float
    search_ms: float
    total_ms: float


class SearchResponse(BaseModel):
    query: QueryInfo
    k: int
    results: list[HitOut]
    timings: Timings


class SimilarResponse(BaseModel):
    product: ProductOut
    k: int
    results: list[HitOut]
    search_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    catalog_size: int
    embedding_dim: int
    model: str


class FiltersResponse(BaseModel):
    gender: list[str] = []
    master_category: list[str] = []
    sub_category: list[str] = []
    article_type: list[str] = []
    base_colour: list[str] = []


class InfoResponse(BaseModel):
    version: str
    model: dict[str, Any]
    catalog: dict[str, Any]
    search: dict[str, Any]
    limits: dict[str, Any]
