#!/usr/bin/env python3
"""Query Qdrant for website and YouTube timestamp retrieval."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import vertexai
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from vertexai.vision_models import MultiModalEmbeddingModel

DEFAULT_COLLECTION = "star_learners_kb"
DEFAULT_TEXT_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_IMAGE_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_CAPTION_MODEL = "gemini-2.0-flash"
TOUR_HINT_KEYWORDS = {"tour", "demo", "show", "watch", "see", "video", "classroom"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Star Learners Qdrant index")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--source-type", choices=["youtube", "website"], default=None)
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION))
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def to_youtube_deeplink(video_id: Optional[str], timestamp_sec: Optional[int]) -> Optional[str]:
    if not video_id or timestamp_sec is None:
        return None
    return f"https://www.youtube.com/watch?v={video_id}&t={timestamp_sec}s"


class GeminiQuery:
    def __init__(self) -> None:
        api_key = require_env("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.text_embed_model = os.getenv("GEMINI_TEXT_EMBED_MODEL", DEFAULT_TEXT_EMBED_MODEL)
        self.image_embed_model = os.getenv("GEMINI_IMAGE_EMBED_MODEL", DEFAULT_IMAGE_EMBED_MODEL)
        self.caption_model = os.getenv("GEMINI_CAPTION_MODEL", DEFAULT_CAPTION_MODEL)

        self._vertex_image_model: Optional[MultiModalEmbeddingModel] = None
        if "multimodalembedding" in self.image_embed_model:
            gcp_project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
            gcp_location = os.getenv("GCP_LOCATION", "us-central1")
            vertexai.init(project=gcp_project, location=gcp_location)
            self._vertex_image_model = MultiModalEmbeddingModel.from_pretrained(self.image_embed_model)

    @staticmethod
    def _first_vector(response: Any) -> List[float]:
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")
        if not embeddings:
            raise ValueError("No embeddings in response")
        first = embeddings[0]
        values = getattr(first, "values", None)
        if values is None and isinstance(first, dict):
            values = first.get("values")
        if not values:
            raise ValueError("Embedding vector missing values")
        return list(values)

    def embed_text_query(self, query: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.text_embed_model,
            contents=[query],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return self._first_vector(response)

    def build_visual_bridge_query(self, query: str) -> str:
        response = self.client.models.generate_content(
            model=self.caption_model,
            contents=[
                (
                    "Rewrite this user query into a short visual scene description suitable "
                    "for matching video frames: "
                    f"{query}"
                )
            ],
        )
        if getattr(response, "text", None):
            return response.text.strip()
        return query

    def embed_for_image_search(self, query: str) -> Optional[List[float]]:
        try:
            if self._vertex_image_model is not None:
                # multimodalembedding@001 can embed text queries directly into 1408-dim space
                result = self._vertex_image_model.get_embeddings(contextual_text=query)
                return list(result.text_embedding)
            # Fallback: text-only model — embed a visual bridge description
            bridge = self.build_visual_bridge_query(query)
            response = self.client.models.embed_content(
                model=self.image_embed_model,
                contents=[bridge],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            return self._first_vector(response)
        except Exception:
            return None


def qdrant_client() -> QdrantClient:
    return QdrantClient(url=require_env("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))


def source_filter(source_type: Optional[str]) -> Optional[qmodels.Filter]:
    if source_type == "youtube":
        value = "youtube_frame"
    elif source_type == "website":
        value = "website_chunk"
    else:
        return None

    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="source_type",
                match=qmodels.MatchValue(value=value),
            )
        ]
    )


def search_points(
    client: QdrantClient,
    collection: str,
    vector_name: str,
    vector: List[float],
    limit: int,
    query_filter: Optional[qmodels.Filter],
) -> List[Any]:
    response = client.query_points(
        collection_name=collection,
        query=vector,
        using=vector_name,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )
    return response.points


def has_tour_intent(query: str) -> bool:
    q = query.lower()
    return any(keyword in q for keyword in TOUR_HINT_KEYWORDS)


def format_result(hit: Any, source: str) -> Dict[str, Any]:
    payload = dict(hit.payload or {})
    content = payload.get("content") or ""
    video_id = payload.get("video_id")
    timestamp_sec = payload.get("timestamp_sec")

    return {
        "score": float(hit.score),
        "source_type": source,
        "content_preview": content[:240],
        "url": payload.get("source_url"),
        "video_id": video_id,
        "timestamp_sec": timestamp_sec,
        "timestamp_hms": payload.get("timestamp_hms"),
        "youtube_deeplink": to_youtube_deeplink(video_id, timestamp_sec),
    }


def main() -> None:
    load_dotenv(Path(__file__).parent / ".env")
    args = parse_args()

    gemini = GeminiQuery()
    client = qdrant_client()
    query_filter = source_filter(args.source_type)

    results: List[Dict[str, Any]] = []

    if args.source_type in (None, "website"):
        text_vec = gemini.embed_text_query(args.query)
        text_hits = search_points(
            client=client,
            collection=args.collection,
            vector_name="text_vector",
            vector=text_vec,
            limit=args.top_k,
            query_filter=query_filter,
        )
        results.extend(format_result(hit, "website_chunk") for hit in text_hits)

    if args.source_type in (None, "youtube"):
        image_query_vec = gemini.embed_for_image_search(args.query)
        if image_query_vec is not None:
            image_hits = search_points(
                client=client,
                collection=args.collection,
                vector_name="image_vector",
                vector=image_query_vec,
                limit=args.top_k,
                query_filter=query_filter,
            )
            results.extend(format_result(hit, "youtube_frame") for hit in image_hits)

    # Prioritize timestamped YouTube hits when intent suggests demo/tour browsing.
    if has_tour_intent(args.query):
        for row in results:
            if row["source_type"] == "youtube_frame":
                row["score"] += 0.1

    results.sort(key=lambda x: x["score"], reverse=True)
    output = {
        "query": args.query,
        "results": results[: args.top_k],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
