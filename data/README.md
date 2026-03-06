# Qdrant + Gemini Data Pipeline

This folder contains a parallel retrieval pipeline (separate from existing Vertex RAG) for:
- Star Learners website ingestion (text chunks)
- YouTube demo video ingestion (timestamped frames)
- Querying Qdrant for website answers and tour/demo timestamps

## Files
- `build_qdrant_index.py`: ingest websites + YouTube into Qdrant
- `query_qdrant.py`: search and return top results with YouTube deeplinks
- `sources.yaml`: exact URLs and YouTube source URL
- `requirements.txt`: Python dependencies for this pipeline

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r data/requirements.txt
```

Required environment variables:

```bash
export GOOGLE_API_KEY="..."
export QDRANT_URL="http://localhost:6333"  # or Qdrant Cloud URL
export QDRANT_API_KEY="..."                # optional for local
export QDRANT_COLLECTION="star_learners_kb"  # optional
```

Optional model overrides:

```bash
export GEMINI_TEXT_EMBED_MODEL="gemini-embedding-001"
export GEMINI_IMAGE_EMBED_MODEL="gemini-embedding-001"
export GEMINI_CAPTION_MODEL="gemini-2.0-flash"
```

## Ingestion Commands

```bash
python data/build_qdrant_index.py --mode all
python data/build_qdrant_index.py --mode websites
python data/build_qdrant_index.py --mode youtube
```

Optional flags:

```bash
python data/build_qdrant_index.py \
  --mode all \
  --frame-interval-sec 5 \
  --collection star_learners_kb \
  --batch-size 32 \
  --recreate-collection
```

## Query Commands

```bash
python data/query_qdrant.py --query "show me tour demo classroom" --top-k 5
python data/query_qdrant.py --query "infant care programme" --top-k 5 --source-type website
python data/query_qdrant.py --query "show the tour video" --top-k 5 --source-type youtube
```

Output format:
- `query`
- `results[]` with fields:
  - `score`
  - `source_type`
  - `content_preview`
  - `url`
  - `video_id`
  - `timestamp_sec`
  - `timestamp_hms`
  - `youtube_deeplink`

## Notes
- `build_qdrant_index.py` uses deterministic `doc_id` hashes, so reruns upsert existing points instead of creating logical duplicates.
- Script is resilient to per-item failures and prints a final JSON summary.
- YouTube query path attempts image-vector retrieval from text query. If the embedding model does not support that input path, the script gracefully skips YouTube vector query.
