# Star Learners Bidi Agent

An AI-powered voice assistant for Star Learners childcare centre, built with Google Gemini Live (bidirectional audio), FastAPI, and Qdrant vector search. The assistant — named **Stella** — answers questions about programmes, facilities, fees, and enrolment, and can reference relevant moments in the centre's video tour.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Step 1 — Qdrant Setup](#step-1--qdrant-setup)
5. [Step 2 — Data Pipeline (Build Knowledge Base)](#step-2--data-pipeline-build-knowledge-base)
6. [Step 3 — Application Setup](#step-3--application-setup)
7. [Running the Application](#running-the-application)
8. [Environment Variables Reference](#environment-variables-reference)
9. [Querying the Knowledge Base Directly](#querying-the-knowledge-base-directly)

---

## Architecture Overview

```
Browser (WebRTC audio + chat)
        │
        ▼
FastAPI Server (app/main.py)
  ├── WebSocket  ──► Google ADK Agent (Stella)
  │                       └── search_knowledge_base()
  │                                  │
  │                                  ▼
  │                          Qdrant Vector DB
  │                    ┌─────────────────────────┐
  │                    │  text_vector  (website)  │
  │                    │  image_vector (video)    │
  │                    └─────────────────────────┘
  │
  └── REST API  ──► /api/qdrant-search
```

**Key technologies:**

| Layer | Technology |
|---|---|
| Voice AI | Gemini Live 2.5 Flash (native audio) via Google ADK |
| Embeddings (text) | `gemini-embedding-001` (Vertex AI) |
| Embeddings (video frames) | `multimodalembedding@001` (Vertex AI) |
| Frame captioning | `gemini-2.5-flash` |
| Vector database | Qdrant |
| Backend | FastAPI + Python |
| Frontend | Vanilla JS + WebSocket |

---

## Prerequisites

- Python 3.10+
- A Google Cloud project with **Vertex AI API** enabled
- `gcloud` CLI authenticated (`gcloud auth application-default login`)
- Qdrant running locally **or** a Qdrant Cloud account
- `ffmpeg` installed (required for YouTube video frame extraction)

---

## Project Structure

```
star_learners_bidi_agent/
├── app/                        # FastAPI application
│   ├── main.py                 # Server entry point
│   ├── .env                    # App environment variables
│   ├── requirements.txt
│   ├── google_search_agent/
│   │   ├── agent.py            # ADK agent definition (Stella)
│   │   └── qdrant_tool.py      # Qdrant search tool
│   └── static/                 # Frontend (HTML/CSS/JS)
│
├── data/                       # Data ingestion pipeline
│   ├── build_qdrant_index.py   # Ingest websites + YouTube into Qdrant
│   ├── query_qdrant.py         # CLI tool to query Qdrant
│   ├── sources.yaml            # Website URLs and YouTube source
│   ├── env.example             # Environment variable template
│   └── requirements.txt
│
└── README.md
```

---

## Step 1 — Qdrant Setup

### Option A: Local Qdrant with Docker (recommended for development)

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Verify it is running:

```bash
curl http://localhost:6333/healthz
# Expected: {"title":"qdrant - vector search engine","version":"..."}
```

### Option B: Qdrant Cloud

1. Create a free cluster at https://cloud.qdrant.io
2. Copy your **Cluster URL** and **API Key** — you will need them in the next steps.

---

## Step 2 — Data Pipeline (Build Knowledge Base)

This step scrapes the Star Learners website and extracts frames from the YouTube tour video, then stores everything in Qdrant with text and image embeddings.

### 2.1 Install data pipeline dependencies

```bash
cd data
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2.2 Configure environment

Copy the example file and fill in your values:

```bash
cp data/env.example data/.env
```

Edit `data/.env`:

```env
# Qdrant
QDRANT_URL=http://localhost:6333          # or your Qdrant Cloud URL
QDRANT_API_KEY=                           # leave empty for local; required for Qdrant Cloud

# Collection name
QDRANT_COLLECTION=star_learners_kb

# Vertex AI (required for multimodal embeddings)
GCP_PROJECT=your_gcp_project_id
GCP_LOCATION=us-central1

# Model overrides (optional — defaults shown)
GEMINI_TEXT_EMBED_MODEL=gemini-embedding-001
GEMINI_IMAGE_EMBED_MODEL=multimodalembedding@001
GEMINI_CAPTION_MODEL=gemini-2.5-flash
```

### 2.3 Configure data sources

`data/sources.yaml` contains the website pages and YouTube video to ingest. Edit this file to add or remove sources:

```yaml
websites:
  - https://starlearners.com.sg/our-centres/yung-ho/
  - https://starlearners.com.sg/our-centres/
  # ... add more pages

youtube:
  url: https://www.youtube.com/watch?v=tkhpVEcBfv0
```

### 2.4 Run ingestion

**Ingest everything (websites + YouTube video):**

```bash
python data/build_qdrant_index.py --mode all
```

**Ingest only website pages:**

```bash
python data/build_qdrant_index.py --mode websites
```

**Ingest only YouTube video frames:**

```bash
python data/build_qdrant_index.py --mode youtube
```

**Advanced options:**

```bash
python data/build_qdrant_index.py \
  --mode all \
  --frame-interval-sec 5 \       # Extract a frame every 5 seconds (default: 10)
  --collection star_learners_kb \
  --batch-size 32 \
  --recreate-collection           # Drop and recreate the collection first
```

> **Note:** Ingestion is idempotent. Re-running uses deterministic `doc_id` hashes so existing points are upserted, not duplicated.

### 2.5 Verify ingestion

```bash
# Check collection info
curl http://localhost:6333/collections/star_learners_kb

# Run a test query
python data/query_qdrant.py --query "infant care programme" --top-k 5
```

---

## Step 3 — Application Setup

### 3.1 Install application dependencies

```bash
cd app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or from the root using the root `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3.2 Configure application environment

Create `app/.env` (you can copy and edit from the data env example):

```env
# Google Cloud / Vertex AI
GOOGLE_CLOUD_PROJECT=your_gcp_project_id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=true
GCP_PROJECT=your_gcp_project_id
GCP_LOCATION=us-central1

# Gemini Live model for the agent
DEMO_AGENT_MODEL=gemini-live-2.5-flash-native-audio

# Qdrant — must match what was used during ingestion
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=star_learners_kb
# QDRANT_API_KEY=          # Required if using Qdrant Cloud

# Embedding models — must match what was used during ingestion
GEMINI_TEXT_EMBED_MODEL=gemini-embedding-001
GEMINI_IMAGE_EMBED_MODEL=multimodalembedding@001
```

### 3.3 Authenticate with Google Cloud

```bash
gcloud auth application-default login
gcloud config set project your_gcp_project_id
```

---

## Running the Application

```bash
cd app
python main.py
```

The server starts at **http://127.0.0.1:8000**.

Open your browser to `http://127.0.0.1:8000` to start a voice conversation with Stella.

**Available endpoints:**

| Endpoint | Description |
|---|---|
| `GET /` | Web frontend |
| `WS /ws/{user_id}/{session_id}` | Bidirectional audio/text stream |
| `POST /api/qdrant-search` | Direct Qdrant search (JSON body: `{"query": "..."}`) |

---

## Environment Variables Reference

### Application (`app/.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | Yes | — | GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | Yes | `us-central1` | GCP region |
| `GOOGLE_GENAI_USE_VERTEXAI` | Yes | `true` | Use Vertex AI backend |
| `GCP_PROJECT` | Yes | — | GCP project ID (used by embedding client) |
| `GCP_LOCATION` | Yes | `us-central1` | GCP region (used by embedding client) |
| `DEMO_AGENT_MODEL` | No | `gemini-live-2.5-flash-native-audio` | Gemini Live model |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | No | `star_learners_kb` | Qdrant collection name |
| `QDRANT_API_KEY` | No | — | Qdrant API key (Qdrant Cloud only) |
| `GEMINI_TEXT_EMBED_MODEL` | No | `gemini-embedding-001` | Text embedding model |
| `GEMINI_IMAGE_EMBED_MODEL` | No | `multimodalembedding@001` | Image embedding model |

### Data Pipeline (`data/.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | No | — | Qdrant API key (Qdrant Cloud only) |
| `QDRANT_COLLECTION` | No | `star_learners_kb` | Qdrant collection name |
| `GCP_PROJECT` | Yes | — | GCP project ID (for multimodal embeddings) |
| `GCP_LOCATION` | No | `us-central1` | GCP region |
| `GEMINI_TEXT_EMBED_MODEL` | No | `gemini-embedding-001` | Text embedding model |
| `GEMINI_IMAGE_EMBED_MODEL` | No | `multimodalembedding@001` | Image/frame embedding model |
| `GEMINI_CAPTION_MODEL` | No | `gemini-2.5-flash` | Model used to caption video frames |

---

## Querying the Knowledge Base Directly

Use `data/query_qdrant.py` to test retrieval without starting the application:

```bash
# Search all sources
python data/query_qdrant.py --query "what programmes are available?" --top-k 5

# Search website content only
python data/query_qdrant.py --query "infant care programme fees" --top-k 5 --source-type website

# Search video frames only (returns YouTube timestamps)
python data/query_qdrant.py --query "classroom tour" --top-k 5 --source-type youtube
```

Example output for a video query:

```json
{
  "query": "classroom tour",
  "results": [
    {
      "score": 0.87,
      "source_type": "youtube",
      "content_preview": "Children exploring art materials at a table...",
      "video_id": "tkhpVEcBfv0",
      "timestamp_sec": 45,
      "timestamp_hms": "00:00:45",
      "youtube_deeplink": "https://www.youtube.com/watch?v=tkhpVEcBfv0&t=45s"
    }
  ]
}
```
