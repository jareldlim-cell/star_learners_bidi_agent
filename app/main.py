"""
FastAPI application for Star Learners AI.
Uses Vertex AI (Gemini Live) for the voice agent and Qdrant for knowledge-base retrieval.
"""

import asyncio
import base64
import json
import logging
import re
import warnings
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ADK and GenAI Imports — wrapped to suppress Pydantic V1/V2 compat warnings at import time
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    from google.adk.agents.live_request_queue import LiveRequestQueue
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

import vertexai

# ========================================
# Phase 0: Environment & SDK Initialization
# ========================================
# Always load env from the app directory first so tools work no matter where
# uvicorn is launched from (repo root vs app/).
app_env_path = Path(__file__).parent / ".env"
load_dotenv(app_env_path)
load_dotenv()

# Force Vertex AI Mode for the GenAI SDK
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]   # KeyError on startup if not set
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")  # Must match corpus region

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Import agent AFTER environment setup
from google_search_agent.agent import agent
from google_search_agent.qdrant_tool import search_qdrant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

APP_NAME = "star-learners-bidi"

# Input validation constants
_SAFE_ID = re.compile(r'^[A-Za-z0-9_-]{1,128}$')
_ALLOWED_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/gif"})

# ========================================
# Phase 1: Application Setup
# ========================================
app = FastAPI()

static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

session_service = InMemorySessionService()
runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)

# ========================================
# Phase 2: Root Endpoint
# ========================================
@app.get("/")
async def root():
    """Serve the main landing page."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


class QdrantSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=20)


@app.post("/api/qdrant-search")
async def qdrant_search_endpoint(payload: QdrantSearchRequest):
    """Search Qdrant for website content and YouTube video frames.

    Returns structured results with text_results and video_results (including YouTube timestamps).
    """
    try:
        results = search_qdrant(payload.query.strip(), top_k=payload.top_k)
        return results
    except Exception as e:
        logger.error("Qdrant search error", exc_info=True)
        raise HTTPException(status_code=500, detail="Qdrant search failed")


# ========================================
# Phase 3: WebSocket Live API Endpoint
# ========================================
@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str):
    if not _SAFE_ID.match(user_id) or not _SAFE_ID.match(session_id):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    logger.info(f"WebSocket session connected: {user_id}/{session_id}")

    # Configuration for Native Audio Model (optimized for Gemini Live)
    run_config = RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Kore"  # Change to "Puck", "Kore", or "Fenrir"
                )
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    # Ensure session exists
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )

    live_request_queue = LiveRequestQueue()

    async def upstream_task() -> None:
        """Handle incoming Browser -> Server messages."""
        while True:
            try:
                message = await websocket.receive()

                # Handle microphone audio (16-bit PCM, 16kHz)
                if "bytes" in message:
                    audio_blob = types.Blob(
                        mime_type="audio/pcm;rate=16000",
                        data=message["bytes"]
                    )
                    live_request_queue.send_realtime(audio_blob)

                # Handle text messages or image frames
                elif "text" in message:
                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError as e:
                        logger.warning("Skipping unparseable text message: %s", e)
                        continue
                    if data.get("type") == "text":
                        content = types.Content(
                            parts=[types.Part(text=data["text"])]
                        )
                        live_request_queue.send_content(content)
                    elif data.get("type") == "image":
                        mime_type = data.get("mimeType", "image/jpeg")
                        if mime_type not in _ALLOWED_IMAGE_MIME_TYPES:
                            logger.warning("Rejected unsupported mimeType: %r", mime_type)
                            continue
                        image_data = base64.b64decode(data["data"])
                        image_blob = types.Blob(mime_type=mime_type, data=image_data)
                        live_request_queue.send_realtime(image_blob)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error("Upstream Error: %s", e, exc_info=True)
                break

    async def downstream_task() -> None:
        """Handle outgoing Server -> Browser events (Audio, Transcripts, Tools)."""
        try:
            async for event in runner.run_live(
                user_id=user_id,
                session_id=session_id,
                live_request_queue=live_request_queue,
                run_config=run_config,
            ):
                # Send raw ADK event as JSON string
                await websocket.send_text(event.model_dump_json(exclude_none=True, by_alias=True))
        except Exception as e:
            logger.error("Downstream Error: %s", e, exc_info=True)
        finally:
            # Close the WebSocket so the frontend reconnects if the live session dies
            try:
                await websocket.close()
            except Exception:
                pass

    # Run tasks concurrently; TaskGroup cancels the sibling when one exits.
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(upstream_task())
            tg.create_task(downstream_task())
    finally:
        logger.info("Closing Live API stream and queue")
        live_request_queue.close()

# ========================================
# Phase 4: Start the Server
# ========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
