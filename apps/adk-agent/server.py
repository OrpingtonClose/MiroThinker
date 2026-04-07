# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""AG-UI FastAPI server for MiroThinker.

Wraps the MiroThinker research_agent with the AG-UI protocol middleware,
enabling rich frontends (CopilotKit, custom React, etc.) to interact with
the agent via Server-Sent Events streaming.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

The AG-UI endpoint is mounted at POST /  (root).
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint

from agents.research import research_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── FastAPI app ─────────────────────────────────────────────────────

app = FastAPI(
    title="MiroThinker AG-UI",
    description="AG-UI protocol endpoint for MiroThinker deep-research agent",
    version="0.1.0",
)

# CORS — allow all origins so any frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── AG-UI agent wrapper ────────────────────────────────────────────

adk_agent = ADKAgent(
    adk_agent=research_agent,
    app_name="mirothinker_adk",
)

# Mount the AG-UI SSE endpoint at root
add_adk_fastapi_endpoint(app, adk_agent, path="/")

logger.info("MiroThinker AG-UI server ready at http://0.0.0.0:8000")
