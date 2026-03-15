from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
import os
import json
import asyncio

load_dotenv()

# ── Load FAISS index at startup ───────────────────────────────────────────
from rag.retriever import load_index
load_index()

app = FastAPI(title="TriageAI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request model ─────────────────────────────────────────────────────────
class NoteInput(BaseModel):
    text: str


# ── Health check ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Main streaming endpoint ───────────────────────────────────────────────
@app.post("/analyze/stream")
async def analyze_stream(note: NoteInput):
    """
    Full LangGraph agent with SSE streaming.
    Frontend receives live step updates as each node completes.
    """
    from agent.graph import agent

    async def event_generator():
        # ── Initial state ─────────────────────────────
        state = {
            "raw_input": note.text,
            "input_type": "text",
            "cleaned_text": "",
            "entities": {},
            "drug_warnings": [],
            "rag_context": "",
            "icd_codes": {},
            "final_report": {},
            "errors": [],
            "confidence_flags": [],
            "current_step": "Starting analysis..."
        }

        # ── Send started event ────────────────────────
        yield {
            "event": "started",
            "data": json.dumps({"message": "Agent started"})
        }

        try:
            # ── Stream each node update ───────────────
            async for chunk in agent.astream(state):
                # chunk is a dict like {"node_name": updated_state}
                node_name = list(chunk.keys())[0]
                node_state = chunk[node_name]

                # Send step update to frontend
                yield {
                    "event": "step",
                    "data": json.dumps({
                        "node": node_name,
                        "step": node_state.get("current_step", ""),
                        "errors": node_state.get("errors", [])
                    })
                }

                # Small delay so frontend can render each step visibly
                await asyncio.sleep(0.3)

                # Track latest state
                state.update(node_state)

            # ── Send final report ─────────────────────
            yield {
                "event": "complete",
                "data": json.dumps({
                    "report": state.get("final_report", {}),
                    "errors": state.get("errors", [])
                })
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_generator())


# ── Non-streaming endpoint (useful for testing) ───────────────────────────
@app.post("/analyze")
async def analyze(note: NoteInput):
    """Full pipeline without streaming — returns complete report at once"""
    from agent.graph import agent

    state = {
        "raw_input": note.text,
        "input_type": "text",
        "cleaned_text": "",
        "entities": {},
        "drug_warnings": [],
        "rag_context": "",
        "icd_codes": {},
        "final_report": {},
        "errors": [],
        "confidence_flags": [],
        "current_step": ""
    }

    result = await agent.ainvoke(state)
    return {
        "report": result.get("final_report", {}),
        "errors": result.get("errors", [])
    }


# ── Run: uvicorn main:app --reload ────────────────────────────────────────