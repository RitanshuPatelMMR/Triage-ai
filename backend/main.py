from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os
import json

load_dotenv()

app = FastAPI(title="TriageAI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LLM instance — shared across all routes ───────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


# ── Request model ─────────────────────────────────────────────────────────
class NoteInput(BaseModel):
    text: str


# ── Health check ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Phase 1: Clean endpoint ───────────────────────────────────────────────
@app.post("/analyze/clean")
async def clean_note(note: NoteInput):
    """Node 1 test — expand abbreviations and return clean text"""
    from agent.prompts import CLEAN_SYSTEM_PROMPT

    messages = [
        SystemMessage(content=CLEAN_SYSTEM_PROMPT),
        HumanMessage(content=note.text)
    ]

    result = await llm.ainvoke(messages)
    return {"cleaned_text": result.content}


# ── Phase 1: Extract endpoint ─────────────────────────────────────────────
@app.post("/analyze/extract")
async def extract_entities(note: NoteInput):
    """Node 2 test — extract structured JSON entities"""
    from agent.prompts import EXTRACT_SYSTEM_PROMPT

    messages = [
        SystemMessage(content=EXTRACT_SYSTEM_PROMPT),
        HumanMessage(content=note.text)
    ]

    result = await llm.ainvoke(messages)

    # Parse JSON safely
    try:
        entities = json.loads(result.content)
    except json.JSONDecodeError:
        # Sometimes LLM adds markdown — strip it
        clean = result.content.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        entities = json.loads(clean.strip())

    return {"entities": entities}


# ── Run instructions ──────────────────────────────────────────────────────
# cd backend && uvicorn main:app --reload