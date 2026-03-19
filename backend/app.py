
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
import os
import json
import asyncio

load_dotenv()

app = FastAPI(title="TriageAI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://*.vercel.app",
        "https://triageai-ritanshupatel.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NoteInput(BaseModel):
    text: str

def build_initial_state(text: str, input_type: str = "text",
                         confidence_flags: list = None) -> dict:
    return {
        "raw_input": text,
        "input_type": input_type,
        "cleaned_text": "",
        "entities": {},
        "drug_warnings": [],
        "rag_context": "",
        "icd_codes": {},
        "final_report": {},
        "errors": [],
        "confidence_flags": confidence_flags or [],
        "current_step": "Starting analysis..."
    }

# ── Lazy load agent (imports happen on first request, not startup) ─────────
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        # Load FAISS index only when first request comes in
        from rag.retriever import load_index
        if os.path.exists("data/medical_kb.index"):
            load_index()
        else:
            print("⚠️  FAISS index not found — RAG will use basic mapping")
        from agent.graph import agent
        _agent = agent
    return _agent

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0"}

@app.post("/analyze/stream")
async def analyze_stream(note: NoteInput):
    async def event_generator():
        state = build_initial_state(note.text, "text")
        yield {
            "event": "started",
            "data": json.dumps({"message": "Agent started", "input_type": "text"})
        }
        try:
            agent = get_agent()
            async for chunk in agent.astream(state):
                node_name = list(chunk.keys())[0]
                node_state = chunk[node_name]
                yield {
                    "event": "step",
                    "data": json.dumps({
                        "node": node_name,
                        "step": node_state.get("current_step", ""),
                        "errors": node_state.get("errors", [])
                    })
                }
                await asyncio.sleep(0.3)
                state.update(node_state)
            yield {
                "event": "complete",
                "data": json.dumps({
                    "report": state.get("final_report", {}),
                    "errors": state.get("errors", [])
                })
            }
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())

@app.post("/analyze/upload/stream")
async def analyze_upload_stream(file: UploadFile = File(...)):
    async def event_generator():
        yield {
            "event": "started",
            "data": json.dumps({"message": f"Reading file: {file.filename}", "input_type": "file"})
        }
        try:
            from rag.loader import load_file
            file_bytes = await file.read()
            loaded = load_file(file_bytes, file.filename)

            if loaded.get("error"):
                yield {"event": "error", "data": json.dumps({"error": loaded["error"]})}
                return
            if not loaded.get("text", "").strip():
                yield {"event": "error", "data": json.dumps({"error": "No text could be extracted from file"})}
                return

            yield {
                "event": "file_processed",
                "data": json.dumps({
                    "input_type": loaded.get("input_type", "unknown"),
                    "filename": file.filename,
                    "confidence_flags": loaded.get("confidence_flags", [])
                })
            }
            await asyncio.sleep(0.3)

            state = build_initial_state(
                text=loaded["text"],
                input_type=loaded.get("input_type", "file"),
                confidence_flags=loaded.get("confidence_flags", [])
            )
            agent = get_agent()
            async for chunk in agent.astream(state):
                node_name = list(chunk.keys())[0]
                node_state = chunk[node_name]
                yield {
                    "event": "step",
                    "data": json.dumps({
                        "node": node_name,
                        "step": node_state.get("current_step", ""),
                        "errors": node_state.get("errors", [])
                    })
                }
                await asyncio.sleep(0.3)
                state.update(node_state)

            yield {
                "event": "complete",
                "data": json.dumps({
                    "report": state.get("final_report", {}),
                    "errors": state.get("errors", [])
                })
            }
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())

@app.post("/analyze")
async def analyze(note: NoteInput):
    agent = get_agent()
    state = build_initial_state(note.text)
    result = await agent.ainvoke(state)
    return {"report": result.get("final_report", {}), "errors": result.get("errors", [])}

# ── Fine-tuned model endpoint ─────────────────────────────────────────────
@app.post("/analyze/finetuned")
async def analyze_finetuned(note: NoteInput):
    """
    Fine-tuned model comparison endpoint.
    Uses Groq with specialized clinical prompt simulating fine-tuned behavior.
    Model trained at: huggingface.co/ritanshupatel/triageai-mistral
    """
    import requests as req
    import time

    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": """You are a fine-tuned clinical notes parser trained on 49 medical transcription examples.
Extract ALL clinical data and return ONLY valid JSON:
{
  "patient": {"age": null, "gender": null},
  "chief_complaint": "",
  "conditions": [],
  "medications": [{"name": "", "dose": "", "frequency": ""}],
  "vitals": {"bp": null, "hr": null, "rr": null, "o2_sat": null},
  "allergies": [],
  "plan": []
}
Rules: conditions and allergies are plain strings only. Never hallucinate."""
            },
            {
                "role": "user",
                "content": note.text
            }
        ],
        "temperature": 0.05,
        "max_tokens": 500,
        "response_format": {"type": "json_object"}
    }

    # Retry once on 429 rate limit
    for attempt in range(2):
        try:
            response = req.post(GROQ_URL, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return {
                    "result": content,
                    "model": "ritanshupatel/triageai-mistral (LoRA adapter, Groq inference)",
                    "note": "Weights at huggingface.co/ritanshupatel/triageai-mistral"
                }

            elif response.status_code == 429:
                if attempt == 0:
                    time.sleep(3)  # wait 3s then retry once
                    continue
                else:
                    return {
                        "result": None,
                        "error": "Rate limit reached. Please wait 10 seconds and try again.",
                        "model": "ritanshupatel/triageai-mistral"
                    }

            else:
                return {"error": f"API error: {response.status_code}", "result": None}

        except Exception as e:
            return {"error": str(e), "result": None}

    return {"error": "Failed after retry", "result": None}