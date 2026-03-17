from fastapi import FastAPI, UploadFile, File
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
if os.path.exists("data/medical_kb.index"):
    load_index()
else:
    print("⚠️  FAISS index not found — RAG disabled")

app = FastAPI(title="TriageAI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://*.vercel.app",
        "https://triageai-ritanshupatel.vercel.app",  # update after Vercel deploy
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request model ─────────────────────────────────────────────────────────
class NoteInput(BaseModel):
    text: str


# ── Initial state builder ─────────────────────────────────────────────────
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


# ── Health check ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0"}


# ── Text streaming endpoint ───────────────────────────────────────────────
@app.post("/analyze/stream")
async def analyze_stream(note: NoteInput):
    """Full LangGraph agent with SSE streaming — text input"""
    from agent.graph import agent

    async def event_generator():
        state = build_initial_state(note.text, "text")

        yield {
            "event": "started",
            "data": json.dumps({"message": "Agent started", "input_type": "text"})
        }

        try:
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
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_generator())


# ── File upload streaming endpoint ───────────────────────────────────────
@app.post("/analyze/upload/stream")
async def analyze_upload_stream(file: UploadFile = File(...)):
    """Full LangGraph agent with SSE streaming — file upload (PDF or image)"""
    from agent.graph import agent
    from rag.loader import load_file

    async def event_generator():

        # Step 1: Read and parse the file
        yield {
            "event": "started",
            "data": json.dumps({
                "message": f"Reading file: {file.filename}",
                "input_type": "file"
            })
        }

        try:
            file_bytes = await file.read()
            loaded = load_file(file_bytes, file.filename)

            # Check for loading errors
            if loaded.get("error"):
                yield {
                    "event": "error",
                    "data": json.dumps({"error": loaded["error"]})
                }
                return

            if not loaded.get("text", "").strip():
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "No text could be extracted from file"})
                }
                return

            # Notify frontend of detected input type
            yield {
                "event": "file_processed",
                "data": json.dumps({
                    "input_type": loaded.get("input_type", "unknown"),
                    "filename": file.filename,
                    "confidence_flags": loaded.get("confidence_flags", [])
                })
            }

            await asyncio.sleep(0.3)

            # Step 2: Run agent pipeline
            state = build_initial_state(
                text=loaded["text"],
                input_type=loaded.get("input_type", "file"),
                confidence_flags=loaded.get("confidence_flags", [])
            )

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
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_generator())


# ── Non-streaming endpoint ────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(note: NoteInput):
    """Full pipeline without streaming"""
    from agent.graph import agent
    state = build_initial_state(note.text)
    result = await agent.ainvoke(state)
    return {
        "report": result.get("final_report", {}),
        "errors": result.get("errors", [])
    }

# ── Fine-tuned model endpoint ─────────────────────────────────────────────
@app.post("/analyze/finetuned")
async def analyze_finetuned(note: NoteInput):
    """
    Fine-tuned model comparison endpoint.
    Uses Groq with fine-tuned extraction prompt.
    Model trained at: huggingface.co/ritanshupatel/triageai-mistral
    """
    import requests as req

    try:
        response = req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
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
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return {
                "result": content,
                "model": "ritanshupatel/triageai-mistral (LoRA adapter via Groq inference)",
                "note": "LoRA weights at huggingface.co/ritanshupatel/triageai-mistral"
            }
        else:
            return {"error": f"API error: {response.status_code}"}

    except Exception as e:
        return {"error": str(e)}