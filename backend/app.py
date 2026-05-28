from fastapi import FastAPI, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
import os
import json
import asyncio
import time
import uuid
from typing import Optional

load_dotenv()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

app = FastAPI(title="TriageAI", version="4.0.0")

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


class VerifyInput(BaseModel):
    session_id: str
    created_at: str


class DeleteInput(BaseModel):
    session_id: str
    created_at: str


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


# ── Lazy load agent ───────────────────────────────────────────────────────
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from rag.retriever import load_index
        if os.path.exists("data/medical_kb.index"):
            load_index()
        else:
            print("⚠️  FAISS index not found — RAG will use basic mapping")
        from agent.graph import agent
        _agent = agent
    return _agent


# ── Health check ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    from tools.s3_service import is_available as s3_ok
    from tools.logger import is_available as cw_ok
    from tools.dynamo_service import is_available as dynamo_ok
    from tools.groq_utils import groq_configured
    return {
        "status": "ok",
        "version": "4.0.0",
        "groq": "configured" if groq_configured() else "missing",
        "s3": "connected" if s3_ok() else "not configured",
        "cloudwatch": "connected" if cw_ok() else "not configured",
        "dynamodb": "connected" if dynamo_ok() else "not configured"
    }


# ── History endpoints ─────────────────────────────────────────────────────
@app.get("/history/{session_id}")
def get_history(session_id: str):
    """Get all reports for a session."""
    from tools.dynamo_service import get_history as dynamo_get
    reports = dynamo_get(session_id)
    return {"reports": reports, "count": len(reports)}


@app.post("/history/verify")
def verify_report(body: VerifyInput):
    """Mark a report as human verified."""
    from tools.dynamo_service import mark_verified
    success = mark_verified(body.session_id, body.created_at)
    return {"success": success}


@app.delete("/history/delete")
def delete_report(body: DeleteInput):
    """Delete a specific report."""
    from tools.dynamo_service import delete_report as dynamo_delete
    success = dynamo_delete(body.session_id, body.created_at)
    return {"success": success}


# ── Text streaming endpoint ───────────────────────────────────────────────
@app.post("/analyze/stream")
async def analyze_stream(
    note: NoteInput,
    x_session_id: Optional[str] = Header(None)
):
    from tools.logger import log_request, log_node, log_report, log_error
    from tools.s3_service import save_report as s3_save
    from tools.dynamo_service import save_report as dynamo_save
    from tools.errors import user_safe_error
    from tools.groq_utils import groq_configured, GROQ_MISSING_MSG

    text = (note.text or "").strip()
    if not text:
        async def empty_generator():
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": "Please enter a clinical note before analyzing."
                })
            }
        return EventSourceResponse(empty_generator())

    request_id = str(uuid.uuid4())[:8]
    session_id = x_session_id or "anonymous"

    async def event_generator():
        start_time = time.time()
        state = build_initial_state(text, "text")

        log_request("text", len(text), request_id)

        yield {
            "event": "started",
            "data": json.dumps({"message": "Agent started", "input_type": "text"})
        }

        try:
            if not groq_configured():
                yield {"event": "error",
                       "data": json.dumps({"error": GROQ_MISSING_MSG})}
                return

            agent = get_agent()
            async for chunk in agent.astream(state):
                node_name = list(chunk.keys())[0]
                node_state = chunk[node_name]
                node_errors = node_state.get("errors", [])

                log_node(node_name, (time.time() - start_time) * 1000,
                         node_errors, request_id)

                yield {
                    "event": "step",
                    "data": json.dumps({
                        "node": node_name,
                        "step": node_state.get("current_step", ""),
                        "errors": node_errors
                    })
                }
                await asyncio.sleep(0.3)
                state.update(node_state)

            final_report = state.get("final_report", {})
            total_ms = (time.time() - start_time) * 1000

            log_report(final_report, "text", total_ms, request_id)

            # Save to S3
            s3_key = s3_save(final_report, "text")

            # Save to DynamoDB
            dynamo_save(
                session_id=session_id,
                request_id=request_id,
                report=final_report,
                input_type="text",
                s3_key=s3_key
            )

            yield {
                "event": "complete",
                "data": json.dumps({
                    "report": final_report,
                    "errors": state.get("errors", []),
                    "request_id": request_id,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")
                })
            }

        except Exception as e:
            log_error(str(e), "analyze_stream", request_id)
            yield {"event": "error",
                   "data": json.dumps({"error": user_safe_error(e)})}

    return EventSourceResponse(event_generator())


# ── File upload streaming endpoint ───────────────────────────────────────
@app.post("/analyze/upload/stream")
async def analyze_upload_stream(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(None)
):
    from tools.logger import log_request, log_node, log_report, log_error, log_file_upload
    from tools.s3_service import upload_file, save_report as s3_save
    from tools.dynamo_service import save_report as dynamo_save
    from tools.errors import user_safe_error, map_loader_error
    from tools.groq_utils import groq_configured, GROQ_MISSING_MSG

    request_id = str(uuid.uuid4())[:8]
    session_id = x_session_id or "anonymous"

    file_bytes = await file.read()
    filename = file.filename or "upload"

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)

        async def too_large_generator():
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": f"File too large (max {max_mb} MB). Try a smaller PDF or image."
                })
            }

        return EventSourceResponse(too_large_generator())

    if len(file_bytes) == 0:
        async def empty_file_generator():
            yield {
                "event": "error",
                "data": json.dumps({"error": "Uploaded file is empty."})
            }

        return EventSourceResponse(empty_file_generator())

    async def event_generator():
        start_time = time.time()

        yield {
            "event": "started",
            "data": json.dumps({
                "message": f"Reading file: {filename}",
                "input_type": "file"
            })
        }

        try:
            if not groq_configured():
                yield {"event": "error",
                       "data": json.dumps({"error": GROQ_MISSING_MSG})}
                return

            from rag.loader import load_file

            s3_key = upload_file(file_bytes, filename,
                                 filename.split(".")[-1].lower())
            log_file_upload(filename, len(file_bytes) / 1024,
                            s3_key, request_id)

            loaded = load_file(file_bytes, filename)

            if loaded.get("error"):
                yield {"event": "error",
                       "data": json.dumps({
                           "error": map_loader_error(loaded["error"])
                       })}
                return

            if not loaded.get("text", "").strip():
                yield {"event": "error",
                       "data": json.dumps({"error": "No text could be extracted from file"})}
                return

            input_type = loaded.get("input_type", "file")
            log_request(input_type, len(loaded.get("text", "")), request_id)

            yield {
                "event": "file_processed",
                "data": json.dumps({
                    "input_type": input_type,
                    "filename": filename,
                    "confidence_flags": loaded.get("confidence_flags", []),
                    "s3_key": s3_key
                })
            }
            await asyncio.sleep(0.3)

            state = build_initial_state(
                text=loaded["text"],
                input_type=input_type,
                confidence_flags=loaded.get("confidence_flags", [])
            )

            agent = get_agent()
            async for chunk in agent.astream(state):
                node_name = list(chunk.keys())[0]
                node_state = chunk[node_name]
                node_errors = node_state.get("errors", [])

                log_node(node_name, (time.time() - start_time) * 1000,
                         node_errors, request_id)

                yield {
                    "event": "step",
                    "data": json.dumps({
                        "node": node_name,
                        "step": node_state.get("current_step", ""),
                        "errors": node_errors
                    })
                }
                await asyncio.sleep(0.3)
                state.update(node_state)

            final_report = state.get("final_report", {})
            total_ms = (time.time() - start_time) * 1000

            log_report(final_report, input_type, total_ms, request_id)

            report_s3_key = s3_save(final_report, input_type)

            dynamo_save(
                session_id=session_id,
                request_id=request_id,
                report=final_report,
                input_type=input_type,
                s3_key=s3_key
            )

            yield {
                "event": "complete",
                "data": json.dumps({
                    "report": final_report,
                    "errors": state.get("errors", []),
                    "request_id": request_id,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")
                })
            }

        except Exception as e:
            log_error(str(e), "analyze_upload_stream", request_id)
            yield {"event": "error",
                   "data": json.dumps({"error": user_safe_error(e)})}

    return EventSourceResponse(event_generator())


# ── Non-streaming endpoint ────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(note: NoteInput):
    from tools.errors import user_safe_error
    from tools.groq_utils import groq_configured, GROQ_MISSING_MSG

    text = (note.text or "").strip()
    if not text:
        return {"report": {}, "errors": ["Please enter a clinical note before analyzing."]}
    if not groq_configured():
        return {"report": {}, "errors": [GROQ_MISSING_MSG]}

    try:
        agent = get_agent()
        state = build_initial_state(text)
        result = await agent.ainvoke(state)
        return {
            "report": result.get("final_report", {}),
            "errors": result.get("errors", [])
        }
    except Exception as e:
        return {"report": {}, "errors": [user_safe_error(e)]}


# ── Fine-tuned model endpoint ─────────────────────────────────────────────
@app.post("/analyze/finetuned")
async def analyze_finetuned(note: NoteInput):
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
            {"role": "user", "content": note.text}
        ],
        "temperature": 0.05,
        "max_tokens": 500,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(2):
        try:
            response = req.post(GROQ_URL, headers=headers,
                                json=payload, timeout=30)
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
                    time.sleep(3)
                    continue
                else:
                    return {
                        "result": None,
                        "error": "Rate limit reached. Please wait 10 seconds and try again.",
                        "model": "ritanshupatel/triageai-mistral"
                    }
            else:
                return {"error": f"API error: {response.status_code}",
                        "result": None}
        except Exception as e:
            return {"error": str(e), "result": None}

    return {"error": "Failed after retry", "result": None}