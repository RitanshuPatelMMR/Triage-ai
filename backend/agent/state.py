from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):

    # ── Input ──────────────────────────────────────────
    raw_input: str          # Original text from user (already OCR'd if image)
    input_type: str         # 'text' | 'pdf' | 'image'

    # ── Node 1 output ──────────────────────────────────
    cleaned_text: str       # Abbreviations expanded, fully readable

    # ── Node 2 output ──────────────────────────────────
    entities: Dict[str, Any]  # Structured JSON — age, gender, meds, diagnoses...

    # ── Node 3 output ──────────────────────────────────
    drug_warnings: List[Dict]  # [{drug_pair, severity, description}]

    # ── Node 4 output ──────────────────────────────────
    rag_context: str           # Retrieved chunks from FAISS
    icd_codes: Dict[str, str]  # {diagnosis_name: ICD10_code}

    # ── Node 5 output ──────────────────────────────────
    final_report: Dict[str, Any]  # Complete structured report

    # ── Meta ───────────────────────────────────────────
    errors: List[str]            # Errors from any node
    confidence_flags: List[str]  # Fields needing human review
    current_step: str            # Which node is running (for SSE streaming)