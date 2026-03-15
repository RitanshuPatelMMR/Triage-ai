import os
import json
import faiss
import numpy as np
from rag.embedder import embed_text

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INDEX_PATH = os.path.join(DATA_DIR, "medical_kb.index")
TEXTS_PATH = os.path.join(DATA_DIR, "medical_kb_texts.json")

_index = None
_texts = []
_metadata = []

# ── Hardcoded primary ICD-10 codes for common conditions ─────────────────
# These are the standard primary codes used in clinical practice
COMMON_ICD_CODES = {
    "hypertension": "I10",
    "essential hypertension": "I10",
    "high blood pressure": "I10",
    "type 2 diabetes": "E11.9",
    "type 2 diabetes mellitus": "E11.9",
    "diabetes mellitus type 2": "E11.9",
    "diabetes type 2": "E11.9",
    "diabetes": "E11.9",
    "type 1 diabetes": "E10.9",
    "type 1 diabetes mellitus": "E10.9",
    "chest pain": "R07.9",
    "chest pain unspecified": "R07.9",
    "shortness of breath": "R06.00",
    "dyspnea": "R06.00",
    "sob": "R06.00",
    "asthma": "J45.909",
    "asthma unspecified": "J45.909",
    "copd": "J44.9",
    "chronic obstructive pulmonary disease": "J44.9",
    "pneumonia": "J18.9",
    "community acquired pneumonia": "J18.9",
    "heart failure": "I50.9",
    "congestive heart failure": "I50.9",
    "chf": "I50.9",
    "atrial fibrillation": "I48.91",
    "afib": "I48.91",
    "depression": "F32.9",
    "major depressive disorder": "F32.9",
    "anxiety": "F41.9",
    "generalized anxiety disorder": "F41.1",
    "hypothyroidism": "E03.9",
    "hyperthyroidism": "E05.90",
    "stroke": "I63.9",
    "ischemic stroke": "I63.9",
    "tia": "G45.9",
    "transient ischemic attack": "G45.9",
    "myocardial infarction": "I21.9",
    "heart attack": "I21.9",
    "mi": "I21.9",
    "urinary tract infection": "N39.0",
    "uti": "N39.0",
    "kidney disease": "N18.9",
    "chronic kidney disease": "N18.9",
    "ckd": "N18.9",
    "obesity": "E66.9",
    "hyperlipidemia": "E78.5",
    "hypercholesterolemia": "E78.0",
    "high cholesterol": "E78.0",
    "gerd": "K21.0",
    "acid reflux": "K21.0",
    "osteoarthritis": "M19.90",
    "rheumatoid arthritis": "M06.9",
    "back pain": "M54.9",
    "lower back pain": "M54.5",
    "migraine": "G43.909",
    "headache": "R51.9",
    "epilepsy": "G40.909",
    "seizure": "G40.909",
    "parkinson": "G20",
    "dementia": "F03.90",
    "alzheimer": "G30.9",
    "anemia": "D64.9",
    "iron deficiency anemia": "D50.9",
    "sepsis": "A41.9",
    "cellulitis": "L03.90",
    "appendicitis": "K37",
    "gallstones": "K80.20",
    "pancreatitis": "K85.90",
    "hepatitis": "K75.9",
    "cirrhosis": "K74.60",
    "influenza": "J11.1",
    "covid": "U07.1",
    "covid-19": "U07.1",
}


def load_index():
    global _index, _texts, _metadata
    if not os.path.exists(INDEX_PATH):
        print("medical_kb.index not found")
        return False
    if not os.path.exists(TEXTS_PATH):
        print("medical_kb_texts.json not found")
        return False
    print("Loading FAISS index...")
    _index = faiss.read_index(INDEX_PATH)
    with open(TEXTS_PATH, "r") as f:
        data = json.load(f)
        _texts = data["texts"]
        _metadata = data["metadata"]
    print(f"FAISS index loaded: {_index.ntotal} vectors")
    return True


def search(query: str, top_k: int = 5) -> list:
    if _index is None:
        return []
    query_vector = np.array([embed_text(query)], dtype="float32")
    scores, indices = _index.search(query_vector, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        result = {
            "text": _texts[idx],
            "score": float(score),
            "source": _metadata[idx].get("source", "unknown"),
            "type": _metadata[idx].get("type", "unknown"),
        }
        if _metadata[idx].get("source") == "icd10":
            result["icd_code"] = _metadata[idx].get("code", "")
            result["icd_description"] = _metadata[idx].get("description", "")
        results.append(result)
    return results


def search_icd_only(query: str, top_k: int = 500) -> list:
    if _index is None:
        return []
    query_vector = np.array([embed_text(query)], dtype="float32")
    scores, indices = _index.search(query_vector, top_k)
    icd_results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if _metadata[idx].get("source") == "icd10":
            icd_results.append({
                "text": _texts[idx],
                "score": float(score),
                "icd_code": _metadata[idx].get("code", ""),
                "icd_description": _metadata[idx].get("description", ""),
                "source": "icd10"
            })
    icd_results.sort(key=lambda x: x["score"], reverse=True)
    return icd_results


def search_icd_codes(conditions: list) -> dict:
    """
    For each condition find the best ICD-10 code.
    Strategy:
    1. Check hardcoded common conditions lookup first (fastest, most accurate)
    2. Fall back to FAISS semantic search for uncommon conditions
    """
    icd_codes = {}

    for condition in conditions:
        condition_lower = condition.lower().strip()

        # Strategy 1: exact match in common codes
        if condition_lower in COMMON_ICD_CODES:
            icd_codes[condition] = COMMON_ICD_CODES[condition_lower]
            continue

        # Strategy 2: partial match in common codes
        matched = False
        for key, code in COMMON_ICD_CODES.items():
            if key in condition_lower or condition_lower in key:
                icd_codes[condition] = code
                matched = True
                break

        if matched:
            continue

        # Strategy 3: FAISS semantic search fallback
        results = search_icd_only(condition)
        if results:
            icd_codes[condition] = results[0].get("icd_code", "Unknown")
        else:
            icd_codes[condition] = "Not found"

    return icd_codes


def is_loaded() -> bool:
    return _index is not None
