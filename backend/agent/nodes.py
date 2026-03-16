import os
import json
import requests
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AgentState
from agent.prompts import CLEAN_SYSTEM_PROMPT, EXTRACT_SYSTEM_PROMPT, SUMMARY_SYSTEM_PROMPT

# ── Shared LLM instance ───────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1  
)


# ── Helper: safe JSON parse ───────────────────────────────────────────────
def safe_json_parse(text: str) -> dict:
    """Strip markdown fences and parse JSON safely"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        clean = text.strip()
        # Strip ```json ... ``` or ``` ... ```
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except:
                    continue
        raise ValueError(f"Could not parse JSON from: {text[:100]}")


# ── NODE 1: Parse and Clean ───────────────────────────────────────────────
async def parse_and_clean(state: AgentState) -> AgentState:
    """Expand all medical abbreviations into full readable text"""
    state["current_step"] = "📋 Parsing and expanding medical abbreviations..."

    try:
        messages = [
            SystemMessage(content=CLEAN_SYSTEM_PROMPT),
            HumanMessage(content=state["raw_input"])
        ]
        result = await llm.ainvoke(messages)
        state["cleaned_text"] = result.content.strip()

    except Exception as e:
        state["errors"].append(f"parse_and_clean: {str(e)}")
        # Fallback — use raw input as-is
        state["cleaned_text"] = state["raw_input"]

    return state


# ── NODE 2: Extract Entities ──────────────────────────────────────────────
async def extract_entities(state: AgentState) -> AgentState:
    """Extract structured clinical entities — schema enforced"""
    state["current_step"] = "🔍 Extracting clinical entities..."

    try:
        from groq import Groq
        import os

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": """You are a clinical data extractor.
Extract ALL medical information from the note.
For allergies always return plain strings like ["PCN", "sulfa"] never objects.
For medications always return objects with name, dose, frequency as strings.
For conditions always return plain strings like ["Hypertension", "Type 2 Diabetes"].
If information is not present use null or empty array. Never hallucinate."""
                },
                {
                    "role": "user",
                    "content": f"Extract clinical data from this note:\n\n{state['cleaned_text']}"
                }
            ],
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        parsed = json.loads(raw)

        # ── Normalize everything regardless of what LLM returned ──────────
        entities = _normalize_entities(parsed)
        state["entities"] = entities

    except Exception as e:
        state["errors"].append(f"extract_entities: {str(e)}")
        state["entities"] = _empty_entities()

    return state


def _normalize_entities(raw: dict) -> dict:
    """
    Force every field into the exact shape the frontend expects.
    Handles any LLM output variation — no more crashes.
    """

    # ── Normalize conditions → always list of strings ─────────────────────
    raw_conditions = raw.get("conditions", []) or []
    conditions = []
    for c in raw_conditions:
        if isinstance(c, str) and c.strip():
            conditions.append(c.strip())
        elif isinstance(c, dict):
            name = c.get("name") or c.get("condition") or c.get("diagnosis") or ""
            if name:
                conditions.append(str(name).strip())

    # ── Normalize allergies → always list of strings ──────────────────────
    raw_allergies = raw.get("allergies", []) or []
    allergies = []
    for a in raw_allergies:
        if isinstance(a, str) and a.strip():
            allergies.append(a.strip())
        elif isinstance(a, dict):
            val = next(iter(a.values()), "")
            if val:
                allergies.append(str(val).strip())

    # ── Normalize medications → always list of {name, dose, frequency} ────
    raw_meds = raw.get("medications", []) or []
    medications = []
    for m in raw_meds:
        if isinstance(m, str) and m.strip():
            medications.append({"name": m.strip(), "dose": "", "frequency": ""})
        elif isinstance(m, dict):
            name = m.get("name") or m.get("medication") or m.get("drug") or ""
            if name:
                medications.append({
                    "name": str(name).strip(),
                    "dose": str(m.get("dose") or m.get("dosage") or "").strip(),
                    "frequency": str(m.get("frequency") or m.get("freq") or "").strip(),
                })

    # ── Normalize vitals → always flat dict of strings ────────────────────
    raw_vitals = raw.get("vitals", {}) or {}
    if isinstance(raw_vitals, dict):
        vitals = {
            "bp": str(raw_vitals.get("bp") or raw_vitals.get("blood_pressure") or "").strip() or None,
            "hr": str(raw_vitals.get("hr") or raw_vitals.get("heart_rate") or "").strip() or None,
            "rr": str(raw_vitals.get("rr") or raw_vitals.get("respiratory_rate") or "").strip() or None,
            "o2_sat": str(raw_vitals.get("o2_sat") or raw_vitals.get("oxygen_saturation") or "").strip() or None,
            "temp": str(raw_vitals.get("temp") or raw_vitals.get("temperature") or "").strip() or None,
            "weight": str(raw_vitals.get("weight") or "").strip() or None,
        }
    else:
        vitals = {}

    # ── Normalize patient ─────────────────────────────────────────────────
    raw_patient = raw.get("patient", {}) or {}
    age = raw_patient.get("age") or raw.get("age")
    gender = raw_patient.get("gender") or raw.get("gender")

    return {
        "patient": {
            "age": int(age) if age and str(age).isdigit() else None,
            "gender": str(gender).lower().strip() if gender else None,
        },
        "chief_complaint": str(raw.get("chief_complaint") or "").strip(),
        "conditions": conditions,
        "medications": medications,
        "vitals": vitals,
        "allergies": allergies,
        "plan": raw.get("plan") or [],
        "follow_up": str(raw.get("follow_up") or "").strip(),
    }


def _empty_entities() -> dict:
    """Safe empty fallback — never crashes frontend"""
    return {
        "patient": {"age": None, "gender": None},
        "chief_complaint": "",
        "conditions": [],
        "medications": [],
        "vitals": {},
        "allergies": [],
        "plan": [],
        "follow_up": "",
    }


# ── NODE 3: Check Drug Interactions ──────────────────────────────────────
async def check_drug_interactions(state: AgentState) -> AgentState:
    """Check each medication against OpenFDA for known interactions"""
    state["current_step"] = "💊 Checking drug interactions via FDA database..."

    warnings = []

    try:
        medications = state["entities"].get("medications", [])

        if not medications:
            state["drug_warnings"] = []
            return state

        # Check each medication against FDA
        for med in medications:
            med_name = med.get("name", "").strip()
            if not med_name:
                continue

            try:
                url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{med_name}&limit=1"
                resp = requests.get(url, timeout=5)

                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])

                    if results:
                        label = results[0]

                        # Extract warnings
                        warning_text = label.get("warnings", [""])[0] if label.get("warnings") else ""
                        interactions = label.get("drug_interactions", [""])[0] if label.get("drug_interactions") else ""

                        if warning_text or interactions:
                            warnings.append({
                                "drug": med_name,
                                "severity": "MODERATE",
                                "warning": warning_text[:300] if warning_text else "",
                                "interactions": interactions[:300] if interactions else "",
                                "source": "FDA Drug Label"
                            })

            except Exception as med_err:
                # One drug failing should never stop the pipeline
                state["errors"].append(f"FDA check for {med_name}: {str(med_err)}")
                continue

    except Exception as e:
        state["errors"].append(f"check_drug_interactions: {str(e)}")

    state["drug_warnings"] = warnings
    return state


async def rag_enrich(state: AgentState) -> AgentState:
    """Search FAISS knowledge base for ICD-10 codes and clinical context"""
    state["current_step"] = "🧠 Searching medical knowledge base..."

    try:
        from rag.retriever import search_icd_codes, search, is_loaded

        raw_conditions = state["entities"].get("conditions", [])

        # ── Normalize conditions to strings ───────────────────────────────
        # LLM sometimes returns list of dicts instead of list of strings
        conditions = []
        for c in raw_conditions:
            if isinstance(c, str):
                conditions.append(c.strip())
            elif isinstance(c, dict):
                # Extract name field if dict
                name = c.get("name") or c.get("condition") or c.get("diagnosis") or str(c)
                conditions.append(name.strip())

        if not conditions or not is_loaded():
            basic_icd = {
                "hypertension": "I10",
                "type 2 diabetes": "E11.9",
                "diabetes": "E11.9",
                "chest pain": "R07.9",
                "shortness of breath": "R06.00",
            }
            icd_codes = {}
            for condition in conditions:
                icd_codes[condition] = basic_icd.get(
                    condition.lower(), "Pending RAG"
                )
            state["icd_codes"] = icd_codes
            state["rag_context"] = "Used basic mapping (index not loaded)"
            return state

        # Real FAISS search for ICD codes
        icd_codes = search_icd_codes(conditions)
        state["icd_codes"] = icd_codes

        # Retrieve clinical context for summary
        chief_complaint = state["entities"].get("chief_complaint", "")
        if chief_complaint:
            context_results = search(chief_complaint, top_k=3)
            context_snippets = [r["text"][:200] for r in context_results
                                 if r.get("source") == "mtsamples"]
            state["rag_context"] = " | ".join(context_snippets[:2])
        else:
            state["rag_context"] = ""

    except Exception as e:
        state["errors"].append(f"rag_enrich: {str(e)}")
        state["icd_codes"] = {}
        state["rag_context"] = ""

    return state


# ── NODE 5: Generate Summary ──────────────────────────────────────────────
async def generate_summary(state: AgentState) -> AgentState:
    """Combine all node outputs into final structured report"""
    state["current_step"] = "📝 Generating final structured report..."

    try:
        context = f"""
PATIENT ENTITIES:
{json.dumps(state['entities'], indent=2)}

ICD-10 CODES:
{json.dumps(state['icd_codes'], indent=2)}

DRUG WARNINGS:
{json.dumps(state['drug_warnings'], indent=2)}

CLEANED NOTE:
{state['cleaned_text']}
"""

        messages = [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=context)
        ]

        result = await llm.ainvoke(messages)
        report = safe_json_parse(result.content)

        # Inject ICD codes and drug warnings
        report["icd_codes"] = state["icd_codes"]
        report["drug_warnings"] = state["drug_warnings"]
        report["errors"] = state["errors"]
        report["confidence_flags"] = state["confidence_flags"]

        # ── Layer 2: Normalize patient_card ──────────────────────────────
        if "patient_card" in report:
            pc = report["patient_card"]

            # Normalize conditions_with_codes
            raw_codes = pc.get("conditions_with_codes", []) or []
            clean_codes = []
            for c in raw_codes:
                if isinstance(c, dict):
                    cname = c.get("condition") or c.get("name") or ""
                    icode = c.get("icd_code") or c.get("code") or ""
                    if cname:
                        clean_codes.append({"condition": str(cname), "icd_code": str(icode)})
                elif isinstance(c, str) and c.strip():
                    clean_codes.append({"condition": c, "icd_code": ""})
            pc["conditions_with_codes"] = clean_codes

            # Normalize medications
            raw_meds = pc.get("medications", []) or []
            clean_meds = []
            for m in raw_meds:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("medication") or ""
                    if name:
                        clean_meds.append({
                            "name": str(name),
                            "dose": str(m.get("dose") or m.get("dosage") or ""),
                            "frequency": str(m.get("frequency") or m.get("freq") or ""),
                        })
                elif isinstance(m, str) and m.strip():
                    clean_meds.append({"name": m, "dose": "", "frequency": ""})
            pc["medications"] = clean_meds

            # Normalize allergies
            raw_allergies = pc.get("allergies", []) or []
            clean_allergies = []
            for a in raw_allergies:
                if isinstance(a, str) and a.strip():
                    clean_allergies.append(a)
                elif isinstance(a, dict):
                    val = next(iter(a.values()), "")
                    if val:
                        clean_allergies.append(str(val))
            pc["allergies"] = clean_allergies

            report["patient_card"] = pc
        # ── End Layer 2 ───────────────────────────────────────────────────

        state["final_report"] = report
        state["current_step"] = "✅ Report complete!"

    except Exception as e:
        state["errors"].append(f"generate_summary: {str(e)}")
        state["final_report"] = {
            "plain_english_summary": state["cleaned_text"],
            "patient_card": state["entities"],
            "icd_codes": state["icd_codes"],
            "drug_warnings": state["drug_warnings"],
            "errors": state["errors"],
            "confidence_flags": []
        }

    return state