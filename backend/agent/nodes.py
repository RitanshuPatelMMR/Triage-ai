import os
import re
import json
import asyncio
import requests
from langchain_groq import ChatGroq
from tools.groq_utils import groq_call_with_retry, is_rate_limit_error
from tools.errors import user_safe_error
from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AgentState
from agent.prompts import CLEAN_SYSTEM_PROMPT, EXTRACT_SYSTEM_PROMPT, SUMMARY_SYSTEM_PROMPT

# ── Shared LLM instance ───────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

# ── BUG #4 FIX: Drug abbreviation mapping ────────────────────────────────
DRUG_ALIASES = {
    "asa": "aspirin",
    "ntg": "nitroglycerin",
    "hep": "heparin",
    "lasix": "furosemide",
    "glucophage": "metformin",
    "lopressor": "metoprolol",
    "zocor": "simvastatin",
    "lipitor": "atorvastatin",
    "norvasc": "amlodipine",
    "prinivil": "lisinopril",
    "zestril": "lisinopril",
    "crestor": "rosuvastatin",
    "plavix": "clopidogrel",
    "coumadin": "warfarin",
    "lanoxin": "digoxin",
    "procardia": "nifedipine",
    "capoten": "captopril",
    "vasotec": "enalapril",
    "accupril": "quinapril",
    "altace": "ramipril",
    "diovan": "valsartan",
    "cozaar": "losartan",
    "hyzaar": "losartan",
    "toprol": "metoprolol",
    "coreg": "carvedilol",
    "tiazac": "diltiazem",
    "cardizem": "diltiazem",
    "bumex": "bumetanide",
    "aldactone": "spironolactone",
    "dyazide": "hydrochlorothiazide",
    "microzide": "hydrochlorothiazide",
    "hctz": "hydrochlorothiazide",
    "solu-medrol": "methylprednisolone",
    "decadron": "dexamethasone",
    "synthroid": "levothyroxine",
    "levoxyl": "levothyroxine",
    "prilosec": "omeprazole",
    "nexium": "esomeprazole",
    "prevacid": "lansoprazole",
    "pepcid": "famotidine",
    "zantac": "ranitidine",
    "benadryl": "diphenhydramine",
    "tylenol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "aleve": "naproxen",
    "oxycontin": "oxycodone",
    "percocet": "oxycodone",
    "vicodin": "hydrocodone",
    "morphine": "morphine sulfate",
    "ms contin": "morphine",
    "ativan": "lorazepam",
    "xanax": "alprazolam",
    "valium": "diazepam",
    "ambien": "zolpidem",
    "glucotrol": "glipizide",
    "amaryl": "glimepiride",
    "actos": "pioglitazone",
    "avandia": "rosiglitazone",
    "jardiance": "empagliflozin",
    "farxiga": "dapagliflozin",
    "ozempic": "semaglutide",
    "victoza": "liraglutide",
    "trulicity": "dulaglutide",
    "insulin": "insulin",
    "lantus": "insulin glargine",
    "humalog": "insulin lispro",
    "novolog": "insulin aspart",
}


# ── Helper: safe JSON parse ───────────────────────────────────────────────
def safe_json_parse(text: str) -> dict:
    """Strip markdown fences and parse JSON safely"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        clean = text.strip()
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except Exception:
                    continue
        raise ValueError(f"Could not parse JSON from: {text[:100]}")


# ── BUG #6 FIX: Robust age parsing ───────────────────────────────────────
def parse_age(age_val) -> int | None:
    """Handle '67', '67.0', '67 years old', 67, 67.0 -> 67"""
    if age_val is None:
        return None
    try:
        # Try direct int
        return int(age_val)
    except (ValueError, TypeError):
        pass
    try:
        # Handle float strings like '67.0'
        return int(float(str(age_val)))
    except (ValueError, TypeError):
        pass
    # Handle '67 years old', '67-year-old', etc.
    match = re.search(r'\b(\d{1,3})\b', str(age_val))
    if match:
        age = int(match.group(1))
        if 0 < age < 120:
            return age
    return None


# ── NODE 1: Parse and Clean ───────────────────────────────────────────────
async def parse_and_clean(state: AgentState) -> AgentState:
    """Expand all medical abbreviations into full readable text"""
    state["current_step"] = "📋 Parsing and expanding medical abbreviations..."

    try:
        messages = [
            SystemMessage(content=CLEAN_SYSTEM_PROMPT),
            HumanMessage(content=state["raw_input"])
        ]
        result = None
        for attempt in range(3):
            try:
                result = await llm.ainvoke(messages)
                break
            except Exception as e:
                if is_rate_limit_error(e) and attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                raise
        state["cleaned_text"] = result.content.strip()

    except Exception as e:
        state["errors"].append(f"parse_and_clean: {user_safe_error(e)}")
        state["cleaned_text"] = state["raw_input"]

    return state


# ── NODE 2: Extract Entities ──────────────────────────────────────────────
async def extract_entities(state: AgentState) -> AgentState:
    """Extract structured clinical entities — schema enforced"""
    state["current_step"] = "🔍 Extracting clinical entities..."

    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = groq_call_with_retry(lambda: client.chat.completions.create(
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
                    "content": f"Extract clinical data from this note and return JSON:\n\n{state['cleaned_text']}"
                }
            ],
            response_format={"type": "json_object"},
            timeout=60,
        ))

        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        state["entities"] = _normalize_entities(parsed)

    except Exception as e:
        state["errors"].append(f"extract_entities: {user_safe_error(e)}")
        state["entities"] = _empty_entities()

    return state


def _normalize_entities(raw: dict) -> dict:
    """Force every field into exact shape. Handles any LLM output variation."""

    # Normalize conditions → always list of strings
    # Check all keys LLM might use for conditions
    raw_conditions = (
        raw.get("conditions") or
        raw.get("diagnoses") or
        raw.get("diagnosis") or
        raw.get("problems") or
        raw.get("problem_list") or
        raw.get("assessment") or
        raw.get("medical_conditions") or
        []
    )
    conditions = []
    for c in raw_conditions:
        if isinstance(c, str) and c.strip():
            conditions.append(c.strip())
        elif isinstance(c, dict):
            name = (
                c.get("name") or c.get("condition") or
                c.get("diagnosis") or c.get("description") or
                c.get("problem") or ""
            )
            if name:
                conditions.append(str(name).strip())

    # Normalize allergies → always list of strings
    raw_allergies = raw.get("allergies", []) or []
    allergies = []
    for a in raw_allergies:
        if isinstance(a, str) and a.strip():
            allergies.append(a.strip())
        elif isinstance(a, dict):
            val = next(iter(a.values()), "")
            if val:
                allergies.append(str(val).strip())

    # Normalize medications → always list of {name, dose, frequency}
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

    # Normalize vitals → always flat dict
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

    # Normalize patient — BUG #6 FIX: robust age parsing
    raw_patient = raw.get("patient", {}) or {}
    age_raw = raw_patient.get("age") or raw.get("age")
    gender_raw = raw_patient.get("gender") or raw.get("gender")

    return {
        "patient": {
            "age": parse_age(age_raw),   # BUG #6 FIX
            "gender": str(gender_raw).lower().strip() if gender_raw else None,
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

        for med in medications:
            med_name = med.get("name", "").strip().lower()
            if not med_name:
                continue

            # BUG #4 FIX: resolve aliases first
            resolved_name = DRUG_ALIASES.get(med_name, med_name)

            try:
                # Try generic_name first, then brand_name as fallback
                fda_result = _query_fda(resolved_name)
                if not fda_result and resolved_name != med_name:
                    fda_result = _query_fda(med_name)

                if fda_result:
                    warnings.append(fda_result)

            except Exception as med_err:
                state["errors"].append(f"FDA check for {med_name}: {str(med_err)}")
                continue

    except Exception as e:
        state["errors"].append(f"check_drug_interactions: {str(e)}")

    state["drug_warnings"] = warnings
    return state


def _query_fda(drug_name: str) -> dict | None:
    """
    Query FDA API for a single drug. Returns warning dict or None.
    BUG #3 FIX: tries multiple text fields, never returns silently empty.
    """
    try:
        # Try generic name search
        url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{drug_name}&limit=1"
        resp = requests.get(url, timeout=8)

        if resp.status_code != 200:
            # Try brand name search as fallback
            url2 = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name}&limit=1"
            resp = requests.get(url2, timeout=8)

        if resp.status_code != 200:
            return None

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None

        label = results[0]

        # BUG #3 FIX: check all warning-related fields, not just 'warnings'
        warning_text = (
            (label.get("boxed_warning") or [""])[0]
            or (label.get("warnings") or [""])[0]
            or (label.get("warnings_and_cautions") or [""])[0]
            or (label.get("precautions") or [""])[0]
            or ""
        )

        interactions_text = (
            (label.get("drug_interactions") or [""])[0]
            or (label.get("drug_and_or_laboratory_test_interactions") or [""])[0]
            or ""
        )

        # Determine severity from text content
        severity = "LOW"
        combined = (warning_text + interactions_text).upper()
        if any(w in combined for w in ["CONTRAINDICATED", "FATAL", "LIFE-THREATENING", "BOXED"]):
            severity = "HIGH"
        elif any(w in combined for w in ["CAUTION", "MONITOR", "AVOID", "INCREASED RISK"]):
            severity = "MODERATE"

        # BUG #3 FIX: always return something if drug was found in FDA
        if not warning_text and not interactions_text:
            # Drug found but no explicit warnings — still useful info
            brand = (label.get("openfda", {}).get("brand_name") or [drug_name])[0]
            warning_text = f"Drug found in FDA database as '{brand}'. No major warnings documented in this label."
            severity = "LOW"

        return {
            "drug": drug_name.title(),
            "severity": severity,
            "warning": warning_text[:400] if warning_text else "",
            "interactions": interactions_text[:400] if interactions_text else "",
            "source": "FDA Drug Label Database"
        }

    except requests.Timeout:
        return None
    except Exception:
        return None


# ── NODE 4: RAG Enrich ───────────────────────────────────────────────────
async def rag_enrich(state: AgentState) -> AgentState:
    """Search FAISS knowledge base for ICD-10 codes and clinical context"""
    state["current_step"] = "🧠 Searching medical knowledge base..."

    try:
        from rag.retriever import search_icd_codes, search, is_loaded

        raw_conditions = state["entities"].get("conditions", [])

        # Normalize conditions to strings
        conditions = []
        for c in raw_conditions:
            if isinstance(c, str):
                conditions.append(c.strip())
            elif isinstance(c, dict):
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
                icd_codes[condition] = basic_icd.get(condition.lower(), "Pending")
            state["icd_codes"] = icd_codes
            state["rag_context"] = "Used basic mapping (index not loaded)"
            return state

        icd_codes = search_icd_codes(conditions)
        state["icd_codes"] = icd_codes

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
PATIENT ENTITIES (already extracted — use these directly):
{json.dumps(state['entities'], indent=2)}

ICD-10 CODES FROM KNOWLEDGE BASE (use these exact codes, do not invent):
{json.dumps(state['icd_codes'], indent=2)}

DRUG WARNINGS FROM FDA:
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

        # ── LAYER 1: Inject authoritative data ───────────────────────────
        report["icd_codes"] = state["icd_codes"]
        report["drug_warnings"] = state["drug_warnings"]
        report["errors"] = state["errors"]
        report["confidence_flags"] = state["confidence_flags"]

        # ── LAYER 2: Normalize + fill nulls from entities ─────────────────
        entities = state["entities"]
        patient_entity = entities.get("patient", {})

        if "patient_card" not in report or not isinstance(report.get("patient_card"), dict):
            report["patient_card"] = {}

        pc = report["patient_card"]

        # BUG #2 FIX: age/gender fallback from entities
        if not pc.get("age"):
            pc["age"] = patient_entity.get("age")
        if not pc.get("gender"):
            pc["gender"] = patient_entity.get("gender")
        if not pc.get("chief_complaint"):
            pc["chief_complaint"] = entities.get("chief_complaint", "")

        # BUG #10 FIX: Build conditions_with_codes programmatically
        # Never let LLM invent ICD codes — use FAISS results
        conditions = entities.get("conditions", [])
        icd_codes = state["icd_codes"]
        pc["conditions_with_codes"] = [
            {
                "condition": c,
                "icd_code": icd_codes.get(c, "")  # BUG #1 + #10 FIX
            }
            for c in conditions if c
        ]

        # BUG #8 FIX: vitals fallback from entities
        if not pc.get("vitals") or not any(pc.get("vitals", {}).values()):
            pc["vitals"] = entities.get("vitals", {})

        # BUG #9 FIX: medications fallback from entities
        pc_meds = pc.get("medications", [])
        if not pc_meds:
            pc["medications"] = entities.get("medications", [])
        else:
            # Normalize whatever LLM returned
            clean_meds = []
            for m in pc_meds:
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
            pc["medications"] = clean_meds if clean_meds else entities.get("medications", [])

        # Normalize allergies
        raw_allergies = pc.get("allergies", []) or entities.get("allergies", [])
        clean_allergies = []
        for a in raw_allergies:
            if isinstance(a, str) and a.strip():
                clean_allergies.append(a)
            elif isinstance(a, dict):
                val = next(iter(a.values()), "")
                if val:
                    clean_allergies.append(str(val))
        pc["allergies"] = clean_allergies

        # Normalize urgent_flags → plain strings
        raw_flags = report.get("urgent_flags", []) or []
        report["urgent_flags"] = [
            f if isinstance(f, str) else str(next(iter(f.values()), ""))
            for f in raw_flags if f
        ]

        report["patient_card"] = pc

        # BUG #5 FIX: plain_english_summary fallback using template
        if not report.get("plain_english_summary"):
            report["plain_english_summary"] = _build_summary_fallback(entities, state["drug_warnings"])

        # BUG #5 FIX: referral_text fallback
        if not report.get("referral_text"):
            report["referral_text"] = _build_referral_fallback(entities, icd_codes)

        state["final_report"] = report
        state["current_step"] = "✅ Report complete!"

    except Exception as e:
        state["errors"].append(f"generate_summary: {str(e)}")
        # BUG #5 FIX: structured fallback, not raw cleaned_text
        entities = state.get("entities", _empty_entities())
        state["final_report"] = {
            "plain_english_summary": _build_summary_fallback(entities, state.get("drug_warnings", [])),
            "patient_card": {
                "age": entities.get("patient", {}).get("age"),
                "gender": entities.get("patient", {}).get("gender"),
                "chief_complaint": entities.get("chief_complaint", ""),
                "conditions_with_codes": [
                    {"condition": c, "icd_code": state["icd_codes"].get(c, "")}
                    for c in entities.get("conditions", []) if c
                ],
                "medications": entities.get("medications", []),
                "vitals": entities.get("vitals", {}),
                "allergies": entities.get("allergies", []),
            },
            "referral_text": _build_referral_fallback(entities, state.get("icd_codes", {})),
            "icd_codes": state.get("icd_codes", {}),
            "drug_warnings": state.get("drug_warnings", []),
            "errors": state["errors"],
            "confidence_flags": state.get("confidence_flags", []),
            "urgent_flags": [],
        }

    return state


# ── BUG #5 FIX: Template-based summary fallback ──────────────────────────
def _build_summary_fallback(entities: dict, drug_warnings: list) -> str:
    """Build a readable plain English summary from structured data."""
    parts = []
    patient = entities.get("patient", {})
    age = patient.get("age")
    gender = patient.get("gender", "")
    complaint = entities.get("chief_complaint", "")
    conditions = entities.get("conditions", [])
    meds = entities.get("medications", [])

    # Patient line
    if age and gender:
        parts.append(f"{age}-year-old {gender}")
    elif age:
        parts.append(f"{age}-year-old patient")
    elif gender:
        parts.append(f"{gender.title()} patient")
    else:
        parts.append("Patient")

    if complaint:
        parts[0] += f" presenting with {complaint.lower()}"
    parts[0] += "."

    # Conditions
    if conditions:
        parts.append(f"Medical history includes {', '.join(conditions[:4])}.")

    # Medications
    if meds:
        med_names = [m.get("name", "") for m in meds if m.get("name")]
        if med_names:
            parts.append(f"Current medications: {', '.join(med_names[:5])}.")

    # Warnings
    if drug_warnings:
        parts.append(f"{len(drug_warnings)} drug interaction warning(s) identified — review before prescribing.")

    return " ".join(parts) if parts else "Clinical note processed. See structured data below."


def _build_referral_fallback(entities: dict, icd_codes: dict) -> str:
    """Build a basic referral text from structured data."""
    patient = entities.get("patient", {})
    age = patient.get("age", "")
    gender = patient.get("gender", "")
    complaint = entities.get("chief_complaint", "")
    conditions = entities.get("conditions", [])
    meds = entities.get("medications", [])

    age_str = f"{age}-year-old " if age else ""
    gender_str = gender if gender else "patient"
    complaint_str = f" presenting with {complaint}" if complaint else ""

    lines = [
        f"Re: {age_str}{gender_str}{complaint_str}",
        "",
        "Dear Colleague,",
        "",
        f"I am referring this {age_str}{gender_str} for further evaluation.",
    ]

    if conditions:
        codes_str = ", ".join(
            f"{c} ({icd_codes.get(c, '')})" if icd_codes.get(c) else c
            for c in conditions[:4]
        )
        lines.append(f"Diagnoses: {codes_str}.")

    if meds:
        med_names = ", ".join(m.get("name", "") for m in meds[:5] if m.get("name"))
        if med_names:
            lines.append(f"Current medications: {med_names}.")

    lines += ["", "Please see attached clinical notes for full details.", "", "Sincerely,"]
    return "\n".join(lines)
