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
    api_key=os.getenv("GROQ_API_KEY")
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
    """Extract structured clinical entities into JSON"""
    state["current_step"] = "🔍 Extracting clinical entities..."

    try:
        messages = [
            SystemMessage(content=EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=state["cleaned_text"])
        ]
        result = await llm.ainvoke(messages)
        state["entities"] = safe_json_parse(result.content)

    except Exception as e:
        state["errors"].append(f"extract_entities: {str(e)}")
        state["entities"] = {
            "patient": {"age": None, "gender": None},
            "chief_complaint": "",
            "conditions": [],
            "medications": [],
            "vitals": {},
            "allergies": [],
            "plan": [],
            "follow_up": ""
        }

    return state


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


# ── NODE 4: RAG Enrich (real FAISS search) ────────────────────────────────
async def rag_enrich(state: AgentState) -> AgentState:
    """Search FAISS knowledge base for ICD-10 codes and clinical context"""
    state["current_step"] = "🧠 Searching medical knowledge base..."

    try:
        from rag.retriever import search_icd_codes, search, is_loaded

        conditions = state["entities"].get("conditions", [])

        if not conditions or not is_loaded():
            # Fallback to basic mapping if index not loaded
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

        # Also retrieve clinical context for summary generation
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
        # Build context for the LLM from all previous nodes
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

        # Inject ICD codes and drug warnings directly
        report["icd_codes"] = state["icd_codes"]
        report["drug_warnings"] = state["drug_warnings"]
        report["errors"] = state["errors"]
        report["confidence_flags"] = state["confidence_flags"]

        state["final_report"] = report
        state["current_step"] = "✅ Report complete!"

    except Exception as e:
        state["errors"].append(f"generate_summary: {str(e)}")
        # Fallback report so UI never gets empty response
        state["final_report"] = {
            "plain_english_summary": state["cleaned_text"],
            "patient_card": state["entities"],
            "icd_codes": state["icd_codes"],
            "drug_warnings": state["drug_warnings"],
            "errors": state["errors"],
            "confidence_flags": []
        }

    return state