# 🏥 TriageAI — Clinical Notes Summarization Agent

![TriageAI Banner](https://img.shields.io/badge/TriageAI-Clinical%20AI%20Agent-B5614A?style=for-the-badge&logo=react)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20App-green?style=for-the-badge)](https://triageai-ritanshupatel.vercel.app)
[![HuggingFace](https://img.shields.io/badge/Backend-HuggingFace%20Spaces-yellow?style=for-the-badge)](https://ritanshupatel-triageai-backend.hf.space)
[![Model](https://img.shields.io/badge/Model-Mistral--7B%20LoRA-blue?style=for-the-badge)](https://huggingface.co/RitanshuPatel/triageai-mistral)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](LICENSE)

> **Paste any messy doctor note — TriageAI reads it, understands it, checks it, and turns it into a professional medical summary in seconds.**

---

## 📸 App Screenshot

![TriageAI App](https://triageai-ritanshupatel.vercel.app/og-image.png)

> 🔗 **[Try the live demo →](https://triageai-ritanshupatel.vercel.app/analyze)**
>
> Paste this note to see it in action:
> ```
> pt 67M c/o CP x2hr rad to L arm, diaphoretic, SOB+, PMH DM2 HTN,
> smoker 20pk/yr, meds metformin 500 BID lisinopril 10 QD atorvastatin
> 40 QHS, EKG ST elev V2-V4, allerg PCN hives, BP 158/94 HR 102 O2 94%
> ```

---

## 🚨 The Problem

Every day, doctors and clinicians write hundreds of notes that look like this:

```
pt 67M c/o CP x2hr rad to L arm, diaphoretic, SOB+, PMH DM2 HTN,
smoker 20pk/yr, meds metformin 500 BID lisinopril 10 QD atorvastatin 40 QHS,
EKG ST elev V2-V4, trop pnd, A: r/o STEMI, P: ASA 325 STAT ntg SL
heparin gtt cath lab activation, BP 158/94 HR 102 RR 18 O2 94% RA, allerg PCN hives
```

This is the reality of clinical documentation — abbreviated, unstructured, and nearly unreadable to anyone outside the medical field. This creates serious problems:

- **Abbreviations are misread** — `CP` can mean chest pain, cerebral palsy, or care plan. Wrong interpretation can be dangerous.
- **Critical information is buried** — Drug allergies, dangerous drug combinations, and urgent flags are scattered across unstructured text.
- **Administrative burden is massive** — A clinician manually structuring one note takes 5–10 minutes. Multiply by 50 patients a day.
- **Downstream systems break** — Hospital software, billing, and referral systems need structured data — not free text.
- **Base LLMs fail at this task** — They return inconsistent JSON structures every time, making them unreliable for production medical applications.

**TriageAI solves all of this with a single paste.**

---

## 🌟 Why TriageAI Stands Out

| What most AI portfolios do | What TriageAI does |
|---|---|
| Simple chatbot — ask question, get answer | Autonomous 5-node agent that works independently |
| RAG over a PDF document | RAG + Vision LLM + external API data fusion |
| Call OpenAI API (costs money) | 100% free: Groq + HuggingFace + local embeddings |
| Single data type — text only | Handles typed notes, PDFs AND handwritten images |
| No fine-tuning | LoRA fine-tuned Mistral-7B on real clinical data |
| Static output | Real-time SSE streaming — watch the agent think live |
| No medical intelligence | ICD-10 codes + FDA drug interaction warnings |

---

## 🎯 Project Objectives

1. Build a production-grade autonomous AI agent using LangGraph — not just a simple prompt chain
2. Demonstrate real-world RAG with domain-specific knowledge (medical ICD-10 + clinical notes)
3. Fine-tune an open-source LLM on medical data to improve extraction consistency
4. Handle multiple input types — typed text, PDF, and handwritten images
5. Stream agent progress to the frontend in real time using Server-Sent Events
6. Deploy a complete full-stack AI application using 100% free infrastructure
7. Integrate real external APIs (OpenFDA) for live drug interaction checking

---

## ✅ What We Achieved

- ✅ 5-node autonomous LangGraph agent running end to end
- ✅ 82,921 vectors in FAISS knowledge base (MTSamples + ICD-10)
- ✅ Real-time SSE streaming with live agent log in the UI
- ✅ LoRA fine-tuned Mistral-7B with training loss 1.74 → 1.26
- ✅ FDA drug interaction warnings per medication
- ✅ Groq Vision OCR for handwritten medical notes
- ✅ PDF parsing and processing
- ✅ Full React frontend with dark mode, history, and export
- ✅ Deployed free on HuggingFace Spaces + Vercel
- ✅ Zero cost — $0/month infrastructure

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                           │
│              React + TypeScript + Tailwind CSS                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │  HTTP / SSE Streaming
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                                │
│              Server-Sent Events (SSE)                           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LangGraph Agent (5 Nodes)                       │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Node 1     │───▶│   Node 2     │───▶│   Node 3     │       │
│  │ parse_and_   │    │  extract_    │    │ check_drug_  │       │
│  │   clean      │    │  entities    │    │interactions  │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│                                                  ▼               │
│  ┌──────────────┐    ┌──────────────┐            │               │
│  │   Node 5     │◀───│   Node 4     │◀───────────┘               │
│  │  generate_   │    │  rag_enrich  │                            │
│  │   summary    │    │              │                            │
│  └──────────────┘    └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
  │  Groq API   │    │ FAISS Index  │    │  OpenFDA API │
  │ Llama 3.1   │    │ 82,921 vecs  │    │ Drug Labels  │
  │ + Vision    │    │ ICD-10 +     │    │              │
  │    OCR      │    │ MTSamples    │    │              │
  └─────────────┘    └──────────────┘    └──────────────┘
```

---

## 🤖 How the Agent Works — Step by Step

### Node 1 — Parse and Clean
Receives raw clinical note. Expands all medical abbreviations into full readable text.
`HTN → Hypertension`, `DM2 → Type 2 Diabetes`, `BID → twice daily`, `c/o → complains of`

### Node 2 — Extract Entities
Uses Groq Llama 3.1 with a strict JSON schema prompt to extract:
- Patient demographics (age, gender)
- Chief complaint
- Conditions / diagnoses
- Medications with dose and frequency
- Vital signs
- Allergies
- Treatment plan

### Node 3 — Check Drug Interactions
For every medication found, calls the **OpenFDA Drug Label API** and checks:
- Boxed warnings (BLACK BOX — highest severity)
- Known drug interactions
- Precautions and contraindications
- Returns severity: HIGH / MODERATE / LOW

### Node 4 — RAG Enrich
Embeds each diagnosis using **sentence-transformers (all-MiniLM-L6-v2)** and searches the FAISS knowledge base built from:
- **MTSamples** — 4,000+ real de-identified clinical notes
- **ICD-10 database** — 70,000+ official diagnosis codes

Returns the correct ICD-10 code for each condition.

### Node 5 — Generate Summary
Combines all previous node outputs into a final structured report:
- Plain English summary (readable by non-doctors)
- Structured patient card with ICD codes
- Drug interaction summary
- Copy-ready referral text
- Urgent flags for critical findings
- Confidence flags for fields needing human review

### Real-Time Streaming
Every node completion fires an **SSE event** to the frontend. Users watch each step complete live — like watching the agent think in real time.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **5-Node LangGraph Agent** | Autonomous multi-step reasoning with live SSE streaming |
| 🔍 **ICD-10 Code Mapping** | 82,921 vector FAISS search over official CMS codes |
| 💊 **FDA Drug Warnings** | Real-time OpenFDA API checks per medication |
| 🧠 **Fine-tuned Model** | LoRA Mistral-7B trained on 49 clinical examples |
| ✍️ **Handwriting OCR** | Groq Vision reads messy handwritten notes |
| 📄 **PDF Support** | Upload and process PDF clinical documents |
| 📊 **Confidence Scoring** | Visual confidence bars per ICD code |
| ⚠️ **Urgent Flag Detection** | Critical findings highlighted automatically |
| 📋 **Referral Text Generator** | Copy-ready specialist referral letters |
| 🕐 **Analysis History** | localStorage with search (swap to DB easily) |
| 🌙 **Dark Mode** | Full dark/light theme toggle |
| 📤 **Export JSON** | Download structured report as JSON |
| ✅ **Human Verification** | Mark low-confidence fields as reviewed |
| 🔄 **Model Comparison** | Side-by-side base vs fine-tuned output |

---

## 🛠️ Tech Stack

| Technology | Role | Why Chosen |
|---|---|---|
| **LangGraph** | Agent orchestration | Stateful multi-node agent, not just a chain |
| **LangChain** | Prompt management | Clean prompt templates and LLM abstraction |
| **Groq API** | Primary LLM inference | 14,400 free req/day, fastest inference available |
| **Llama 3.1 8B** | Text extraction & summary | Fast, accurate, free on Groq |
| **Llama 4 Scout** | Vision OCR | Best free vision model for handwriting |
| **Mistral-7B LoRA** | Fine-tuned extraction | Consistent JSON output after fine-tuning |
| **FAISS** | Vector similarity search | Local, fast, no API cost |
| **sentence-transformers** | Text embeddings | Local model, no API rate limits |
| **OpenFDA API** | Drug interaction data | Official FDA database, completely free |
| **FastAPI** | Backend REST API | Async support, native SSE streaming |
| **React + TypeScript** | Frontend UI | Type-safe, component-based architecture |
| **Tailwind CSS v3** | Styling | Rapid UI development, dark mode support |
| **Vite** | Frontend build tool | Fast HMR, optimized production builds |
| **pypdf** | PDF text extraction | Lightweight, no external dependencies |
| **HuggingFace Spaces** | Backend hosting | Free Docker deployment, permanent URL |
| **Vercel** | Frontend hosting | Free React deployment, one command |
| **Google Colab** | Model fine-tuning | Free T4 GPU for LoRA training |
| **HuggingFace Hub** | Model storage | Free model hosting, version control |

---

## 🔬 Fine-Tuning Details

### Why We Fine-Tuned
Base LLMs extract clinical entities inconsistently. Sometimes `conditions` is a list of strings, sometimes objects, sometimes nested. This breaks the entire pipeline. Fine-tuning forces consistent JSON schema on every call.

### Training Process
- **Base model:** mistralai/Mistral-7B-Instruct-v0.3
- **Method:** QLoRA (4-bit quantization + LoRA)
- **Dataset:** 49 examples from MTSamples (10 specialties)
- **Label generation:** llama-3.3-70b-versatile auto-labeling
- **Platform:** Google Colab T4 GPU (free)
- **Duration:** ~45 minutes
- **Training loss:** 1.74 → 1.26 (genuine learning)
- **Adapter size:** 13.6 MB

### What Improved
- JSON structure is consistent on every call
- Medical abbreviations expanded correctly
- Age returned as number not string
- Allergies returned as plain strings not objects
- No hallucinated fields

### LoRA Configuration
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)
```

---

## 📚 RAG Pipeline

### Knowledge Base Contents
- **MTSamples** — 4,000+ de-identified real clinical transcription notes
- **ICD-10 CMS database** — 70,000+ official diagnosis codes with descriptions
- **Total vectors:** 82,921
- **Index size:** ~121 MB (stored via Git LFS on HuggingFace)
- **Embedding model:** all-MiniLM-L6-v2 (384 dimensions)

### How ICD Codes Are Retrieved
1. Node 2 extracts condition names (e.g. "Hypertension", "Type 2 Diabetes")
2. Node 4 embeds each condition name into a 384-dimensional vector
3. FAISS searches 82,921 vectors for closest matches
4. Top result from ICD-10 source → returns official code (e.g. I10, E11.9)
5. Hardcoded lookup for 60+ common conditions as primary strategy
6. FAISS semantic search as fallback for rare conditions

### Drug Context Enrichment
- Chief complaint embedded and searched against MTSamples chunks
- Returns top 3 most similar clinical note excerpts
- Injected as context into Node 5 summary generation

---

## 🔧 Challenges and Solutions

| Challenge | What Happened | Solution |
|---|---|---|
| **OOM crash on HuggingFace** | Exit code 137 — sentence-transformers loaded 500MB at startup | Lazy loading with `get_model()` — loads only on first request |
| **Inconsistent LLM JSON** | Agent crashed when LLM returned objects instead of strings | 3-layer normalization: strict prompts + `_normalize_entities()` + frontend guards |
| **Gemini quota zero in India** | Gemini free tier had no quota | Switched to Groq Vision (llama-4-scout) for OCR |
| **FAISS index too large for GitHub** | 121MB file rejected by GitHub | Git LFS on HuggingFace Spaces |
| **ICD codes never showing** | Codes found in Node 4 but lost in Node 5 | Programmatically build `conditions_with_codes` from entities + icd_codes dict |
| **Drug abbreviations missed by FDA** | ASA, ntg, lasix not recognized | 60+ drug alias mapping dictionary |
| **EOF literal in app.py** | `cat << 'EOF'` marker accidentally written into file | Removed with `head -n -1` |
| **Page refresh 404 on Vercel** | React Router routes not handled by Vercel | Added `vercel.json` with SPA rewrites |

---

## 🚀 Local Setup Instructions

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/RitanshuPatelMMR/Triage-ai.git
cd Triage-ai
```

### Step 2 — Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### Step 3 — Create .env File
```bash
touch .env
```

Add these keys to `.env`:
```
GROQ_API_KEY=your_groq_key_here
HF_API_KEY=your_huggingface_key_here
FDA_API_KEY=your_fda_key_here
```

### Step 4 — Build FAISS Knowledge Base (one time only)
```bash
# Download mtsamples.csv from kaggle.com/datasets/tboyle10/medicaltranscriptions
# Download icd10_codes.csv from cms.gov
# Place both in backend/data/

python rag/build_index.py
# Takes 5-10 minutes, builds 82,921 vectors
```

### Step 5 — Start Backend
```bash
uvicorn main:app --reload
# Runs at http://localhost:8000
```

### Step 6 — Frontend Setup
```bash
cd ../frontend
npm install
```

Create `.env.local`:
```
VITE_API_URL=http://localhost:8000
```

### Step 7 — Start Frontend
```bash
npm run dev
# Runs at http://localhost:5173
```

---

## 🔑 API Keys Required

| Service | Where to Get | Free Limit |
|---|---|---|
| **Groq** | console.groq.com | 14,400 req/day |
| **HuggingFace** | huggingface.co/settings/tokens | 1,000 req/day |
| **OpenFDA** | api.fda.gov/api_key | Unlimited with key |

---

## 📁 Project Structure

```
triageai/
├── backend/
│   ├── main.py                 ← FastAPI app (local)
│   ├── app.py                  ← FastAPI app (HuggingFace)
│   ├── Dockerfile              ← Docker config for HF Spaces
│   ├── requirements.txt
│   ├── agent/
│   │   ├── graph.py            ← LangGraph 5-node pipeline
│   │   ├── nodes.py            ← All 5 node functions
│   │   ├── state.py            ← AgentState TypedDict
│   │   └── prompts.py          ← All LLM system prompts
│   ├── rag/
│   │   ├── embedder.py         ← sentence-transformers wrapper
│   │   ├── retriever.py        ← FAISS search + ICD lookup
│   │   ├── loader.py           ← PDF + image file handler
│   │   └── build_index.py      ← One-time index builder
│   ├── tools/
│   │   ├── fda_api.py          ← OpenFDA drug checker
│   │   └── ocr.py              ← Groq Vision OCR
│   └── data/                   ← gitignored — large files
│       ├── mtsamples.csv
│       ├── icd10_codes.csv
│       └── medical_kb.index    ← 121MB FAISS index
│
├── frontend/
│   └── src/
│       ├── pages/              ← Home, Analyzer, History
│       ├── components/         ← UI components
│       │   ├── analyzer/       ← UploadPanel, AgentLog
│       │   ├── report/         ← ReportView, DrugWarnings
│       │   └── layout/         ← Navbar, Footer
│       ├── hooks/              ← useSSE, useHistory, useTheme
│       ├── services/           ← historyService (repository pattern)
│       └── types/              ← TypeScript interfaces
│
└── fine_tuning/
    ├── prepare_data.py         ← MTSamples → JSONL converter
    ├── generate_labels.py      ← Auto-label with Groq 70B
    └── colab_train.ipynb       ← Google Colab training notebook
```

---

## 🎓 Conclusion

TriageAI demonstrates that a complete, production-grade AI agent application can be built and deployed for **$0** using modern open-source tools and free API tiers.

**Skills demonstrated through this project:**

- **AI Agent Design** — Multi-node LangGraph orchestration with error handling and state management
- **LLM Fine-Tuning** — QLoRA training pipeline from data preparation to HuggingFace deployment
- **RAG Architecture** — Vector embedding pipeline, FAISS indexing, semantic search
- **Full-Stack Development** — FastAPI backend + React frontend, end to end
- **Real-Time Systems** — Server-Sent Events for live streaming
- **Production Deployment** — Docker, HuggingFace Spaces, Vercel, Git LFS
- **Domain Knowledge** — Medical NLP, ICD-10 coding, FDA drug data
- **Problem Solving** — OOM crashes, inconsistent LLM output, API rate limits

**What can be built on top of this:**
- Hospital EHR integration
- Automated billing code generation
- Multi-language clinical note support
- HIPAA-compliant version with proper data handling
- Real-time clinical decision support system

---

## 📄 Resume Line

```
Built TriageAI, a clinical notes summarization agent using LangGraph for 
5-node autonomous reasoning, LoRA fine-tuned Mistral-7B on 49 clinical 
examples, RAG with FAISS over 82,000+ medical knowledge vectors, Groq Vision 
for handwritten OCR, real-time SSE streaming — deployed free on HuggingFace 
Spaces and Vercel.
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ using LangGraph · Groq · FAISS · React · FastAPI**

[Live Demo](https://triageai-ritanshupatel.vercel.app) · [HuggingFace Model](https://huggingface.co/RitanshuPatel/triageai-mistral) · [Backend API](https://ritanshupatel-triageai-backend.hf.space/health)

</div>
