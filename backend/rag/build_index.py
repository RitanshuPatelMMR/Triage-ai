import os
import sys
import json
import faiss
import numpy as np
import pandas as pd

# Add backend to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embedder import embed_batch

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INDEX_PATH = os.path.join(DATA_DIR, "medical_kb.index")
TEXTS_PATH = os.path.join(DATA_DIR, "medical_kb_texts.json")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """Split long text into overlapping chunks"""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 30:  # skip tiny chunks
            chunks.append(chunk)
    return chunks


def load_mtsamples() -> tuple:
    """Load MTSamples clinical notes"""
    path = os.path.join(DATA_DIR, "mtsamples.csv")

    if not os.path.exists(path):
        print(f"⚠️  mtsamples.csv not found at {path}")
        print("   Download from: kaggle.com/datasets/tboyle10/medicaltranscriptions")
        return [], []

    print("Loading MTSamples...")
    df = pd.read_csv(path)
    print(f"   Found {len(df)} clinical notes")

    texts, metadata = [], []
    for _, row in df.iterrows():
        transcription = str(row.get("transcription", "")).strip()
        specialty = str(row.get("medical_specialty", "General")).strip()

        if len(transcription) < 50:
            continue

        for chunk in chunk_text(transcription):
            texts.append(chunk)
            metadata.append({
                "source": "mtsamples",
                "specialty": specialty,
                "type": "clinical_note"
            })

    print(f"   Generated {len(texts)} chunks from MTSamples")
    return texts, metadata


def load_icd10() -> tuple:
    """Load ICD-10 diagnosis codes"""
    path = os.path.join(DATA_DIR, "icd10_codes.csv")

    if not os.path.exists(path):
        print(f"⚠️  icd10_codes.csv not found at {path}")
        return [], []

    print("Loading ICD-10 codes...")

    texts, metadata = [], []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Format: col0,col1,CODE,"Description","Full Description","Category"
                # We need column index 2 (code) and index 3 (description)
                parts = line.split(",")
                if len(parts) < 4:
                    continue

                code = parts[2].strip().strip('"')
                description = parts[3].strip().strip('"')

                if not code or not description or len(code) < 3:
                    continue

                entry = f"ICD-10 code {code}: {description}"
                texts.append(entry)
                metadata.append({
                    "source": "icd10",
                    "code": code,
                    "description": description,
                    "type": "diagnosis_code"
                })

            except Exception:
                continue

    print(f"   Loaded {len(texts)} ICD-10 entries")
    return texts, metadata

def build_index():
    print("\n🏗️  Building TriageAI Medical Knowledge Base")
    print("=" * 50)

    all_texts, all_metadata = [], []

    # Load both data sources
    mt_texts, mt_meta = load_mtsamples()
    all_texts.extend(mt_texts)
    all_metadata.extend(mt_meta)

    icd_texts, icd_meta = load_icd10()
    all_texts.extend(icd_texts)
    all_metadata.extend(icd_meta)

    if not all_texts:
        print("❌ No data loaded — check your data files")
        return

    print(f"\n📊 Total chunks to embed: {len(all_texts)}")
    print("⏳ Embedding... (this takes 5-10 minutes)")
    print("   Tip: Go grab a coffee — this runs once and saves forever\n")

    # Embed in batches with progress
    batch_size = 128
    all_vectors = []

    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i + batch_size]
        vectors = embed_batch(batch)
        all_vectors.extend(vectors)

        # Progress indicator
        progress = min(i + batch_size, len(all_texts))
        percent = (progress / len(all_texts)) * 100
        print(f"   Progress: {progress}/{len(all_texts)} ({percent:.1f}%)")

    print("\n🔨 Building FAISS index...")
    vectors_np = np.array(all_vectors, dtype="float32")
    dimension = vectors_np.shape[1]  # 384 for all-MiniLM-L6-v2

    # IndexFlatIP = Inner Product (cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_np)

    # Save index to disk
    faiss.write_index(index, INDEX_PATH)
    print(f"✅ FAISS index saved: {INDEX_PATH}")
    print(f"   Vectors stored: {index.ntotal}")

    # Save texts and metadata for lookup
    with open(TEXTS_PATH, "w") as f:
        json.dump({"texts": all_texts, "metadata": all_metadata}, f)
    print(f"✅ Texts saved: {TEXTS_PATH}")

    # File size info
    index_size = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    texts_size = os.path.getsize(TEXTS_PATH) / (1024 * 1024)
    print(f"\n📁 Index size: {index_size:.1f} MB")
    print(f"📁 Texts size: {texts_size:.1f} MB")
    print(f"\n🎉 Knowledge base ready! {index.ntotal} vectors indexed.")


if __name__ == "__main__":
    build_index()