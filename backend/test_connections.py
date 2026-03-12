import os
import requests
from dotenv import load_dotenv

load_dotenv()
print("Testing all API connections...\n")

# ── Test 1: Groq ──────────────────────────────────────
try:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say the word: CONNECTED"}]
    )
    print("✅ Groq:", resp.choices[0].message.content)
except Exception as e:
    print("❌ Groq failed:", e)


# Test 2: Groq Vision (replaces Gemini — handles handwriting OCR)
try:
    from groq import Groq
    vision_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = vision_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": "Say the word: CONNECTED"}]
    )
    print("✅ Groq Vision:", resp.choices[0].message.content.strip())
except Exception as e:
    print("❌ Groq Vision failed:", e)

# Test 3: HuggingFace
try:
    import time
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    urls = [
        "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction",
        "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
        "https://router.huggingface.co/hf-inference/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
    ]
    
    success = False
    for url in urls:
        resp = requests.post(url, headers=headers, json={"inputs": "test medical text"})
        if resp.status_code == 200:
            result = resp.json()
            # result can be list of lists or list of floats
            if isinstance(result[0], list):
                print(f"✅ HuggingFace: vector length = {len(result[0])}")
            else:
                print(f"✅ HuggingFace: vector length = {len(result)}")
            success = True
            break
        elif resp.status_code == 503:
            wait = resp.json().get("estimated_time", 10)
            print(f"⏳ Model loading... waiting {int(wait)+2}s")
            time.sleep(int(wait) + 2)
            # retry same URL after wait
            resp2 = requests.post(url, headers=headers, json={"inputs": "test medical text"})
            if resp2.status_code == 200:
                result = resp2.json()
                print(f"✅ HuggingFace: connected after wait")
                success = True
                break
    
    if not success:
        print(f"❌ HuggingFace: all URLs failed — last status {resp.status_code}")
        print(f"   Response: {resp.text[:150]}")

except Exception as e:
    print("❌ HuggingFace failed:", e)
# ── Test 4: OpenFDA ───────────────────────────────────
try:
    resp = requests.get(
        "https://api.fda.gov/drug/label.json?search=openfda.generic_name:metformin&limit=1"
    )
    if resp.status_code == 200:
        print("✅ OpenFDA: status 200 — connected")
    else:
        print("❌ OpenFDA: status", resp.status_code)
except Exception as e:
    print("❌ OpenFDA failed:", e)

print("\n--- Done. Fix any ❌ before moving to Phase 1 ---")