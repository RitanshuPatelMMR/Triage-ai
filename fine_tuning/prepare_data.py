"""
Run this to generate training_data.jsonl from MTSamples.
Command: python fine_tuning/prepare_data.py
"""

import pandas as pd
import json
import os

# Path to your MTSamples file
MTSAMPLES_PATH = "backend/data/mtsamples.csv"
OUTPUT_PATH = "fine_tuning/training_data.jsonl"

SYSTEM_PROMPT = """You are a clinical notes parser.
Extract structured JSON from medical notes.
Return ONLY valid JSON with these fields:
{
  "patient": {"age": null, "gender": null},
  "chief_complaint": "",
  "conditions": [],
  "medications": [{"name": "", "dose": "", "frequency": ""}],
  "vitals": {"bp": null, "hr": null, "rr": null, "o2_sat": null},
  "allergies": [],
  "plan": []
}
Conditions and allergies must be plain strings.
Never hallucinate. Use null if not mentioned."""


def clean_text(text: str) -> str:
    """Clean and truncate transcription text"""
    if not text or str(text) == "nan":
        return ""
    text = str(text).strip()
    # Truncate to 800 words max
    words = text.split()
    if len(words) > 800:
        text = " ".join(words[:800])
    return text


def make_example(transcription: str, specialty: str) -> dict | None:
    """Create one JSONL training example"""
    text = clean_text(transcription)
    if len(text) < 50:
        return None

    # The assistant output — we ask the model to extract
    # For training we use a structured extraction
    # This creates the input/output pair
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Extract clinical data from this {specialty} note:\n\n{text}"
            },
            {
                "role": "assistant",
                "content": "__PLACEHOLDER__"
            }
        ]
    }


def main():
    if not os.path.exists(MTSAMPLES_PATH):
        print(f"❌ MTSamples not found at {MTSAMPLES_PATH}")
        return

    print("Loading MTSamples...")
    df = pd.read_csv(MTSAMPLES_PATH)
    print(f"Total notes: {len(df)}")

    # Pick diverse samples across specialties
    specialties = df['medical_specialty'].value_counts().head(10).index.tolist()
    selected = []

    for specialty in specialties:
        subset = df[df['medical_specialty'] == specialty].head(5)
        selected.append(subset)

    sample_df = pd.concat(selected).head(50)
    print(f"Selected {len(sample_df)} samples across {len(specialties)} specialties")

    # Generate examples
    examples = []
    for _, row in sample_df.iterrows():
        ex = make_example(
            str(row.get('transcription', '')),
            str(row.get('medical_specialty', 'General'))
        )
        if ex:
            examples.append(ex)

    print(f"Generated {len(examples)} valid examples")
    print()
    print("⚠️  IMPORTANT: You need to fill in the __PLACEHOLDER__ outputs manually")
    print("    Open training_data.jsonl and replace each __PLACEHOLDER__ with")
    print("    the correct JSON extraction for that note.")
    print()
    print("    OR use the auto-generate script below to let GPT fill them in.")

    # Save with placeholders
    with open(OUTPUT_PATH, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"✅ Saved to {OUTPUT_PATH}")
    print(f"   {len(examples)} examples ready for review")


if __name__ == "__main__":
    main()