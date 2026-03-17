"""
Auto-generates the assistant outputs for training_data.jsonl using Groq.
Run AFTER prepare_data.py.
Command: python fine_tuning/generate_labels.py
"""

import json
import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv("backend/.env")

INPUT_PATH = "fine_tuning/training_data.jsonl"
OUTPUT_PATH = "fine_tuning/training_data_labeled.jsonl"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EXTRACT_PROMPT = """Extract clinical data from this note and return ONLY valid JSON.
Follow this exact structure:
{
  "patient": {"age": null, "gender": null},
  "chief_complaint": "",
  "conditions": ["string only", "no objects"],
  "medications": [{"name": "", "dose": "", "frequency": ""}],
  "vitals": {"bp": null, "hr": null, "rr": null, "o2_sat": null},
  "allergies": ["string only"],
  "plan": []
}
Return ONLY JSON. No explanation. No markdown."""


def generate_label(note_text: str) -> str:
    """Use Groq llama-3.1-70b to generate high quality label"""
    try:
        resp = client.chat.completions.create(
model="llama-3.3-70b-versatile",
            temperature=0.1,
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": note_text}
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"   Error: {e}")
        return json.dumps({
            "patient": {"age": None, "gender": None},
            "chief_complaint": "",
            "conditions": [],
            "medications": [],
            "vitals": {},
            "allergies": [],
            "plan": []
        })


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Run prepare_data.py first")
        return

    with open(INPUT_PATH, 'r') as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Generating labels for {len(examples)} examples...")
    print("Using llama-3.1-70b-versatile for high quality labels\n")

    labeled = []
    for i, ex in enumerate(examples):
        print(f"[{i+1}/{len(examples)}] Generating label...")

        # Get the user message (the note)
        user_msg = next(m['content'] for m in ex['messages'] if m['role'] == 'user')

        # Generate label
        label = generate_label(user_msg)

        # Replace placeholder with real label
        new_ex = {
            "messages": [
                ex['messages'][0],  # system
                ex['messages'][1],  # user
                {"role": "assistant", "content": label}  # labeled output
            ]
        }
        labeled.append(new_ex)

        # Rate limit — Groq allows ~30 req/min on free tier
        time.sleep(2)

    # Save labeled data
    with open(OUTPUT_PATH, 'w') as f:
        for ex in labeled:
            f.write(json.dumps(ex) + '\n')

    print(f"\n✅ Labeled data saved to {OUTPUT_PATH}")
    print(f"   {len(labeled)} examples ready for fine-tuning")
    print("\nNext step: Upload training_data_labeled.jsonl to Google Colab")


if __name__ == "__main__":
    main()