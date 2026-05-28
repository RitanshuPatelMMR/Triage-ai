import os
import base64
from groq import Groq
from dotenv import load_dotenv
from tools.groq_utils import groq_call_with_retry, groq_configured, require_groq
from tools.errors import user_safe_error

load_dotenv()

_client = None

TRANSCRIBE_PROMPT = """
You are a medical transcription assistant.
Look at this handwritten medical note image carefully.

Your job:
1. Read and transcribe ALL visible text exactly as written
2. For words you cannot read clearly, write [unclear] in that position
3. Preserve the original structure and order of the text
4. Do NOT interpret, expand, or add anything
5. Return ONLY the transcribed text, nothing else

Be thorough — even partially visible words should be captured.
"""

INFER_PROMPT = """
You are a medical transcription assistant.
This medical note has some [unclear] words from difficult handwriting.

Using the surrounding medical context, infer the most likely word for each [unclear].
Mark every inferred word like this: word(inferred)

Rules:
- Only infer if you are confident from medical context
- If you cannot confidently infer, leave it as [unclear]
- Do not change any clearly readable words
- Return ONLY the corrected text, nothing else

Example:
Input:  "patient has [unclear]ension and [unclear]betes"
Output: "patient has hypertension(inferred) and diabetes(inferred)"
"""


def _get_client() -> Groq:
    global _client
    if _client is None:
        require_groq()
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def transcribe_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Transcribe a handwritten medical note image using Groq Vision.
    Returns dict with transcribed text and confidence info.
    """
    if not groq_configured():
        return {
            "text": "",
            "input_type": "image",
            "error": "AI service is not configured (GROQ_API_KEY missing).",
            "confidence_flags": []
        }

    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        client = _get_client()

        response = groq_call_with_retry(lambda: client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": TRANSCRIBE_PROMPT
                        }
                    ]
                }
            ],
            max_tokens=1000,
            timeout=60,
        ))

        raw_text = response.choices[0].message.content.strip()

        has_unclear = "[unclear]" in raw_text
        final_text = raw_text

        if has_unclear:
            infer_response = groq_call_with_retry(lambda: client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": f"{INFER_PROMPT}\n\nText to fix:\n{raw_text}"
                    }
                ],
                max_tokens=1000,
                timeout=60,
            ))
            final_text = infer_response.choices[0].message.content.strip()

        inferred_count = final_text.count("(inferred)")
        remaining_unclear = final_text.count("[unclear]")

        confidence_flags = []
        if inferred_count > 0:
            confidence_flags.append(
                f"{inferred_count} word(s) were inferred from context — verify before clinical use"
            )
        if remaining_unclear > 0:
            confidence_flags.append(
                f"{remaining_unclear} word(s) could not be read — marked as [unclear]"
            )

        return {
            "text": final_text,
            "input_type": "image",
            "has_unclear": remaining_unclear > 0,
            "inferred_count": inferred_count,
            "confidence_flags": confidence_flags,
            "raw_transcription": raw_text
        }

    except Exception as e:
        safe = user_safe_error(e)
        return {
            "text": "",
            "input_type": "image",
            "error": safe,
            "confidence_flags": [f"OCR failed: {safe}"]
        }
