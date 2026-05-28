"""User-safe error messages for API responses (recruiter-facing)."""


def user_safe_error(exc: Exception) -> str:
    msg = str(exc).lower()

    if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
        return "Rate limited — wait 10 seconds and try again."

    if "401" in msg or "invalid_api_key" in msg or "api key" in msg:
        return "AI service is not configured. Please contact the administrator."

    if "groq_api_key" in msg or "not set" in msg:
        return "AI service is not configured (missing API key)."

    if "read of closed file" in msg:
        return "File upload failed — please try again."

    if "poppler" in msg or "pdfinfo" in msg or "unable to get page count" in msg:
        return "Could not process this PDF scan. Try a clearer PDF or paste the note as text."

    if "password" in msg or "encrypted" in msg:
        return "Password-protected PDFs are not supported."

    if "timeout" in msg or "timed out" in msg:
        return "Request timed out — the server may be waking up. Please try again."

    if "connection" in msg or "network" in msg:
        return "Could not reach the AI service. Please try again in a moment."

    # Avoid leaking long stack traces
    raw = str(exc).strip()
    if len(raw) > 200:
        return "Something went wrong during analysis. Please try again."
    if raw.startswith("{") or "traceback" in msg:
        return "Something went wrong during analysis. Please try again."

    return raw or "Something went wrong during analysis. Please try again."


def map_loader_error(error: str) -> str:
    """Map loader/OCR error strings to recruiter-friendly copy."""
    lower = error.lower()
    if "unsupported file" in lower:
        return error
    if "ocr" in lower or "extract" in lower:
        return "Could not read text from this file. Try a clearer scan or paste the note as text."
    if "poppler" in lower:
        return "PDF scan processing is unavailable. Paste the note as text instead."
    return user_safe_error(Exception(error))
