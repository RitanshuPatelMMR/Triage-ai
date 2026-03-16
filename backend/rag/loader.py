import io
from tools.ocr import transcribe_image


SUPPORTED_IMAGE_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}

SUPPORTED_PDF_TYPES = {"pdf"}


def load_file(file_bytes: bytes, filename: str) -> dict:
    """
    Universal file loader — handles text, PDF, and images.
    Returns dict with text content and metadata.
    """
    filename_lower = filename.lower().strip()
    extension = filename_lower.split(".")[-1] if "." in filename_lower else ""

    # ── PDF ──────────────────────────────────────────────────────────────
    if extension in SUPPORTED_PDF_TYPES:
        return _load_pdf(file_bytes, filename)

    # ── Image ─────────────────────────────────────────────────────────────
    if extension in SUPPORTED_IMAGE_TYPES:
        mime_type = SUPPORTED_IMAGE_TYPES[extension]
        return _load_image(file_bytes, mime_type)

    # ── Plain text ────────────────────────────────────────────────────────
    if extension in {"txt", "text", ""}:
        return _load_text(file_bytes)

    # ── Unknown type ──────────────────────────────────────────────────────
    return {
        "text": "",
        "input_type": "unknown",
        "error": f"Unsupported file type: .{extension}. Supported: PDF, JPG, PNG, TXT",
        "confidence_flags": []
    }


def _load_pdf(file_bytes: bytes, filename: str) -> dict:
    """Extract text from PDF using pypdf"""
    try:
        import pypdf

        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages_text = []

        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages_text.append(page_text.strip())

        if not pages_text:
            return {
                "text": "",
                "input_type": "pdf",
                "error": "No text could be extracted from PDF. It may be a scanned image PDF.",
                "confidence_flags": ["PDF appears to be image-based — text extraction failed"]
            }

        full_text = "\n\n".join(pages_text)

        return {
            "text": full_text,
            "input_type": "pdf",
            "page_count": len(pdf_reader.pages),
            "pages_with_text": len(pages_text),
            "confidence_flags": []
        }

    except Exception as e:
        return {
            "text": "",
            "input_type": "pdf",
            "error": str(e),
            "confidence_flags": [f"PDF loading failed: {str(e)}"]
        }


def _load_image(file_bytes: bytes, mime_type: str) -> dict:
    """Transcribe handwritten image using Groq Vision OCR"""
    result = transcribe_image(file_bytes, mime_type)
    return result


def _load_text(file_bytes: bytes) -> dict:
    """Decode plain text file"""
    try:
        text = file_bytes.decode("utf-8")
        return {
            "text": text.strip(),
            "input_type": "text",
            "confidence_flags": []
        }
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("latin-1")
            return {
                "text": text.strip(),
                "input_type": "text",
                "confidence_flags": []
            }
        except Exception as e:
            return {
                "text": "",
                "input_type": "text",
                "error": str(e),
                "confidence_flags": []
            }