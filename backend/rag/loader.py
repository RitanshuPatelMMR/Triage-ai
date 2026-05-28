import io
from tools.ocr import transcribe_image
from tools.errors import user_safe_error


SUPPORTED_IMAGE_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}

SUPPORTED_PDF_TYPES = {"pdf"}

OCR_MAX_PAGES = 3


def _pdf_total_pages(file_bytes: bytes) -> int | None:
    try:
        import pypdf
        return len(pypdf.PdfReader(io.BytesIO(file_bytes)).pages)
    except Exception:
        return None


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
    """Extract text from PDF using pypdf, fallback to OCR for scanned PDFs"""
    try:
        import pypdf

        pdf_stream = io.BytesIO(file_bytes)  # Bug A fix: pin reference
        pdf_reader = pypdf.PdfReader(pdf_stream)
        pages_text = []

        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages_text.append(page_text.strip())

        # Bug B fix: detect junk/empty extraction → fallback to OCR
        full_text = "\n\n".join(pages_text)
        if _is_meaningful_text(full_text):
            flags = []
            total = len(pdf_reader.pages)
            if total > OCR_MAX_PAGES:
                flags.append(
                    f"PDF has {total} pages — all pages were read from embedded text."
                )
            return {
                "text": full_text,
                "input_type": "pdf",
                "page_count": total,
                "pages_with_text": len(pages_text),
                "confidence_flags": flags
            }

        # Fallback: render PDF as image → Groq Vision OCR
        return _load_pdf_via_ocr(file_bytes)

    except Exception as e:
        # Last resort: try OCR anyway
        try:
            return _load_pdf_via_ocr(file_bytes)
        except Exception:
            return {
                "text": "",
                "input_type": "pdf",
                "error": user_safe_error(e),
                "confidence_flags": [f"PDF loading failed: {user_safe_error(e)}"]
            }


def _is_meaningful_text(text: str) -> bool:
    """Check if extracted text is real content vs junk metadata"""
    if not text or len(text.strip()) < 50:
        return False
    # Junk signals: URLs, pixel dimensions, image filenames
    junk_signals = ["http", ".jpg", ".png", "pixels", "Page 1 of 1"]
    junk_count = sum(1 for s in junk_signals if s in text)
    return junk_count < 2


def _load_pdf_via_ocr(file_bytes: bytes) -> dict:
    """Render PDF pages as images → Groq Vision OCR"""
    try:
        from pdf2image import convert_from_bytes
        import io as _io

        total_pages = _pdf_total_pages(file_bytes)
        pages = convert_from_bytes(
            file_bytes, dpi=200, first_page=1, last_page=OCR_MAX_PAGES
        )

        all_texts = []
        all_flags = []
        if total_pages and total_pages > OCR_MAX_PAGES:
            all_flags.append(
                f"Only first {OCR_MAX_PAGES} of {total_pages} pages were analyzed "
                "(scanned PDF limit)."
            )

        for i, page_img in enumerate(pages):
            # Convert PIL image → bytes
            img_buffer = _io.BytesIO()
            page_img.save(img_buffer, format="JPEG", quality=90)
            img_bytes = img_buffer.getvalue()

            result = transcribe_image(img_bytes, "image/jpeg")

            if result.get("text", "").strip():
                all_texts.append(result["text"])
            all_flags.extend(result.get("confidence_flags", []))

        if not all_texts:
            return {
                "text": "",
                "input_type": "pdf_ocr",
                "error": "OCR could not extract text from PDF pages",
                "confidence_flags": ["PDF OCR failed — no text found"]
            }

        return {
            "text": "\n\n".join(all_texts),
            "input_type": "pdf_ocr",
            "page_count": len(pages),
            "confidence_flags": all_flags or ["PDF processed via OCR — verify before clinical use"]
        }

    except Exception as e:
        safe = user_safe_error(e)
        return {
            "text": "",
            "input_type": "pdf_ocr",
            "error": safe,
            "confidence_flags": [f"PDF OCR failed: {safe}"]
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