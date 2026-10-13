"""PDF Document Extractor using PyMuPDF (fitz) with OCR architecture."""
import logging
import shutil
from typing import List, Tuple
from pathlib import Path
import pymupdf as fitz

from models.schemas import DocumentBlock
from config.settings import OCR_MIN_CHAR_COUNT

logger = logging.getLogger(__name__)

# Check Tesseract availability once at module level
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image
    import io
    # Test PATH first, then standard Windows installation locations. This
    # keeps OCR working when Streamlit was started before Tesseract was added
    # to the user's PATH.
    try:
        tesseract_candidates = [
            shutil.which("tesseract"),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for candidate in tesseract_candidates:
            if candidate and Path(candidate).exists():
                pytesseract.pytesseract.tesseract_cmd = candidate
                break
        pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
    except Exception:
        TESSERACT_AVAILABLE = False
except ImportError:
    TESSERACT_AVAILABLE = False


def extract_pdf(
    file_bytes: bytes,
    file_name: str,
    document_id: str
) -> Tuple[List[DocumentBlock], bool]:
    """Extract structured document blocks from a PDF file.

    Returns:
        Tuple of (blocks, requires_ocr_flag)
    """
    blocks: List[DocumentBlock] = []
    requires_ocr = False
    order_counter = 0

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        logger.error(f"Failed to open PDF stream for {file_name}: {e}")
        raise ValueError(f"Corrupted or invalid PDF file: {e}")

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        page_text = page.get_text("text").strip()

        # Check if page has sufficient text or appears scanned
        if len(page_text) < OCR_MIN_CHAR_COUNT:
            requires_ocr = True
            ocr_text = ""
            
            if TESSERACT_AVAILABLE:
                try:
                    pix = page.get_pixmap(dpi=150)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img).strip()
                except Exception as ocr_err:
                    logger.warning(f"OCR execution failed on page {page_num} of {file_name}: {ocr_err}")
                    ocr_text = ""

            if ocr_text:
                blocks.append(
                    DocumentBlock(
                        document_id=document_id,
                        file_name=file_name,
                        file_type="pdf",
                        location=f"page_{page_num}_ocr",
                        block_type="page_ocr",
                        text=ocr_text,
                        order=order_counter,
                        metadata={"page": page_num, "source": "tesseract_ocr"}
                    )
                )
                order_counter += 1
            else:
                placeholder_text = f"[SCANNED_PAGE_REQUIRES_OCR: Page {page_num} contains minimal or no extractable text]"
                blocks.append(
                    DocumentBlock(
                        document_id=document_id,
                        file_name=file_name,
                        file_type="pdf",
                        location=f"page_{page_num}",
                        block_type="scanned_page",
                        text=placeholder_text,
                        order=order_counter,
                        metadata={"page": page_num, "requires_ocr": True}
                    )
                )
                order_counter += 1
        else:
            # Extract structured blocks from the page
            page_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
            if page_blocks:
                for b_idx, b in enumerate(page_blocks):
                    b_text = b[4].strip() if len(b) > 4 else ""
                    if not b_text:
                        continue
                    blocks.append(
                        DocumentBlock(
                            document_id=document_id,
                            file_name=file_name,
                            file_type="pdf",
                            location=f"page_{page_num}_block_{b_idx + 1}",
                            block_type="paragraph",
                            text=b_text,
                            order=order_counter,
                            metadata={"page": page_num, "block_index": b_idx}
                        )
                    )
                    order_counter += 1
            else:
                blocks.append(
                    DocumentBlock(
                        document_id=document_id,
                        file_name=file_name,
                        file_type="pdf",
                        location=f"page_{page_num}",
                        block_type="page",
                        text=page_text,
                        order=order_counter,
                        metadata={"page": page_num}
                    )
                )
                order_counter += 1

    doc.close()
    return blocks, requires_ocr
