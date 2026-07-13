"""
extractor.py
------------
Handles PDF text extraction using PyMuPDF (fitz).
Falls back to OCR via pytesseract for scanned/image-based PDFs.
"""

import os
import fitz  # PyMuPDF
from typing import Optional

# Optional OCR support
try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class PDFExtractor:
    """
    Extracts text from PDF files.

    Supports:
    - Native text extraction (digital PDFs)
    - OCR fallback for scanned/image-based PDFs
    - Page-range limiting
    """

    def __init__(self, ocr_fallback: bool = True, min_text_threshold: int = 50):
        """
        Args:
            ocr_fallback: Whether to use OCR if native extraction yields little text.
            min_text_threshold: Minimum characters per page to consider native text valid.
        """
        self.ocr_fallback = ocr_fallback and OCR_AVAILABLE
        self.min_text_threshold = min_text_threshold

    def extract(self, pdf_path: str, max_pages: Optional[int] = None) -> dict:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Absolute or relative path to the PDF.
            max_pages: Maximum number of pages to read. None = all pages.

        Returns:
            dict with keys:
                - text (str): Full extracted text.
                - pages (int): Total pages in the document.
                - pages_processed (int): Number of pages actually read.
                - used_ocr (bool): Whether OCR was used.
                - title (str): Document title (if available).
                - author (str): Document author (if available).
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        pages_to_read = min(total_pages, max_pages) if max_pages else total_pages

        # Metadata
        meta = doc.metadata or {}
        title = meta.get("title", "").strip() or os.path.basename(pdf_path)
        author = meta.get("author", "").strip() or "Unknown"

        full_text_parts = []
        used_ocr = False

        for page_num in range(pages_to_read):
            page = doc[page_num]
            page_text = page.get_text("text").strip()

            # If text is too sparse, try OCR
            if len(page_text) < self.min_text_threshold and self.ocr_fallback:
                ocr_text = self._ocr_page(page)
                if ocr_text:
                    page_text = ocr_text
                    used_ocr = True

            if page_text:
                full_text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

        doc.close()

        full_text = "\n\n".join(full_text_parts)

        return {
            "text": full_text,
            "pages": total_pages,
            "pages_processed": pages_to_read,
            "used_ocr": used_ocr,
            "title": title,
            "author": author,
            "file_path": pdf_path,
            "file_name": os.path.basename(pdf_path),
        }

    def _ocr_page(self, page) -> str:
        """
        Render a PDF page as an image and run OCR on it.

        Args:
            page: A fitz.Page object.

        Returns:
            OCR-extracted text string.
        """
        if not OCR_AVAILABLE:
            return ""
        try:
            mat = fitz.Matrix(2, 2)  # 2x scale for better OCR accuracy
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print(f"[OCR Warning] Failed on page: {e}")
            return ""
