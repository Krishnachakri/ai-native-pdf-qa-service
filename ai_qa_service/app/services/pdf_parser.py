import fitz  # PyMuPDF

class EmptyDocumentError(Exception):
    """Exception raised when the PDF has no pages or content."""
    pass

class OCRNotSupportedError(Exception):
    """Exception raised when the PDF contains no extractable text characters (e.g. scanned PDF)."""
    pass

def parse_pdf(file_bytes: bytes) -> list[dict]:
    """
    Parses a PDF from a bytes stream page-by-page.
    
    Args:
        file_bytes: The raw binary data of the PDF file.
        
    Returns:
        list[dict]: A list of dicts containing "page" (1-based index) and "text" (extracted string).
        
    Raises:
        EmptyDocumentError: If the PDF contains zero pages.
        OCRNotSupportedError: If the PDF has pages but zero extractable text characters.
    """
    try:
        # Open PDF from memory stream
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Failed to open PDF file: {str(e)}")
        
    pages_data = []
    total_characters = 0
    
    try:
        # Check if the document has any pages
        if len(doc) == 0:
            raise EmptyDocumentError("The uploaded PDF contains zero pages.")
            
        for page_idx, page in enumerate(doc):
            text = page.get_text("text") or ""
            stripped_text = text.strip()
            total_characters += len(stripped_text)
            
            pages_data.append({
                "page": page_idx + 1,  # 1-based page numbering
                "text": stripped_text
            })
    finally:
        doc.close()
        
    # If the combined text across all pages is completely empty, it's a scanned PDF
    if total_characters == 0:
        raise OCRNotSupportedError(
            "Uploaded PDF contains no extractable text. Please upload a searchable PDF (OCR is not supported by this backend)."
        )
        
    return pages_data
