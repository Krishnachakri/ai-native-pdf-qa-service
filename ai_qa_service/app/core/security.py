from app.core.config import settings

class FileTooLargeError(Exception):
    def __init__(self, size: int):
        self.size = size
        self.max_size = settings.MAX_FILE_SIZE_BYTES

class InvalidPDFSignatureError(Exception):
    def __init__(self, signature: bytes):
        self.signature = signature

def validate_pdf_file(content: bytes) -> None:
    """
    Validates the size and PDF magic bytes signature of a file's binary content.
    Raises:
        FileTooLargeError: If file exceeds the configured size limit.
        InvalidPDFSignatureError: If magic bytes do not match `%PDF-`.
    """
    # 1. Size Validation
    if len(content) > settings.MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(len(content))
        
    # 2. Magic Bytes Validation
    # Valid PDF file signature starts with "%PDF-" (ASCII) or hex 25 50 44 46 2d
    if not content.startswith(b"%PDF-"):
        # Extract starting bytes for reporting
        sig = content[:5]
        raise InvalidPDFSignatureError(sig)
