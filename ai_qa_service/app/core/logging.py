import contextvars
import json
import logging
import sys
from typing import Any

# Context variables for request tracing
request_id_ctx = contextvars.ContextVar("request_id", default="")
document_id_ctx = contextvars.ContextVar("document_id", default="")

class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Get request context
        req_id = request_id_ctx.get()
        doc_id = document_id_ctx.get()
        
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": req_id,
            "document_id": doc_id,
        }
        
        # Include extra fields if passed (e.g. latency, error codes)
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_record.update(record.extra_fields)
            
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

def setup_logging():
    # Setup our custom logger
    app_logger = logging.getLogger("ai_qa_service")
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False  # Avoid propagating to root to prevent duplicate logs
    
    if not app_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = StructuredFormatter(datefmt="%Y-%m-%dT%H:%M:%SZ")
        handler.setFormatter(formatter)
        app_logger.addHandler(handler)
        
    # Also configure root logger
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_handler = logging.StreamHandler(sys.stdout)
    root_handler.setFormatter(StructuredFormatter(datefmt="%Y-%m-%dT%H:%M:%SZ"))
    root_logger.addHandler(root_handler)
    root_logger.setLevel(logging.INFO)

logger = logging.getLogger("ai_qa_service")
