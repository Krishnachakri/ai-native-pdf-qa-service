import uuid
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from app.core.logging import request_id_ctx

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that assigns a unique Request ID to each incoming request.
    It reads 'X-Request-ID' from incoming headers, or generates a new UUID.
    It binds the ID to the logging context and appends it to response headers.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Retrieve or generate Request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
            
        # Store in logging context variable
        token = request_id_ctx.set(request_id)
        
        # Set in request state for downstream use
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
        except Exception as e:
            # Propagate exception up to standard FastAPI handlers
            raise e
        finally:
            # Always reset contextvar to avoid leaking across concurrent requests on the event loop
            request_id_ctx.reset(token)
            
        # Append request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response
