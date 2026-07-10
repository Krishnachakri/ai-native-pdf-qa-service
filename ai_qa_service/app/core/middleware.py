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

        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())


        token = request_id_ctx.set(request_id)


        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception as e:

            raise e
        finally:

            request_id_ctx.reset(token)


        response.headers["X-Request-ID"] = request_id
        return response
