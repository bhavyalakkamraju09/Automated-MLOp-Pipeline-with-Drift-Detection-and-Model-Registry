"""ASGI middleware that times every request and mirrors the latency into
the shared Prometheus histogram, independent of the /predict handler's
own instrumentation (covers /health and any future routes for free)."""
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.monitoring.metrics import LATENCY_HISTOGRAM


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        if request.url.path == "/predict":
            LATENCY_HISTOGRAM.observe(time.monotonic() - start)
        return response
