from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.background import BackgroundTask
from logging.handlers import RotatingFileHandler

import logging
import httpx
import time
import uuid
import json

metrics_log = logging.getLogger("ollama.metrics")
metrics_log.setLevel(logging.INFO)
metrics_log.propagate = False  # keep it out of service.log
_h = RotatingFileHandler("src/logs/requests.jsonl", maxBytes=10_000_000, backupCount=5)
_h.setFormatter(logging.Formatter("%(message)s"))
metrics_log.addHandler(_h)

proxy_router = APIRouter(prefix="/ollama")

HOP_BY_HOP = {"host", "content-length", "connection", "transfer-encoding",
              "accept-encoding"}

def per_sec(count, dur_ns):
    return round(count / dur_ns * 1e9, 2) if count and dur_ns else None

def extract_metrics(final: dict) -> dict:
    return {
        "total_s": round(final.get("total_duration", 0) / 1e9, 3),
        "load_s": round(final.get("load_duration", 0) / 1e9, 3),
        "prompt_tokens": final.get("prompt_eval_count"),
        "gen_tokens": final.get("eval_count"),
        "prompt_tps": per_sec(final.get("prompt_eval_count"), final.get("prompt_eval_duration")),
        "gen_tps": per_sec(final.get("eval_count"), final.get("eval_duration")),
    }

async def relay_and_measure(upstream, path: str, start: float, model_hint=None):
    request_id = uuid.uuid4().hex
    ttft, buf, final = None, b"", None
    parse = path in ("api/chat", "api/generate")
    try:
        async for chunk in upstream.aiter_raw():
            if ttft is None:
                ttft = time.perf_counter() - start
            if parse:
                buf += chunk
                *lines, buf = buf.split(b"\n")   # chunks can split a line in half
                for line in lines:
                    final = _final_or(line, final)
            yield chunk                           # forward bytes untouched
        if parse and buf:                         # non-streaming reply has no trailing newline
            final = _final_or(buf, final)
    finally:
        record = {
            "request_id": request_id,
            "ts": time.time(),
            "path": path,
            "status": upstream.status_code,
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "latency_s": round(time.perf_counter() - start, 3),
            **(extract_metrics(final) if final else {}),
            "model": final.get("model") if final else None,
            "completed": final is not None,
        }
        metrics_log.info(json.dumps(record))
        await upstream.aclose()

def _final_or(line: bytes, current):
    try:
        obj = json.loads(line)
    except ValueError:
        return current
    return obj if obj.get("done") else current

@proxy_router.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
async def forward(path: str, request: Request):
    start = time.perf_counter() # start time counter
    client: httpx.AsyncClient = request.app.state.http
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    upstream_req = client.build_request(
        request.method, f"/{path}",
        params=request.query_params,
        headers=headers,
        content=await request.body(),
    )
    try:
        upstream = await client.send(upstream_req, stream=True)
    except httpx.ConnectError:
        return JSONResponse({"detail": "Ollama unreachable"}, status_code=502)

    return StreamingResponse(
        relay_and_measure(upstream, path, start),
        status_code=upstream.status_code,
        headers={k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP},
        background=BackgroundTask(upstream.aclose),
    )