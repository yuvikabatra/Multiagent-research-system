import hmac
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


app = FastAPI()

OLLAMA_URL = "http://127.0.0.1:11434"
PROXY_TOKEN = os.getenv("OLLAMA_PROXY_TOKEN")

if not PROXY_TOKEN or len(PROXY_TOKEN) < 32:
    raise RuntimeError("A secure OLLAMA_PROXY_TOKEN is required.")


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def ollama_proxy(path: str, request: Request) -> StreamingResponse:
    supplied_token = request.headers.get("authorization", "")
    expected_token = f"Bearer {PROXY_TOKEN}"

    if not hmac.compare_digest(supplied_token, expected_token):
        raise HTTPException(status_code=401, detail="Unauthorized")

    blocked_headers = {"host", "authorization", "content-length", "connection"}
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in blocked_headers
    }

    timeout = httpx.Timeout(600.0, connect=10.0)
    client = httpx.AsyncClient(timeout=timeout)
    try:
        upstream_request = client.build_request(
            method=request.method,
            url=f"{OLLAMA_URL}/{path}",
            params=request.query_params,
            content=await request.body(),
            headers=headers,
        )
        upstream = await client.send(upstream_request, stream=True)
    except Exception:
        await client.aclose()
        raise

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in {"transfer-encoding", "connection"}
    }

    async def stream_response():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        content=stream_response(),
        status_code=upstream.status_code,
        headers=response_headers,
    )
