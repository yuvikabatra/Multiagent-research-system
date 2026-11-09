import hmac
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response


app = FastAPI()

OLLAMA_URL = "http://127.0.0.1:11434"
PROXY_TOKEN = os.getenv("OLLAMA_PROXY_TOKEN")

if not PROXY_TOKEN or len(PROXY_TOKEN) < 32:
    raise RuntimeError("A secure OLLAMA_PROXY_TOKEN is required.")


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def ollama_proxy(path: str, request: Request) -> Response:
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
    async with httpx.AsyncClient(timeout=timeout) as client:
        upstream = await client.request(
            method=request.method,
            url=f"{OLLAMA_URL}/{path}",
            params=request.query_params,
            content=await request.body(),
            headers=headers,
        )

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower()
        not in {"content-encoding", "content-length", "transfer-encoding", "connection"}
    }

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
    )
