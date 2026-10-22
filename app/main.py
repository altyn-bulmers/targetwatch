import os
from pathlib import Path

import psycopg
from fastapi import FastAPI
from fastapi.responses import FileResponse

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="TargetWatch")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    url = os.environ.get("DATABASE_URL")
    if not url:
        return {"status": "ok", "database": "not configured"}
    try:
        async with await psycopg.AsyncConnection.connect(url, connect_timeout=5) as conn:
            row = await (await conn.execute("select version()")).fetchone()
        return {"status": "ok", "database": "ok: " + row[0].split(" on ")[0]}
    except Exception as e:
        return {"status": "ok", "database": f"error: {e}"}
