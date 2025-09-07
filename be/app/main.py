import os, httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from db import SessionLocal

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8000")
app = FastAPI(title="BE Server")

class PredictIn(BaseModel):
    corp_code: str
    features: dict

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/info/{corp_code}")
def info(corp_code: str):
    with SessionLocal() as s:
        row = s.execute(
            "SELECT corp_code, corp_name FROM companies WHERE corp_code=%s",
            (corp_code,)
        ).first()
    return {"corp_code": corp_code, "corp_name": row[1] if row else None}

@app.post("/predict")
def predict(body: PredictIn):
    try:
        r = httpx.post(f"{INFERENCE_URL}/predict", json=body.dict(), timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Inference error: {e}")
