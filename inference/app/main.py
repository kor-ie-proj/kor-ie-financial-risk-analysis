from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import text
from db import engine
from model import load_model

app = FastAPI(title="Inference Server")
_eng = engine()
_model = load_model()

class PredictIn(BaseModel):
    corp_code: str
    features: dict

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(body: PredictIn):
    fs_pred = _model.predict_fs(body.features)
    score = _model.risk_score(fs_pred, body.features)
    with _eng.begin() as conn:
        conn.execute(
            text("INSERT INTO predictions (corp_code, predicted_roe, risk_score) VALUES (:c,:r,:s)"),
            {"c": body.corp_code, "r": fs_pred["next_quarter_roe"], "s": score},
        )
    return {
        "corp_code": body.corp_code,
        "fs_prediction": fs_pred,
        "final_risk_score": score,
        "explain": "prototype score using ccsi/rate/leverage"
    }
