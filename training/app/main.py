import os, mlflow, tempfile, pickle
from fastapi import FastAPI

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT_PATH", "model.pkl")

from model import train_dummy
app = FastAPI(title="Training Server")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@app.get("/health")
def health(): return {"ok": True}

@app.post("/train")
def train():
    model = train_dummy()
    with tempfile.TemporaryDirectory() as td:
        pkl = os.path.join(td, "model.pkl")
        with open(pkl, "wb") as f:
            pickle.dump(model, f)
        with mlflow.start_run(run_name="dummy-train"):
            mlflow.log_artifact(pkl, artifact_path="artifacts")
            # 실제론 S3/MinIO로 올라감 (mlflow+S3 설정 기준)
    return {"status": "trained", "artifact": ARTIFACT_PATH}
