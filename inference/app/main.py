import os
import pickle
import mlflow
import torch
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# 로직 및 모델 클래스 import
from .inference_logic import predict_next_step_from_db
from .model import MultivariateLSTM

# --- 환경 변수 및 설정 ---
# (docker-compose.yml에서 주입되지만, 코드 내에서 변수로 사용하기 위해 로드)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MODEL_NAME = os.environ.get("MODEL_NAME", "lstm_financial_forecast_model")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")
DATABASE_URI = os.environ.get("DATABASE_URI") # DB 연결을 위해 추가
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행될 코드
    print("--- Connecting to MLflow and loading model ---")
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        latest_version = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0]
        run_id = latest_version.run_id
        model_uri = f"runs:/{run_id}/lstm_model"
        
        print(f"Loading model from: {model_uri} (Version: {latest_version.version}, Stage: {MODEL_STAGE})")

        # 모델과 아티팩트를 app.state에 저장하여 전역적으로 사용
        app.state.model = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
        
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model_artifacts")
        with open(os.path.join(local_path, "artifacts.pkl"), 'rb') as f:
            app.state.artifacts = pickle.load(f)

        print("--- Model and artifacts loaded successfully! ---")
    except Exception as e:
        print(f"FATAL: Error loading model during startup: {e}")
        raise RuntimeError(f"Failed to load model from MLflow: {e}") from e
    
    yield
    
    # 종료 시 실행될 코드 (필요 시)
    print("--- Shutting down inference server ---")
    app.state.model = None
    app.state.artifacts = None

# --- FastAPI 앱 초기화 (lifespan 적용) ---
app = FastAPI(title="Financial Forecasting Inference API", lifespan=lifespan)

# --- 데이터 유효성 검사를 위한 Pydantic 모델 ---
# DB에서 데이터를 가져오므로 입력이 필요 없음 (단, 예측 기간 등 옵션은 받을 수 있음)
class PredictionOptions(BaseModel):
    months_to_predict: int = Field(1, ge=1, le=12, description="Number of future months to predict")

class PredictionOutput(BaseModel):
    predictions: Dict[str, List[float]]

# --- API 엔드포인트 ---
@app.get("/")
def read_root():
    return {"message": "Inference Server is running."}

@app.post("/predict", response_model=PredictionOutput)
async def predict(request: Request, options: PredictionOptions = PredictionOptions()):
    # app.state에서 모델과 아티팩트를 안전하게 가져옴
    model = request.app.state.model
    artifacts = request.app.state.artifacts
    
    if not model or not artifacts:
        raise HTTPException(status_code=503, detail="Model is not available. Check server startup logs.")
    
    try:
        # DB에서 직접 데이터를 가져와 예측하는 로직 호출
        result = predict_next_step_from_db(
            db_uri=DATABASE_URI,
            model=model, 
            artifacts=artifacts,
            months_to_predict=options.months_to_predict
        )
        return result

    except ValueError as ve:
         raise HTTPException(status_code=400, detail=f"Data processing error: {str(ve)}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")