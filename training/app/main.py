import os
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# training_logic.py의 run_training 함수를 import
from .training_logic import run_training

# --- 환경 변수 로드 ---
DATABASE_URI = os.environ.get("DATABASE_URI")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME")

# --- 상태 관리를 위한 전역 변수 ---
# 참고: 이 방식은 단일 워커 환경에서 유효. Gunicorn 등으로 여러 워커를 사용할 경우, 상태 공유를 위해 Redis나 데이터베이스 같은 외부 저장소가 필요.
training_status = {
    "status": "idle",  # idle, running, success, failed
    "last_run_id": None,
    "start_time": None,
    "end_time": None,
    "error_message": None
}

# --- API 응답 모델 정의 (Pydantic) ---
class TrainingStatus(BaseModel):
    status: str = Field(..., example="idle")
    last_run_id: Optional[str] = Field(None, example="a1b2c3d4e5f6")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

class ServerConfig(BaseModel):
    mlflow_tracking_uri: Optional[str]
    mlflow_experiment_name: Optional[str]
    database_uri_is_set: bool

# --- 백그라운드 작업을 위한 함수 ---
def background_training_task():
    """
    실제 훈련 로직을 실행하고 전역 상태를 업데이트하는 백그라운드 작업.
    """
    if not DATABASE_URI:
        print("DATABASE_URI environment variable not set.")
        training_status["status"] = "failed"
        training_status["error_message"] = "DATABASE_URI environment variable not set."
        training_status["end_time"] = datetime.utcnow()
        return

    # 훈련 시작 전 상태 업데이트
    training_status["status"] = "running"
    training_status["start_time"] = datetime.utcnow()
    training_status["end_time"] = None
    training_status["last_run_id"] = None
    training_status["error_message"] = None

    try:
        # 훈련 로직 실행
        result = run_training(db_uri=DATABASE_URI)
        
        # 성공 시 상태 업데이트
        training_status["status"] = "success"
        training_status["last_run_id"] = result.get("run_id")

    except Exception as e:
        # 실패 시 상태 업데이트
        print(f"An error occurred during training: {e}")
        training_status["status"] = "failed"
        training_status["error_message"] = str(e)
    
    finally:
        # 훈련 종료 시간 기록
        training_status["end_time"] = datetime.utcnow()


# --- FastAPI 앱 초기화 ---
app = FastAPI(
    title="Financial Forecasting Training API",
    description="API to trigger, monitor, and configure the model training process."
)

# --- API 엔드포인트 ---
@app.post("/train", status_code=202)
async def train_model_endpoint(background_tasks: BackgroundTasks):
    if training_status["status"] == "running":
        raise HTTPException(status_code=409, detail="A training process is already running.")
    
    background_tasks.add_task(background_training_task)
    return {"message": "Training started in the background. Check /status for progress."}

@app.get("/status", response_model=TrainingStatus)
def get_training_status():
    return training_status

@app.get("/config", response_model=ServerConfig)
def get_server_config():
    return {
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "mlflow_experiment_name": MLFLOW_EXPERIMENT_NAME,
        "database_uri_is_set": bool(DATABASE_URI)
    }

@app.get("/")
def read_root():
    return {"message": "Training Server is running."}