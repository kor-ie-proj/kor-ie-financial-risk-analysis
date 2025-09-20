# training/app/main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import os
import mlflow
from .training_logic import run_training

app = FastAPI()

# MLflow 설정
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "LSTM_Financial_Forecast")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

class TrainRequest(BaseModel):
    table_name: str = "ecos_data"

# DB URI는 보안을 위해 환경 변수에서 직접 가져옴
DATABASE_URI = os.environ.get("DATABASE_URI")

def background_training_task(table_name: str):
    if not DATABASE_URI:
        print("DATABASE_URI environment variable not set.")
        return
    print(f"Starting training for table: {table_name}")
    run_training(db_uri=DATABASE_URI, table_name=table_name)

@app.post("/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """모델 학습을 백그라운드에서 시작합니다."""
    background_tasks.add_task(background_training_task, request.table_name)
    return {"message": "Model training started in the background. Check MLflow UI for progress."}

@app.get("/")
def read_root():
    return {"message": "Training Server is running."}