# training/app/main.py
from fastapi import FastAPI, BackgroundTasks
import os
import mlflow
from .training_logic import run_training

app = FastAPI()

# MLflow는 다음 환경변수들을 자동으로 읽어서 사용:
# - MLFLOW_TRACKING_URI (tracking server 주소)
# - MLFLOW_EXPERIMENT_NAME (실험명, 기본값: "Default")
# - MLFLOW_S3_ENDPOINT_URL (MinIO 엔드포인트)
# - AWS_ACCESS_KEY_ID (MinIO 접근 키)
# - AWS_SECRET_ACCESS_KEY (MinIO 비밀 키)
# - AWS_S3_VERIFY_SSL (SSL 검증 여부)

# DB 환경 변수 설정
DATABASE_URI = os.environ.get("DATABASE_URI")

def background_training_task():
    if not DATABASE_URI:
        print("DATABASE_URI environment variable not set.")
        return
    run_training(db_uri=DATABASE_URI)

@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_training_task)
    return {"message": "Training started in the background."}

@app.get("/")
def read_root():
    return {"message": "Training Server is running."}