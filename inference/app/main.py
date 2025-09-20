# inference/app/main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import torch
import mlflow
import os
import cloudpickle

from .schemas import PredictionInput, PredictionOutput
from .model import MultivariateLSTM
from .preprocessing import preprocess_for_inference
from .predict import predict_future, format_predictions

ml_models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    model_name = os.environ.get("MLFLOW_MODEL_NAME", "lstm_model")
    model_version = os.environ.get("MLFLOW_MODEL_VERSION", "latest")
    model_uri = f"models:/{model_name}/{model_version}"
    
    print(f"Loading model from: {model_uri}")
    try:
        local_path = mlflow.artifacts.download_artifacts(model_uri)
        artifacts = torch.load(os.path.join(local_path, "artifacts.pth"), map_location=device)
        
        hyperparams = artifacts['hyperparameters']
        input_size = len(artifacts['final_features'])
        output_size = len(artifacts['target_columns'])

        model = MultivariateLSTM(
            input_size, hyperparams['hidden_size'], hyperparams['num_layers'], output_size, hyperparams['dropout_rate']
        ).to(device)
        model.load_state_dict(artifacts['model_state_dict'])
        
        ml_models['model'] = model
        ml_models['artifacts'] = artifacts
        print("Model and artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if 'model' not in ml_models:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        raw_df = pd.DataFrame(input_data.raw_data)
        
        model = ml_models['model']
        artifacts = ml_models['artifacts']
        
        X, df_processed = preprocess_for_inference(
            raw_df, artifacts['final_features'], artifacts['target_columns']
        )
        
        seq_len = artifacts['hyperparameters']['seq_length']
        if len(X) < seq_len:
            raise HTTPException(status_code=400, detail=f"Insufficient data. Need at least {seq_len} data points after preprocessing.")

        predictions = predict_future(
            model, X, artifacts['scaler_X'], artifacts['scaler_y'], df_processed, artifacts
        )
        
        last_date = df_processed.index[-1]
        monthly_preds, quarterly_preds = format_predictions(
            predictions, last_date, artifacts['target_columns']
        )
        
        return {"monthly_predictions": monthly_preds, "quarterly_predictions": quarterly_preds}
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/")
def read_root():
    return {"message": "Inference Server is running."}