# be/app/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
import httpx
import os
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime
from pydantic import BaseModel

app = FastAPI()

INFERENCE_SERVER_URL = os.environ.get("INFERENCE_SERVER_URL")
DATABASE_URI = os.environ.get("DATABASE_URI")
engine = create_engine(DATABASE_URI)

class PredictResponse(BaseModel):
    monthly_predictions: dict
    quarterly_predictions: dict

def save_predictions_to_db(predictions: dict):
    """예측 결과를 model_output 테이블에 저장 (UPSERT)"""
    try:
        df_to_save = pd.DataFrame.from_dict(predictions, orient='index')
        df_to_save.index = pd.to_datetime(df_to_save.index).strftime('%Y%m')
        df_to_save.reset_index(inplace=True)
        df_to_save.rename(columns={'index': 'date'}, inplace=True)
        
        with engine.connect() as conn:
            for _, row in df_to_save.iterrows():
                # 간단한 UPSERT 로직 (MySQL 8.0+ 기준)
                insert_stmt = f"""
                INSERT INTO model_output (date, {', '.join(row.index[1:])})
                VALUES ('{row['date']}', {', '.join(['%s'] * len(row.values[1:]))})
                ON DUPLICATE KEY UPDATE
                {', '.join([f'{col}=VALUES({col})' for col in row.index[1:]])}
                """
                conn.execute(insert_stmt, tuple(row.values[1:]))
            conn.commit()
        print("Successfully saved predictions to DB.")
    except Exception as e:
        print(f"Error saving predictions to DB: {e}")

@app.get("/predict", response_model=PredictResponse)
async def get_financial_prediction(background_tasks: BackgroundTasks):
    if not DATABASE_URI or not INFERENCE_SERVER_URL:
        raise HTTPException(status_code=500, detail="Server configuration error.")

    try:
        # 1. DB에서 예측에 필요한 충분한 Raw Data 조회 (최근 40개월)
        query = "SELECT * FROM ecos_data ORDER BY date DESC LIMIT 40"
        raw_df = pd.read_sql(query, engine)
        raw_df_sorted = raw_df.sort_values('date').reset_index(drop=True)
        
        # 2. Inference 서버에 요청
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                INFERENCE_SERVER_URL,
                json={"raw_data": raw_df_sorted.to_dict('records')}
            )
        response.raise_for_status()
        predictions = response.json()
        
        # 3. DB에 예측 결과 비동기 저장
        background_tasks.add_task(save_predictions_to_db, predictions['monthly_predictions'])
        
        return predictions

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred in BE server: {e}")