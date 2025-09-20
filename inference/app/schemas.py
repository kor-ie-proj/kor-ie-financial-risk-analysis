# inference_server/schemas.py

from pydantic import BaseModel
from typing import List, Dict, Any

class PredictionInput(BaseModel):
    # DB에서 받은 Raw Data를 리스트 형태로 가정
    # 각 딕셔너리는 한 시점의 데이터를 의미
    raw_data: List[Dict[str, Any]]

class PredictionOutput(BaseModel):
    # 월별 예측 결과를 딕셔너리 형태로 반환
    monthly_predictions: Dict[str, Any]
    # 분기별 예측 결과를 딕셔너리 형태로 반환
    quarterly_predictions: Dict[str, Any]