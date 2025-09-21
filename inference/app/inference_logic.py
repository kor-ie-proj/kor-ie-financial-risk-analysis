import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, List
from sqlalchemy import create_engine

def preprocess_for_inference(df_raw: pd.DataFrame, final_features: list, available_targets: list) -> pd.DataFrame:
    """
    추론을 위해 입력 데이터를 전처리하고 피처를 생성하는 함수.
    훈련 시 사용한 로직과 동일해야 합니다.
    """
    df = df_raw.copy()
    
    # 'date' 컬럼이 문자열이면 datetime으로 변환
    if df['date'].dtype == 'object':
        df['date'] = pd.to_datetime(df['date'], format='%Y%m')
    df = df.sort_values('date').set_index('date')
    
    # 결측치 처리 (선형 보간)
    df = df.interpolate(method='linear', limit_direction='both')
    
    diff_targets = [f'{col}_diff' for col in available_targets]
    
    # 1차 차분
    for col in available_targets:
        df[f'{col}_diff'] = df[col].diff()
    
    # 피처 엔지니어링 (이동평균, 변화율, 지연 특성)
    for col in available_targets:
        diff_col = f'{col}_diff'
        df[f'{diff_col}_ma3'] = df[diff_col].rolling(window=3, min_periods=1).mean()
        df[f'{diff_col}_ma6'] = df[diff_col].rolling(window=6, min_periods=1).mean()
        df[f'{diff_col}_pct_change'] = df[diff_col].pct_change().fillna(0)
        for lag in [1, 3, 6]:
            df[f'{diff_col}_lag{lag}'] = df[diff_col].shift(lag)
    
    for col in available_targets:
        for lag in [1, 3, 6]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
            
    # 결측치 제거
    df = df.dropna()
    
    # 훈련 시 사용된 최종 피처들이 존재하는지 확인
    missing_features = [f for f in final_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Required features are missing from the data after preprocessing: {missing_features}")
        
    return df[final_features]

def predict_next_step(
    db_uri: str, 
    model: torch.nn.Module, 
    artifacts: Dict[str, Any], 
    months_to_predict: int
) -> Dict[str, Any]:
    """
    DB에서 데이터를 로드하고, n개월 후를 예측하는 메인 함수
    """
    device = next(model.parameters()).device
    seq_length = artifacts['hyperparameters']['seq_length']
    
    # 1. DB에서 예측에 필요한 최신 데이터 로드
    # seq_length에 피처 엔지니어링 시 발생하는 결측치(최대 lag=6)를 고려하여 넉넉하게 데이터를 가져옵니다.
    required_rows = seq_length + 10 
    try:
        engine = create_engine(db_uri)
        query = f"SELECT * FROM ecos_data ORDER BY date DESC LIMIT {required_rows}"
        df_raw = pd.read_sql(query, engine)
        # 시간 순서를 맞추기 위해 다시 정렬
        df_raw = df_raw.sort_values('date', ascending=True).reset_index(drop=True)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to the database or query data: {e}")

    if len(df_raw) < seq_length:
        raise ValueError(f"Not enough data in DB. Required at least {seq_length} rows, but got {len(df_raw)}.")
        
    model.eval()

    # 2. 다중 스텝 예측(Multi-step Forecasting) 로직
    final_predictions = []
    current_df = df_raw.copy()

    for _ in range(months_to_predict):
        # 2-1. 현재 데이터프레임으로 전처리 수행
        X_processed = preprocess_for_inference(
            current_df, 
            artifacts['final_features'], 
            artifacts['target_columns']
        )
        
        # 2-2. 마지막 시퀀스 준비
        last_sequence = X_processed.iloc[-seq_length:].values
        last_sequence_scaled = artifacts['scaler_X'].transform(last_sequence)
        X_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(device)

        # 2-3. 1스텝 예측
        with torch.no_grad():
            prediction_scaled = model(X_tensor)
        
        # 2-4. 예측값을 차분값으로 역정규화
        prediction_diff = artifacts['scaler_y'].inverse_transform(prediction_scaled.cpu().numpy())[0]
        
        # 2-5. 차분값을 원본 값으로 복원
        last_original_values = current_df[artifacts['target_columns']].iloc[-1].values
        predicted_values = last_original_values + prediction_diff
        final_predictions.append(predicted_values)

        # 2-6. 다음 예측을 위해 예측값을 입력 데이터에 추가
        # 다음 달의 date 생성
        next_date = (pd.to_datetime(current_df['date'].iloc[-1], format='%Y%m') + pd.DateOffset(months=1)).strftime('%Y%m')
        
        # 예측된 값을 포함하는 새로운 행(row) 생성
        new_row = {'date': next_date}
        for i, col in enumerate(artifacts['target_columns']):
            new_row[col] = predicted_values[i]
        
        # current_df에 새로운 행 추가하여 다음 반복에 사용
        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

    # 3. 최종 결과 포맷팅
    final_predictions = np.array(final_predictions)
    response = {
        target: final_predictions[:, i].tolist() 
        for i, target in enumerate(artifacts['target_columns'])
    }
    
    return {"predictions": response}