# inference_server/predict.py

import torch
import numpy as np
import pandas as pd

def predict_future(model, X_latest, scaler_X, scaler_y, df_original, artifacts, n_months=3):
    model.eval()
    
    # 아티팩트에서 필요 정보 추출
    seq_length = artifacts['hyperparameters']['seq_length']
    target_columns = artifacts['target_columns']
    device = next(model.parameters()).device

    # 마지막 시퀀스 준비
    last_sequence = X_latest[-seq_length:].copy()
    last_sequence_scaled = scaler_X.transform(last_sequence)
    
    with torch.no_grad():
        # 1-step 미래 차분값 예측
        X_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(device)
        pred_scaled = model(X_tensor).cpu().numpy()
        pred_diff = scaler_y.inverse_transform(pred_scaled)[0]
        
        # 차분값을 원본 값으로 복원
        last_original_values = df_original[target_columns].iloc[-1].values
        
        predictions_original = []
        current_values = last_original_values.copy()
        
        for _ in range(n_months):
            next_values = current_values + pred_diff
            predictions_original.append(next_values.copy())
            current_values = next_values # 다음 예측을 위해 현재 값 업데이트
            
    return np.array(predictions_original)

def format_predictions(predictions, last_date, target_columns):
    # 월별 예측 결과
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=len(predictions), freq='MS')
    monthly_df = pd.DataFrame(predictions, columns=target_columns, index=future_dates)
    
    # 분기별 평균 계산
    quarterly_df = monthly_df.resample('Q').mean()
    quarterly_df.index = [f"{d.year}Q{d.quarter}" for d in quarterly_df.index]
    
    return monthly_df.to_dict('index'), quarterly_df.to_dict('index')