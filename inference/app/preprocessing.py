# inference_server/preprocessing.py

import pandas as pd
import numpy as np

def preprocess_for_inference(df_raw: pd.DataFrame, final_features: list, target_columns: list):
    df = df_raw.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')

    available_targets = [col for col in target_columns if col in df.columns]
    
    df = df.interpolate(method='linear', limit_direction='both')

    # 학습 시 사용된 피처 엔지니어링과 동일하게 적용
    for col in available_targets:
        df[f'{col}_diff'] = df[col].diff()

    diff_targets = [f'{col}_diff' for col in available_targets]
    
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
            
    # 학습에 사용된 피처만 선택
    # 결측치 제거 전에 필요한 모든 피처가 생성되었는지 확인
    # 일부 피처는 dropna 이후에 사라질 수 있으므로, dropna 전에 필요한 피처를 모두 생성
    df_processed = df.dropna(subset=final_features)
    
    X = df_processed[final_features].values
    
    return X, df_processed