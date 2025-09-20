# training/app/training_logic.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import optuna
import mlflow
import mlflow.pytorch
import warnings
from sqlalchemy import create_engine

from .model import MultivariateLSTM

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 데이터 전처리 및 피처 엔지니어링 ---
def preprocess_and_feature_engineer(df_raw: pd.DataFrame):
    df = df_raw.copy()
    # ECOS date는 YYYYMM 형식이므로 datetime으로 변환
    df['date'] = pd.to_datetime(df['date'], format='%Y%m')
    df = df.sort_values('date').set_index('date')
    
    target_columns = ['construction_bsi_actual', 'base_rate', 'housing_sale_price', 'm2_growth', 'credit_spread']
    available_targets = [col for col in target_columns if col in df.columns]
    
    df = df.interpolate(method='linear', limit_direction='both')

    # 1차 차분
    for col in available_targets:
        df[f'{col}_diff'] = df[col].diff()
    diff_targets = [f'{col}_diff' for col in available_targets]
    
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
            
    df = df.dropna()

    all_features = [col for col in df.columns if col not in available_targets and col not in diff_targets]
    correlation_matrix = df[all_features].corr()
    
    # 높은 상관관계 피처 제거
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) > 0.95)]
    selected_features = [col for col in all_features if col not in to_drop]

    # 타겟과의 상관관계 기반 최종 피처 선택
    target_correlations = []
    for target in diff_targets:
        if target in df.columns:
            corr_with_target = df[selected_features + [target]].corr()[target].abs().sort_values(ascending=False)
            target_correlations.append(corr_with_target[:-1])

    avg_correlation = pd.concat(target_correlations, axis=1).mean(axis=1).sort_values(ascending=False)
    n_features = max(20, int(len(selected_features) * 0.5))
    final_features = avg_correlation.head(n_features).index.tolist()

    X = df[final_features].values
    y = df[diff_targets].values
    
    return X, y, final_features, available_targets, diff_targets

# --- 데이터 준비 (시퀀스 생성, 분할, 스케일링) ---
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

def prepare_data_for_training(X, y, seq_length, batch_size):
    X_seq, y_seq = create_sequences(X, y, seq_length)
    
    total_samples = len(X_seq)
    train_size = int(total_samples * 0.8)
    
    X_train, X_val = X_seq[:train_size], X_seq[train_size:]
    y_train, y_val = y_seq[:train_size], y_seq[train_size:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # 중요: 스케일러는 train 데이터로만 fit 해야 함
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler_X.fit(X_train_flat)
    scaler_y.fit(y_train)

    X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train.shape)
    y_train_scaled = scaler_y.transform(y_train)
    
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    y_val_scaled = scaler_y.transform(y_val)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled).to(device), torch.FloatTensor(y_train_scaled).to(device)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_scaled).to(device), torch.FloatTensor(y_val_scaled).to(device)), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler_X, scaler_y

# --- Optuna Objective ---
def objective(trial, X, y):
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 96, 128]),
        'num_layers': trial.suggest_int('num_layers', 1, 2),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.4),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
        'seq_length': trial.suggest_categorical('seq_length', [12, 24]),
        'optimizer_type': trial.suggest_categorical('optimizer_type', ['Adam', 'AdamW']),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
    }

    train_loader, val_loader, _, _ = prepare_data_for_training(X, y, params['seq_length'], params['batch_size'])
    
    model = MultivariateLSTM(X.shape[1], params['hidden_size'], params['num_layers'], y.shape[1], params['dropout_rate']).to(device)
    optimizer = getattr(optim, params['optimizer_type'])(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    criterion = nn.HuberLoss()

    for epoch in range(30):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item()
    return val_loss / len(val_loader)


# --- 메인 학습 함수 ---
def run_training(db_uri: str, table_name: str):
    with mlflow.start_run() as run:
        try:
            # 1. 데이터 로드 및 전처리
            engine = create_engine(db_uri)
            raw_df = pd.read_sql(f"SELECT * FROM {table_name} ORDER BY date ASC", engine)
            X, y, final_features, available_targets, _ = preprocess_and_feature_engineer(raw_df)
            
            # 2. 하이퍼파라미터 튜닝
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, X, y), n_trials=30, timeout=600)
            best_params = study.best_params
            mlflow.log_params(best_params)

            # 3. 최종 모델 학습
            train_loader, _, scaler_X, scaler_y = prepare_data_for_training(X, y, best_params['seq_length'], best_params['batch_size'])
            
            final_model = MultivariateLSTM(X.shape[1], best_params['hidden_size'], best_params['num_layers'], y.shape[1], best_params['dropout_rate']).to(device)
            optimizer = getattr(optim, best_params['optimizer_type'])(final_model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
            criterion = nn.HuberLoss()

            for epoch in range(150):
                final_model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = final_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # 4. 아티팩트 로깅
            artifacts = {
                "model_state_dict": {k: v.cpu() for k, v in final_model.state_dict().items()}, # CPU로 변환
                "hyperparameters": best_params,
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "final_features": final_features,
                "target_columns": available_targets
            }
            mlflow.pytorch.log_model(pytorch_model=final_model.cpu(), artifact_path="lstm_model", pickle_module="cloudpickle", extra_files={"artifacts.pth": artifacts})
            print(f"Training successful. Run ID: {run.info.run_id}")
            return {"status": "success", "run_id": run.info.run_id}
        except Exception as e:
            print(f"Training failed: {e}")
            mlflow.log_param("error", str(e))
            raise e