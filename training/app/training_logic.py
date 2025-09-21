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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import tempfile
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

from model import MultivariateLSTM

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 데이터 전처리 및 피처 엔지니어링 ---
def preprocess_and_feature_engineer(df_raw: pd.DataFrame):
    df = df_raw.copy()
    
    # ECOS date는 YYYYMM 형식이므로 datetime으로 변환
    df['date'] = pd.to_datetime(df['date'], format='%Y%m')
    df = df.sort_values('date').set_index('date')
    
    # 타겟 컬럼 확인
    target_columns = ['construction_bsi_actual', 'base_rate', 'housing_sale_price', 'm2_growth']
    available_targets = [col for col in target_columns if col in df.columns]
    
    # 결측치 처리 (선형 보간)
    df = df.interpolate(method='linear', limit_direction='both')

    # 1차 차분 (비정상성 제거)
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

    # 결측치 제거            
    df = df.dropna()

    # 상관관계 기반 피처 선택을 위해 모든 피처 추출 후 상관관계 계산
    all_features = [col for col in df.columns if col not in available_targets and col not in diff_targets]
    correlation_matrix = df[all_features].corr()
    
    # 높은 상관관계 피처 제거 (임계값 0.95)
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) > 0.95)]
    selected_features = [col for col in all_features if col not in to_drop]

    # 타겟과의 상관관계 기반 최종 피처 선택
    target_correlations = []
    for target in diff_targets:
        if target in df.columns:
            corr_with_target = df[selected_features + [target]].corr()[target].abs().sort_values(ascending=False)
            target_correlations.append(corr_with_target[:-1])

    # 평균 상관관계가 높은 피처 선택
    avg_correlation = pd.concat(target_correlations, axis=1).mean(axis=1).sort_values(ascending=False)
    n_features = max(20, int(len(selected_features) * 0.5))

    # 평균 상관관계가 높은 피처 선택
    final_features = avg_correlation.head(n_features).index.tolist()

    # 최종 피처에 타겟 컬럼 추가
    X = df[final_features].values
    y = df[diff_targets].values
    return X, y, final_features, available_targets, diff_targets

# --- 데이터 준비 (시퀀스 생성, 분할, 스케일링) ---
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length]) # 시퀀스 생성
        y_seq.append(y[i + seq_length]) # 다음 시점의 타겟 예측
    return np.array(X_seq), np.array(y_seq)

def prepare_data_no_leakage(X, y, seq_length, batch_size, test_size=0.2, val_size=0.1):
    # 시퀀스 생성
    X_seq, y_seq = create_sequences(X, y, seq_length)
    
    # 시계열 분할 (시간 순서 유지)
    total_samples = len(X_seq)
    test_start = int(total_samples * (1 - test_size))
    val_start = int(test_start * (1 - val_size))
    
    # 데이터 분할 (train 80%, val 10%, test 10%)
    X_train = X_seq[:val_start]
    y_train = y_seq[:val_start]
    X_val = X_seq[val_start:test_start]
    y_val = y_seq[val_start:test_start]
    X_test = X_seq[test_start:]
    y_test = y_seq[test_start:]
    
    # 스케일러 fit (train 데이터로만)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # train 데이터로 스케일러 학습
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    y_train_flat = y_train.reshape(-1, y_train.shape[-1])
    
    scaler_X.fit(X_train_flat)
    scaler_y.fit(y_train_flat)
    
    # 각 세트에 스케일링 적용
    X_train_scaled = np.array([scaler_X.transform(seq) for seq in X_train])
    X_val_scaled = np.array([scaler_X.transform(seq) for seq in X_val])
    X_test_scaled = np.array([scaler_X.transform(seq) for seq in X_test])
    y_train_scaled = scaler_y.transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Tensor 변환
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)
    
    # Dataset 생성 (shuffle=False로 시간 순서 유지)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler_X, scaler_y, y_train, y_test

# --- 베이스라인 모델 (비교를 위한 기준선) ---
def create_baseline_models(y_train, y_test):
    baselines = {}
    
    # Naive (마지막 값 예측)
    naive_pred = np.tile(y_train[-1], (len(y_test), 1))
    baselines['Naive'] = naive_pred
    
    # 계절성 Naive (12개월 전 값, 월별 데이터 가정)
    if len(y_train) >= 12:
        seasonal_pred = np.tile(y_train[-12:], (len(y_test) // 12 + 1, 1))[:len(y_test)]
        baselines['Seasonal_Naive'] = seasonal_pred
    
    # 평균값 예측
    mean_pred = np.tile(np.mean(y_train, axis=0), (len(y_test), 1))
    baselines['Mean'] = mean_pred
    
    return baselines

# 베이스라인 모델 평가 함수
def evaluate_baselines(baselines, y_test, target_names):    
    baseline_metrics = {}
    
    for name, pred in baselines.items():
        metrics = {}
        for i, target in enumerate(target_names):
            rmse = np.sqrt(mean_squared_error(y_test[:, i], pred[:, i]))
            mae = mean_absolute_error(y_test[:, i], pred[:, i])
            r2 = r2_score(y_test[:, i], pred[:, i])
            
            metrics[f'{target}_RMSE'] = rmse
            metrics[f'{target}_MAE'] = mae
            metrics[f'{target}_R2'] = r2
        
        baseline_metrics[name] = metrics
        
    return baseline_metrics

# --- Optuna Objective ---
def objective(trial, X, y):
    # parameters for optimization
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 96, 128]),
        'num_layers': trial.suggest_int('num_layers', 1, 2),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.4, step=0.1),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'seq_length': trial.suggest_categorical('seq_length', [12, 18, 24, 30]),
        'optimizer_type': trial.suggest_categorical('optimizer_type', ['Adam', 'AdamW']),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'loss_function': trial.suggest_categorical('loss_function', ['MSE', 'MAE', 'Huber'])
    }


    try:
        # 데이터 준비
        train_loader, val_loader, _, _, _, _, _ = prepare_data_no_leakage(
            X, y, 
            seq_length=params['seq_length'], 
            batch_size=params['batch_size']
        )
        
        # 모델 초기화
        model = MultivariateLSTM(X.shape[1], params['hidden_size'], params['num_layers'], y.shape[1], params['dropout_rate']).to(device)
        
        # 옵티마이저 설정 (Adam or AdamW)
        if params['optimizer_type'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        else:
            optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        
        # 손실 함수 선택
        if params['loss_function'] == 'MSE':
            criterion = nn.MSELoss()
        elif params['loss_function'] == 'MAE':
            criterion = nn.L1Loss()
        else:
            criterion = nn.HuberLoss(delta=1.0)

        # 모델 학습 w/ early stopping
        best_val_loss = float('inf')
        patience = 0
        max_patience = 10

        for epoch in range(30):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
                optimizer.step()
            
            # Validation loss 계산
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Early stopping for trial
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    break
            
            # Pruning
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return best_val_loss
        
    except Exception as e:
        return float('inf')

# --- 모델 학습 함수 ---
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, early_stopping_patience=15, grad_clip=1.0):
    # ReduceLROnPlateau 스케줄러 설정
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 손실 기록을 위한 리스트 초기화
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Epoch별 학습
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()    
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 학습률 스케줄러 업데이트
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
        
        if (epoch + 1) % 10 == 0:
            pass  # Training progress can be monitored via MLflow
    
    return train_losses, val_losses, best_val_loss

# --- 모델 평가 함수들 ---
def calculate_directional_accuracy(actual, predicted):
    # 방향성 정확도 계산
    if len(actual) <= 1:
        return 0.0
    actual_direction = np.sign(np.diff(actual.flatten()))
    predicted_direction = np.sign(np.diff(predicted.flatten()))
    return np.mean(actual_direction == predicted_direction)

def evaluate_model_performance(model, test_loader, scaler_y, target_names):
    # 모델 성능 평가 함수
    model.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(batch_y.cpu().numpy())

    # 예측 결과 결합
    test_predictions = np.vstack(test_predictions)
    test_targets = np.vstack(test_targets)

    # 역정규화 (차분값으로 변환)
    test_predictions_diff = scaler_y.inverse_transform(test_predictions)
    test_targets_diff = scaler_y.inverse_transform(test_targets)

    # 성능 메트릭 계산
    metrics = {}
    for i, target in enumerate(target_names):
        if i < test_predictions_diff.shape[1] and i < test_targets_diff.shape[1]:
            rmse = np.sqrt(mean_squared_error(test_targets_diff[:, i], test_predictions_diff[:, i]))
            mae = mean_absolute_error(test_targets_diff[:, i], test_predictions_diff[:, i])
            r2 = r2_score(test_targets_diff[:, i], test_predictions_diff[:, i])
            dir_acc = calculate_directional_accuracy(test_targets_diff[:, i], test_predictions_diff[:, i])
            
            metrics[f'{target}_RMSE'] = rmse
            metrics[f'{target}_MAE'] = mae
            metrics[f'{target}_R2'] = r2
            metrics[f'{target}_Directional_Accuracy'] = dir_acc

    return metrics, test_predictions_diff, test_targets_diff


# --- 메인 학습 함수 (MLflow 통합 버전) ---
def run_training(db_uri: str):
    with mlflow.start_run() as run:
        try:
            # 1. 데이터 로드 및 전처리
            # MySQL URI 형식: mysql+pymysql://user:password@host:port/database
            engine = create_engine(
                db_uri, 
                pool_pre_ping=True,  # 연결 확인
                pool_recycle=300,    # 5분마다 연결 재생성
                echo=False           # SQL 로그 비활성화
            )
            raw_df = pd.read_sql(f"SELECT * FROM ecos_data ORDER BY date ASC", engine)
            X, y, final_features, available_targets, _ = preprocess_and_feature_engineer(raw_df)
        
            # 2. 하이퍼파라미터 튜닝
            study = optuna.create_study(
                direction='minimize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
            )
            study.optimize(lambda trial: objective(trial, X, y), n_trials=50, timeout=600)
            best_params = study.best_params
            
            for key, value in best_params.items():
                pass  # Parameters logged to MLflow below
            
            mlflow.log_params(best_params)
            mlflow.log_metric("best_trial_score", study.best_trial.value)

            # 3. 데이터 준비 (시간 순서 고려한 분할)
            train_loader, val_loader, test_loader, scaler_X, scaler_y, y_train_final, y_test_final = prepare_data_no_leakage(
                X, y, best_params['seq_length'], best_params['batch_size']
            )

            # 4. 베이스라인 모델 평가
            baselines = create_baseline_models(y_train_final, y_test_final)
            baseline_metrics = evaluate_baselines(baselines, y_test_final, available_targets)
            
            for model_name, metrics in baseline_metrics.items():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"baseline_{model_name}_{metric_name}", value)
            
            # 5. 최종 LSTM 모델 학습
            final_model = MultivariateLSTM(
                X.shape[1], best_params['hidden_size'], best_params['num_layers'], 
                y.shape[1], best_params['dropout_rate']
            ).to(device)
            
            # 옵티마이저 설정
            if best_params['optimizer_type'] == 'Adam':
                optimizer = optim.Adam(final_model.parameters(), 
                                     lr=best_params['learning_rate'], 
                                     weight_decay=best_params['weight_decay'])
            else:
                optimizer = optim.AdamW(final_model.parameters(), 
                                      lr=best_params['learning_rate'], 
                                      weight_decay=best_params['weight_decay'])
            
            # 손실 함수 설정
            if best_params['loss_function'] == 'MSE':
                criterion = nn.MSELoss()
            elif best_params['loss_function'] == 'MAE':
                criterion = nn.L1Loss()
            else:
                criterion = nn.HuberLoss(delta=1.0)
            
            # 모델 학습 (Early Stopping 포함)
            train_losses, val_losses, best_val_loss = train_model(
                final_model, train_loader, val_loader, optimizer, criterion, 
                epochs=200, early_stopping_patience=20
            )
            
            # 학습 곡선 MLflow 로깅
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            mlflow.log_metric("final_train_loss", train_losses[-1])
            mlflow.log_metric("final_val_loss", val_losses[-1])
            mlflow.log_metric("best_val_loss", best_val_loss)

            # 6. 테스트 세트 성능 평가
            test_metrics, test_predictions, test_targets = evaluate_model_performance(
                final_model, test_loader, scaler_y, available_targets
            )
            
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)

            # 7. 모델 및 아티팩트 저장
            artifacts = {
                "model_state_dict": {k: v.cpu() for k, v in final_model.state_dict().items()},
                "hyperparameters": best_params,
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "final_features": final_features,
                "target_columns": available_targets,
                "test_metrics": test_metrics,
                "baseline_metrics": baseline_metrics,
                "training_losses": train_losses,
                "validation_losses": val_losses
            }
            
            # MLflow에 모델과 아티팩트 저장            
            with tempfile.TemporaryDirectory() as temp_dir:
                # 아티팩트 파일 저장
                artifacts_path = os.path.join(temp_dir, "artifacts.pkl")
                with open(artifacts_path, 'wb') as f:
                    pickle.dump(artifacts, f)
                
                # MLflow에 로깅
                # 현재 training 컨테이너의 requirements.txt 경로
                requirements_path = "requirements.txt" 
                mlflow.pytorch.log_model(
                    pytorch_model=final_model.cpu(), 
                    artifact_path="lstm_model",
                    # requirements.txt 파일을 직접 지정하여 requirements.txt 추측 시 에러 fallback 방지
                    pip_requirements=requirements_path 
                )

                # 추가 아티팩트 저장
                mlflow.log_artifact(artifacts_path, artifact_path="model_artifacts")
            
            return {
                "status": "success", 
                "run_id": run.info.run_id,
                "test_metrics": test_metrics,
                "best_params": best_params
            }
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            raise e