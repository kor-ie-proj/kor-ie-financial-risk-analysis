import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Iterable, List, Optional, Tuple
from sqlalchemy import create_engine, text


_QUARTER_MONTHS: Dict[int, Tuple[int, int, int]] = {
    1: (1, 2, 3),
    2: (4, 5, 6),
    3: (7, 8, 9),
    4: (10, 11, 12),
}


class RiskInferenceConfig:
    """Static configuration for heuristic risk inference.

    Edit the class attributes below to adjust which columns participate in the
    risk aggregation and how much influence they have. When the column lists are
    empty, the inference logic falls back to the available columns detected at
    runtime (ECOS target columns from artifacts and numeric DART columns).
    """

    # Names from ECOS target columns to use for both actual/predicted features.
    # Example: ["gdp_growth", "cpi"]
    ecos_columns: List[str] = []

    # Column names from the DART aggregation vector.
    dart_columns: List[str] = [
        "debt_ratio",
        "equity_ratio",
        "roa",
        "roe",
        "revenue_growth",
        "operating_profit_growth",
        "net_income_growth",
    ]

    # Optional per-feature weights. Keys should match the generated feature
    # names (e.g. "actual_gdp_growth", "predicted_cpi").
    ecos_weights: Dict[str, float] = {}

    # Optional per-feature weights for DART columns (keys are column names as-is).
    dart_weights: Dict[str, float] = {}

    ecos_default_weight: float = 1.0
    dart_default_weight: float = 1.0
    flag_weight: float = 1.0
    bias: float = 0.0

    @classmethod
    def resolve_ecos_columns(cls, available: Iterable[str]) -> List[str]:
        available_set = set(available)
        if not cls.ecos_columns:
            return list(available)
        missing = [col for col in cls.ecos_columns if col not in available_set]
        if missing:
            raise ValueError(
                f"Configured ECOS columns missing from available targets: {missing}"
            )
        return cls.ecos_columns

    @classmethod
    def resolve_dart_columns(cls, available: Iterable[str]) -> List[str]:
        available_set = set(available)
        if not cls.dart_columns:
            return list(available)
        missing = [col for col in cls.dart_columns if col not in available_set]
        if missing:
            raise ValueError(
                f"Configured DART columns missing from available data: {missing}"
            )
        return cls.dart_columns

    @classmethod
    def build_ecos_weights(cls, feature_keys: Iterable[str]) -> Dict[str, float]:
        return {
            key: cls.ecos_weights.get(key, cls.ecos_default_weight)
            for key in feature_keys
        }

    @classmethod
    def build_dart_weights(cls, feature_keys: Iterable[str]) -> Dict[str, float]:
        return {
            key: cls.dart_weights.get(key, cls.dart_default_weight)
            for key in feature_keys
        }

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

# === 재귀적 예측이 아닌 고정된 차분값 누적 예측
def predict_next_step(
    db_uri: str, 
    model: torch.nn.Module, 
    artifacts: Dict[str, Any], 
    months_to_predict: int
) -> Dict[str, Any]:
    """
    DB에서 데이터를 로드하고, n개월 후를 예측하는 함수
    - 훈련 시 사용한 predict_future_improved 방식과 동일하게 동작
    """
    device = next(model.parameters()).device
    seq_length = artifacts['hyperparameters']['seq_length']
    available_targets = artifacts['target_columns']
    scaler_X = artifacts['scaler_X']
    scaler_y = artifacts['scaler_y']

    # 1. DB에서 최신 데이터 가져오기
    try:
        engine = create_engine(db_uri)
        query = f"SELECT * FROM ecos_data ORDER BY date ASC"
        df_raw = pd.read_sql(query, engine)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to the database or query data: {e}")

    if len(df_raw) < seq_length:
        raise ValueError(f"Not enough data in DB. Required at least {seq_length} rows, but got {len(df_raw)}.")

    # 시간 순서 맞추기
    df_raw['date'] = pd.to_datetime(df_raw['date'], format='%Y%m')
    df_raw = df_raw.sort_values('date').set_index('date')

    # 2. 마지막 시퀀스 준비
    X_processed = preprocess_for_inference(df_raw.reset_index(), artifacts['final_features'], available_targets)
    last_sequence = X_processed.iloc[-seq_length:].values
    last_sequence_scaled = scaler_X.transform(last_sequence)

    # 3. 모델 예측 (차분값)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(device)
        pred_scaled = model(X_tensor).cpu().numpy()
        pred_diff = scaler_y.inverse_transform(pred_scaled)[0]
        print("Prediction diff sample:", pred_diff[:5])

    # 4. 차분값을 원시값으로 복원 (고정된 pred_diff를 순차 누적)
    last_original_values = df_raw[available_targets].iloc[-1].values
    predictions_original = []
    current_values = last_original_values.copy()

    for _ in range(months_to_predict):
        next_values = current_values + pred_diff
        predictions_original.append(next_values.copy())
        current_values = next_values

    # 5. 결과 반환
    predictions_original = np.nan_to_num(
        np.array(predictions_original), nan=0.0, posinf=1e12, neginf=-1e12
    )
    response = {
        target: predictions_original[:, i].tolist()
        for i, target in enumerate(available_targets)
    }

    return {"predictions": response}


def calculate_flag_score_from_dart(dart_df: pd.DataFrame) -> float:
    """Placeholder for DART-based flag score calculation logic."""
    if dart_df.empty:
        return 0.0
    return 0.0


def _extract_latest_quarterly_ecos(
    ecos_df: pd.DataFrame,
    target_columns: List[str],
    reference_year: Optional[int] = None,
    reference_quarter: Optional[int] = None,
) -> pd.Series:
    """Return the most recent quarter mean for the requested target columns.

    (최근 분기 3개월 평균을 구해 최신 경기지표 벡터로 사용)
    """
    ecos_df = ecos_df.copy()
    ecos_df['date'] = pd.to_datetime(ecos_df['date'], format='%Y%m', errors='coerce')
    ecos_df = ecos_df.dropna(subset=['date']).sort_values('date')
    if ecos_df.empty:
        raise ValueError("ECOS data is empty after parsing dates.")

    missing_targets = [col for col in target_columns if col not in ecos_df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in ECOS data: {missing_targets}")

    if reference_year is not None and reference_quarter is not None:
        months = _QUARTER_MONTHS.get(int(reference_quarter))
        if not months:
            raise ValueError(
                f"Unsupported reference quarter requested: {reference_quarter}"
            )
        mask = (
            (ecos_df['date'].dt.year == int(reference_year))
            & (ecos_df['date'].dt.month.isin(months))
        )
        quarter_df = ecos_df.loc[mask]
        if quarter_df.empty:
            raise ValueError(
                f"ECOS data does not contain records for reference quarter "
                f"{reference_year}Q{reference_quarter}."
            )
        aggregated = quarter_df[target_columns].mean()
        aggregated.name = pd.Period(f"{int(reference_year)}Q{int(reference_quarter)}")
        return aggregated

    ecos_df['quarter'] = ecos_df['date'].dt.to_period('Q')
    quarterly = ecos_df.groupby('quarter')[target_columns].mean().sort_index()
    if quarterly.empty:
        raise ValueError("Unable to compute quarterly aggregates for ECOS data.")

    return quarterly.iloc[-1]


def _get_latest_dart_subset(
    dart_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int, int]:
    """Return rows for the latest reported DART quarter alongside year/quarter info."""
    if dart_df.empty:
        raise ValueError("DART data is empty.")

    df = dart_df.copy()
    df['quarter_order'] = (
        df['quarter']
        .astype(str)
        .str.upper()
        .str.replace('Q', '', regex=False)
        .astype(int)
    )
    df = df.sort_values(['year', 'quarter_order'])
    latest_year = int(df['year'].max())
    latest_quarter = int(
        df.loc[df['year'] == latest_year, 'quarter_order'].max()
    )

    latest_rows = df[
        (df['year'] == latest_year) & (df['quarter_order'] == latest_quarter)
    ].copy()
    if latest_rows.empty:
        raise ValueError("Unable to isolate latest DART quarter data.")

    return latest_rows, latest_year, latest_quarter


def _increment_quarter(year: int, quarter: int) -> Tuple[int, int]:
    """Return the next year/quarter pair after the provided quarter."""
    year = int(year)
    quarter = int(quarter)
    if quarter not in _QUARTER_MONTHS:
        raise ValueError(f"Unsupported quarter value: {quarter}")
    if quarter == 4:
        return year + 1, 1
    return year, quarter + 1


def _extract_predicted_quarterly_ecos(
    ecos_df: pd.DataFrame,
    predicted_months: Dict[str, List[float]],
    target_columns: List[str],
    reference_year: int,
    reference_quarter: int,
) -> pd.Series:
    """Aggregate predicted monthly values into a single future quarter vector.

    (LSTM 예측 월 데이터를 활용해 다음 분기 평균을 작성)
    """
    if not predicted_months:
        raise ValueError("Predicted ECOS values are empty.")

    prediction_df = pd.DataFrame(predicted_months)
    if prediction_df.empty:
        raise ValueError("Predicted ECOS DataFrame is empty.")

    prediction_df = prediction_df[target_columns]
    lengths = {len(values) for values in predicted_months.values()}
    if len(lengths) != 1:
        raise ValueError("Predicted ECOS columns have inconsistent horizon lengths.")
    num_predicted_months = lengths.pop()
    if num_predicted_months == 0:
        raise ValueError("Predicted ECOS values contain empty horizons.")

    ecos_df = ecos_df.copy()
    ecos_df['date'] = pd.to_datetime(ecos_df['date'], format='%Y%m', errors='coerce')
    ecos_df = ecos_df.dropna(subset=['date']).sort_values('date')
    if ecos_df.empty:
        raise ValueError("ECOS data is empty when aligning predictions.")

    reference_quarter = int(reference_quarter)
    next_year, next_quarter = _increment_quarter(reference_year, reference_quarter)
    target_period = pd.Period(f"{next_year}Q{next_quarter}")
    quarter_months = _QUARTER_MONTHS[next_quarter]

    last_month_of_reference = _QUARTER_MONTHS[reference_quarter][-1]
    base_date = pd.Timestamp(year=int(reference_year), month=last_month_of_reference, day=1)
    future_dates = [base_date + pd.DateOffset(months=offset) for offset in range(1, num_predicted_months + 1)]
    prediction_df['date'] = future_dates
    prediction_df = prediction_df.set_index('date').sort_index()

    predicted_quarter_df = prediction_df[prediction_df.index.to_period('Q') == target_period]
    predicted_by_month = {
        ts.month: row.astype(float)
        for ts, row in predicted_quarter_df.iterrows()
    }

    actual_quarter_df = ecos_df[ecos_df['date'].dt.to_period('Q') == target_period]
    actual_by_month = {
        int(row['date'].month): row[target_columns].astype(float)
        for _, row in actual_quarter_df.iterrows()
    }

    monthly_vectors: List[pd.Series] = []
    for month in quarter_months:
        if month in actual_by_month:
            monthly_vectors.append(actual_by_month[month])
        elif month in predicted_by_month:
            monthly_vectors.append(predicted_by_month[month])
        else:
            raise ValueError(
                "Insufficient data to assemble next quarter ECOS vector: "
                f"missing month {next_year}-{month:02d}."
            )

    combined_df = pd.DataFrame(monthly_vectors)
    quarter_series = combined_df.mean()
    quarter_series.name = target_period
    return quarter_series


def _extract_latest_dart_vector(
    dart_df: pd.DataFrame, latest_rows: Optional[pd.DataFrame] = None
) -> pd.Series:
    """Return the numeric vector for the most recent quarter of the given corporation.

    (요청한 기업의 최신 분기 재무지표 벡터를 계산)
    """
    if latest_rows is None:
        latest_rows, _, _ = _get_latest_dart_subset(dart_df)

    exclude_cols = {'id', 'corp_name', 'quarter', 'created_at', 'updated_at', 'year', 'quarter_order'}
    value_cols = [col for col in latest_rows.columns if col not in exclude_cols]

    if not value_cols:
        raise ValueError("No usable columns found in DART data for vector extraction.")

    value_df = latest_rows[value_cols].apply(pd.to_numeric, errors='coerce')
    value_df = value_df.dropna(axis=1, how='all')

    if value_df.empty:
        raise ValueError("All candidate DART columns are non-numeric or missing.")

    return value_df.mean()


def run_heuristic_risk_inference(
    db_uri: str,
    model: torch.nn.Module,
    artifacts: Dict[str, Any],
    corp_name: str,
    months_to_predict: int = 3
) -> Dict[str, Any]:
    """Compute a heuristic risk score combining ECOS forecasts and DART fundamentals.

    (ECOS 분기 지표와 DART 재무지표를 선형 결합하여 위험도를 추정)
    """
    target_columns = artifacts.get('target_columns', [])
    if not target_columns:
        raise ValueError("Artifacts are missing target columns for ECOS data.")

    try:
        engine = create_engine(db_uri)
        ecos_df = pd.read_sql("SELECT * FROM ecos_data ORDER BY date ASC", engine)
        dart_df = pd.read_sql(
            text("SELECT * FROM dart_data WHERE corp_name = :corp_name"),
            engine,
            params={"corp_name": corp_name},
        )
    except Exception as exc:
        raise ConnectionError(f"Failed to load data for heuristic inference: {exc}")

    if dart_df.empty:
        raise ValueError(f"No DART records found for corporation '{corp_name}'.")

    latest_dart_rows, latest_dart_year, latest_dart_quarter = _get_latest_dart_subset(dart_df)
    latest_ecos_quarter = _extract_latest_quarterly_ecos(
        ecos_df,
        target_columns,
        reference_year=latest_dart_year,
        reference_quarter=latest_dart_quarter,
    )

    prediction_payload = predict_next_step(
        db_uri=db_uri,
        model=model,
        artifacts=artifacts,
        months_to_predict=months_to_predict
    )
    predicted_quarter = _extract_predicted_quarterly_ecos(
        ecos_df=ecos_df,
        predicted_months=prediction_payload.get('predictions', {}),
        target_columns=target_columns,
        reference_year=latest_dart_year,
        reference_quarter=latest_dart_quarter,
    )

    ecos_columns = RiskInferenceConfig.resolve_ecos_columns(target_columns)
    missing_actual = [col for col in ecos_columns if col not in latest_ecos_quarter]
    if missing_actual:
        raise ValueError(
            f"Configured ECOS columns missing in latest quarter data: {missing_actual}"
        )
    missing_predicted = [col for col in ecos_columns if col not in predicted_quarter]
    if missing_predicted:
        raise ValueError(
            f"Configured ECOS columns missing in predicted quarter data: {missing_predicted}"
        )

    ecos_features = {
        f"actual_{col}": float(latest_ecos_quarter[col])
        for col in ecos_columns
    }
    ecos_features.update({
        f"predicted_{col}": float(predicted_quarter[col])
        for col in ecos_columns
    })

    dart_vector = _extract_latest_dart_vector(dart_df, latest_rows=latest_dart_rows)
    dart_columns = RiskInferenceConfig.resolve_dart_columns(dart_vector.index.tolist())
    missing_dart = [col for col in dart_columns if col not in dart_vector]
    if missing_dart:
        raise ValueError(
            f"Configured DART columns missing in latest vector: {missing_dart}"
        )
    dart_features = {col: float(dart_vector[col]) for col in dart_columns}

    flag_score = float(calculate_flag_score_from_dart(dart_df))

    # === Linear heuristic weights (선형 모델 가중치: 사용자 정의 가능) ===
    ecos_weights = RiskInferenceConfig.build_ecos_weights(ecos_features.keys())
    dart_weights = RiskInferenceConfig.build_dart_weights(dart_features.keys())
    flag_weight = RiskInferenceConfig.flag_weight
    bias = RiskInferenceConfig.bias

    # === Weighted contributions (각 벡터에 대한 가중 합) ===
    ecos_score = sum(ecos_features[key] * ecos_weights.get(key, 0.0) for key in ecos_features)
    dart_score = sum(dart_features[key] * dart_weights.get(key, 0.0) for key in dart_features)
    flag_contribution = flag_weight * flag_score

    risk_score = ecos_score + dart_score + flag_contribution + bias

    return {
        "corp_name": corp_name,
        "risk_score": risk_score,
        "components": {
            "ecos_score": ecos_score,
            "dart_score": dart_score,
            "flag_score": flag_score,
            "bias": bias,
        },
        "ecos_quarters": {
            "latest_actual": {
                "quarter": str(latest_ecos_quarter.name),
                "values": {col: float(val) for col, val in latest_ecos_quarter.to_dict().items()},
            },
            "predicted": {
                "quarter": str(predicted_quarter.name),
                "values": {col: float(val) for col, val in predicted_quarter.to_dict().items()},
            },
        },
        "dart_vector": dart_features,
        "corp_name": corp_name,
        "weights": {
            "ecos": ecos_weights,
            "dart": dart_weights,
            "flag": flag_weight,
            "bias": bias,
        },
    }


# === 재귀적 예측
# def predict_next_step(
#     db_uri: str, 
#     model: torch.nn.Module, 
#     artifacts: Dict[str, Any], 
#     months_to_predict: int
# ) -> Dict[str, Any]:
#     """
#     DB에서 데이터를 로드하고, n개월 후를 예측하는 메인 함수
#     """
#     device = next(model.parameters()).device
#     seq_length = artifacts['hyperparameters']['seq_length']
    
#     # 1. DB에서 예측에 필요한 최신 데이터 로드
#     # seq_length에 피처 엔지니어링 시 발생하는 결측치(최대 lag=6)를 고려하여 넉넉하게 데이터를 가져옵니다.
#     required_rows = seq_length + 10 
#     try:
#         engine = create_engine(db_uri)
#         query = f"SELECT * FROM ecos_data ORDER BY date DESC LIMIT {required_rows}"
#         df_raw = pd.read_sql(query, engine)
#         # 시간 순서를 맞추기 위해 다시 정렬
#         df_raw = df_raw.sort_values('date', ascending=True).reset_index(drop=True)
#     except Exception as e:
#         raise ConnectionError(f"Failed to connect to the database or query data: {e}")

#     if len(df_raw) < seq_length:
#         raise ValueError(f"Not enough data in DB. Required at least {seq_length} rows, but got {len(df_raw)}.")
        
#     model.eval()

#     # 2. 다중 스텝 예측(Multi-step Forecasting) 로직
#     final_predictions = []
#     current_df = df_raw.copy()

#     for _ in range(months_to_predict):
#         # 2-1. 현재 데이터프레임으로 전처리 수행
#         X_processed = preprocess_for_inference(
#             current_df, 
#             artifacts['final_features'], 
#             artifacts['target_columns']
#         )
        
#         # 2-2. 마지막 시퀀스 준비
#         last_sequence = X_processed.iloc[-seq_length:].values
#         last_sequence_scaled = artifacts['scaler_X'].transform(last_sequence)
#         X_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(device)

#         # 2-3. 1스텝 예측
#         with torch.no_grad():
#             prediction_scaled = model(X_tensor)
        
#         # 2-4. 예측값을 차분값으로 역정규화
#         prediction_diff = artifacts['scaler_y'].inverse_transform(prediction_scaled.cpu().numpy())[0]
        
#         # 2-5. 차분값을 원본 값으로 복원
#         last_original_values = current_df[artifacts['target_columns']].iloc[-1].values
#         predicted_values = last_original_values + prediction_diff
#         predicted_values = np.nan_to_num(predicted_values, nan=0.0, posinf=1e12, neginf=-1e12)
#         final_predictions.append(predicted_values)

#         # 2-6. 다음 예측을 위해 예측값을 입력 데이터에 추가
#         # 다음 달의 date 생성
#         next_date = (pd.to_datetime(current_df['date'].iloc[-1], format='%Y%m') + pd.DateOffset(months=1)).strftime('%Y%m')
        
#         # 예측된 값을 포함하는 새로운 행(row) 생성
#         new_row = {'date': next_date}
#         for i, col in enumerate(artifacts['target_columns']):
#             new_row[col] = predicted_values[i]
        
#         # current_df에 새로운 행 추가하여 다음 반복에 사용
#         current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

#     # 3. 최종 결과 포맷팅
#     final_predictions = np.array(final_predictions)
#     response = {
#         target: final_predictions[:, i].tolist() 
#         for i, target in enumerate(artifacts['target_columns'])
#     }
    
#     return {"predictions": response}
