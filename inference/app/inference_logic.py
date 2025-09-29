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

# 문서의 13개 휴리스틱 가중치 (합=1.0)
HEURISTIC_WEIGHTS = {
    # 재무지표 73.4%
    "flag_score": 0.2186,
    "debt_score": 0.1648,
    "equity_score": 0.1041,
    "roa_score": 0.1165,
    "roe_score": 0.0219,
    "sales_growth_score": 0.0219,
    "profit_growth_score": 0.0112,
    "net_growth_score": 0.0476,
    
    # 경제지표 26.6%
    "bsi_score": 0.0852,
    "rate_score": 0.0749,
    "housing_score": 0.0088,
    "m2_score": 0.0667,
    #"spread_score": 0.0577,
}

# 재무 지표별 클리핑 범위 (모델 명세대로)
CLIP_BOUNDS = {
    "debt_ratio": (0, 500),            # %
    "equity_ratio": (0, 100),          # %
    "roa": (-50, 50),                  # %
    "roe": (-100, 100),                # %
    "revenue_growth": (-200, 200),     # 넉넉히
    "operating_profit_growth": (-200, 200),
    "net_income_growth": (-200, 200),
}

def _clip_series(s: pd.Series, low: float, high: float) -> pd.Series:
    return s.clip(lower=low, upper=high)

def _percentile_score_from_population(pop: pd.Series, value: float, reverse: bool) -> float:
    """
    모집단(pop) 대비 value의 퍼센타일 점수(0~10)를 계산.
    reverse=True면 값이 높을수록 '위험' → 점수가 커짐 (모델 명세 정의대로)
    """
    pop = pop.dropna().astype(float)
    if pop.empty:
        return 5.0  # 정보 부족 시 중립
    rank = (pop <= value).mean()  # 누적분포 F(x)
    if reverse:
        # 높을수록 위험: 퍼센타일(낮음=안전) 그대로 0~10로 스케일
        return float(rank * 10)
    else:
        # 높을수록 안전: 상위일수록 점수 낮아지도록 (1 - rank)*10
        return float((1.0 - rank) * 10)

def _company_size_from_assets(total_assets: Optional[float]) -> str:
    if total_assets is None or np.isnan(total_assets):
        return "중견기업"  # 정보 없으면 중간 가중치
    # 단위는 입력 데이터에 맞춰야 함(예: 억/조 등). 여기선 '원 단위'라고 가정.
    # 10조 = 10_0000_0000_0000, 1조 = 1_0000_0000_0000 (독자 데이터 단위에 맞춰 조정)
    TEN_TRILLION = 10_0000_0000_0000
    ONE_TRILLION = 1_0000_0000_0000
    if total_assets >= TEN_TRILLION:
        return "대기업"
    elif total_assets >= ONE_TRILLION:
        return "중견기업"
    else:
        return "중소기업"

def _debt_weight_by_size(size: str) -> float:
    return {"대기업": 0.5, "중견기업": 0.75, "중소기업": 1.0}.get(size, 0.75)

class RiskInferenceConfig:
    """
    문서 사양을 반영한 정적 설정.
    - ECOS: 상대평가(0~10) 점수로 환산하여 사용
    - DART: 상대평가(0~10) 점수 + 기업규모별 부채 가중치
    """

    ecos_columns: List[str] = [
        'construction_bsi_actual',
        'base_rate',
        'housing_sale_price',
        'm2_growth',
        #'credit_spread',
    ]

    dart_columns: List[str] = [
        "debt_ratio",
        "equity_ratio",
        "roa",
        "roe",
        "revenue_growth",
        "operating_profit_growth",
        "net_income_growth",
        # 상대평가용 자산총계(기업규모 분류용) 컬럼명 가정: total_assets
        # 없으면 중견기업으로 처리
        "total_assets",
        # 플래그 산출에 필요한 열들이 별도로 dart_df에 존재한다고 가정
        "total_equity",
        "operating_profit",
    ]

    # 경제지표의 '위험 방향' 정의 (reverse=True면 높을수록 위험)
    ecos_reverse = {
        "construction_bsi_actual": False,  # 높을수록 안전
        "base_rate": True,                 # 높을수록 위험
        "housing_sale_price": False,       # 높을수록 안전
        "m2_growth": False,                # 높을수록 안전
        # "credit_spread": True,             # 높을수록 위험
    }

    # 재무지표의 '위험 방향'
    dart_reverse = {
        "debt_ratio": True,                # 높을수록 위험
        "equity_ratio": False,             # 높을수록 안전
        "roa": False,                      # 높을수록 안전
        "roe": False,                      # 높을수록 안전
        "revenue_growth": False,           # 높을수록 안전
        "operating_profit_growth": False,  # 높을수록 안전
        "net_income_growth": False,        # 높을수록 안전
    }

def preprocess_for_inference(df_raw: pd.DataFrame, final_features: list, available_targets: list) -> pd.DataFrame:
    df = df_raw.copy()
    if df['date'].dtype == 'object':
        df['date'] = pd.to_datetime(df['date'], format='%Y%m', errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').set_index('date')
    # 경제지표 결측 → 선형보간(문서: 중앙값 대체였지만, 시계열 예측 파이프라인과의 일관성을 위해 보간 유지)
    df = df.interpolate(method='linear', limit_direction='both')

    # 1차 차분 및 파생특성
    for col in available_targets:
        df[f'{col}_diff'] = df[col].diff()
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
    missing_features = [f for f in final_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Required features are missing after preprocessing: {missing_features}")
    return df[final_features]

def predict_next_step(
    db_uri: str,
    model: torch.nn.Module,
    artifacts: Dict[str, Any],
    months_to_predict: int
) -> Dict[str, Any]:
    """
    차분 예측값을 고정 누적하는 간단 다중스텝 예측 (기존 버전 유지)
    """
    device = next(model.parameters()).device
    seq_length = artifacts['hyperparameters']['seq_length']
    available_targets = artifacts['target_columns']
    scaler_X = artifacts['scaler_X']
    scaler_y = artifacts['scaler_y']

    try:
        engine = create_engine(db_uri)
        df_raw = pd.read_sql("SELECT * FROM ecos_data ORDER BY date ASC", engine)
    except Exception as e:
        raise ConnectionError(f"Failed to load ECOS data: {e}")

    if len(df_raw) < seq_length:
        raise ValueError(f"Not enough data in DB. Need >= {seq_length}, got {len(df_raw)}.")

    df_raw['date'] = pd.to_datetime(df_raw['date'], format='%Y%m', errors='coerce')
    df_raw = df_raw.dropna(subset=['date']).sort_values('date').set_index('date')

    X_processed = preprocess_for_inference(df_raw.reset_index(), artifacts['final_features'], available_targets)
    last_sequence = X_processed.iloc[-seq_length:].values
    last_sequence_scaled = scaler_X.transform(last_sequence)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(device)
        pred_scaled = model(X_tensor).cpu().numpy()
        pred_diff = scaler_y.inverse_transform(pred_scaled)[0]

    last_original_values = df_raw[available_targets].iloc[-1].values
    predictions_original = []
    current_values = last_original_values.copy()

    for _ in range(months_to_predict):
        next_values = current_values + pred_diff
        predictions_original.append(next_values.copy())
        current_values = next_values

    predictions_original = np.nan_to_num(np.array(predictions_original), nan=0.0, posinf=1e12, neginf=-1e12)
    response = {target: predictions_original[:, i].tolist() for i, target in enumerate(available_targets)}
    return {"predictions": response}

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

def _get_latest_dart_subset(dart_df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    if dart_df.empty:
        raise ValueError("DART data is empty.")
    df = dart_df.copy()
    df['quarter_order'] = (
        df['quarter'].astype(str).str.upper().str.replace('Q', '', regex=False).astype(int)
    )
    df = df.sort_values(['year', 'quarter_order'])
    latest_year = int(df['year'].max())
    latest_quarter = int(df.loc[df['year'] == latest_year, 'quarter_order'].max())
    latest_rows = df[(df['year'] == latest_year) & (df['quarter_order'] == latest_quarter)].copy()
    if latest_rows.empty:
        raise ValueError("Unable to isolate latest DART quarter data.")
    return latest_rows, latest_year, latest_quarter

def _increment_quarter(year: int, quarter: int) -> Tuple[int, int]:
    year = int(year); quarter = int(quarter)
    if quarter == 4: return year + 1, 1
    if quarter in (1, 2, 3): return year, quarter + 1
    raise ValueError(f"Unsupported quarter value: {quarter}")

def _extract_latest_quarterly_ecos(
    ecos_df: pd.DataFrame,
    target_columns: List[str],
    reference_year: Optional[int] = None,
    reference_quarter: Optional[int] = None,
) -> pd.Series:
    ecos_df = ecos_df.copy()
    ecos_df['date'] = pd.to_datetime(ecos_df['date'], format='%Y%m', errors='coerce')
    ecos_df = ecos_df.dropna(subset=['date']).sort_values('date')
    if ecos_df.empty:
        raise ValueError("ECOS data empty after date parsing.")
    for c in target_columns:
        if c not in ecos_df.columns:
            raise ValueError(f"Missing ECOS column: {c}")
    if reference_year is not None and reference_quarter is not None:
        months = _QUARTER_MONTHS.get(int(reference_quarter))
        if not months:
            raise ValueError(f"Unsupported quarter: {reference_quarter}")
        mask = (ecos_df['date'].dt.year == int(reference_year)) & (ecos_df['date'].dt.month.isin(months))
        quarter_df = ecos_df.loc[mask]
        if quarter_df.empty:
            raise ValueError(f"No ECOS for {reference_year}Q{reference_quarter}.")
        aggregated = quarter_df[target_columns].mean()
        aggregated.name = pd.Period(f"{int(reference_year)}Q{int(reference_quarter)}")
        return aggregated
    ecos_df['quarter'] = ecos_df['date'].dt.to_period('Q')
    quarterly = ecos_df.groupby('quarter')[target_columns].mean().sort_index()
    if quarterly.empty:
        raise ValueError("Cannot compute ECOS quarterly aggregates.")
    return quarterly.iloc[-1]

def _extract_predicted_quarterly_ecos(
    ecos_df: pd.DataFrame,
    predicted_months: Dict[str, List[float]],
    target_columns: List[str],
    reference_year: int,
    reference_quarter: int,
) -> pd.Series:
    if not predicted_months:
        raise ValueError("Predicted ECOS is empty.")
    prediction_df = pd.DataFrame(predicted_months)
    if prediction_df.empty:
        raise ValueError("Predicted ECOS DataFrame is empty.")
    prediction_df = prediction_df[target_columns]
    lengths = {len(v) for v in predicted_months.values()}
    if len(lengths) != 1:
        raise ValueError("Inconsistent horizons in predicted ECOS.")
    horizon = lengths.pop()
    if horizon == 0:
        raise ValueError("Empty ECOS horizon.")

    ecos_df = ecos_df.copy()
    ecos_df['date'] = pd.to_datetime(ecos_df['date'], format='%Y%m', errors='coerce')
    ecos_df = ecos_df.dropna(subset=['date']).sort_values('date')

    next_year, next_quarter = _increment_quarter(reference_year, reference_quarter)
    target_period = pd.Period(f"{next_year}Q{next_quarter}")
    quarter_months = _QUARTER_MONTHS[next_quarter]
    base_last_month = _QUARTER_MONTHS[reference_quarter][-1]
    base_date = pd.Timestamp(year=int(reference_year), month=base_last_month, day=1)
    future_dates = [base_date + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
    prediction_df['date'] = future_dates
    prediction_df = prediction_df.set_index('date').sort_index()

    predicted_quarter_df = prediction_df[prediction_df.index.to_period('Q') == target_period]
    predicted_by_month = {ts.month: row.astype(float) for ts, row in predicted_quarter_df.iterrows()}

    actual_quarter_df = ecos_df[ecos_df['date'].dt.to_period('Q') == target_period]
    actual_by_month = {int(r['date'].month): r[target_columns].astype(float) for _, r in actual_quarter_df.iterrows()}

    monthly = []
    for m in quarter_months:
        if m in actual_by_month: monthly.append(actual_by_month[m])
        elif m in predicted_by_month: monthly.append(predicted_by_month[m])
        else:
            raise ValueError(f"Missing month {next_year}-{m:02d} for next quarter ECOS vector.")
    combined_df = pd.DataFrame(monthly)
    s = combined_df.mean()
    s.name = target_period
    return s

def calculate_flag_score_from_dart(dart_df: pd.DataFrame) -> float:
    if dart_df.empty:
        return 0.0
    df = dart_df.copy()

    # 시계열 정렬
    df['quarter_order'] = df['quarter'].astype(str).str.replace('Q', '', regex=False).astype(int)
    df = df.sort_values(['year', 'quarter_order']).reset_index(drop=True)

    # 안전하게 수치화
    num = lambda c: pd.to_numeric(df.get(c), errors='coerce')

    # 1. 완전자본잠식
    df['flag_total_equity_erosion'] = (num('total_equity') < 0).astype(int)
    # 2. 연속매출감소 (최근 3개 중 2개 이상 < 0)
    df['flag_revenue_decline_streak'] = (num('revenue_growth') < 0).astype('int8').rolling(3, min_periods=2).sum().ge(2).astype(int)
    # 3. 고부채비율
    df['flag_high_debt_ratio'] = (num('debt_ratio') >= 200).astype(int)
    # 4. 영업손실연속 (최근 4개 중 2개 이상 손실)  ※ 문서 설명에 맞춰 ge(2)
    df['flag_operating_loss_streak'] = (num('operating_profit') < 0).astype('int8').rolling(4, min_periods=3).sum().ge(2).astype(int)
    # 5. ROA 악화
    df['flag_roa_deterioration'] = (num('roa') < -5).astype(int)
    # 6. 자기자본부족
    df['flag_low_equity_ratio'] = (num('equity_ratio') < 20).astype(int)
    # 7. 매출급감
    df['flag_revenue_sharp_drop'] = (num('revenue_growth') < -30).astype(int)
    # 8. 영업이익성장률 악화 (최근 3개 중 2개 이상 < 0)
    df['flag_profit_growth_deterioration'] = (num('operating_profit_growth') < 0).astype('int8').rolling(3, min_periods=2).sum().ge(2).astype(int)

    weights = {
        'flag_total_equity_erosion': 10,
        'flag_roa_deterioration': 6,
        'flag_operating_loss_streak': 5,
        'flag_high_debt_ratio': 4,
        'flag_revenue_decline_streak': 3,
        'flag_low_equity_ratio': 3,
        'flag_revenue_sharp_drop': 2,
        'flag_profit_growth_deterioration': 1
    }
    df['flag_score'] = 0
    for k, w in weights.items():
        if k in df.columns:
            df['flag_score'] += df[k] * w
    return float(df['flag_score'].iloc[-1])

def _compute_dart_relative_scores_for_latest_quarter(
    all_dart_df: pd.DataFrame,
    latest_year: int,
    latest_quarter: int,
    corp_name: str
) -> Dict[str, float]:
    """
    같은 '연/분기'의 모든 기업을 모집단으로 삼아 상대평가(0~10) 스코어를 계산.
    클리핑 후 방향(reverse)에 맞게 점수를 산출.
    """
    # 해당 분기의 전체 기업 rows
    dfq = all_dart_df.copy()
    dfq['quarter_order'] = dfq['quarter'].astype(str).str.upper().str.replace('Q', '', regex=False).astype(int)
    dfq = dfq[(dfq['year'] == latest_year) & (dfq['quarter_order'] == latest_quarter)]

    if dfq.empty:
        # 모집단이 없으면 중립값으로
        return {k: 5.0 for k in ["debt_score","equity_score","roa_score","roe_score",
                                 "sales_growth_score","profit_growth_score","net_growth_score"]}

    # 필요한 열만 수치화 + 클리핑
    cols = ["debt_ratio","equity_ratio","roa","roe","revenue_growth","operating_profit_growth","net_income_growth","total_assets"]
    for c in cols:
        if c in dfq.columns:
            dfq[c] = pd.to_numeric(dfq[c], errors='coerce')
        else:
            dfq[c] = np.nan

    for k, (lo, hi) in CLIP_BOUNDS.items():
        if k in dfq.columns:
            dfq[k] = _clip_series(dfq[k], lo, hi)

    # 타깃 기업의 벡터
    row_target = dfq[dfq['corp_name'] == corp_name].tail(1)
    if row_target.empty:
        # 해당 분기에 이 기업 데이터 없음 → 중립
        return {k: 5.0 for k in ["debt_score","equity_score","roa_score","roe_score",
                                 "sales_growth_score","profit_growth_score","net_growth_score"]}

    t = row_target.iloc[0]
    reverse = RiskInferenceConfig.dart_reverse

    scores = {}
    # 각 항목 상대평가
    for raw, key in [
        ("debt_ratio","debt_score"),
        ("equity_ratio","equity_score"),
        ("roa","roa_score"),
        ("roe","roe_score"),
        ("revenue_growth","sales_growth_score"),
        ("operating_profit_growth","profit_growth_score"),
        ("net_income_growth","net_growth_score"),
    ]:
        val = float(t.get(raw)) if pd.notna(t.get(raw)) else np.nan
        pop = dfq[raw]
        scores[key] = _percentile_score_from_population(pop, val, reverse=reverse.get(raw, False))

    # 기업 규모별 부채 가중치 적용
    size = _company_size_from_assets(float(t.get("total_assets")) if pd.notna(t.get("total_assets")) else None)
    debt_w = _debt_weight_by_size(size)
    scores["debt_score"] = float(scores["debt_score"] * debt_w)

    # 스코어들은 0~10 범위에 머물도록 클리핑
    for k in scores:
        scores[k] = float(np.clip(scores[k], 0.0, 10.0))
    return scores

def _compute_ecos_relative_scores_from_history(
    ecos_quarterly_history: pd.DataFrame,
    predicted_quarter_vector: pd.Series
) -> Dict[str, float]:
    """
    ECOS 분기평균의 과거 히스토리를 모집단으로 삼아,
    예측 분기값의 상대평가(0~10)를 산출.
    """
    # quarterly_history: index=Period('YYYYQn'), columns=ecos columns
    reverse = RiskInferenceConfig.ecos_reverse
    scores = {}
    for col in RiskInferenceConfig.ecos_columns:
        if col not in ecos_quarterly_history.columns:
            scores_key = {
                "construction_bsi_actual":"bsi_score",
                "base_rate":"rate_score",
                "housing_sale_price":"housing_score",
                "m2_growth":"m2_score",
                # "credit_spread":"spread_score",
            }.get(col, f"{col}_score")
            scores[scores_key] = 5.0
            continue
        pop = ecos_quarterly_history[col].astype(float)
        val = float(predicted_quarter_vector.get(col, np.nan))
        key = {
            "construction_bsi_actual":"bsi_score",
            "base_rate":"rate_score",
            "housing_sale_price":"housing_score",
            "m2_growth":"m2_score",
            # "credit_spread":"spread_score",
        }[col]
        sc = _percentile_score_from_population(pop, val, reverse=reverse.get(col, False))
        scores[key] = float(np.clip(sc, 0.0, 10.0))
    return scores

def run_heuristic_risk_inference(
    db_uri: str,
    model: Optional[torch.nn.Module],
    artifacts: Dict[str, Any],
    corp_name: str,
    months_to_predict: int = 3,
    manual_adjustments: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    모델 명세:
      - 위험 플래그(flag_score) 산출
      - DART/ECOS 상대평가 0~10 스코어 산출
      - 기업규모별 부채 가중치
      - 최종 Heuristic Score = Σ(weights * score) * 10   → 0~100
      - ECOS는 (t-1 분기 재무, t 분기 경제) 구조를 반영하기 위해
        최신 DART 분기를 기준으로 다음 분기 ECOS 벡터를 구성(예측 또는 수기 조정)
    """
    target_columns = artifacts.get('target_columns', [])
    if not target_columns:
        raise ValueError("Artifacts missing target_columns for ECOS.")
    # ECOS 기준 컬럼 확인/정렬
    ecos_columns = RiskInferenceConfig.ecos_columns

    try:
        engine = create_engine(db_uri)
        ecos_df = pd.read_sql("SELECT * FROM ecos_data ORDER BY date ASC", engine)
        # 전체 기업 DART를 읽어와 상대평가 모집단으로 사용
        all_dart_df = pd.read_sql("SELECT * FROM dart_data ORDER BY year, quarter", engine)
        # 대상 기업 슬라이스
        dart_df = pd.read_sql(
            text("SELECT * FROM dart_data WHERE corp_name = :corp_name ORDER BY year, quarter"),
            engine,
            params={"corp_name": corp_name},
        )
    except Exception as exc:
        raise ConnectionError(f"Failed to load data for heuristic inference: {exc}")

    if dart_df.empty:
        raise ValueError(f"No DART records found for '{corp_name}'.")

    # 최신 분기 파악
    latest_dart_rows, latest_year, latest_quarter = _get_latest_dart_subset(dart_df)

    # 해당 최신 분기의 '실제' ECOS 분기평균 (설명용/검증용)
    latest_ecos_quarter = _extract_latest_quarterly_ecos(
        ecos_df,
        ecos_columns,
        reference_year=latest_year,
        reference_quarter=latest_quarter,
    )

    # 다음 분기 ECOS 벡터 (예측 or 수기조정)
    next_year, next_quarter = _increment_quarter(latest_year, latest_quarter)

    if manual_adjustments:
        # 문서: "경제지표 시프트"의 수기 조정 경로
        adjustments = {k: float(v) for k, v in manual_adjustments.items() if v is not None}
        base = latest_ecos_quarter.copy().astype(float)
        # base_rate는 절대값 증감, 나머지는 % 변화(문서 정의)
        predicted_quarter = base.copy()
        for c in ecos_columns:
            base_val = float(base.get(c, 0.0))
            delta = float(adjustments.get(c, 0.0) or 0.0)
            if c == "base_rate":
                predicted_quarter[c] = base_val + delta
            else:
                predicted_quarter[c] = base_val * (1.0 + delta / 100.0)
        predicted_quarter.name = pd.Period(f"{int(next_year)}Q{int(next_quarter)}")
        prediction_mode = "manual"
    else:
        if model is None:
            raise ValueError("Model is required when manual adjustments are not provided.")
        # LSTM 기반 월별 예측 → 다음 분기 평균 구성
        prediction_payload = predict_next_step(
            db_uri=db_uri,
            model=model,
            artifacts=artifacts,
            months_to_predict=months_to_predict
        )
        predicted_quarter = _extract_predicted_quarterly_ecos(
            ecos_df=ecos_df,
            predicted_months=prediction_payload.get('predictions', {}),
            target_columns=ecos_columns,
            reference_year=latest_year,
            reference_quarter=latest_quarter,
        )
        prediction_mode = "forecast"

    # ========= 상대평가 점수 산출 =========

    # 1) DART 상대평가 (동분기 전체 기업을 모집단으로)
    dart_scores = _compute_dart_relative_scores_for_latest_quarter(
        all_dart_df=all_dart_df,
        latest_year=latest_year,
        latest_quarter=latest_quarter,
        corp_name=corp_name
    )

    # 2) ECOS 상대평가 (과거 분기 히스토리 대비 예측 분기값)
    ecos_df_q = ecos_df.copy()
    ecos_df_q['date'] = pd.to_datetime(ecos_df_q['date'], format='%Y%m', errors='coerce')
    ecos_df_q = ecos_df_q.dropna(subset=['date']).sort_values('date')
    ecos_df_q['quarter'] = ecos_df_q['date'].dt.to_period('Q')
    ecos_quarterly_history = ecos_df_q.groupby('quarter')[ecos_columns].mean().sort_index()
    ecos_scores = _compute_ecos_relative_scores_from_history(
        ecos_quarterly_history=ecos_quarterly_history,
        predicted_quarter_vector=predicted_quarter
    )

    # 3) 위험 플래그 점수 (원시 점수, 상대평가 X)
    flag_score_raw = calculate_flag_score_from_dart(dart_df)

    # ========= 최종 휴리스틱 점수 (0~100) =========
    # 모델 명세: 13개 점수의 가중평균(합1.0) × 10
    # flag_score는 '원시 점수'라서 0~10 스케일로 맞춰줄 필요가 있음.
    # 간단히 min-max 클리핑 후 0~10로 압축(경험적). 필요시 사전 캘리브레이션 테이블로 교체 가능.
    # 여기서는 상한 20 가정 → 20 이상이면 10점, 0이면 0점.
    FLAG_CAP = 20.0
    flag_score_scaled_0_10 = float(np.clip(flag_score_raw / FLAG_CAP * 10.0, 0.0, 10.0))

    # 가중합 구성용 벡터
    combined_scores = {
        "flag_score": flag_score_scaled_0_10,
        **dart_scores,   # debt/equity/roa/roe/sales/profit/net
        **ecos_scores,   # bsi/rate/housing/m2/spread
    }

    # 누락 항목이 있으면 중립(5.0)로 채움
    for key in HEURISTIC_WEIGHTS.keys():
        if key not in combined_scores:
            combined_scores[key] = 5.0

    weighted_sum = 0.0
    for k, w in HEURISTIC_WEIGHTS.items():
        weighted_sum += w * combined_scores[k]

    heuristic_score = float(np.clip(weighted_sum * 10.0, 0.0, 100.0))

    # 4단계 라벨링
    def _label(score: float) -> str:
        # 문서 예시(안전/주의/위험/매우위험) 기준 스레숄드는 업무정책에 맞춰 조정.
        # 여기서는 경험적 기준: 0-25 안전, 25-50 주의, 50-75 위험, 75-100 매우위험
        if score <= 21:
            return "Low"
        elif score <= 35.45:
            return "Moderate"
        elif score <= 56.42:
            return "High"
        else:
            return "Critical"

    # 결과
    result = {
        "corp_name": corp_name,
        "latest_dart_quarter": f"{latest_year}Q{latest_quarter}",
        "next_ecos_quarter": str(predicted_quarter.name),
        "mode": prediction_mode,
        "heuristic_score": heuristic_score,     # 0~100
        "risk_level": _label(heuristic_score),  # 4단계
        "components": {
            "flag_score_raw": flag_score_raw,               # 원시 플래그 점수
            "flag_score_scaled_0_10": flag_score_scaled_0_10,
            **dart_scores,
            **ecos_scores,
        },
        "weights": HEURISTIC_WEIGHTS,
        "ecos_quarters": {
            "latest_actual": {
                "quarter": str(latest_ecos_quarter.name),
                "values": {c: float(latest_ecos_quarter[c]) for c in ecos_columns if c in latest_ecos_quarter},
            },
            "predicted": {
                "quarter": str(predicted_quarter.name),
                "values": {c: float(predicted_quarter[c]) for c in ecos_columns if c in predicted_quarter},
                "source": prediction_mode,
            },
        },
    }
    return result
