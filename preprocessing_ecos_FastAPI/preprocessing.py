"""
ECOS 데이터 전처리 FastAPI
DB에서 ECOS 데이터를 불러와 전처리 후 final_features 테이블에 저장하는 API
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
import mysql.connector
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
import logging

# 경고 무시
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DB 모듈 import (상위 폴더의 DB 디렉토리에서)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DB'))
from db_query import DatabaseConnection

# FastAPI 앱 생성
app = FastAPI(
    title="ECOS Data Preprocessing API",
    description="ECOS 경제데이터 전처리 API - DB에서 데이터 로드 → 전처리 → final_features 저장",
    version="1.0.0"
)

# 응답 모델 정의
class PreprocessingStatus(BaseModel):
    success: bool
    message: str
    processed_rows: Optional[int] = None
    feature_count: Optional[int] = None
    date_range: Optional[Dict[str, str]] = None

class FeatureInfo(BaseModel):
    feature_name: str
    correlation_with_targets: Optional[float] = None
    selected: bool

# 타겟 변수 정의 (predict.ipynb와 동일하게)
TARGET_COLUMNS = [
    'construction_bsi_actual',
    'base_rate', 
    'housing_sale_price',
    'm2_growth',
    'credit_spread'
]

NUMERIC_COLUMNS = [
    'base_rate', 'ccsi', 'construction_bsi_actual', 'construction_bsi_forecast', 
    'cpi', 'esi', 'exchange_usd_krw_close', 'housing_lease_price', 
    'housing_sale_price', 'import_price_non_metal_mineral', 'import_price_steel_primary', 
    'leading_index', 'm2_growth', 'market_rate_treasury_bond_10yr', 
    'market_rate_treasury_bond_3yr', 'market_rate_corporate_bond_3yr_AA', 
    'market_rate_corporate_bond_3yr_BBB', 'ppi_non_metal_mineral', 'ppi_steel_primary'
]

def get_db_connection():
    """DB 연결을 위한 의존성 함수"""
    db = DatabaseConnection()
    if not db.connect():
        raise HTTPException(status_code=500, detail="데이터베이스 연결에 실패했습니다.")
    try:
        yield db
    finally:
        db.disconnect()

def load_ecos_data(db: DatabaseConnection) -> pd.DataFrame:
    """
    DB에서 ECOS 데이터를 로드하고 기본 전처리 수행
    preprocessing.ipynb의 데이터 로드 부분과 동일한 로직
    """
    try:
        # DB에서 ECOS 데이터 조회
        ecos_data = db.get_ecos_data()
        
        if ecos_data is None or ecos_data.empty:
            raise HTTPException(status_code=404, detail="ECOS 데이터를 찾을 수 없습니다.")
        
        logger.info(f"ECOS 데이터 로드 완료: {ecos_data.shape}")
        
        # 수치형 컬럼 변환
        for col in NUMERIC_COLUMNS:
            if col in ecos_data.columns:
                ecos_data[col] = pd.to_numeric(ecos_data[col], errors='coerce')
        
        # 날짜 처리 (YYYYMM 형식을 datetime으로 변환)
        if ecos_data['date'].dtype == 'object' or ecos_data['date'].dtype == 'int64':
            ecos_data['date'] = pd.to_datetime(ecos_data['date'].astype(str), format='%Y%m')
        else:
            ecos_data['date'] = pd.to_datetime(ecos_data['date'])
        
        ecos_data = ecos_data.sort_values('date').set_index('date')
        
        # 필요한 파생 변수 생성 (predict.ipynb에서 사용하는 것들)
        # credit_spread 계산 (회사채 3년 AA- - 국고채 3년)
        if 'market_rate_corporate_bond_3yr_AA' in ecos_data.columns and 'market_rate_treasury_bond_3yr' in ecos_data.columns:
            ecos_data['credit_spread'] = ecos_data['market_rate_corporate_bond_3yr_AA'] - ecos_data['market_rate_treasury_bond_3yr']
        
        # term_spread 계산 (국고채 10년 - 국고채 3년)
        if 'market_rate_treasury_bond_10yr' in ecos_data.columns and 'market_rate_treasury_bond_3yr' in ecos_data.columns:
            ecos_data['term_spread'] = ecos_data['market_rate_treasury_bond_10yr'] - ecos_data['market_rate_treasury_bond_3yr']
        
        # housing_sale_price가 없으면 기본값 생성 (실제 데이터에 맞춰 조정 필요)
        if 'housing_sale_price' not in ecos_data.columns:
            ecos_data['housing_sale_price'] = 100.0  # 기본값
        
        # housing_lease_price가 있으면 housing_sale_price로 사용할 수도 있음
        if 'housing_lease_price' in ecos_data.columns and 'housing_sale_price' not in ecos_data.columns:
            ecos_data['housing_sale_price'] = ecos_data['housing_lease_price']
        
        return ecos_data
        
    except Exception as e:
        logger.error(f"ECOS 데이터 로드 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"데이터 로드 오류: {str(e)}")

def feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    피쳐 엔지니어링 수행 (predict.ipynb TASK 2와 동일한 로직)
    """
    try:
        logger.info("피쳐 엔지니어링 시작")
        
        # 사용 가능한 타겟 변수 확인
        available_targets = [col for col in TARGET_COLUMNS if col in df.columns]
        logger.info(f"Available target columns: {available_targets}")
        
        # 결측치 처리 (선형 보간)
        df = df.interpolate(method='linear', limit_direction='both')
        
        # 타겟 변수 차분 적용 (비정상성 제거)
        for col in available_targets:
            df[f'{col}_diff'] = df[col].diff()
        
        diff_targets = [f'{col}_diff' for col in available_targets]
        
        # 피쳐 엔지니어링: 이동평균, 지연, 변화율
        for col in available_targets:
            diff_col = f'{col}_diff'
            
            # 이동평균
            df[f'{diff_col}_ma3'] = df[diff_col].rolling(window=3, min_periods=1).mean()
            df[f'{diff_col}_ma6'] = df[diff_col].rolling(window=6, min_periods=1).mean()
            
            # 변화율
            df[f'{diff_col}_pct_change'] = df[diff_col].pct_change().fillna(0)
            
            # 지연 특징
            for lag in [1, 3, 6]:
                df[f'{diff_col}_lag{lag}'] = df[diff_col].shift(lag)
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        # 추가 파생 변수들
        # construction_bsi_mom (전월대비)
        if 'construction_bsi_actual' in df.columns:
            df['construction_bsi_mom'] = df['construction_bsi_actual'].pct_change()
            df['construction_bsi_ma3'] = df['construction_bsi_actual'].rolling(window=3, min_periods=1).mean()
        
        # base_rate 관련 파생 변수
        if 'base_rate' in df.columns:
            df['base_rate_mdiff_bp'] = df['base_rate'].diff() * 100  # 베이시스포인트
        
        # 결측치 제거
        df = df.dropna()
        
        # 피쳐 선택: 상관관계 기반
        all_features = [col for col in df.columns if col not in available_targets and col not in diff_targets]
        
        # 높은 상관관계 피쳐 제거
        def remove_highly_correlated_features(corr_matrix, threshold=0.95):
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) > threshold)]
            return to_drop
        
        correlation_matrix = df[all_features].corr()
        highly_corr_features = remove_highly_correlated_features(correlation_matrix)
        selected_features = [col for col in all_features if col not in highly_corr_features]
        
        # 타겟과의 상관관계 기반 최종 피쳐 선택
        target_correlations = []
        for target in diff_targets:
            if target in df.columns:
                corr_with_target = df[selected_features + [target]].corr()[target].abs().sort_values(ascending=False)
                target_correlations.append(corr_with_target[:-1])
        
        if target_correlations:
            avg_correlation = pd.concat(target_correlations, axis=1).mean(axis=1).sort_values(ascending=False)
            n_features = max(20, int(len(selected_features) * 0.5))
            final_features = avg_correlation.head(n_features).index.tolist()
        else:
            final_features = selected_features[:30]  # 기본값
        
        logger.info(f"Original features: {len(all_features)}")
        logger.info(f"Final selected features: {len(final_features)}")
        
        return df, final_features
        
    except Exception as e:
        logger.error(f"피쳐 엔지니어링 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"피쳐 엔지니어링 오류: {str(e)}")

def prepare_final_features(df: pd.DataFrame, final_features: List[str]) -> pd.DataFrame:
    """
    최종 피쳐를 DB 저장 형식에 맞게 준비
    """
    try:
        # 데이터 준비
        df_save = df.copy()
        df_save.reset_index(inplace=True)
        df_save['date'] = pd.to_datetime(df_save['date']).dt.strftime('%Y%m')
        
        # DDL 스키마에 맞는 컬럼들 (db_query.py의 save_final_features 함수 참조)
        schema_columns = [
            'date', 'construction_bsi_actual_diff', 'housing_sale_price_diff', 
            'm2_growth_diff', 'credit_spread_diff', 'base_rate_diff',
            'construction_bsi_mom', 'housing_sale_price_diff_ma3', 'm2_growth_lag1',
            'base_rate_mdiff_bp', 'credit_spread_diff_ma3', 'construction_bsi_ma3',
            'leading_index', 'housing_sale_price_diff_lag6', 'construction_bsi_actual_lag3',
            'construction_bsi_actual_diff_ma3', 'base_rate_diff_ma6', 'term_spread',
            'construction_bsi_actual_diff_ma6', 'credit_spread_diff_lag1',
            'market_rate_treasury_bond_3yr', 'credit_spread_diff_ma6', 'base_rate_diff_ma3',
            'base_rate_lag1', 'esi', 'base_rate_diff_lag3', 'm2_growth_diff_ma6'
        ]
        
        available_cols = [col for col in schema_columns if col in df_save.columns]
        df_final = df_save[available_cols].copy()
        
        # 결측치를 0으로 채움
        df_final = df_final.fillna(0)
        
        logger.info(f"최종 피쳐 데이터 준비 완료: {df_final.shape}")
        
        return df_final
        
    except Exception as e:
        logger.error(f"최종 피쳐 준비 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"최종 피쳐 준비 오류: {str(e)}")

@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "message": "ECOS Data Preprocessing API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/preprocess", response_model=PreprocessingStatus)
async def preprocess_ecos_data(db: DatabaseConnection = Depends(get_db_connection)):
    """
    ECOS 데이터 전처리 메인 엔드포인트
    1. DB에서 ECOS 데이터 로드
    2. 피쳐 엔지니어링 수행
    3. final_features 테이블에 저장
    """
    try:
        logger.info("ECOS 데이터 전처리 시작")
        
        # 1. 데이터 로드
        df = load_ecos_data(db)
        
        # 2. 피쳐 엔지니어링
        df_processed, final_features = feature_engineering(df)
        
        # 3. 최종 피쳐 준비
        df_final = prepare_final_features(df_processed, final_features)
        
        # 4. DB에 저장
        success = db.save_final_features(df_final)
        
        if not success:
            raise HTTPException(status_code=500, detail="DB 저장에 실패했습니다.")
        
        # 날짜 범위 계산
        date_range = {
            "start_date": df_final['date'].min(),
            "end_date": df_final['date'].max()
        }
        
        logger.info("ECOS 데이터 전처리 완료")
        
        return PreprocessingStatus(
            success=True,
            message="ECOS 데이터 전처리 및 저장 완료",
            processed_rows=len(df_final),
            feature_count=len(df_final.columns) - 1,  # date 컬럼 제외
            date_range=date_range
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"전처리 중 예상치 못한 오류: {e}")
        raise HTTPException(status_code=500, detail=f"전처리 오류: {str(e)}")

@app.get("/features/info")
async def get_feature_info(db: DatabaseConnection = Depends(get_db_connection)):
    """
    저장된 final_features 정보 조회
    """
    try:
        # DB에서 final_features 조회
        features_data = db.get_final_features()
        
        if features_data is None or features_data.empty:
            return {"message": "저장된 피쳐 데이터가 없습니다.", "features": []}
        
        # 컬럼 정보
        feature_columns = [col for col in features_data.columns if col != 'date']
        
        features_info = []
        for col in feature_columns:
            features_info.append(FeatureInfo(
                feature_name=col,
                selected=True
            ))
        
        return {
            "total_features": len(feature_columns),
            "total_rows": len(features_data),
            "date_range": {
                "start_date": str(features_data['date'].min()),
                "end_date": str(features_data['date'].max())
            },
            "features": features_info
        }
        
    except Exception as e:
        logger.error(f"피쳐 정보 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"피쳐 정보 조회 오류: {str(e)}")

@app.get("/data/preview")
async def preview_processed_data(limit: int = 10, db: DatabaseConnection = Depends(get_db_connection)):
    """
    전처리된 데이터 미리보기
    """
    try:
        # DB에서 final_features 조회
        features_data = db.get_final_features()
        
        if features_data is None or features_data.empty:
            return {"message": "저장된 피쳐 데이터가 없습니다.", "data": []}
        
        # 최신 데이터부터 limit 개수만 반환
        preview_data = features_data.tail(limit)
        
        return {
            "total_rows": len(features_data),
            "preview_rows": len(preview_data),
            "data": preview_data.to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"데이터 미리보기 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"데이터 미리보기 오류: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)