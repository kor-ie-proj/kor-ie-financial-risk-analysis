# be/app/main.py
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional
from sqlalchemy import create_engine, inspect, text
import pandas as pd
from pydantic import BaseModel

INFERENCE_SERVER_URL = os.environ.get("INFERENCE_SERVER_URL")
DATABASE_URI = os.environ.get("DATABASE_URI")

if not DATABASE_URI:
    raise RuntimeError("DATABASE_URI must be set for the BE service to function.")

engine = create_engine(DATABASE_URI)

app = FastAPI()

allowed_origins_env = os.environ.get("ALLOWED_ORIGINS")
if allowed_origins_env:
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_INDICATOR_COLUMNS = [
    col.strip()
    for col in os.environ.get("DEFAULT_INDICATORS", "construction_bsi_actual,base_rate,housing_sale_price").split(",")
    if col.strip()
]

class PredictResponse(BaseModel):
    monthly_predictions: dict
    quarterly_predictions: dict


class IndicatorRecord(BaseModel):
    date: str
    values: Dict[str, float]


class IndicatorResponse(BaseModel):
    columns: List[str]
    data: List[IndicatorRecord]


class CompanyListResponse(BaseModel):
    companies: List[str]


class FinancialRecord(BaseModel):
    period: str
    year: int
    quarter: str
    metrics: Dict[str, float]


class FinancialResponse(BaseModel):
    corp_name: str
    available_metrics: List[str]
    data: List[FinancialRecord]


class RiskResponse(BaseModel):
    corp_name: str
    risk_score: float
    risk_level: str
    thresholds: Dict[str, float]
    components: Dict[str, float]
    ecos_quarters: Dict[str, Any]
    dart_vector: Dict[str, float]
    mode: Optional[str] = None
    manual_adjustments: Optional[Dict[str, float]] = None


class ManualAdjustments(BaseModel):
    construction_bsi_actual: float = 0.0
    base_rate: float = 0.0
    housing_sale_price: float = 0.0
    m2_growth: float = 0.0


class ManualRiskRequest(BaseModel):
    adjustments: ManualAdjustments


@lru_cache()
def _ecos_value_columns() -> List[str]:
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("ecos_data")]
    ignore = {"id", "created_at", "updated_at", "date"}
    return [col for col in columns if col not in ignore]


@lru_cache()
def _dart_value_columns() -> List[str]:
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("dart_data")]
    ignore = {"id", "corp_name", "year", "quarter", "created_at", "updated_at"}
    return [col for col in columns if col not in ignore]


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None




def _build_risk_response(corp_name: str, risk_payload: Dict[str, Any]) -> RiskResponse:
    if not risk_payload.get("corp_name"):
        risk_payload["corp_name"] = corp_name

    components: Dict[str, float] = {}
    for key, value in risk_payload.get("components", {}).items():
        num = _coerce_float(value)
        if num is not None:
            components[key] = num

    dart_vector: Dict[str, float] = {}
    for key, value in risk_payload.get("dart_vector", {}).items():
        num = _coerce_float(value)
        if num is not None:
            dart_vector[key] = num

    return RiskResponse(
        corp_name=risk_payload.get("corp_name", corp_name),
        risk_score=risk_payload.get("risk_score", 0.0),
        risk_level=risk_payload.get("risk_level", "Unknown"),
        thresholds=risk_payload.get("thresholds", {
            "safe": 35,
            "caution": 59,
            "danger": 85
        }),
        components=components,
        ecos_quarters=risk_payload.get("ecos_quarters", {}),
        dart_vector=dart_vector,
        mode=risk_payload.get("mode"),
        manual_adjustments=risk_payload.get("manual_adjustments"),
    )

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
                value_columns = [col for col in row.index if col != 'date']
                if not value_columns:
                    continue
                update_clause = ', '.join([f"{col}=VALUES({col})" for col in value_columns])
                insert_stmt = text(
                    f"""
                    INSERT INTO model_output (date, {', '.join(value_columns)})
                    VALUES (:date, {', '.join([':'+col for col in value_columns])})
                    ON DUPLICATE KEY UPDATE {update_clause}
                    """
                )
                params = {col: row[col] for col in value_columns}
                params['date'] = row['date']
                conn.execute(insert_stmt, params)
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


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/indicators", response_model=IndicatorResponse)
def get_indicators(
    columns: Optional[str] = Query(
        None,
        description="Comma-separated list of ECOS indicator column names to include.",
    ),
    limit: int = Query(120, ge=1, le=240, description="Maximum number of rows to return."),
) -> IndicatorResponse:
    available_columns = _ecos_value_columns()

    if columns:
        if columns.strip().lower() in {"all", "*"}:
            requested = available_columns
        else:
            requested = []
            for candidate in columns.split(','):
                candidate = candidate.strip()
                if candidate not in available_columns:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown ECOS indicator column: {candidate}",
                    )
                requested.append(candidate)
    else:
        requested = [col for col in DEFAULT_INDICATOR_COLUMNS if col in available_columns]
        if not requested:
            requested = available_columns[: min(5, len(available_columns))]

    if not requested:
        raise HTTPException(status_code=404, detail="No indicator columns available.")

    column_clause = ", ".join(["date"] + requested)
    query = text(
        f"SELECT {column_clause} FROM ecos_data ORDER BY date DESC LIMIT :limit"
    )
    df = pd.read_sql(query, engine, params={"limit": limit})
    if df.empty:
        return IndicatorResponse(columns=requested, data=[])

    df = df.iloc[::-1].reset_index(drop=True)
    records: List[IndicatorRecord] = []
    for row in df.to_dict(orient='records'):
        date_str = str(row.get('date'))
        values = {}
        for col in requested:
            num = _coerce_float(row.get(col))
            if num is not None:
                values[col] = num
        records.append(IndicatorRecord(date=date_str, values=values))

    return IndicatorResponse(columns=requested, data=records)


@app.get("/companies", response_model=CompanyListResponse)
def list_companies(
    q: Optional[str] = Query(None, description="Substring to match against corp_name."),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of companies to return."),
) -> CompanyListResponse:
    base_query = "SELECT DISTINCT corp_name FROM dart_data"
    params: Dict[str, Any] = {"limit": limit}

    if q:
        base_query += " WHERE corp_name LIKE :pattern"
        params["pattern"] = f"%{q}%"

    base_query += " ORDER BY corp_name ASC LIMIT :limit"

    with engine.connect() as conn:
        result = conn.execute(text(base_query), params)
        companies = [row[0] for row in result.fetchall()]

    return CompanyListResponse(companies=companies)


@app.get("/companies/{corp_name}/financials", response_model=FinancialResponse)
def get_company_financials(
    corp_name: str,
    metrics: Optional[str] = Query(
        None,
        description="Comma-separated list of DART metric column names to include.",
    ),
) -> FinancialResponse:
    available_metrics = _dart_value_columns()

    if metrics:
        requested_metrics = []
        for candidate in metrics.split(','):
            candidate = candidate.strip()
            if candidate not in available_metrics:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown financial metric column: {candidate}",
                )
            requested_metrics.append(candidate)
    else:
        requested_metrics = available_metrics

    if not requested_metrics:
        raise HTTPException(status_code=404, detail="No financial metrics available.")

    column_clause = ", ".join(["year", "quarter"] + requested_metrics)
    query = text(
        f"""
        SELECT {column_clause}
        FROM dart_data
        WHERE corp_name = :corp_name
        ORDER BY year ASC, quarter ASC
        """
    )

    df = pd.read_sql(query, engine, params={"corp_name": corp_name})

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No financial records found for corporation '{corp_name}'.",
        )

    data: List[FinancialRecord] = []
    for row in df.to_dict(orient='records'):
        year = int(row['year'])
        quarter = str(row['quarter']).upper()
        period = f"{year}{quarter}"
        metrics_payload = {}
        for metric in requested_metrics:
            num = _coerce_float(row.get(metric))
            if num is not None:
                metrics_payload[metric] = num
        data.append(
            FinancialRecord(
                period=period,
                year=year,
                quarter=quarter,
                metrics=metrics_payload,
            )
        )

    return FinancialResponse(
        corp_name=corp_name,
        available_metrics=requested_metrics,
        data=data,
    )


@app.get("/companies/{corp_name}/risk", response_model=RiskResponse)
async def get_company_risk(
    corp_name: str,
    months_to_predict: int = Query(
        3,
        ge=1,
        le=12,
        description="Number of future months to include in the risk inference request.",
    ),
) -> RiskResponse:
    if not INFERENCE_SERVER_URL:
        raise HTTPException(status_code=500, detail="Inference server URL is not configured.")

    inference_url = INFERENCE_SERVER_URL.rstrip('/') + "/risk_inference"
    payload = {"corp_name": corp_name, "months_to_predict": months_to_predict}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(inference_url, json=payload)
        response.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Risk inference request timed out.")
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json() if exc.response.content else exc.response.text
        raise HTTPException(status_code=exc.response.status_code, detail=detail)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch risk inference result: {exc}",
        )

    result = response.json()
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="Unexpected response format from inference service.")

    # Map new inference result fields to legacy risk_payload structure
    risk_payload = {
        "corp_name": result.get("corp_name", corp_name),
        "risk_score": result.get("heuristic_score"),
        "risk_level": result.get("risk_level"),
        "components": result.get("components", {}),
        "weights": result.get("weights", {}),
        "ecos_quarters": result.get("ecos_quarters", {}),
        "dart_vector": result.get("dart_vector", {}),
        "mode": result.get("mode"),
        "manual_adjustments": result.get("manual_adjustments"),
    }
    # For backward compatibility: if latest_dart_quarter or next_ecos_quarter needed, add here

    return _build_risk_response(corp_name, risk_payload)


@app.post("/companies/{corp_name}/risk/manual", response_model=RiskResponse)
async def get_company_manual_risk(
    corp_name: str,
    request_body: ManualRiskRequest,
) -> RiskResponse:
    if not INFERENCE_SERVER_URL:
        raise HTTPException(status_code=500, detail="Inference server URL is not configured.")

    inference_url = INFERENCE_SERVER_URL.rstrip('/') + "/risk_inference/manual"
    payload = {"corp_name": corp_name, "adjustments": request_body.adjustments.dict()}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(inference_url, json=payload)
        response.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Risk inference request timed out.")
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json() if exc.response.content else exc.response.text
        raise HTTPException(status_code=exc.response.status_code, detail=detail)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch manual risk inference result: {exc}",
        )

    result = response.json()
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="Unexpected response format from inference service.")

    risk_payload = {
        "corp_name": result.get("corp_name", corp_name),
        "risk_score": result.get("heuristic_score"),
        "risk_level": result.get("risk_level"),
        "components": result.get("components", {}),
        "ecos_quarters": result.get("ecos_quarters", {}),
        "dart_vector": result.get("dart_vector", {}),
        "mode": result.get("mode"),
        "manual_adjustments": result.get("manual_adjustments"),
    }

    return _build_risk_response(corp_name, risk_payload)

