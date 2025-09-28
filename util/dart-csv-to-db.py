# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
import math

# -------- 설정 --------

# CSV → DB 컬럼 매핑 (영문)
CSV_TO_DB_COL_MAP = {
    "corp_name": "corp_name",
    "year": "year",
    "quarter": "quarter",
    "자산총계": "total_assets",
    "부채총계": "total_liabilities",
    "자본총계": "total_equity",
    "매출액": "revenue",
    "영업이익": "operating_profit",
    "분기순이익": "quarterly_profit",
    "부채비율": "debt_ratio",
    "자기자본비율": "equity_ratio",
    "ROA": "roa",
    "ROE": "roe",
    "매출액성장률": "revenue_growth",
    "영업이익성장률": "operating_profit_growth",
    "순이익성장률": "net_income_growth",
}

# INSERT 컬럼 순서
DB_COL_ORDER = [
    "corp_name",
    "year",
    "quarter",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "revenue",
    "operating_profit",
    "quarterly_profit",
    "debt_ratio",
    "equity_ratio",
    "roa",
    "roe",
    "revenue_growth",
    "operating_profit_growth",
    "net_income_growth",
]

TABLE_NAME = "dart_data"
BATCH_SIZE = 500  # INSERT 묶음 크기

# -------- 유틸 --------

def _escape_sql_str(s: str) -> str:
    """SQL 문자열 이스케이프 (단일 인용부호만 처리)"""
    return s.replace("'", "''")

def _coerce_number(x):
    """숫자 변환: NaN/None → None, 나머지는 float→str로"""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x
    try:
        v = float(str(x).strip())
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

# -------- 메인 로직 --------

def load_and_normalize(csv_path: Path) -> pd.DataFrame:
    """dart_final.csv를 읽어 영문 헤더로 정규화한 DataFrame 반환"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 필요한 컬럼만 남기고 매핑
    missing = [c for c in CSV_TO_DB_COL_MAP.keys() if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV에 필요한 컬럼 누락: {missing}")

    df = df[list(CSV_TO_DB_COL_MAP.keys())].rename(columns=CSV_TO_DB_COL_MAP)

    # 타입 정리
    df["corp_name"] = df["corp_name"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["quarter"] = df["quarter"].astype(str).str.strip().str.upper()
    df["quarter"] = df["quarter"].str.replace(r"[^Q1-4]", "", regex=True)  # 안전하게

    # 숫자 컬럼 coercion
    num_cols = [c for c in DB_COL_ORDER if c not in ("corp_name", "year", "quarter")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 정렬(선택) : 기업명, 연도, 분기
    df = df.sort_values(["corp_name", "year", "quarter"], na_position="last").reset_index(drop=True)
    return df

def write_normalized_csv(df: pd.DataFrame, out_path: Path):
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"O normalized CSV saved: {out_path}")

def create_insert_sql(df: pd.DataFrame, out_sql: Path, table: str = TABLE_NAME, batch_size: int = BATCH_SIZE):
    """정규화된 DF → INSERT SQL 파일 생성 (트랜잭션 + 배치)"""
    cols_sql = ", ".join(f"`{c}`" for c in DB_COL_ORDER)
    prefix = f"INSERT INTO `{table}` ({cols_sql}) VALUES\n"

    with open(out_sql, "w", encoding="utf-8") as f:
        f.write(f"-- Auto-generated from dart_final.csv\n")
        f.write("BEGIN;\n\n")

        rows = []
        n = len(df)
        for i, row in df.iterrows():
            values = []
            for col in DB_COL_ORDER:
                val = row[col]
                if col in ("corp_name", "quarter"):
                    if pd.isna(val):
                        values.append("NULL")
                    else:
                        values.append(f"'{_escape_sql_str(str(val))}'")
                elif col == "year":
                    if pd.isna(val):
                        values.append("NULL")
                    else:
                        values.append(str(int(val)))
                else:
                    num = _coerce_number(val)
                    values.append("NULL" if num is None else str(num))

            rows.append(f"({', '.join(values)})")

            # 배치 출력
            if (i + 1) % batch_size == 0 or (i + 1) == n:
                f.write(prefix)
                f.write(",\n".join(rows))
                f.write(";\n\n")
                rows = []

        f.write("COMMIT;\n")

    print(f"O INSERT SQL saved: {out_sql}")

if __name__ == "__main__":
    script_dir = Path(__file__).parent

    src_csv = script_dir / "dart" / "dart_final.csv"                  # 입력
    normalized_csv = script_dir / "dart_data_normalized.csv" # 영문 헤더 CSV
    insert_sql = script_dir / "dart_data_insert.sql"         # INSERT SQL

    try:
        df_norm = load_and_normalize(src_csv)
        write_normalized_csv(df_norm, normalized_csv)
        create_insert_sql(df_norm, insert_sql)
    except Exception as e:
        print(f"X Error: {e}")
