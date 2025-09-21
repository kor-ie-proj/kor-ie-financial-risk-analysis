# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import unicodedata
from pathlib import Path

# --- 설정 (Configuration) ---

# 1) 파일명을 데이터베이스 컬럼명으로 매핑합니다.
#    - CSV 파일 이름 (확장자 제외)을 키로, DB 컬럼명을 값으로 가집니다.
#    - 파일명의 한글은 NFC 정규화 형식을 사용합니다.
FILE_TO_DB_COLUMN_MAP = {
    "base_rate": "base_rate",
    "ccsi": "ccsi",
    "construction_bsi_actual": "construction_bsi_actual",
    "construction_bsi_forecast": "construction_bsi_forecast",
    "cpi": "cpi",
    "esi": "esi",
    "exchange_usd_원_달러종가_15_30": "exchange_usd_krw_close",
    "housing_lease_price": "housing_lease_price",
    "housing_sale_price": "housing_sale_price",
    "import_price_비금속광물": "import_price_non_metal_mineral",
    "import_price_철강1차제품": "import_price_steel_primary",
    "leading_index": "leading_index",
    "m2_growth": "m2_growth",
    "market_rate_국고채3년": "market_rate_treasury_bond_3yr",
    "market_rate_국고채10년": "market_rate_treasury_bond_10yr",
    "market_rate_회사채3년_AA_": "market_rate_corporate_bond_3yr_AA",
    "market_rate_회사채3년_BBB_": "market_rate_corporate_bond_3yr_BBB",
    "ppi_비금속광물": "ppi_non_metal_mineral",
    "ppi_철강1차제품": "ppi_steel_primary",
}

# 2) 데이터베이스 테이블에 삽입될 컬럼의 순서를 정의합니다.
#    - 최종 CSV 파일과 SQL INSERT 구문에서 이 순서를 따릅니다.
DB_COLUMN_ORDER = [
    "date", "base_rate", "ccsi", "construction_bsi_actual", "construction_bsi_forecast",
    "cpi", "esi", "exchange_usd_krw_close", "housing_lease_price", "housing_sale_price",
    "import_price_non_metal_mineral", "import_price_steel_primary", "leading_index", "m2_growth",
    "market_rate_treasury_bond_10yr", "market_rate_treasury_bond_3yr", "market_rate_corporate_bond_3yr_AA",
    "market_rate_corporate_bond_3yr_BBB", "ppi_non_metal_mineral", "ppi_steel_primary",
]

# --- 함수 정의 (Functions) ---

def normalize_filename(filename: str) -> str:
    """
    파일 경로에서 확장자를 제거하고, 파일명만 반환합니다.

    Args:
        filename (str): 원본 파일명 (예: 'base_rate.csv')

    Returns:
        str: 확장자가 제거된 파일명 (예: 'base_rate')
    """
    return Path(filename).stem

def merge_csv_files(source_dir: Path, output_csv_path: Path) -> pd.DataFrame:
    """
    지정된 디렉토리의 모든 CSV 파일을 하나의 데이터프레임으로 병합합니다.

    - 각 CSV 파일은 'date'와 'value' 컬럼을 가져야 합니다.
    - 파일명을 기반으로 'value' 컬럼의 이름을 DB 컬럼명으로 변경합니다.
    - 모든 데이터를 'date' 기준으로 outer join하여 병합합니다.

    Args:
        source_dir (Path): CSV 파일들이 있는 디렉토리 경로
        output_csv_path (Path): 병합된 결과가 저장될 CSV 파일 경로

    Returns:
        pd.DataFrame: 병합 및 정렬된 데이터프레임
    """
    all_dataframes = []
    processed_files, skipped_files = [], []

    for fpath in source_dir.glob("*.csv"):
        # 최종 결과물 파일은 건너뜁니다.
        if fpath.name in ("ecos_data.csv", "ecos_data_insert.sql"):
            continue

        # 파일명 정규화 (자음/모음 조합 문제 해결)
        cleaned_name = unicodedata.normalize('NFC', normalize_filename(fpath.name))
        
        # 파일명 매핑 확인
        if cleaned_name not in FILE_TO_DB_COLUMN_MAP:
            skipped_files.append((fpath.name, f"매핑 정보 없음: '{cleaned_name}'"))
            continue

        try:
            df = pd.read_csv(fpath, encoding="utf-8-sig")
        except Exception as e:
            skipped_files.append((fpath.name, f"파일 읽기 오류: {e}"))
            continue

        # 필수 컬럼 ('date', 'value') 존재 여부 확인
        if not {"date", "value"}.issubset(df.columns):
            skipped_files.append((fpath.name, "필수 컬럼(date, value) 없음"))
            continue
        
        # 데이터 정제
        # 1. 'date' 컬럼: YYYYMM 형식의 6자리 문자열로 변환
        df["date"] = df["date"].astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 6)
        # 2. 'value' 컬럼: 숫자 타입으로 변환 (변환 불가 시 NaT으로 처리)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        
        # DB 컬럼명으로 변경 후 리스트에 추가
        db_col_name = FILE_TO_DB_COLUMN_MAP[cleaned_name]
        df.rename(columns={"value": db_col_name}, inplace=True)
        all_dataframes.append(df[["date", db_col_name]])
        processed_files.append((fpath.name, db_col_name))

    if not all_dataframes:
        raise RuntimeError("병합할 CSV 파일이 하나도 없습니다.")

    # 데이터프레임 병합
    # reduce와 유사하게, 리스트의 모든 데이터프레임을 'date' 기준으로 순차적으로 병합
    merged_df = all_dataframes[0]
    for df in all_dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on="date", how="outer")

    # DB 컬럼 순서에 맞게 정렬 및 누락된 컬럼 추가
    for col in DB_COLUMN_ORDER:
        if col not in merged_df.columns:
            merged_df[col] = pd.NA  # 누락된 컬럼은 결측치로 채움

    merged_df = merged_df[DB_COLUMN_ORDER]
    merged_df.sort_values("date", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    # 결과 저장
    merged_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    # 처리 결과 출력
    print(f"O 통합 CSV 생성 완료: {output_csv_path}")
    print(f"   - 포함된 파일 수: {len(processed_files)}")
    for orig, col in processed_files:
        print(f"     • {orig} -> {col}")
    if skipped_files:
        print("\n   - 스킵된 파일:")
        for orig, reason in skipped_files:
            print(f"     • {orig} (사유: {reason})")

    return merged_df

def create_insert_sql(merged_df: pd.DataFrame, sql_output_path: Path, batch_size: int = 500):
    """
    병합된 데이터프레임을 기반으로 SQL INSERT 문을 생성합니다.

    - 대용량 데이터 삽입을 위해 'batch_size' 단위로 INSERT 문을 분할합니다.
    - 모든 작업은 하나의 트랜잭션(BEGIN-COMMIT)으로 묶입니다.

    Args:
        merged_df (pd.DataFrame): 병합된 데이터프레임
        sql_output_path (Path): 생성될 SQL 파일의 경로
        batch_size (int): 한 번의 INSERT 문에 포함될 데이터 행의 수
    """
    # SQL 쿼리 작성을 위해 데이터프레임 복사
    df_for_sql = merged_df.copy()

    # INSERT 구문의 컬럼 목록 생성
    columns_str = ",\n    ".join([f"`{col}`" for col in DB_COLUMN_ORDER])
    insert_prefix = f"INSERT INTO `ecos_data` (\n    {columns_str}\n) VALUES\n"

    with open(sql_output_path, "w", encoding="utf-8") as f:
        f.write("-- Auto-generated from merged ecos_data.csv\n")
        f.write("BEGIN;\n\n")

        # 데이터를 batch_size 만큼 나누어 처리
        for i in range(0, len(df_for_sql), batch_size):
            chunk = df_for_sql.iloc[i : i + batch_size]
            
            value_rows = []
            for _, row in chunk.iterrows():
                row_values = []
                for col in DB_COLUMN_ORDER:
                    value = row.get(col)
                    if pd.isna(value):
                        row_values.append("NULL")
                    elif col == "date":
                        # YYYYMM 형식의 문자열로 삽입
                        row_values.append(f"'{str(value)[:6]}'")
                    else:
                        # 나머지 컬럼은 숫자로 처리
                        row_values.append(str(value))
                value_rows.append(f"({', '.join(row_values)})")
            
            # INSERT 구문 작성
            f.write(insert_prefix)
            f.write(",\n".join(value_rows) + ";\n\n")

        f.write("COMMIT;\n")
    
    print(f"O INSERT SQL 생성 완료: {sql_output_path}")

# --- 실행 (Main Execution) ---

if __name__ == "__main__":
    # pathlib을 사용하여 현재 스크립트 파일의 위치를 기준으로 경로를 설정합니다.
    script_dir = Path(__file__).parent
    source_data_dir = script_dir / "ecos"
    
    # 최종 결과물이 저장될 파일 경로
    merged_csv_path = script_dir / "ecos_data.csv"
    sql_output_path = script_dir / "ecos_data_insert.sql"

    # 1. 개별 CSV 파일 병합
    try:
        df_merged = merge_csv_files(source_data_dir, merged_csv_path)

        # 2. 병합된 CSV를 SQL로 변환
        create_insert_sql(df_merged, sql_output_path, batch_size=500)
        
    except Exception as e:
        print(f"X 처리 중 오류 발생: {e}")