CREATE DATABASE IF NOT EXISTS IE_project;
USE IE_project;

-- 1. ECOS 경제지표 데이터 테이블
CREATE TABLE ecos_data (
id INT AUTO_INCREMENT PRIMARY KEY,
-- YYYYMM 형식
date VARCHAR(6) NOT NULL,
-- 기준금리
base_rate DECIMAL(5, 3),
-- 소비자동향지수
ccsi DECIMAL(5, 1),
-- 건설업경기실사지수(실적)
construction_bsi_actual DECIMAL(5, 1),
-- 건설업경기실사지수(전망)
construction_bsi_forecast DECIMAL(5, 1),
-- 소비자물가지수
cpi DECIMAL(8, 3),
-- 경제심리지수
esi DECIMAL(5, 1),
-- 원달러환율 (종가 15:30)
exchange_usd_krw_close DECIMAL(8, 2),
-- 주택전세가격지수
housing_lease_price DECIMAL(8, 3),
-- 주택매매가격지수
housing_sale_price DECIMAL(8, 3),
-- 수입물가지수_비금속광물
import_price_non_metal_mineral DECIMAL(8, 2),
-- 수입물가지수_철강1차제품
import_price_steel_primary DECIMAL(8, 2),
-- 선행종합지수
leading_index DECIMAL(8, 1),
-- 통화유동성_M2증가율
m2_growth DECIMAL(8, 2),
-- 국고채10년수익률
market_rate_treasury_bond_10yr DECIMAL(8, 3),
-- 국고채3년수익률
market_rate_treasury_bond_3yr DECIMAL(8, 3),
-- 회사채3년AA
market_rate_corporate_bond_3yr_AA DECIMAL(8, 3),
-- 회사채3년BBB
market_rate_corporate_bond_3yr_BBB DECIMAL(8, 3),
-- 생산자물가지수_비금속광물
ppi_non_metal_mineral DECIMAL(8, 2),
-- 생산자물가지수_철강1차제품
ppi_steel_primary DECIMAL(8, 2),
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
UNIQUE KEY unique_date (date)
);

-- 2. dart 제무제표 데이터 테이블
DROP TABLE IF EXISTS dart_data;
CREATE TABLE dart_data (
  id INT AUTO_INCREMENT PRIMARY KEY,

  -- identity
  corp_name VARCHAR(100) NOT NULL,
  year INT NOT NULL,
  quarter VARCHAR(2) NOT NULL,            -- e.g., 'Q1', 'Q2', 'Q3', 'Q4'

  -- absolute values
  total_assets DECIMAL(20, 2),
  total_liabilities DECIMAL(20, 2),
  total_equity DECIMAL(20, 2),
  revenue DECIMAL(20, 2),
  operating_profit DECIMAL(20, 2),
  quarterly_profit DECIMAL(20, 2),

  -- ratios / growth (비율/성장률)
  debt_ratio DECIMAL(12, 6),              -- 부채비율 (% 단위면 그대로 수치 삽입)
  equity_ratio DECIMAL(12, 6),            -- 자기자본비율
  roa DECIMAL(12, 6),
  roe DECIMAL(12, 6),
  revenue_growth DECIMAL(12, 6),
  operating_profit_growth DECIMAL(12, 6),
  net_income_growth DECIMAL(12, 6),

  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

  UNIQUE KEY unique_corp_period (corp_name, year, quarter),
  INDEX idx_corp_name (corp_name)
);


-- 3. 모델 예측 결과 테이블
-- 예측 결과: construction_bsi_actual, base_rate, housing_sale_price, m2_growth, credit_spread
CREATE TABLE model_output (
	id INT AUTO_INCREMENT PRIMARY KEY,
	-- YYYYMM 형식
	date VARCHAR(6) NOT NULL,
	-- 예측값 컬럼들
	construction_bsi_actual DECIMAL(5, 1),
	base_rate DECIMAL(5, 3),
	housing_sale_price DECIMAL(8, 3),
	m2_growth DECIMAL(8, 2),
	credit_spread DECIMAL(8, 3),
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	UNIQUE KEY unique_date (date)
);