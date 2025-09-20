CREATE DATABASE IF NOT EXISTS IE_project;
USE IE_project;

-- 1. ECOS 경제지표 데이터 테이블
-- 데이터 소스: ECOS_data.py (ECOS API)
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

-- 2. DART 건설업 재무데이터 테이블
-- 데이터 소스: 건설10_11년로드_2015~2024_연결_분기재무_정규화.csv (DART API)
CREATE TABLE dart_data (
id INT AUTO_INCREMENT PRIMARY KEY,
-- 기업명
corp_name VARCHAR(50) NOT NULL,
-- 기업코드
corp_code VARCHAR(8) NOT NULL,
-- 연도
year INT NOT NULL,
-- 분기(Q1,Q2,Q3,Q4)
quarter VARCHAR(2) NOT NULL,
-- 보고서기준일
report_date DATE NOT NULL,
-- 자산총계
total_assets DECIMAL(20, 2),
-- 부채총계
total_liabilities DECIMAL(20, 2),
-- 자본총계
total_equity DECIMAL(20, 2),
-- 매출액
revenue DECIMAL(20, 2),
-- 영업이익
operating_profit DECIMAL(20, 2),
-- 분기순이익
quarterly_profit DECIMAL(20, 2),
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
UNIQUE KEY unique_corp_period (corp_name, year, quarter),
INDEX idx_corp_name (corp_name),
INDEX idx_report_date (report_date)
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