# ECOS 데이터 전처리 FastAPI

ECOS 경제데이터를 DB에서 불러와 전처리 후 `final_features` 테이블에 저장하는 FastAPI 애플리케이션

## 주요 기능

1. **DB에서 ECOS 데이터 로드**: `ecos_data` 테이블에서 경제지표 데이터를 조회
2. **피쳐 엔지니어링**: 
   - 타겟 변수 차분 적용 (비정상성 제거)
   - 이동평균, 지연 특징, 변화율 생성
   - 상관관계 기반 피쳐 선택
3. **final_features 저장**: 전처리된 데이터를 DB의 `final_features` 테이블에 저장

## 기존 코드와의 호환성

FastAPI는 다음 노트북들의 로직을 재현:
- `preprocessing/preprocessing.ipynb`: ECOS 데이터 로드 및 전처리
- `modeling/predict.ipynb`: TASK 1-3 (데이터 로드, 피쳐 엔지니어링, DB 연결)

## 설치 및 실행

### 1. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

주요 패키지:
- FastAPI
- uvicorn
- pydantic
- pandas, numpy
- scikit-learn
- mysql-connector-python

### 2. 환경 설정

`DB/.env` 파일에 데이터베이스 연결 정보가 설정:

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=IE_project
```

### 3. API 서버 실행

방법 1: 직접 실행
```bash
cd final
python preprocessing.py
```

방법 2: uvicorn 사용
```bash
cd final
uvicorn preprocessing:app --reload --host 0.0.0.0 --port 8000
```

### 4. API 테스트

```bash
cd final
python test_api.py
```

## API 엔드포인트

### 1. `GET /`
- 설명: API 상태 확인
- 응답: API 정보 및 버전

### 2. `GET /health`
- 설명: 헬스 체크
- 응답: 서버 상태 및 타임스탬프

### 3. `POST /preprocess`
- 설명: ECOS 데이터 전처리 실행
- 기능:
  1. DB에서 ECOS 데이터 로드
  2. 피쳐 엔지니어링 수행
  3. final_features 테이블에 저장
- 응답:
  ```json
  {
    "success": true,
    "message": "ECOS 데이터 전처리 및 저장 완료",
    "processed_rows": 180,
    "feature_count": 26,
    "date_range": {
      "start_date": "201501",
      "end_date": "202409"
    }
  }
  ```

### 4. `GET /features/info`
- 설명: 저장된 final_features 정보 조회
- 응답: 피쳐 개수, 데이터 행 수, 날짜 범위, 피쳐 목록

### 5. `GET /data/preview?limit=10`
- 설명: 전처리된 데이터 미리보기
- 매개변수: `limit` (선택, 기본값: 10)
- 응답: 최신 데이터 n개 행

## 데이터 흐름

```
ecos_data (DB 테이블)
    ↓
[load_ecos_data] - DB에서 데이터 로드 및 기본 전처리
    ↓
[feature_engineering] - 피쳐 엔지니어링 (차분, 이동평균, 지연 등)
    ↓
[prepare_final_features] - DB 스키마에 맞게 데이터 준비
    ↓
final_features (DB 테이블) - 최종 저장
```

## 주요 전처리 과정

1. **데이터 타입 변환**: 수치형 컬럼을 numeric으로 변환
2. **날짜 처리**: YYYYMM 형식을 datetime으로 변환
3. **파생 변수 생성**:
   - `credit_spread`: 회사채 AA- - 국고채 3년
   - `term_spread`: 국고채 10년 - 국고채 3년
4. **타겟 변수 차분**: 비정상성 제거를 위한 1차 차분
5. **피쳐 엔지니어링**:
   - 이동평균 (3개월, 6개월)
   - 지연 특징 (1개월, 3개월, 6개월)
   - 변화율 계산
6. **피쳐 선택**:
   - 높은 상관관계 피쳐 제거 (임계값: 0.95)
   - 타겟과의 상관관계 기반 선택

## 타겟 변수

```python
TARGET_COLUMNS = [
    'construction_bsi_actual',  # 건설업 BSI 실적
    'base_rate',               # 한국은행 기준금리
    'housing_sale_price',      # 주택매매가격
    'm2_growth',              # M2 통화량 증가율
    'credit_spread'           # 신용 스프레드
]
```

## 최종 피쳐 스키마

`final_features` 테이블에 저장되는 컬럼들:
- `date`: 날짜 (YYYYMM 형식)
- `construction_bsi_actual_diff`: 건설업 BSI 차분
- `housing_sale_price_diff`: 주택가격 차분
- `m2_growth_diff`: M2 증가율 차분
- `credit_spread_diff`: 신용스프레드 차분
- `base_rate_diff`: 기준금리 차분
- 기타 파생 변수들 (이동평균, 지연 등)

## 사용 예시

```python
import requests

# 전처리 실행
response = requests.post("http://localhost:8000/preprocess")
result = response.json()

if result["success"]:
    print(f"처리 완료: {result['processed_rows']}행")
    print(f"피쳐 개수: {result['feature_count']}개")
else:
    print(f"처리 실패: {result['message']}")

# 결과 확인
info_response = requests.get("http://localhost:8000/features/info")
info = info_response.json()
print(f"저장된 데이터: {info['total_rows']}행, {info['total_features']}개 피쳐")
```

## 문제 해결

### 1. DB 연결 오류
- `.env` 파일의 DB 연결 정보 확인
- MySQL 서버 실행 상태 확인
- 네트워크 연결 확인

### 2. 데이터 없음 오류
- `ecos_data` 테이블에 데이터가 있는지 확인
- ECOS API 실행하여 데이터 수집 후 재시도

### 3. 패키지 오류
```bash
pip install fastapi uvicorn pydantic
```

## 향후 개선 사항

1. **비동기 처리**: 대용량 데이터 처리를 위한 비동기 작업
2. **캐싱**: Redis를 활용한 결과 캐싱
3. **스케줄링**: 정기적인 데이터 업데이트
4. **모니터링**: 로깅 및 성능 모니터링 추가
5. **검증**: 데이터 품질 검증 로직 추가