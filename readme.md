# KOR-IE Finance Risk Assessment

> local의 경우 `local-ver` 브랜치, production 배포의 경우 `prod-ver` 브랜치를 사용 필요

## model Design and Operation

- [Model Design and Operation (English)](model-en.md)
- [모델 설계 및 운영 (한국어)](model-kr.md)

## Local Architecture

The `local-ver` branch runs the entire workflow on a single machine via Docker containers. The services communicate over the `risknet` bridge network created by `docker-compose`:

```
frontend (Next.js, :3000)
   ↓ REST (CORS)
be (FastAPI, :8003)
   ↓ REST
inference (FastAPI, :8001) ← MLflow model registry (:5001, backed by MinIO)
   ↘ SQLAlchemy
    MySQL (:3306)

training (FastAPI, :8002)
   ↘ writes runs to MLflow → artifacts stored in MinIO (:9000/:9001)
   ↘ reads/writes features in MySQL
```

### Service Inventory

| Service     | Framework / Image         | Exposed Port | Key Dependencies     | Responsibility                                               |
| ----------- | ------------------------- | ------------ | -------------------- | ------------------------------------------------------------ |
| `mysql`     | `mysql:8.0`               | 3306         | –                    | Primary OLTP store for ECOS/DART series and company metadata |
| `minio`     | `minio/minio:latest`      | 9000, 9001   | –                    | Object storage for MLflow artifacts                          |
| `mlflow`    | Custom image (`./mlflow`) | 5001         | mysql, minio         | Tracking UI + model registry backed by MySQL / MinIO         |
| `training`  | FastAPI (`./training`)    | 8002         | mysql, mlflow, minio | Launches LSTM fitting jobs and logs runs                     |
| `inference` | FastAPI (`./inference`)   | 8001         | mysql, mlflow, minio | Serves the latest promoted model for risk scoring            |
| `be`        | FastAPI (`./be`)          | 8003         | mysql, inference     | Aggregates risk outputs and raw indicators for the frontend  |
| `frontend`  | Next.js (`./frontend`)    | 3000         | be                   | Operator dashboard                                           |

> 참고: `fetcher` 와 같은 데이터 수집 마이크로서비스는 아직 별도의 컨테이너로 분리되지 않았습니다. 필요한 경우 `util/` 스크립트를 통해 수동으로 데이터를 적재합니다.

## Local Environment Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/shj1081/kor-ie-finance-risk-analysis.git
   ```

2. Navigate to the project directory and set up environment:

   ```bash
    cd kor-ie-finance-risk
    cp .env.example .env
   ```

3. ECOS/DART data migration (only for the first time / db/init 디렉토리에 초기화 스크립트가 없을 때만):

   - `util` 디렉토리의 `ecos_insert.sql`, `dart_insert.sql` 생성 스크립트를 통해 sql 파일을 생성
   - `db/init` 에 `0x_nameOfSqlFile.sql` 형식으로 복사 시 자동으로 초기화 시점에 반영됨.
   - MySQL 컨테이너가 이미 실행 중이라면, `docker-compose down -v` 로 볼륨을 삭제 후 재기동 필요.

4. 서비스 빌드 (모든 컨테이너를 한 번 준비)

   ```bash
   docker-compose build
   ```

5. Start the Docker containers (launches the full local architecture):

   ```bash
    docker-compose up
    docker-compose up -d  # 백그라운드 실행
   ```

   > Note:
   >
   > - 이미 사용중인 포트나 컨테이너 이름이 없도록 주의.
   > - 초기 실행 시, inference 서버는 production model의 부재로 down 상태가 되는 것이 정상.

6. 초기 MinIO bucket 생성 (minio)

   - `http://localhost:9000` 접속 후, ID/PW: `minioadmin/minioadminpass`로 로그인
   - `mlflow-artifacts` 버킷 생성

7. 학습 테스트 (local training 서비스 확인)

   ```bash
   curl -X POST http://localhost:8002/train
   ```

   - `8002` 번 포트의 `train` 서버로 학습 요청
   - `mlflow-artifacts` 버킷에 모델이 저장됨.
   - `mlflow` 서버에 `LSTM_Financial_Forecast` 실험이 등록됨. (학습 상태 및 최종 parameter, metric 등 확인 가능)

8. 초기 production model 설정 (MLflow UI)

   - `http://localhost:5000` 접속 후 experiment: `LSTM_Financial_Forecast` 선택
   - 가장 최근에 학습된 run 선택 후, `Register Model` 클릭 (model name: `lstm_financial_forecast_model` 으로 생성 후 등록)
   - `model` 탭에서 해당 모델의 version 클릭 후, `Stage`를 `Production`으로 변경

9. 추론 테스트 (inference 서비스 확인)

   - `8001` 번 포트의 `inference` 서버 컨테이너를 재시작 (또는 `docker-compose down` 후 `docker-compose up`)
   - 다음과 같이 추론 요청

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"months_to_predict": 3}' http://localhost:8001/predict
   ```

10. 프론트 엔드 접속

    - `http://localhost:3000` 접속
    - `8003` 번 포트의 `be` 서버를 통해 MySQL에서 raw indicator 데이터를 읽고, `8001` 번 포트의 `inference` 서버에서 risk score를 받아와서 화면에 표시

## Production (`prod-ver`) Deployment

`prod-ver` 브랜치는 동일한 컨테이너 구성을 Cloudflare 터널 뒤쪽에서 외부로 노출하는 것을 전제로 합니다. 프런트엔드와 API를 하나의 도메인으로 합치기 위해 **모든 FastAPI 엔드포인트는 `/api` 프리픽스**를 갖습니다.

### 공개 엔드포인트 구조

- `GET https://korie.hyzoon.dev/api/health` — 백엔드 상태 확인
- `GET https://korie.hyzoon.dev/api/companies` — 기업 목록 조회
- `GET https://korie.hyzoon.dev/api/companies/{corp_name}/risk` — 위험도 조회
- `POST https://korie.hyzoon.dev/api/companies/{corp_name}/risk/manual` — 수동 조정 기반 위험도 재계산

프런트엔드는 동일한 도메인의 루트 경로(`https://korie.hyzoon.dev`)에서 렌더링되며, 내부 요청은 위의 `/api/*` 경로로 라우팅됩니다. `be` 서비스의 `ALLOWED_ORIGINS` 환경 변수에 해당 도메인을 반드시 포함시켜야 합니다.

### Cloudflared 터널 설정

1. Cloudflare 대시보드에서 터널을 생성하고 자격 증명(`*.json`)을 받아 로컬에 저장합니다.
2. 다음과 같이 `cloudflared` 설정 파일을 준비합니다(예: `~/.cloudflared/config.yml`).

   ```yaml
   tunnel: <tunnel_name>
   credentials-file: ~/.cloudflared/<tunnel_uuid>.json

   ingress:
   - hostname: korie.hyzoon.dev
      path: /api/*
      service: http://localhost:8003

   - hostname: korie.hyzoon.dev
      service: http://localhost:3000

   - hostname: korie_mlflow.hyzoon.dev
      service: http://localhost:5001

   - hostname: korie_minio.hyzoon.dev
      service: http://localhost:9001

   - service: http_status:404
   ```

   - 루트 경로는 Next.js 프런트엔드(포트 3000)로 전달합니다.
   - `/api/*` 트래픽은 FastAPI `be` 서비스(포트 8003)로 프록시되어 CORS 없이 동일 오리진을 유지합니다.

3. 터널 실행:

   ```bash
   # dns 레코드 설정 (최초 1회)
   cLoudflared tunnel route dns <tunnel_uuid> korie.hyzoon.dev
   cLoudflared tunnel route dns <tunnel_uuid> korie_mlflow.hyzoon.dev
   cloudflared tunnel route dns <tunnel_uuid> korie_minio.hyzoon.dev

   # 터널 시작
   cLoudflared tunnel --config ./cLoudflared/config.yml run
   ```

### 프로덕션 환경 변수 체크리스트

- `NEXT_PUBLIC_API_BASE_URL=https://korie.hyzoon.dev/api`
- `ALLOWED_ORIGINS=https://korie.hyzoon.dev`
- `INFERENCE_SERVER_URL=http://inference:8000` (Docker 네트워크 내 주소)
- 민감 정보(`MYSQL_ROOT_PASSWORD`, `AWS_SECRET_ACCESS_KEY` 등)는 `.env` 에서만 관리하고 Git에 포함하지 않습니다.
- cloudflared 자격 증명 파일(`*.json`)도 Git에 포함하지 않습니다.

Cloudflared 터널이 올라와 있으면 외부 사용자는 `https://korie.hyzoon.dev` 에 접속해 대시보드와 API를 동시에 이용할 수 있고, 내부 모델 업데이트 루틴은 기존과 동일하게 Docker Compose 스택을 통해 수행합니다.

## Prod Environment Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/shj1081/kor-ie-finance-risk-analysis.git
   ```

2. Navigate to the project directory and set up environment:

   ```bash
    cd kor-ie-finance-risk
    cp .env.example .env
   ```

3. ECOS/DART data migration (only for the first time / db/init 디렉토리에 초기화 스크립트가 없을 때만):

   - `util` 디렉토리의 `ecos_insert.sql`, `dart_insert.sql` 생성 스크립트를 통해 sql 파일을 생성
   - `db/init` 에 `0x_nameOfSqlFile.sql` 형식으로 복사 시 자동으로 초기화 시점에 반영됨.
   - MySQL 컨테이너가 이미 실행 중이라면, `docker-compose down -v` 로 볼륨을 삭제 후 재기동 필요.

4. 서비스 빌드 (모든 컨테이너를 한 번 준비)

   ```bash
   docker-compose build
   ```

5. Start the Docker containers (launches the full local architecture):

   ```bash
    docker-compose up
    docker-compose up -d  # 백그라운드 실행
   ```

   > Note:
   >
   > - 이미 사용중인 포트나 컨테이너 이름이 없도록 주의.
   > - 초기 실행 시, inference 서버는 production model의 부재로 down 상태가 되는 것이 정상.

6. 초기 MinIO bucket 생성 (minio)

   - `https://korie_minio.hyzoon.dev` 접속 후, ID/PW: `minioadmin/minioadminpass`로 로그인
   - `mlflow-artifacts` 버킷 생성

7. 학습 테스트 (prod 서버에서 실행해야 함)

   ```bash
   curl -X POST http://localhost:8002/train
   ```

   - `8002` 번 포트의 `train` 서버로 학습 요청
   - `mlflow-artifacts` 버킷에 모델이 저장됨.
   - `mlflow` 서버에 `LSTM_Financial_Forecast` 실험이 등록됨. (학습 상태 및 최종 parameter, metric 등 확인 가능)

8. 초기 production model 설정 (MLflow UI)

   - `https://korie_mlflow.hyzoon.dev` 접속 후 experiment: `LSTM_Financial_Forecast` 선택
   - 가장 최근에 학습된 run 선택 후, `Register Model` 클릭 (model name: `lstm_financial_forecast_model` 으로 생성 후 등록)
   - `model` 탭에서 해당 모델의 version 클릭 후, `Stage`를 `Production`으로 변경

9. 추론 테스트 (inference 서비스 확인)

   - `8001` 번 포트의 `inference` 서버 컨테이너를 재시작 (또는 `docker-compose down` 후 `docker-compose up`)
   - 다음과 같이 추론 요청

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"months_to_predict": 3}' http://localhost:8001/predict
   ```

10. 프론트 엔드 접속

    - `https://korie.hyzoon.dev` 접속
    - `8003` 번 포트의 `be` 서버를 통해 MySQL에서 raw indicator 데이터를 읽고, `8001` 번 포트의 `inference` 서버에서 risk score를 받아와서 화면에 표시
