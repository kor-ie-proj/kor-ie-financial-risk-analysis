# KOR-IE Finance Risk Assessment

## Implemented containers

- **db**: MySQL database container (3306:3306)
- **mlflow**: MLflow tracking server (5001:5000)
- **minio**: MinIO object storage for MLflow artifacts (9000:9000, 9001:9001)
- **train**: Model training server (8002:8000)
- **inference**: Model inference server (8001:8000)

## Not Implemented containers (yet)

- **inference**: risk assessment logic in inference server
- **fetcher**: Data fetching and preprocessing server
- **be**: Central backend server for orchestrating tasks
- **fe**: Frontend server for user interface

## Initial Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/shj1081/kor-ie-finance-risk.git
   ```

2. Navigate to the project directory and set up environment:

   ```bash
    cd kor-ie-finance-risk
    cp .env.example .env
   ```

3. ECOS data migration (only for the first time):

   - `util` 디렉토리의 `ecos_insert.sql` 생성 스크립트를 통해 sql 파일을 생성
   - `db` 디렉토리의 `init.sql` 파일에 `ecos_insert.sql` 파일 내용을 붙여 넣으면 자동으로 컨테이너 실행 시 DB에 반영됨.
   - datagrip 같은 DB 관리 툴을 통해 컨테이너 뜬 후에 sql 문을 실행하거나 `docker exec -it <db_container_name> bash`로 접속 후 `mysql -u root -p < ecos_insert.sql` 명령어로 실행하여도 무방.

4. docker-compose build

   ```bash
   docker-compose build
   ```

5. Start the Docker containers:

   ```bash
    docker-compose up
   ```

   > Note:
   >
   > - 이미 사용중인 포트나 컨테이너 이름이 없도록 주의.
   > - 초기 실행 시, inference 서버는 production model의 부재로 down 상태가 되는 것이 정상.

6. 초기 minio bucket 생성 (minio)

   - `http://localhost:9000` 접속 후, ID/PW: `minioadmin/minioadminpass`로 로그인
   - `mlflow-artifacts` 버킷 생성

7. 학습 테스트

   ```bash
   curl -X POST http://localhost:8002/train
   ```

   - `8002` 번 포트의 `train` 서버로 학습 요청
   - `mlflow-artifacts` 버킷에 모델이 저장됨.
   - `mlflow` 서버에 `LSTM_Financial_Forecast` 실험이 등록됨. (학습 상태 및 최종 parameter, metric 등 확인 가능)

8. 초기 production model 설정 (mlflow)

   - `http://localhost:5000` 접속 후 experiment: `LSTM_Financial_Forecast` 선택
   - 가장 최근에 학습된 run 선택 후, `Register Model` 클릭 (model name: `lstm_financial_forecast_model` 으로 생성 후 등록)
   - `model` 탭에서 해당 모델의 version 클릭 후, `Stage`를 `Production`으로 변경

9. 추론 테스트

   - `8001` 번 포트의 `inference` 서버 컨테이너를 재시작 (또는 `docker-compose down` 후 `docker-compose up`)
   - 다음과 같이 추론 요청

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"months_to_predict": 3}' http://localhost:8001/predict
   ```
