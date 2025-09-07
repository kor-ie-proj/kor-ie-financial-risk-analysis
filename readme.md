# Corp Financial Risk Analysis Project

## Overview

기업의 재무제표 및 외부 지표들을 통한 금융 리스크 분석 프로젝트

## Structure

```
risk-analytics/
├─ docker-compose.yaml
├─ .env.example
├─ cloudflared/
│  └─ config.yml
├─ be/                      # BE Server (FastAPI, API 게이트웨이 역할)
│  ├─ Dockerfile
│  └─ app/
│     ├─ main.py
│     └─ db.py
├─ inference/               # Inference Server (FastAPI)
│  ├─ Dockerfile
│  └─ app/
│     ├─ main.py
│     ├─ model.py
│     └─ db.py
├─ training/                # Training Server (FastAPI)
│  ├─ Dockerfile
│  └─ app/
│     ├─ main.py
│     └─ model.py
├─ mlflow/                  # MLflow Server
│  ├─ Dockerfile
│  └─ start.sh
├─ fetcher/                 # (optional) 주기적 데이터 fetch 워커
│  ├─ Dockerfile
│  └─ app.py
└─ db/                      # mysql 데이터베이스
   └─ init/
      └─ 01_schema.sql
```

- 외부 트래픽: cloudflared(Cloudflare Tunnel) ↔︎ BE/MLflow/MinIO(내부 포트만)
- 내부 네트워크: risknet (compose 네트워크 하나)
- 상태 저장: mysql(RDB로 대체 가능), minio(S3 대체 가능)
- 모델 자산: MLflow(Tracking DB=MySQL, Artifacts=MinIO)
- Inference는 DB 읽고 예측/리스크 산출, 결과는 DB에 저장
- Training은 모델 재학습 후 MLflow/MinIO에 등록(프로토타입)

## how to run

### cloudflare

- 도메인 네임서버를 Cloudflare로 전환
- Zero Trust → Tunnels에서 새 터널 생성, 토큰 발급
- ./cloudflared/config.yml 편집 후(서브도메인 매핑) .env에 토큰 넣기
- Public Hostname(예: api.example.com, mlflow.example.com 등) 추가

### set up .env

```bash
cp .env.example .env
# 필요시 값 수정
docker compose up -d --build
```

### retrain trigger

```bash
# Tunnel에서 training도 열고 싶다면 (선택) ingress에 추가 후 실행
curl -X POST https://train.example.com/train
```

## note

### mlflow python library

- 오픈소스 MLOps 프레임워크
- 주요 기능
  - 실험 관리(Experiment Tracking): 학습 시 어떤 하이퍼파라미터, 어떤 데이터셋, 어떤 성능 지표가 나왔는지 기록.
  - 모델 저장 & 버저닝(Model Registry): 학습된 모델을 저장하고 버전별로 관리.
  - 모델 배포(Serving): 저장된 모델을 REST API 서버 등으로 서빙 가능.
  - 아티팩트 저장(Artifacts): 모델 파일뿐 아니라 학습 로그, 시각화 파일 등도 저장.
- MLflow는 Tracking 서버와 Python 라이브러리(mlflow 패키지) 두 가지로 구성
  - Tracking 서버 = mlflow server로 띄운 HTTP 서버
  - Python 라이브러리 = import mlflow 해서 코드 안에서 로그/모델 저장/불러오기 하는 API

### how MLflow works

**Training 컨테이너**

- 학습 끝난 모델을 mlflow.log_artifact() 같은 함수로 MLflow 서버에 업로드.
- 모델 파일(model.pkl)은 실제로는 **MinIO(S3 호환 저장소)**에 저장되고, MLflow DB(MySQL)에 “이 run에서 이 파일을 이 위치에 저장했음” 메타데이터가 기록됨.

**Inference 컨테이너**

- 원래라면 MLflow에서 “가장 최신 버전 모델” 정보를 조회해서 MinIO에서 다운로드 → 로딩.
- 지금은 프로토타입이라 더미 모델(DummyModel)만 쓰지만, 실제 모델을 연결하면 여기서 MLflow/MinIO를 통해 가져와서 서빙하게 돼요.

**즉, MLflow = 모델 관리 허브**

- Training 쪽에서 “이런 모델을 학습했다” → MLflow에 등록
- Inference 쪽에서 “최신 모델을 달라” → MLflow 통해 불러옴
