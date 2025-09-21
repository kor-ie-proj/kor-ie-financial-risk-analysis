#!/usr/bin/env bash
set -e

# 환경변수로 받은 설정을 사용하여 MLflow 서버를 실행합니다.
mlflow server \
  --backend-store-uri "${BACKEND_STORE_URI}" \
  --default-artifact-root "${ARTIFACT_ROOT}" \
  --host 0.0.0.0 --port 5000