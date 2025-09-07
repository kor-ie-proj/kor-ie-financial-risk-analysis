#!/usr/bin/env bash
set -e
# 환경변수로 받은 설정 사용
mlflow server \
  --backend-store-uri "${BACKEND_STORE_URI}" \ # mysql db
  --default-artifact-root "${ARTIFACT_ROOT}" \ # minio
  --host 0.0.0.0 --port 5000
