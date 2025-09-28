# Repository Guidelines

## Project Structure & Module Organization
Core services live in dedicated folders: `be/` for orchestration APIs, `training/` for model fitting, `inference/` for serving, and `mlflow/` for the tracking server wrapper. Shared data tooling sits in `util/` (CSV loaders for DART and ECOS), while database bootstrap SQL is under `db/init/`. Notebooks and exploratory checks belong in `testing/`. The Compose stack and environment configuration (`docker-compose.yaml`, `.env`) bind the services together; update them in lockstep when adding new components.

## Build, Test, and Development Commands
Bootstrap the stack with `cp .env.example .env`, then `docker-compose build` to rebuild service images and `docker-compose up` to launch MySQL, MinIO, MLflow, training, and inference containers. During service work, run individuals locally via `uvicorn app.main:app --reload --port 8000` from the target module folder. Exercise training with `curl -X POST http://localhost:8002/train`; refresh inference after a successful run and probe it with `curl -X POST -H 'Content-Type: application/json' -d '{"months_to_predict": 3}' http://localhost:8001/predict`.

## Coding Style & Naming Conventions
Python code follows PEP 8: four-space indentation, snake_case for functions and variables, CapWords for classes, and ENV_VAR casing for configuration keys. Prefer type hints and FastAPI/Pydantic models for request or response payloads. Mirror existing module naming (`training_logic.py`, `model.py`) and keep inline comments bilingual only when user-facing (default to English).

## Testing Guidelines
Formal pytest suites are absent; keep regression notebooks in `testing/` and name them with the feature under review, e.g., `inference-latency.ipynb`. For automated additions, colocate unit tests beside the module with a `test_*.py` prefix and ensure they run against a locally started stack. Always confirm MLflow records by checking the `LSTM_Financial_Forecast` experiment after training.

## Commit & Pull Request Guidelines
Follow the existing conventional prefix: `<scope>: <imperative summary>` (examples: `docs: update readme`, `inference: impl update_model endpoint`). Write commits that focus on a single concern. Pull requests should summarize service-level impacts, link related issues or tracking docs, list test evidence (curl samples, notebook outputs), and note any environment variable changes so reviewers can mirror them.

## Security & Configuration Tips
Never commit secrets—populate `.env` locally and document required keys instead. Validate MinIO and MySQL credentials before pushing changes, and prefer referencing S3 bucket names and database URIs via environment variables rather than literals in code.
