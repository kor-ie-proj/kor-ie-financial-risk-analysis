class DummyModel:
    def predict_fs(self, features: dict) -> dict:
        base = float(features.get("base_rate", 3.0))
        ccsi = float(features.get("ccsi", 100))
        next_roe = (ccsi / 100.0) - (base * 0.05)
        return {"next_quarter_roe": round(next_roe, 4)}

    def risk_score(self, fs_pred: dict, features: dict) -> float:
        roe = float(fs_pred["next_quarter_roe"])
        liabilities = float(features.get("total_liabilities", 0))
        assets = float(features.get("total_assets", 1))
        leverage = liabilities / max(assets, 1e-6)
        raw = (1 - max(min(roe, 1), -1)) * 0.5 + min(leverage, 2.0) * 0.25
        return round(min(max(raw, 0.0), 1.0), 4)

def load_model():
    # TODO: MLflow/MinIO에서 최신 모델 로드
    return DummyModel()
