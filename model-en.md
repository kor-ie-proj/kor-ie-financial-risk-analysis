# Model Design and Operation

## Data Sources
- **ECOS macro indicators** (`ecos_data` table) supply monthly time series for `construction_bsi_actual`, `base_rate`, `housing_sale_price`, and `m2_growth`.
- **DART financial statements** (`dart_data` table) provide quarterly corporate metrics and raw values used for risk flags and percentile scoring.
- Training and inference share the same ECOS feature space; inference extends results with DART-driven heuristics.

## Training Pipeline (`training/app/training_logic.py`)
### Preprocessing & Feature Engineering
- Parse ECOS `date` strings (YYYYMM) to timestamps, sort, and index by date.
- Interpolate missing values linearly in both directions.
- Derive first differences for every available target, then build feature blocks per target:
  - Rolling means over 3 and 6 months, percentage change of the differenced signal, and lags at 1, 3, and 6 months.
  - Original level lags (1, 3, 6) to retain low-frequency context.
- Drop rows with remaining gaps and prune highly collinear features by removing any pair with |corr| > 0.95.
- Rank surviving features by average absolute correlation to all differenced targets; keep `max(20, 50% of features)` as the final input vector.

### Sequence Preparation & Scaling
- Build sliding windows of length `seq_length` (candidate hyperparameter) for inputs and next-step targets.
- Split sequences chronologically into train (≈80%), validation (≈10%), and test (≈10%).
- Fit `StandardScaler` instances on flattened train windows for X and y, then transform every split.
- Convert to `TensorDataset`/`DataLoader` with GPU placement and maintain temporal order (no shuffling for validation/test).

### Baselines & Metrics
- Produce naive, seasonal naive (12-month), and historical mean baselines for differenced targets.
- Evaluate each baseline with RMSE, MAE, and R² per target to benchmark the LSTM.
- During final evaluation, also compute directional accuracy on the denormalized differenced series.

### Model & Optimization
- Model: `MultivariateLSTM(input_dim=len(final_features), hidden_size, num_layers, output_dim=len(targets), dropout)`.
- Hyperparameter search via Optuna (50 trials or 10‑minute timeout) over hidden size, layer count, dropout, learning rate, batch size, sequence length, optimizer type (Adam/AdamW), weight decay, and loss function (MSE/MAE/Huber).
- Each trial trains for up to 30 epochs with gradient clipping (norm 1.0), early stopping (patience 10), and validation loss pruning.
- Final training reuses best params, runs up to 200 epochs with ReduceLROnPlateau (factor 0.5, patience 5), early stopping (patience 20), and MLflow logging of per-epoch losses.

### Artifacts & Logging
- Record best trial metrics, baseline scores, final losses, and test metrics to MLflow (`LSTM_Financial_Forecast`).
- Persist pickled artifacts (`model_state_dict`, scalers, feature lists, hyperparameters, loss history, metrics) plus the PyTorch model (with explicit `requirements.txt`) for downstream inference.

## Inference Pipeline (`inference/app/inference_logic.py`)
### Forecasting Monthly Targets
- Load ECOS history, enforce identical preprocessing (date parsing, interpolation, diff-based features, lag construction).
- Validate that every stored `final_feature` exists post-transformation; abort if any are missing.
- Slice the most recent `seq_length` rows, scale with persisted `scaler_X`, and predict next-step differenced targets via the trained LSTM.
- Invert scaling on outputs, then iteratively accumulate differenced predictions to produce `months_to_predict` forecasts on the original target scale.

### Heuristic Risk Aggregation
- Fetch DART snapshots for the selected `corp_name` and the full peer population to support percentile scoring; identify the latest reporting quarter.
- Compute raw flag scores from DART time series (equity erosion, revenue streaks, leverage, profitability deterioration, etc.) using weighted rule-based checks.
- For the latest quarter, derive percentile scores (0–10) of leverage, equity, ROA/ROE, and growth metrics relative to peers, applying size-based debt penalties and clipping.
- Derive actual ECOS quarterly averages for the reference quarter and assemble the next-quarter ECOS vector either by:
  - **Forecast mode**: map LSTM monthly forecasts into the next quarter, filling gaps with actuals where available, or
  - **Manual mode**: apply analyst-provided adjustments (absolute delta for `base_rate`, percentage shifts for others).
- Score the predicted quarter against historical ECOS quarters via percentiles, respecting indicator risk directions.
- Scale the raw flag score onto 0–10 (capped at 20) and combine the 13 components with prescribed weights; final heuristic score is `(Σ weights × score) × 10`, bounded to 0–100.
- Map the composite score to discrete risk levels (`Low`, `Moderate`, `High`, `Critical`) using calibrated thresholds.

## Key Assumptions & Failure Modes
- ECOS preprocessing requires sufficient history (≥ `seq_length` rows) and all feature columns present; otherwise inference raises explicit errors.
- Peer benchmarking assumes quarterly DART coverage for the target firm; missing quarters default to neutral scores but degrade interpretability.
- The differenced-target design means level forecasts are generated by cumulative sums; long horizons will compound any bias, so downstream consumers should monitor drift.
- Manual adjustments bypass the LSTM, but still rely on the same weighting and percentile infrastructure for consistent scoring.
