# =============================================================================
# Proactive Scam Intelligence — Pipeline Makefile
# =============================================================================
# Usage:
#   make install       Install all dependencies (dev + main)
#   make pipeline      Run the full 11-stage training pipeline
#   make infer         Score new transactions in data/new/
#   make test          Run the test suite with coverage
#   make lint          Run ruff + black --check
#   make format        Auto-format code with black
#   make monitor       Generate data/model drift report
#   make serve         Start the FastAPI scoring API (localhost:8000)
#   make clean         Remove all generated artefacts
# =============================================================================

.DEFAULT_GOAL := help
PYTHON        := python
REPORTS_DIR   := reports
MODELS_DIR    := models
DATA_DIR      := data

.PHONY: help install pipeline data features train evaluate infer test lint \
        format monitor serve clean mlflow-ui

# ─── Help ────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Proactive Scam Intelligence – Available Commands"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  make install     Install project in editable mode + dev deps"
	@echo "  make pipeline    Run full training pipeline (stages 01–11)"
	@echo "  make data        Run data ingestion + audit + cleaning (01–03)"
	@echo "  make features    Run feature engineering + labelling + split (04–06)"
	@echo "  make train       Train baseline + Torch MLP + calibration (07–08b)"
	@echo "  make evaluate    Evaluate models + risk bands (09–10)"
	@echo "  make infer       Score new transactions (11)"
	@echo "  make test        Run pytest with coverage"
	@echo "  make lint        Run ruff + black --check"
	@echo "  make format      Auto-format code with black"
	@echo "  make monitor     Generate drift monitoring report"
	@echo "  make serve       Start FastAPI inference server on :8000"
	@echo "  make mlflow-ui   Open MLflow experiment dashboard"
	@echo "  make clean       Remove generated artefacts"
	@echo ""

# ─── Install ─────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"
	pre-commit install
	@echo "Project installed. Run 'make pipeline' to start."

# ─── Full Pipeline ────────────────────────────────────────────────────────────
pipeline: data features train evaluate
	@echo ""
	@echo "  Pipeline complete. Artefacts:"
	@echo "  Models  → $(MODELS_DIR)/"
	@echo "  Reports → $(REPORTS_DIR)/"
	@echo ""

# ─── Stage Groups ────────────────────────────────────────────────────────────
data:
	@echo "[01/11] Loading and merging raw data..."
	$(PYTHON) scripts/01_load_merge.py
	@echo "[02/11] Auditing merged dataset..."
	$(PYTHON) scripts/02_audit_merged.py
	@echo "[03/11] Cleaning merged dataset..."
	$(PYTHON) scripts/03_clean_merged.py

features:
	@echo "[04/11] Engineering features..."
	$(PYTHON) scripts/04_build_features.py
	@echo "[05/11] Constructing early-warning labels (72h horizon)..."
	$(PYTHON) scripts/05_label_early_warning.py
	@echo "[06/11] Time-based dataset split..."
	$(PYTHON) scripts/06_time_split.py

train:
	@echo "[07/11] Training baseline LogisticRegression..."
	$(PYTHON) scripts/07_train_baseline.py
	@echo "[08/11] Training Torch MLP with embeddings..."
	$(PYTHON) scripts/08_train_torch_mlp.py
	@echo "[08b/11] Fitting temperature scaling calibration..."
	$(PYTHON) scripts/08b_fit_temperature.py

evaluate:
	@echo "[09/11] Evaluating models + SHAP explanations..."
	$(PYTHON) scripts/09_evaluate_models.py
	@echo "[10/11] Assigning capacity-based risk bands..."
	$(PYTHON) scripts/10_risk_bands.py

infer:
	@echo "[11/11] Batch inference on new transactions..."
	$(PYTHON) scripts/11_infer_batch.py \
		--input $(DATA_DIR)/new/new_transactions.parquet \
		--output $(DATA_DIR)/outputs/scored_new_transactions.parquet

# ─── Quality & Testing ────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ scripts/
	black --check src/ scripts/ --line-length 100

format:
	black src/ scripts/ --line-length 100
	ruff check --fix src/ scripts/

# ─── Monitoring ───────────────────────────────────────────────────────────────
monitor:
	@echo "Generating drift monitoring report..."
	$(PYTHON) scripts/12_monitor_drift.py
	@echo "Report saved to reports/drift_monitoring_report.html"

# ─── Serving ─────────────────────────────────────────────────────────────────
serve:
	@echo "Starting FastAPI inference server on http://localhost:8000"
	@echo "Interactive docs: http://localhost:8000/docs"
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# ─── MLflow ──────────────────────────────────────────────────────────────────
mlflow-ui:
	mlflow ui --host 127.0.0.1 --port 5001

# ─── Clean ───────────────────────────────────────────────────────────────────
clean:
	@echo "Removing generated artefacts..."
	rm -rf $(DATA_DIR)/interim/*.parquet
	rm -rf $(DATA_DIR)/processed/*.parquet
	rm -rf $(DATA_DIR)/outputs/*.parquet
	rm -rf $(MODELS_DIR)/*.pt $(MODELS_DIR)/*.joblib $(MODELS_DIR)/*.json
	rm -rf $(REPORTS_DIR)/*.json $(REPORTS_DIR)/*.png $(REPORTS_DIR)/*.html
	rm -rf mlruns/ .coverage htmlcov/ __pycache__/
	find . -type d -name "__pycache__" -not -path "./.venv/*" | xargs rm -rf
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete
	@echo "Clean complete."
