# MLOps Monitoring Platform

Automated retraining, drift detection, and safe model promotion for a production churn-prediction service — so a model degrading in the field gets caught and fixed before it costs revenue, not after.

![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2?logo=mlflow&logoColor=white)
![Airflow](https://img.shields.io/badge/Airflow-2.8-017CEE?logo=apacheairflow&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Build](https://img.shields.io/badge/build-passing-brightgreen)

---

## Problem

Models don't fail loudly — they decay quietly. Customer behavior shifts, feature distributions drift, and a model that scored well at launch silently degrades until someone notices churn predictions stopped being useful, usually well after the damage is done. Most teams either retrain on a fixed schedule regardless of need, or don't retrain until a human notices something's wrong. This platform closes that gap: it versions data and models, gates every promotion behind a measured quality bar, and is built to catch distribution drift automatically rather than relying on someone to notice.

## Key Results

| Metric | Baseline | Tuned | Delta |
|---|---|---|---|
| AUC-ROC | 0.832 | 0.848 | +1.5 pts |
| AUC-PR | 0.631 | 0.663 | +3.2 pts |
| F1 | 0.619 | 0.631 | +1.2 pts |
| Serving latency (p50, end-to-end incl. preprocessing) | — | ~9 ms | — |
| Data validation gate accuracy | — | 8/8 checks, 4/4 injected defects caught | — |

Tuning was a 50-trial Optuna Bayesian search; the reported delta is measured on a held-out test set (the search itself optimized against a separate validation split, where the best trial reached 0.852 AUC-ROC). Latency is measured via the serving container's own Prometheus histogram, not a synthetic benchmark.

## What Makes This Different

- **Stage-gated promotion, not "train and deploy."** A new model version sits in `None`/`Staging` until it clears an AUC-ROC threshold against the current run's own metrics — promotion to `Production` is a `MlflowClient` API call gated by that check, not a manual copy-paste of a model file.
- **PSI over KS-test/chi-squared for drift.** Population Stability Index gives an interpretable, industry-standard scale (< 0.10 no drift, 0.10–0.20 moderate, > 0.20 significant) that's legible to non-ML stakeholders reading a dashboard, not just to the person who wrote the detector.
- **Data quality is a pipeline stage, not an assumption.** An 8-check validation gate runs before feature engineering and fails the DVC pipeline outright on schema or range violations, rather than letting bad rows silently flow into training.
- **Per-service dependency isolation.** Airflow's containers and the FastAPI serving container each get their own `requirements-*.txt` — installing the full orchestrator's dependency tree inside a lightweight serving image is a common source of version conflicts (it caused a real protobuf conflict between MLflow and OpenTelemetry here) and was deliberately avoided.

## Architecture

```
Raw data → DVC-tracked ingest → validation gate → feature pipeline
                                                        │
                                                        ▼
                                        Airflow: training_pipeline DAG
                                        (Optuna HPO → train → register)
                                                        │
                                                        ▼
                                  MLflow registry: Staging → [AUC gate] → Production
                                                        │
                                                        ▼
                              FastAPI serving → Prometheus metrics → Grafana
                                                        │
                                                        ▼
                          Airflow: drift_monitor DAG (Evidently PSI) → Slack alert → retrain trigger
```

**Key tradeoffs:**
- **DVC + MLflow instead of one tool doing both.** MLflow has no native concept of data versioning or pipeline reproducibility; DVC has no experiment tracking or model registry. Running both means two systems to operate, but avoids bending either tool to do a job it wasn't designed for.
- **Local artifact storage (shared Docker volume) instead of S3/object storage.** Faster to stand up and zero-cost for a single-node deployment, at the cost of not being multi-host-ready — any container writing MLflow artifacts must share the exact same mounted volume as the MLflow server, which is a real constraint documented below.

## Tech Stack

**Data & pipeline** — DVC (versioning, reproducible stages), custom validation gate (schema/range checks)
**Training** — XGBoost, Optuna (Bayesian HPO), SHAP (explainability, logged as MLflow artifacts)
**Tracking & registry** — MLflow (experiments, staged model promotion)
**Orchestration** — Apache Airflow (3 DAGs: data pipeline, training pipeline, drift monitor)
**Serving** — FastAPI + uvicorn
**Monitoring** — Evidently AI (PSI drift detection), Prometheus (metrics), Grafana (dashboards), Alertmanager (Slack routing)
**Infra** — Docker Compose (9-service local stack)

## Project Structure

```
mlops-platform/
├── docker-compose.yml     # full 9-service stack
├── dvc.yaml               # pipeline stage definitions
├── params.yaml            # DVC-tracked hyperparameters
├── src/
│   ├── data/               # ingest, validate, feature engineering
│   ├── training/           # train, tune (Optuna), register (MLflow)
│   ├── serving/             # FastAPI app, model loader, metrics middleware
│   └── monitoring/           # drift detector, alerting
├── dags/                     # Airflow DAGs
├── monitoring/                 # Prometheus/Alertmanager config, Grafana dashboards
└── tests/                      # pytest suite
```

## Configuration

Configuration is layered, in order of precedence (highest first):

1. **Environment variables** (set directly, or via `docker-compose.yml`'s `environment:` blocks) — override everything, used for per-container settings like `MLFLOW_TRACKING_URI` and `MODEL_NAME`.
2. **`params.yaml`** — DVC-tracked hyperparameters and thresholds (training params, monitoring thresholds). Changes here invalidate and re-trigger the relevant `dvc.yaml` pipeline stages.
3. **`.env`** — service credentials and endpoints for local Docker Compose runs.

No real secrets are committed. Copy `.env.example` to `.env` and fill in your own values before running Docker Compose; `.env` is gitignored.

Example `params.yaml` excerpt:

```yaml
train:
  max_depth: 4
  learning_rate: 0.0127
  n_estimators: 475
  scale_pos_weight: 2.46

monitoring:
  psi_threshold: 0.20     # triggers alert
  psi_critical: 0.25      # triggers auto-retrain
```

## Quickstart

```bash
git clone https://github.com/bhavyalakkamraju09/mlops-monitoring-platform.git
cd mlops-monitoring-platform
cp .env.example .env
pip install -r requirements.txt
python src/data/ingest.py
dvc repro
docker-compose up -d
```

Once the stack is up:

```bash
# Train against the containerized MLflow and promote to Production
docker-compose exec airflow-scheduler python src/training/train.py
docker-compose exec airflow-scheduler python -c \
  "from src.training.register import promote_to_staging; promote_to_staging()"
docker-compose exec airflow-scheduler python -c \
  "from src.training.register import promote_to_production; promote_to_production(min_auc=0.85)"

# Verify serving
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "tenure": 12, "MonthlyCharges": 70.5, "TotalCharges": 846.0,
  "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "TechSupport": "No",
  "MultipleLines": "No", "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
  "PhoneService": 1, "PaperlessBilling": 1
}'
```

| Service | URL | Auth |
|---|---|---|
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| FastAPI docs | http://localhost:8000/docs | — |

## Limitations

- **Single-node artifact storage.** MLflow artifacts live on a shared local Docker volume, not object storage — this works for a single-host deployment but doesn't scale to multiple training/serving nodes without moving to S3-backed storage first.
- **Airflow DAGs are implemented and unit-tested but not yet exercised end-to-end in production-like conditions.** The orchestration logic (HPO → train → register → gate → promote, and the drift monitor's alert/retrain path) is built and validated for correctness, but hasn't run a full live cycle against real production traffic.
- **No authentication on internal service UIs.** Airflow, MLflow, and Grafana run with default or basic credentials suitable for local development — not hardened for exposure beyond a private network.

## Future Directions

- Move MLflow's artifact store to S3 (or equivalent object storage) to support multi-node training and remove the shared-volume constraint.
- Run the `drift_monitor` DAG against live serving traffic and validate the auto-retrain trigger fires correctly on real (not synthetic) distribution shift.
- Add a feature store (Feast) to serve consistent features between training and inference instead of duplicating transformation logic in both paths.
- Replace `LocalExecutor` with `KubernetesExecutor` for Airflow to support parallel DAG runs and horizontal scaling beyond a single machine.

## License / Contact

MIT License. Built by Bhavya Lakkamraju — [GitHub](https://github.com/bhavyalakkamraju09) · Project 07 of a 10-project AI/ML engineering portfolio.
