# MLOps Monitoring Platform

Complete, zero-cost MLOps platform: **DVC + MLflow + Airflow + Evidently + Prometheus + Grafana**, fully local via Docker Compose.

Use-case model is customer churn prediction (XGBoost on the Telco Customer Churn dataset) — deliberately simple so the platform, not the model, is the thing being demonstrated.

## Architecture

```
Data Ingestion (raw CSV) ──► DVC Track ──► Feature Pipeline (Airflow DAG)
                                                    │
                                                    ▼
Feature Store (local) ──► Training Pipeline (Airflow DAG) ──► MLflow Experiment
        │                                                              │
        ▼                                                              ▼
Model Registry (MLflow) ──► Staging ──► [Manual approval / AUC gate] ──► Production
                                                    │
                                                    ▼
Serving (FastAPI) ──► Prometheus metrics ──► Grafana dashboard
                                                    │
                                                    ▼
Evidently drift monitor ──► Alert (if PSI > threshold) ──► Airflow retrain DAG trigger
```

**3 Airflow DAGs:**
- `data_pipeline` (daily) — ingest → validate → featurize → DVC track/push → trigger training
- `training_pipeline` (triggered/weekly) — HPO (Optuna) → train (XGBoost + MLflow) → register → smoke test → AUC quality gate → promote
- `drift_monitor` (hourly) — fetch recent predictions → Evidently PSI report → alert Slack → auto-retrain if critical

## Stack

| Layer | Tech |
|---|---|
| Data versioning | DVC 3.x |
| Orchestration | Apache Airflow 2.8 |
| Experiment tracking / registry | MLflow 2.x |
| Model | XGBoost + scikit-learn |
| Drift detection | Evidently AI (PSI) |
| Metrics | Prometheus + prometheus-client |
| Dashboards | Grafana |
| Serving | FastAPI + uvicorn |
| Data quality | Great Expectations-style validation gate |
| HPO | Optuna (50-trial Bayesian search) |
| Explainability | SHAP (logged as MLflow artifact) |

## Quickstart

```bash
# 1. Environment
conda create -n mlops python=3.11 -y
conda activate mlops
pip install -r requirements.txt

# 2. Get the data (Kaggle CLI if configured, otherwise auto-fallback)
python src/data/ingest.py

# 3. Init DVC (first time only)
dvc init
dvc remote add -d localremote /tmp/dvc-storage

# 4. Run the pipeline locally (no Docker needed for this part)
dvc repro

# 5. Bring up the full platform (Airflow + MLflow + Prometheus + Grafana + FastAPI)
docker-compose up -d
```

| Service | URL | Login |
|---|---|---|
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| FastAPI | http://localhost:8000/docs | — |

```bash
# Or via Makefile
make up      # start everything
make repro   # run the DVC pipeline
make train   # train only
make monitor # run a drift check
make down    # tear down
```

## Repository layout

```
mlops-platform/
├── docker-compose.yml       # entire platform
├── Dockerfile*               # app / airflow / mlflow images
├── Makefile
├── params.yaml               # DVC-tracked hyperparameters
├── dvc.yaml                  # DVC pipeline stages
├── src/
│   ├── data/                 # ingest, validate, features
│   ├── training/             # train, evaluate, tune (Optuna), register (MLflow)
│   ├── serving/               # FastAPI app, model loader, Prometheus middleware
│   └── monitoring/            # Evidently drift detector, metrics, Slack alerting
├── dags/                      # 3 Airflow DAGs
├── monitoring/                 # prometheus.yml, alertmanager.yml, Grafana dashboards
└── tests/                      # pytest suite (features, model, api, drift)
```

## Validating the platform

- `dvc repro` from a clean checkout reproduces `metrics/eval.json` within ±0.001 AUC
- All 3 DAGs show green runs in the Airflow UI
- Staging → Production is blocked when AUC-ROC < 0.85 (test with a deliberately bad model)
- Injecting shifted data into `drift_monitor`'s `fetch_predictions` task triggers a PSI alert → Slack → auto-retrain
- `curl localhost:9090/metrics` (via `app:8000/metrics`) exposes `predictions_total`, `prediction_latency_seconds`, `churn_rate_rolling`

## Interview talking points

**DVC vs MLflow — don't they overlap?** MLflow tracks experiments (hyperparameters, metrics, model artifacts, code version at training time). DVC tracks data — large binaries Git can't handle — and defines reproducible pipeline stages (`dvc.yaml`) so `dvc repro` only re-runs stages whose inputs changed. They're complementary: DVC owns the data/pipeline layer, MLflow owns the experiment/registry layer.

**How does the Staging → Production gate work?** After training, the model registers as `Staging`. The training DAG's `BranchPythonOperator` reads AUC-ROC from the run's MLflow metrics; if ≥ 0.85 it calls `MlflowClient.transition_model_version_stage()` to promote to Production and archive the previous version. Below threshold, it holds in Staging for manual review. Serving always loads the `Production` stage, so a bad model never reaches users automatically.

**Why PSI over KS-test or chi-squared?** PSI gives an interpretable, industry-standard scale (< 0.10 no drift, 0.10–0.20 moderate, > 0.20 significant) that's been used in financial risk modeling for decades and translates to business language non-technical stakeholders understand.

**Scaling to a 10-engineer team?** Swap the local DVC remote for S3, move MLflow's SQLite backend to RDS Postgres + S3 artifacts, run Airflow on Kubernetes (KubernetesExecutor), add a feature store (Feast/Tecton), add delayed-ground-truth model monitoring, and gate CI/CD on `dvc repro` + pytest before merge. The architecture doesn't change — only the backends do.

---
*Project 07: MLOps Monitoring Platform — build guide implementation for Bhavya Lakkamraju. Zero-cost infrastructure edition.*
