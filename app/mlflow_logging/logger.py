"""
Thin MLflow wrapper. Every eval run logs its config (chunk size, embedding
model, vector store, LLM provider) as params and its RAGAS scores as
metrics, so you can browse the full experiment history with:

    mlflow ui --backend-store-uri ./.codecompass/mlruns
"""
from __future__ import annotations

from app.config import settings


def log_eval_run(params: dict, metrics: dict) -> None:
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
