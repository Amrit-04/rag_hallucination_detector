from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from app.core.config import Settings

logger = logging.getLogger(__name__)


@contextmanager
def mlflow_run(settings: Settings, run_name: str) -> Iterator[object | None]:
    if not settings.enable_mlflow:
        yield None
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            yield run
    except Exception as exc:
        logger.warning("MLflow logging skipped: %s", exc)
        yield None


def log_query_metrics(settings: Settings, metrics: dict[str, float], params: dict[str, str]) -> None:
    if not settings.enable_mlflow:
        return
    try:
        import mlflow

        with mlflow_run(settings, "online-query"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
    except Exception as exc:
        logger.debug("Unable to log query metrics to MLflow: %s", exc)

