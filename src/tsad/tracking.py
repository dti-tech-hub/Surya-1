"""MLflow wrapper with W&B option behind a flag. All methods no-op when backend=off.

Secrets policy: never read keys from code — W&B key comes from env WANDB_API_KEY only.
"""

from __future__ import annotations

from typing import Any


class Tracker:
    def __init__(self, backend: str = "off", experiment: str = "tsad"):
        assert backend in ("mlflow", "wandb", "off"), f"unknown backend {backend}"
        self.backend = backend
        self.experiment = experiment
        self._run = None

    def start(self, run_name: str = "run", config: Any = None):
        if self.backend == "mlflow":
            import mlflow

            mlflow.set_experiment(self.experiment)
            self._run = mlflow.start_run(run_name=run_name)
            if config is not None:
                try:
                    mlflow.log_params(_flatten(config))
                except Exception:
                    pass
        elif self.backend == "wandb":
            import os

            import wandb

            if "WANDB_API_KEY" not in os.environ:
                raise RuntimeError("WANDB_API_KEY env var required for backend=wandb")
            self._run = wandb.init(project=self.experiment, name=run_name, config=_flatten(config))
        return self

    def log_metrics(self, metrics: dict, step: int | None = None):
        if self.backend == "mlflow":
            import mlflow

            mlflow.log_metrics(metrics, step=step)
        elif self.backend == "wandb":
            assert self._run is not None, "call start() before log_metrics()"
            self._run.log(metrics, step=step)

    def stop(self):
        if self.backend == "mlflow":
            import mlflow

            mlflow.end_run()
        elif self.backend == "wandb":
            self._run.finish()
        self._run = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()


def _flatten(config: Any) -> dict:
    try:
        from omegaconf import OmegaConf

        c = OmegaConf.to_container(config, resolve=True)
        return dict(c) if isinstance(c, dict) else {}
    except Exception:
        if isinstance(config, dict):
            return config
        return {}
