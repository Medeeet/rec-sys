"""MLflow experiment tracking wrapper."""

from pathlib import Path

from src.utils import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Wraps MLflow for experiment tracking."""

    def __init__(
        self, experiment_name: str, tracking_uri: str = "outputs/mlruns"
    ):
        import mlflow

        self.mlflow = mlflow
        tracking_path = Path(tracking_uri).resolve()
        tracking_path.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{tracking_path}")
        mlflow.set_experiment(experiment_name)
        self._run = None
        logger.info(
            f"MLflow experiment: {experiment_name}, URI: {tracking_path}"
        )

    def start_run(self, run_name: str = None, tags: dict = None):
        """Start a new MLflow run."""
        self._run = self.mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"MLflow run started: {run_name}")
        return self._run

    def log_params(self, params: dict, prefix: str = ""):
        """Log parameters (flattens nested dicts)."""
        flat = self._flatten_dict(params, prefix)
        for k, v in flat.items():
            try:
                self.mlflow.log_param(k[:250], str(v)[:500])
            except Exception:
                pass

    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics."""
        for k, v in metrics.items():
            try:
                self.mlflow.log_metric(k, float(v), step=step)
            except Exception:
                pass

    def log_artifact(self, filepath: str):
        """Log an artifact file."""
        self.mlflow.log_artifact(filepath)

    def end_run(self):
        """End the current run."""
        if self._run:
            self.mlflow.end_run()
            self._run = None

    @staticmethod
    def _flatten_dict(d: dict, prefix: str = "") -> dict:
        """Flatten nested dict with dot notation."""
        items = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(ExperimentTracker._flatten_dict(v, key))
            else:
                items[key] = v
        return items

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end_run()
