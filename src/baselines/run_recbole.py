"""Run RecBole baseline models (BPR, Pop)."""

import os
from pathlib import Path

from src.utils import get_logger

logger = get_logger(__name__)


def run_baseline(model_name: str, config_path: str, data_path: str = "data/recbole") -> dict:
    """Run a RecBole model and return metrics dict.

    Args:
        model_name: 'BPR' or 'Pop'
        config_path: path to RecBole YAML config
        data_path: path to directory containing dataset folder

    Returns:
        dict with test metrics {Recall@10, NDCG@10, Precision@10}
    """
    from recbole.quick_start import run_recbole as _run_recbole

    logger.info(f"Running RecBole baseline: {model_name}")

    result = _run_recbole(
        model=model_name,
        dataset="amazon-electronics",
        config_file_list=[config_path],
        config_dict={"data_path": data_path},
    )

    # result is a dict with 'best_valid_result' and 'test_result'
    test_result = result.get("test_result", {})
    metrics = {}
    for key, value in test_result.items():
        # Keys are like 'recall@10', 'ndcg@10', etc.
        metrics[str(key)] = float(value)

    logger.info(f"{model_name} test results: {metrics}")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RecBole baseline")
    parser.add_argument("--model", choices=["BPR", "Pop"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="data/recbole")
    args = parser.parse_args()

    metrics = run_baseline(args.model, args.config, args.data_path)
    print(f"\n{'='*50}")
    print(f"{args.model} Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"{'='*50}")
