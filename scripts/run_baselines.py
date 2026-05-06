#!/usr/bin/env python3
"""Run RecBole baseline models (BPR, Pop) and save results."""

import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.config import Config
from src.utils import set_seed, get_logger
from src.data.recbole_converter import create_inter_file

logger = get_logger(__name__)


def run_baseline_model(model_name: str, config_path: str, data_path: str) -> dict:
    """Run a single RecBole baseline and return metrics."""
    from recbole.quick_start import run_recbole

    logger.info(f"Running baseline: {model_name}")

    result = run_recbole(
        model=model_name,
        dataset="amazon-electronics",
        config_file_list=[config_path],
        config_dict={"data_path": data_path},
    )

    test_result = result.get("test_result", {})
    metrics = {}
    for key, value in test_result.items():
        clean_key = str(key).replace("@", "@")
        metrics[clean_key] = float(value)

    logger.info(f"{model_name} results: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument(
        "--models", nargs="+", default=["Pop", "BPR"],
        help="Baseline models to run",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    config = Config.load("configs/base.yaml", "configs/data.yaml")
    processed_dir = config.get("dataset", {}).get("processed_dir", "data/processed")
    recbole_dir = config.get("dataset", {}).get("recbole_dir", "data/recbole")

    # Convert data to RecBole format if needed
    interactions_path = Path(processed_dir) / "interactions.parquet"
    if not interactions_path.exists():
        logger.error(f"No processed data found at {interactions_path}. Run train.py first (or just the data prep step).")
        sys.exit(1)

    interactions = pd.read_parquet(interactions_path)
    recbole_dataset_dir = str(Path(recbole_dir) / "amazon-electronics") if "amazon-electronics" not in recbole_dir else recbole_dir
    create_inter_file(interactions, recbole_dataset_dir)

    # Run baselines
    all_results = {}
    config_map = {
        "BPR": "configs/recbole_bpr.yaml",
        "Pop": "configs/recbole_pop.yaml",
    }

    for model_name in args.models:
        config_path = config_map.get(model_name)
        if not config_path or not Path(config_path).exists():
            logger.warning(f"No config found for {model_name}, skipping")
            continue

        metrics = run_baseline_model(
            model_name,
            config_path,
            str(Path(recbole_dir).parent),
        )
        all_results[model_name] = metrics

    # Save results
    results_path = Path("outputs/baseline_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    for model, metrics in all_results.items():
        print(f"\n{model}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
