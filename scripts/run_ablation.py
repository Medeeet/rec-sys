#!/usr/bin/env python3
"""Ablation study: test model with components removed."""

import sys
import json
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.utils import set_seed, get_logger
from scripts.train import prepare_data, run_training

logger = get_logger(__name__)

# Ablation variants: name -> overrides
ABLATION_VARIANTS = {
    "full_model": {
        "model.use_dynamic": True,
        "model.use_content": True,
        "model.use_attention": True,
    },
    "no_dynamic": {
        "model.use_dynamic": False,
        "model.use_content": True,
        "model.use_attention": True,
    },
    "no_content": {
        "model.use_dynamic": True,
        "model.use_content": False,
        "model.use_attention": True,
    },
    "no_attention": {
        "model.use_dynamic": True,
        "model.use_content": True,
        "model.use_attention": False,
    },
    "static_only": {
        "model.use_dynamic": False,
        "model.use_content": False,
        "model.use_attention": False,
    },
}


def apply_flat_overrides(config: dict, overrides: dict) -> dict:
    """Apply dot-notation overrides to nested config."""
    config = deepcopy(config)
    for key, value in overrides.items():
        parts = key.split(".")
        d = config
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return config


def main():
    set_seed(42)

    base_config = Config.load(
        "configs/base.yaml", "configs/data.yaml", "configs/hybrid_model.yaml"
    )

    # Prepare data once
    logger.info("Preparing data for ablation study...")
    data = prepare_data(base_config)

    all_results = {}

    for variant_name, overrides in ABLATION_VARIANTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"ABLATION: {variant_name}")
        logger.info(f"{'='*60}")

        config = apply_flat_overrides(base_config, overrides)
        try:
            results = run_training(config, data, experiment_name=f"ablation_{variant_name}")
            all_results[variant_name] = {
                "test_metrics": results["test_metrics"],
                "best_val_metrics": results["best_val_metrics"],
                "overrides": overrides,
            }
        except Exception as e:
            logger.error(f"Ablation {variant_name} failed: {e}")
            all_results[variant_name] = {"error": str(e), "overrides": overrides}

    # Save results
    output_path = Path("outputs/ablation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"{'Variant':<20} {'Recall@10':>10} {'NDCG@10':>10} {'Precision@10':>12}")
    print("-" * 55)
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name:<20} {'ERROR':>10}")
        else:
            m = res["test_metrics"]
            r = m.get("Recall@10", 0)
            n = m.get("NDCG@10", 0)
            p = m.get("Precision@10", 0)
            print(f"{name:<20} {r:>10.4f} {n:>10.4f} {p:>12.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
