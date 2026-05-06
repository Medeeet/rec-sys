#!/usr/bin/env python3
"""Main training script: data prep -> train -> evaluate -> save results."""

import sys
import argparse
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import numpy as np
import json

from src.config import Config
from src.utils import set_seed, get_logger
from src.data.download import download_amazon_electronics
from src.data.preprocess import preprocess_pipeline
from src.data.splitter import temporal_split
from src.data.feature_engineering import FeatureBuilder
from src.data.dataset import get_dataloaders
from src.models.hybrid_model import AdaptiveHybridModel
from src.training.trainer import Trainer
from src.experiments.mlflow_tracking import ExperimentTracker

logger = get_logger(__name__)


def load_preprocessed_data(processed_dir: str) -> dict:
    """Load data preprocessed by Colab notebook. Supports sparse TF-IDF."""
    import pickle
    import scipy.sparse as sp
    P = Path(processed_dir)

    # Check required files (sparse or dense TF-IDF)
    base_required = ["train.parquet", "val.parquet", "test.parquet",
                     "user_sequences.npz", "train_temporal.parquet",
                     "val_temporal.parquet", "test_temporal.parquet",
                     "dataset_stats.json"]
    missing = [f for f in base_required if not (P / f).exists()]
    has_sparse = (P / "item_content_sparse.npz").exists()
    has_dense  = (P / "item_content_features.npy").exists()
    if not has_sparse and not has_dense:
        missing.append("item_content_sparse.npz (or item_content_features.npy)")
    if missing:
        raise FileNotFoundError(
            f"Missing files in {processed_dir}: {missing}. "
            f"Run colab/01_data_pipeline.ipynb first."
        )

    logger.info(f"Loading preprocessed data from {processed_dir}")
    train_df = pd.read_parquet(P / "train.parquet")
    val_df   = pd.read_parquet(P / "val.parquet")
    test_df  = pd.read_parquet(P / "test.parquet")

    train_temporal = pd.read_parquet(P / "train_temporal.parquet")
    val_temporal   = pd.read_parquet(P / "val_temporal.parquet")
    test_temporal  = pd.read_parquet(P / "test_temporal.parquet")

    # Load TF-IDF: prefer sparse (saves RAM)
    if has_sparse:
        item_content = sp.load_npz(str(P / "item_content_sparse.npz"))
        logger.info(f"Loaded sparse TF-IDF: {item_content.shape}, nnz={item_content.nnz:,}")
    else:
        item_content = np.load(P / "item_content_features.npy")
        logger.info(f"Loaded dense TF-IDF: {item_content.shape}")

    seq_data       = np.load(P / "user_sequences.npz", allow_pickle=True)
    user_sequences = dict(seq_data["sequences"].item())

    with open(P / "dataset_stats.json") as f:
        stats = json.load(f)

    # Build train_user_items efficiently
    train_user_items = {}
    for uid, iid in zip(train_df["user_idx"].values, train_df["item_idx"].values):
        train_user_items.setdefault(int(uid), set()).add(int(iid))

    logger.info(
        f"Loaded: {stats['n_users']:,} users, {stats['n_items']:,} items | "
        f"train={stats['train_size']:,}, val={stats['val_size']:,}, test={stats['test_size']:,}"
    )
    return {
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "user_sequences": user_sequences, "item_content": item_content,
        "train_temporal": train_temporal, "val_temporal": val_temporal, "test_temporal": test_temporal,
        "train_user_items": train_user_items,
        "n_users": stats["n_users"], "n_items": stats["n_items"],
        "content_dim": stats["content_dim"],
    }


def prepare_data(config: dict) -> dict:
    """Full data preparation pipeline. Uses preprocessed data if available."""
    data_cfg = config.get("dataset", {})
    preproc_cfg = config.get("preprocessing", {})
    split_cfg = config.get("split", {})
    feat_cfg = config.get("features", {})

    processed_dir = data_cfg.get("processed_dir", "data/processed")

    # If already preprocessed (e.g. from Colab), load directly
    stats_path = Path(processed_dir) / "dataset_stats.json"
    if stats_path.exists():
        logger.info("Found preprocessed data, loading directly (skipping download/preprocess).")
        return load_preprocessed_data(processed_dir)

    raw_dir = data_cfg.get("raw_dir", "data/raw")

    # Step 1: Download
    logger.info("=== Step 1: Download data ===")
    reviews_path, metadata_path = download_amazon_electronics(raw_dir)

    # Step 2: Preprocess
    logger.info("=== Step 2: Preprocess ===")
    interactions, metadata, user_map, item_map = preprocess_pipeline(
        reviews_path, metadata_path, output_dir=processed_dir,
        min_user=preproc_cfg.get("min_user_interactions", 5),
        min_item=preproc_cfg.get("min_item_interactions", 5),
    )
    n_users = len(user_map)
    n_items = len(item_map)

    # Step 3: Temporal split
    logger.info("=== Step 3: Temporal split ===")
    train_df, val_df, test_df = temporal_split(
        interactions,
        train_ratio=split_cfg.get("train_ratio", 0.7),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
    )

    # Step 4: Feature engineering
    logger.info("=== Step 4: Feature engineering ===")
    fb = FeatureBuilder(processed_dir)
    content_dim = feat_cfg.get("tfidf_max_features", 5000)
    item_content    = fb.build_item_content_features(metadata, max_features=content_dim)
    max_seq_len     = config.get("model", {}).get("dynamic", {}).get("max_seq_len", 50)
    user_sequences  = fb.build_user_sequences(interactions, max_length=max_seq_len)
    train_temporal  = fb.build_temporal_features(train_df)
    val_temporal    = fb.build_temporal_features(val_df)
    test_temporal   = fb.build_temporal_features(test_df)

    train_user_items = {}
    for _, row in train_df.iterrows():
        uid = int(row["user_idx"])
        iid = int(row["item_idx"])
        train_user_items.setdefault(uid, set()).add(iid)

    return {
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "user_sequences": user_sequences, "item_content": item_content,
        "train_temporal": train_temporal, "val_temporal": val_temporal, "test_temporal": test_temporal,
        "train_user_items": train_user_items,
        "n_users": n_users, "n_items": n_items,
        "content_dim": item_content.shape[1],
    }


def run_training(config: dict, data: dict, experiment_name: str = None) -> dict:
    """Build model, train, evaluate on test set. Returns test metrics."""
    device = Config.get_device()
    config["device"] = str(device)

    # Build dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data["train_df"], data["val_df"], data["test_df"],
        data["user_sequences"], data["item_content"],
        data["train_temporal"], data["val_temporal"], data["test_temporal"],
        data["n_items"], config,
    )

    # Update config with actual data dimensions
    config["n_users"] = data["n_users"] + 1  # +1 for padding idx 0
    config["n_items"] = data["n_items"] + 1
    config["content_dim"] = data["content_dim"]
    config["context_dim"] = 5  # hour_sin, hour_cos, dow_sin, dow_cos, is_weekend

    # Build model
    model = AdaptiveHybridModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model.__class__.__name__}, params: {total_params:,}")

    # MLflow tracking (optional)
    tracker = None
    if experiment_name:
        tracking_uri = config.get("logging", {}).get("mlflow_tracking_uri", "outputs/mlruns")
        tracker = ExperimentTracker(experiment_name, tracking_uri)
        tracker.start_run(run_name=experiment_name)
        tracker.log_params(config)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        train_user_items=data["train_user_items"],
        n_items=data["n_items"],
        mlflow_tracker=tracker,
    )
    best_val_metrics = trainer.train()

    # Evaluate on test set
    logger.info("=== Final evaluation on test set ===")
    from src.training.evaluator import Evaluator
    evaluator = Evaluator(k_values=config.get("evaluation", {}).get("topk", [10]))
    eval_k = config.get("evaluation", {}).get("topk", [10])[0]

    test_metrics = evaluator.evaluate_model(
        model, test_loader, data["train_user_items"],
        data["n_items"], device, n_sample_items=100, k=eval_k,
    )
    logger.info(f"Test metrics: {test_metrics}")

    # Log final metrics
    if tracker:
        tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        tracker.end_run()

    # Save results
    results = {
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "n_users": data["n_users"],
        "n_items": data["n_items"],
        "total_params": total_params,
        "train_size": len(data["train_df"]),
        "val_size": len(data["val_df"]),
        "test_size": len(data["test_df"]),
    }

    results_path = Path(config.get("paths", {}).get("output_dir", "outputs")) / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train AdaptiveHybridModel")
    parser.add_argument(
        "--configs", nargs="+",
        default=["configs/base.yaml", "configs/data.yaml", "configs/hybrid_model.yaml"],
        help="YAML config files to merge (in order)",
    )
    parser.add_argument("--experiment", type=str, default="hybrid_main", help="Experiment name for MLflow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. training.epochs=10")
    args = parser.parse_args()

    # Load config
    overrides = Config.parse_cli_overrides(args.overrides)
    config = Config.load(*args.configs, overrides=overrides)
    config["seed"] = args.seed

    set_seed(args.seed)
    logger.info(f"Config loaded from: {args.configs}")
    logger.info(f"Device: {Config.get_device()}")

    # Prepare data
    data = prepare_data(config)

    # Train and evaluate
    results = run_training(config, data, experiment_name=args.experiment)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Test metrics: {results['test_metrics']}")
    print(f"Best val metrics: {results['best_val_metrics']}")
    print(f"Dataset: {results['n_users']} users, {results['n_items']} items")
    print(f"Model params: {results['total_params']:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
