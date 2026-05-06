#!/usr/bin/env python3
"""Hyperparameter optimization with Optuna."""

import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
import torch

from src.config import Config
from src.utils import set_seed, get_logger
from scripts.train import prepare_data, run_training

logger = get_logger(__name__)


def create_objective(base_config: dict, data: dict, optuna_cfg: dict):
    """Create Optuna objective function with data closure."""
    search_space = optuna_cfg.get("search_space", {})

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        config = base_config.copy()

        if "learning_rate" in search_space:
            sp = search_space["learning_rate"]
            lr = trial.suggest_float("learning_rate", sp["low"], sp["high"], log=sp.get("type") == "log_float")
            config.setdefault("training", {})["learning_rate"] = lr

        if "embed_dim" in search_space:
            sp = search_space["embed_dim"]
            embed_dim = trial.suggest_categorical("embed_dim", sp["choices"])
            config.setdefault("model", {})["embed_dim"] = embed_dim

        if "gru_hidden_dim" in search_space:
            sp = search_space["gru_hidden_dim"]
            hidden_dim = trial.suggest_categorical("gru_hidden_dim", sp["choices"])
            config.setdefault("model", {}).setdefault("dynamic", {})["hidden_dim"] = hidden_dim

        if "gru_num_layers" in search_space:
            sp = search_space["gru_num_layers"]
            n_layers = trial.suggest_int("gru_num_layers", sp["low"], sp["high"])
            config.setdefault("model", {}).setdefault("dynamic", {})["num_layers"] = n_layers

        if "dropout" in search_space:
            sp = search_space["dropout"]
            dropout = trial.suggest_float("dropout", sp["low"], sp["high"])
            config.setdefault("model", {}).setdefault("dynamic", {})["dropout"] = dropout
            config.setdefault("model", {}).setdefault("content", {})["dropout"] = dropout

        if "batch_size" in search_space:
            sp = search_space["batch_size"]
            batch_size = trial.suggest_categorical("batch_size", sp["choices"])
            config.setdefault("training", {})["batch_size"] = batch_size

        # Override training params for faster HPO
        train_override = optuna_cfg.get("training_override", {})
        for k, v in train_override.items():
            config.setdefault("training", {})[k] = v

        # Run training
        try:
            results = run_training(config, data, experiment_name=f"optuna_trial_{trial.number}")
            ndcg_key = f"NDCG@{config.get('evaluation', {}).get('topk', [10])[0]}"
            val_ndcg = results["best_val_metrics"].get(ndcg_key, 0.0)
            return val_ndcg
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=None, help="Override number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Override timeout in seconds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load configs
    base_config = Config.load("configs/base.yaml", "configs/data.yaml", "configs/hybrid_model.yaml")
    optuna_cfg = Config.load("configs/optuna.yaml")

    n_trials = args.n_trials or optuna_cfg.get("n_trials", 50)
    timeout = args.timeout or optuna_cfg.get("timeout", 86400)

    # Prepare data once
    logger.info("Preparing data for HPO...")
    data = prepare_data(base_config)

    # Create Optuna study
    storage_path = Path("outputs/optuna")
    storage_path.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path}/study.db"

    study = optuna.create_study(
        study_name=optuna_cfg.get("experiment_name", "hybrid_tuning"),
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=optuna_cfg.get("pruner", {}).get("n_warmup_steps", 5)
        ),
    )

    objective = create_objective(base_config, data, optuna_cfg)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Report results
    print("\n" + "=" * 60)
    print("OPTUNA RESULTS")
    print("=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value (NDCG): {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")
    print("=" * 60)

    # Save best params
    results = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    with open(storage_path / "best_params.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {storage_path}/best_params.json")


if __name__ == "__main__":
    main()
