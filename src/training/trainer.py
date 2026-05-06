"""Main training loop with early stopping and logging."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.training.losses import bpr_loss
from src.training.evaluator import Evaluator
from src.utils import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score if self.mode == "max" else score < self.best_score
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


class Trainer:
    """Training loop with early stopping, checkpointing, and optional MLflow."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        train_user_items: dict = None,
        n_items: int = 0,
        mlflow_tracker=None,
    ):
        self.config = config
        self.device = config.get("device", torch.device("cpu"))
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_user_items = train_user_items or {}
        self.n_items = n_items
        self.mlflow_tracker = mlflow_tracker

        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 50)
        self.lr = train_cfg.get("learning_rate", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 0.0001)
        self.patience = train_cfg.get("early_stopping_patience", 5)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        # Evaluator
        eval_cfg = config.get("evaluation", {})
        self.evaluator = Evaluator(k_values=eval_cfg.get("topk", [10]))
        self.eval_k = eval_cfg.get("topk", [10])[0]

        # Checkpoint path
        self.checkpoint_dir = Path(
            config.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {"train_loss": [], "val_metrics": []}

    def _train_epoch(self, epoch: int) -> float:
        """One training epoch with BPR loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            # Move to device
            user_ids = batch["user_idx"].to(self.device)
            pos_items = batch["pos_item_idx"].to(self.device)
            neg_items = batch["neg_item_idx"].to(self.device)
            sequences = batch["sequence"].to(self.device)
            seq_lens = batch["seq_len"].to(self.device)
            pos_content = batch["pos_content"].to(self.device)
            neg_content = batch["neg_content"].to(self.device)
            context = batch["context"].to(self.device)

            # Forward: positive items
            pos_scores = self.model(
                user_ids, pos_items, sequences, seq_lens, pos_content, context
            )

            # Forward: negative items
            neg_scores = self.model(
                user_ids, neg_items, sequences, seq_lens, neg_content, context
            )

            # BPR Loss
            loss = bpr_loss(pos_scores, neg_scores)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    def _validate(self) -> dict:
        """Compute metrics on validation set."""
        return self.evaluator.evaluate_model(
            self.model,
            self.val_loader,
            self.train_user_items,
            self.n_items,
            self.device,
            n_sample_items=100,
            k=self.eval_k,
        )

    def train(self) -> dict:
        """Full training loop.

        Returns: best validation metrics dict
        """
        early_stopping = EarlyStopping(patience=self.patience, mode="max")
        best_metrics = {}
        best_epoch = 0

        logger.info(
            f"Starting training: {self.epochs} epochs, lr={self.lr}, "
            f"device={self.device}"
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params:,}")

        for epoch in range(self.epochs):
            start_time = time.time()

            # Train
            train_loss = self._train_epoch(epoch)
            self.history["train_loss"].append(train_loss)

            # Validate (every epoch or every N epochs for speed)
            val_metrics = self._validate()
            self.history["val_metrics"].append(val_metrics)

            # Scheduler step
            self.scheduler.step()

            elapsed = time.time() - start_time
            metric_key = f"NDCG@{self.eval_k}"
            val_ndcg = val_metrics.get(metric_key, 0)

            logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val {metric_key}: {val_ndcg:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # MLflow logging
            if self.mlflow_tracker:
                self.mlflow_tracker.log_metrics(
                    {"train_loss": train_loss, **val_metrics}, step=epoch
                )

            # Check for improvement
            if early_stopping.best_score is None or val_ndcg > early_stopping.best_score:
                best_metrics = val_metrics.copy()
                best_epoch = epoch + 1
                self.save_checkpoint(self.checkpoint_dir / "best_model.pt")

            # Early stopping
            if early_stopping.step(val_ndcg):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info(
            f"Training complete. Best epoch: {best_epoch}, "
            f"Best metrics: {best_metrics}"
        )
        return best_metrics

    def save_checkpoint(self, path: str | Path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "history": self.history,
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str | Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        logger.info(f"Checkpoint loaded: {path}")
