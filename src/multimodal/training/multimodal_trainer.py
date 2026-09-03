"""
Multimodal Training Controller with Modality-Ablation Benchmarking.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import ExperimentConfig
from src.utils.device import resolve_device, set_seed
from src.utils.logging import setup_logger


class MultimodalTrainer:
    """Trains Multimodal Fusion networks with support for class weighting, scheduler, and checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ExperimentConfig,
        class_weights: Optional[torch.Tensor] = None,
        logger: Optional[Any] = None,
    ):
        self.config = config
        self.logger = logger or setup_logger("MultimodalTrainer", level=config.system.log_level)

        set_seed(config.system.seed, config.system.deterministic)
        self.device = resolve_device(config.system.device)
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        if config.training.use_class_weights and class_weights is not None:
            weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.training.epochs
        )

        self.checkpoint_dir = Path(config.system.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for embs, masks, targets in self.train_loader:
            embs = {k: v.to(self.device) for k, v in embs.items()}
            masks = {k: v.to(self.device) for k, v in masks.items()}
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(embs, masks=masks)
            loss = self.criterion(outputs, targets)
            loss.backward()

            if self.config.training.gradient_clip_val:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_val
                )

            self.optimizer.step()

            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return {
            "loss": running_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for embs, masks, targets in self.val_loader:
            embs = {k: v.to(self.device) for k, v in embs.items()}
            masks = {k: v.to(self.device) for k, v in masks.items()}
            targets = targets.to(self.device)

            outputs = self.model(embs, masks=masks)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return {
            "loss": running_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }

    def train(self) -> Dict[str, Any]:
        epochs = self.config.training.epochs
        self.logger.info("Starting multimodal training for %d epochs on %s", epochs, self.device)
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            t_m = self.train_epoch(epoch)
            v_m = self.validate()
            self.scheduler.step()

            history["train_loss"].append(t_m["loss"])
            history["train_acc"].append(t_m["accuracy"])
            history["val_loss"].append(v_m["loss"])
            history["val_acc"].append(v_m["accuracy"])

            self.logger.info(
                "Epoch %d/%d | Train Loss: %.4f | Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f",
                epoch, epochs, t_m["loss"], t_m["accuracy"], v_m["loss"], v_m["accuracy"]
            )

            if v_m["accuracy"] > self.best_val_acc:
                self.best_val_acc = v_m["accuracy"]
                self.best_epoch = epoch
                ckpt_path = self.checkpoint_dir / f"{self.config.experiment_name}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "val_accuracy": self.best_val_acc,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "class_names": self.config.dataset.classes,
                }, ckpt_path)
                self.logger.info("Saved best multimodal checkpoint: %s (Val Acc: %.4f)", ckpt_path, self.best_val_acc)

        return history
