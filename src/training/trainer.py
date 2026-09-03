"""
Enhanced PyTorch Trainer supporting Class-Weighted Loss, Discriminative LR, and Experiment Tracking.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import ExperimentConfig
from src.utils.device import resolve_device, set_seed
from src.utils.logging import setup_logger


class Trainer:
    """Industrial Vision PyTorch model trainer supporting transfer learning and fine-tuning."""

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
        self.logger = logger or setup_logger("Trainer", level=config.system.log_level)

        # 1. Device and reproducibility
        set_seed(config.system.seed, config.system.deterministic)
        self.device = resolve_device(config.system.device)
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # 2. Loss Criterion with Class Imbalance Weighting
        if config.training.use_class_weights and class_weights is not None:
            weights = class_weights.to(self.device)
            self.logger.info("Applying inverse-frequency class weights: %s", weights.cpu().tolist())
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # 3. Optimizer with Discriminative Learning Rates for fine-tuning
        self.optimizer = self._build_optimizer()

        # 4. Learning Rate Scheduler
        sched_name = config.training.scheduler.lower()
        if sched_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.training.epochs
            )
        elif sched_name == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, config.training.epochs // 3), gamma=0.5
            )
        else:
            self.scheduler = None

        # 5. Mixed Precision Setup
        self.use_amp = config.training.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # 6. Checkpoint Directory
        self.checkpoint_dir = Path(config.system.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_f1 = 0.0
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Construct optimizer supporting discriminative layer-wise learning rates."""
        opt_name = self.config.training.optimizer.lower()
        head_lr = self.config.training.learning_rate
        backbone_lr = self.config.training.backbone_learning_rate or (head_lr * 0.1)

        head_params = []
        backbone_params = []

        if hasattr(self.model, "classifier"):
            head_params = [p for p in self.model.classifier.parameters() if p.requires_grad]
        if hasattr(self.model, "features"):
            backbone_params = [p for p in self.model.features.parameters() if p.requires_grad]

        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr})

        if not param_groups:
            param_groups = [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": head_lr}]

        if opt_name == "adamw":
            return torch.optim.AdamW(param_groups, weight_decay=self.config.training.weight_decay)
        elif opt_name == "adam":
            return torch.optim.Adam(param_groups, weight_decay=self.config.training.weight_decay)
        elif opt_name == "sgd":
            return torch.optim.SGD(
                param_groups, weight_decay=self.config.training.weight_decay, momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                if self.config.training.gradient_clip_val:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip_val
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                if self.config.training.gradient_clip_val:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip_val
                    )
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return {"loss": epoch_loss, "accuracy": epoch_acc}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Execute validation epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / max(total, 1)
        val_acc = correct / max(total, 1)
        return {"loss": val_loss, "accuracy": val_acc}

    def save_checkpoint(self, filename: str, epoch: int, val_acc: float) -> Path:
        """Save state dicts and raw dictionary metadata (safe for PyTorch 2.6+ weights_only)."""
        ckpt_path = self.checkpoint_dir / filename
        payload = {
            "epoch": epoch,
            "val_accuracy": val_acc,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config_dict": asdict(self.config),
        }
        torch.save(payload, ckpt_path)
        self.logger.info("Saved checkpoint: %s (Val Acc: %.4f)", ckpt_path, val_acc)
        return ckpt_path

    def train(self, resume_path: Optional[Path] = None) -> Dict[str, Any]:
        """Run complete training cycle."""
        start_epoch = 1
        epochs = self.config.training.epochs
        self.logger.info("Starting training for %d epochs on %s", epochs, self.device)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            if self.scheduler:
                self.scheduler.step()

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            self.logger.info(
                "Epoch %d/%d | Train Loss: %.4f | Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f",
                epoch,
                epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
            )

            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.best_epoch = epoch
                self.save_checkpoint(
                    f"{self.config.experiment_name}_best.pt", epoch, self.best_val_acc
                )

            self.save_checkpoint(
                f"{self.config.experiment_name}_latest.pt", epoch, val_metrics["accuracy"]
            )

        return history
