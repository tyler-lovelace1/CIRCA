from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import lightning as L

from torch.optim.swa_utils import get_ema_avg_fn

# from L.pytorch import Callback


class EMAWeightAveraging(L.pytorch.callbacks.WeightAveraging):
    def __init__(self, decay, start_step=50):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.start_step = start_step
        super().__init__(avg_fn=get_ema_avg_fn(decay))

    def should_update(self, step_idx=None, epoch_idx=None):
        # Keep the legacy 50-step delay unless explicitly configured.
        return (step_idx is not None) and (step_idx >= self.start_step)

class StopOnMinLR(L.Callback):
    def __init__(self, min_lr: float):
        super().__init__()
        self.min_lr = min_lr

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module) -> None:
        # Check the current learning rate from the optimizer
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        
        # Stop training the moment it falls below your threshold
        if current_lr < self.min_lr:
            print(f"\n[Early Stopping] LR {current_lr} dropped below threshold {self.min_lr}.")
            trainer.should_stop = True

class StopAfterLRReductions(L.Callback):
    """Count actual LR decreases after scheduler updates, ignoring warmup increases."""

    def __init__(self, max_reductions=7):
        if not isinstance(max_reductions, int) or max_reductions < 1:
            raise ValueError('max_reductions must be a positive integer')
        self.max_reductions = max_reductions
        self.reductions = 0
        self.previous_lr = None

    def on_train_start(self, trainer, pl_module):
        if self.previous_lr is None:
            self.previous_lr = trainer.optimizers[0].param_groups[0]['lr']

    def on_train_epoch_start(self, trainer, pl_module):
        # Lightning steps ReduceLROnPlateau after validation/epoch callbacks.
        current = trainer.optimizers[0].param_groups[0]['lr']
        if self.previous_lr is not None and current < self.previous_lr * (1 - 1e-8):
            self.reductions += 1
        self.previous_lr = current
        if self.reductions >= self.max_reductions:
            trainer.should_stop = True

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.reductions >= self.max_reductions:
            return -1  # End the epoch before another optimizer step.

    def state_dict(self):
        return dict(reductions=self.reductions, previous_lr=self.previous_lr)

    def load_state_dict(self, state_dict):
        self.reductions = state_dict['reductions']
        self.previous_lr = state_dict['previous_lr']
