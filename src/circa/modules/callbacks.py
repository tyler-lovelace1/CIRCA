from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import lightning as L

from torch.optim.swa_utils import get_ema_avg_fn

# from L.pytorch import Callback


class EMAWeightAveraging(L.pytorch.callbacks.WeightAveraging):
    def __init__(self, decay):
        super().__init__(avg_fn=get_ema_avg_fn(decay))

    def should_update(self, step_idx=None, epoch_idx=None):
        # Start after 50 steps.
        return (step_idx is not None) and (step_idx >= 50)

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