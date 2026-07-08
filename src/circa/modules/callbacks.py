from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import lightning as L

from torch.optim.swa_utils import get_ema_avg_fn


class EMAWeightAveraging(L.pytorch.callbacks.WeightAveraging):
    def __init__(self, decay):
        super().__init__(avg_fn=get_ema_avg_fn(decay))

    def should_update(self, step_idx=None, epoch_idx=None):
        # Start after 50 steps.
        return (step_idx is not None) and (step_idx >= 50)
