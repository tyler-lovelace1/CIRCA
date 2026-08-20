# CIRCA

## Training input performance

Spatial batches gather one center cell and its neighbors from the AnnData expression
matrix. For a sparse matrix this includes sparse-to-dense conversion, which can make
the input pipeline CPU-bound even with only a few data-loader workers. CIRCA batches
the index and metadata gathers to avoid per-cell tensor construction.

Use `LoaderConfig` to tune the remaining host-side work:

```python
from circa.data.datamodule import LoaderConfig

train_loader = LoaderConfig(
    batch_size=256,
    num_workers=2,
    pin_memory=True,       # recommended when training on CUDA
    persistent_workers=True,
    prefetch_factor=1,     # reduce to limit CPU/memory read-ahead
)
```

Leave `log_sklearn_metrics=False` (the model default) for normal training. Enabling
it transfers predictions to the CPU and runs scikit-learn metrics on every batch;
it is intended for diagnostics rather than throughput-sensitive runs.
CIRCA (Causal Interaction Representations for Cellular Architectures) uses causal representation learning to model cell-microenvironment interactions by leveraging spatial transcriptomics data.
