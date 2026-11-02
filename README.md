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

## High-level training API

```python
from circa import CIRCA

run = CIRCA(adata, latent_dim=50, n_neighbors=10, seed=0)
run.fit('runs/k10_z50_seed0', accelerator='gpu')
print(run.best_checkpoint_path, run.best_validation_loss)
cell_embedding = run.embeddings['cell']
# run.adata is a copy, with X_circa_cell/state/niche in .obsm
```

Defaults include EMA (decay 0.999, from the first optimizer step), linear LR
warmup, ReduceLROnPlateau (factor 0.5, patience 50), and stopping after seven
actual LR reductions. Only the minimum `valid_loss` checkpoint is retained;
embeddings are computed from its EMA weights, rather than the last epoch.
Inference runs on CPU and covers all rows, including reference cells. This
convenience API currently supports one training device. Existing low-level
Lightning classes remain available.

For additional modalities, explicitly transform numeric observation columns
before setup, then declare their groups and latent sizes:

```python
run = CIRCA(
    adata, latent_dim=50, n_neighbors=10, seed=0,
    feature_groups={'amyloid': ['log1p_Mean.BetaAmyloid', 'log1p_Max.BetaAmyloid']},
    feature_latent_dims={'amyloid': 2},
    reference_indices=reference_indices,
    model_kwargs={'patience': 50, 'learning_rate': 0.001},
    data_kwargs={'val_fraction': 0.1},
)
run.fit('runs/amyloid_seed0', max_epochs=1000)
amyloid_embedding = run.embeddings['extra_amyloid']
# Also stored as run.adata.obsm['X_circa_extra_amyloid'].
```

Use a new output directory for every latent-dimension/neighborhood/seed
combination. Setup infers input dimensions and encodes slide/panel labels;
missing slide/panel columns default to one category. For multiple panels, pass
an explicit boolean `panel_mask` in pandas categorical order. Zeros alone do
not identify unmeasured genes. `x_layer` selects the expression layer.

Use `model_kwargs` for architecture/optimizer settings and `data_kwargs` for
sampling, reference splits and loader settings. Structural dimensions are
inferred by the API. Training and validation use complete batches because the
contrastive objective requires fixed blocks; reduce batch size if the split
cannot provide one. The block size must divide the batch size and be at least
2 plus the number of extra groups. Prediction retains every row.

`fit(..., ema_decay=None)` disables EMA; `max_lr_reductions=None` disables
LR-based stopping. `compute_embeddings=False` postpones inference until
`run.compute_embeddings()`. Additional Lightning options (e.g. logging) go in
`trainer_kwargs`. Restore training through the low-level Lightning API; this
wrapper creates fresh experiments and does not implement resume.

### Public and low-level classes

`from circa import CIRCA` is the high-level API for data setup, training, and
best-checkpoint embeddings. The underlying Lightning class is available as
`from circa.model import CIRCAModel` (also exported from `circa`):

```python
from circa import CIRCA, CIRCAModel

experiment = CIRCA(adata, latent_dim=50, n_neighbors=10)
experiment.fit('runs/example')
model = experiment.model  # CIRCAModel
restored = CIRCAModel.load_from_checkpoint(
    experiment.best_checkpoint_path, weights_only=False,
)
```

Existing low-level imports `from circa.model.CIRCA import CIRCA` remain an
alias for `CIRCAModel`; they do not resolve to the high-level API. Existing
Lightning checkpoint parameter names and hyperparameters are unchanged. The
initial high-level name `CIRCARun` also remains available as an alias for `CIRCA`.
