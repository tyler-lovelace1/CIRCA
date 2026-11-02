"""High-level setup, training, and best-checkpoint embedding API."""
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from circa.data.datamodule import LoaderConfig, SpatialNeighborhoodDataModule
from circa.data.dataset import NeighborSamplingConfig
from circa.model import CIRCAModel
from circa.modules.callbacks import EMAWeightAveraging, StopAfterLRReductions


class CIRCA:
    """One seeded experiment on AnnData; input data are copied before setup.

    Extra modalities are named groups of numeric ``adata.obs`` columns. Transform
    (e.g. log1p/standardize) these explicitly before setup. Panel masks describe
    measured genes, not genes with nonzero counts; supply them for multiple panels
    in pandas categorical order. Embeddings always use the best validation loss.
    """

    def __init__(self, adata, *, latent_dim=50, n_neighbors=10, seed=0,
                 slide_key='slide_id', panel_key='panel', x_layer=None,
                 panel_mask=None, feature_groups=None, feature_latent_dims=None,
                 batch_size=256, num_workers=0, reference_indices=None,
                 model_kwargs=None, data_kwargs=None):
        if latent_dim < 1 or n_neighbors < 1 or batch_size < 2:
            raise ValueError('latent_dim/neighbors must be positive; batch_size must be >= 2')
        self.adata = adata.copy()
        self.seed = seed
        L.seed_everything(seed, workers=True)
        for key in (slide_key, panel_key):
            if key not in self.adata.obs:
                self.adata.obs[key] = 'default'
            if self.adata.obs[key].isna().any():
                raise ValueError(f'{key} contains missing labels')
            self.adata.obs[key] = pd.Categorical(self.adata.obs[key]).remove_unused_categories()
        groups = {k: tuple(v) for k, v in (feature_groups or {}).items()}
        dims = dict(feature_latent_dims or {k: min(2, latent_dim) for k in groups})
        if set(groups) != set(dims):
            raise ValueError('feature_groups and feature_latent_dims must have matching names')
        for name, cols in groups.items():
            if not cols or not 1 <= dims[name] <= latent_dim:
                raise ValueError(f'Invalid columns or latent dimension for {name}')
            values = self.adata.obs[list(cols)].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f'Feature group {name} contains nonfinite values')
        if x_layer is not None and x_layer not in self.adata.layers:
            raise KeyError(x_layer)
        n_panels = len(self.adata.obs[panel_key].cat.categories)
        if panel_mask is None:
            if n_panels > 1:
                raise ValueError('Supply panel_mask for multiple panels (categorical order)')
            panel_mask = np.ones((1, self.adata.n_vars), dtype=bool)
        panel_mask = np.asarray(panel_mask, dtype=bool)
        if panel_mask.shape != (n_panels, self.adata.n_vars) or not panel_mask.any(axis=1).all():
            raise ValueError('panel_mask must have shape (n_panels, n_genes), with measured genes in each panel')
        hp = dict(hidden_dims=[256]*3, proj_hidden_dims=[128]*2, psi_hidden_dims=[64]*2,
                  activation='leaky_relu', phi_type='gauss-maxout', learning_rate=1e-3,
                  dropout_rate=0.1, weight_decay=1e-4, max_epochs=1000, patience=50,
                  opt_algo='adamw', mask_prop=0.2, tau=0.2, l1_lam=0.001,
                  sym_lam=0.002, state_lam=1., phi_share=True, pool_neighbors=True,
                  embed_norm=True, n_perms=1, use_adv=False, use_slide_adv=False)
        hp.update(model_kwargs or {})
        hp.update(input_dim=self.adata.n_vars, latent_dim=latent_dim, n_neighbors=n_neighbors,
                  seed=seed, slide_key=slide_key, num_slides=len(self.adata.obs[slide_key].cat.categories),
                  num_panels=n_panels, panel_mask=panel_mask, batch_size=batch_size,
                  extra_group_input_dims={k: len(v) for k, v in groups.items()}, extra_group_latent_dims=dims)
        hp.setdefault('block_size', next((d for d in range(max(2 + len(groups), batch_size//4), batch_size + 1) if batch_size % d == 0), batch_size))
        if hp['block_size'] < 2 + len(groups) or batch_size % hp['block_size']:
            raise ValueError('block_size must divide batch_size and be >= 2 + number of feature groups')
        self.model = CIRCAModel(hp)
        loader = dict(batch_size=batch_size, block_size=hp['block_size'], num_workers=num_workers)
        data = dict(sampling_train=NeighborSamplingConfig(k=n_neighbors, expansion_factor=1.2,
                    sample_with_invdist=True, weight_power=0.),
                    val_fraction=0.1, split_seed=seed, stratify_split_by=slide_key,
                    squidpy_kwargs={'coord_type': 'generic', 'n_neighs': max(6, n_neighbors)},
                    train_loader=LoaderConfig(**loader, drop_last=True),
                    val_loader=LoaderConfig(**loader, drop_last=True),
                    predict_loader=LoaderConfig(**loader, drop_last=False), seed=seed)
        data.update(data_kwargs or {})
        if data['sampling_train'].k != n_neighbors:
            raise ValueError('sampling_train.k must match n_neighbors')
        data.update(obs_keys=(slide_key, panel_key), slide_key=slide_key, panel_key=panel_key,
                    x_layer=x_layer, obs_feature_groups=groups, reference_indices=reference_indices)
        self.datamodule = SpatialNeighborhoodDataModule(self.adata, **data)
        self.trainer = None
        self.best_checkpoint_path = None
        self.best_validation_loss = None
        self.embeddings = None

    def fit(self, output_dir, *, max_epochs=None, accelerator='auto', devices=1,
            precision='32-true', ema_decay=0.999, max_lr_reductions=7,
            compute_embeddings=True, trainer_kwargs=None):
        """Train, reload minimum-valid_loss EMA weights, and optionally embed all cells.

        Plateau patience is ``model_kwargs['patience']`` (50 by default); the
        learning rate halves on plateaus. Stop after seven reductions by default.
        Set ema_decay or max_lr_reductions to None to disable that behavior.
        output_dir must be new or empty, so repeated searches cannot mix runs.
        Single-device training/prediction is supported by this convenience API.
        """
        if devices != 1:
            raise ValueError('CIRCA currently supports devices=1')
        path = Path(output_dir)
        if path.exists() and any(path.iterdir()):
            raise ValueError('output_dir must be new or empty')
        options = dict(trainer_kwargs or {})
        reserved = {'callbacks', 'enable_checkpointing', 'fast_dev_run', 'max_epochs',
                    'accelerator', 'devices', 'precision', 'strategy'} & options.keys()
        if reserved:
            raise ValueError(f'Use the explicit fit options; unsupported trainer overrides: {reserved}')
        checkpoint = ModelCheckpoint(dirpath=path/'checkpoints', monitor='valid_loss',
                                     mode='min', save_top_k=1, save_on_train_epoch_end=False,
                                     filename='best-{epoch:03d}-{valid_loss:.6f}')
        callbacks = []
        if ema_decay is not None:
            callbacks.append(EMAWeightAveraging(ema_decay, start_step=0))
        if max_lr_reductions is not None:
            if self.model.opt_algo not in ('adamw', 'sgd'):
                raise ValueError('LR reduction stopping requires adamw or sgd plateau scheduling')
            callbacks.append(StopAfterLRReductions(max_lr_reductions))
        callbacks.append(checkpoint)
        options.setdefault('logger', CSVLogger(str(path), name='logs'))
        options.setdefault('gradient_clip_val', 1.)
        options.setdefault('deterministic', True)
        options.setdefault('num_sanity_val_steps', 0)
        self.trainer = L.Trainer(max_epochs=max_epochs or self.model.max_epochs,
                                accelerator=accelerator, devices=devices, precision=precision,
                                callbacks=callbacks, enable_checkpointing=True, **options)
        self.datamodule.setup('fit')
        if not len(self.datamodule.train_dataloader()) or not len(self.datamodule.val_dataloader()):
            raise ValueError('Train/validation split has no complete batches; reduce batch_size or increase val_fraction')
        self.trainer.fit(self.model, datamodule=self.datamodule)
        if not checkpoint.best_model_path:
            raise RuntimeError('No best checkpoint was saved; ensure validation ran and logged valid_loss')
        self.best_checkpoint_path = checkpoint.best_model_path
        self.best_validation_loss = float(checkpoint.best_model_score)
        # Load model only: restoring training callbacks can replace EMA weights with raw weights.
        self.model = CIRCAModel.load_from_checkpoint(self.best_checkpoint_path, map_location='cpu', weights_only=False)
        self.model.eval()
        if compute_embeddings:
            self.embeddings = self.compute_embeddings()
        return self

    def compute_embeddings(self, *, write_to_adata=True):
        """Return cell/state/niche and extra modality arrays in original row order.

        Writes X_circa_cell, X_circa_state, X_circa_niche and
        X_circa_<group> into this run's copied AnnData.obsm by default.
        """
        if self.best_checkpoint_path is None:
            raise RuntimeError('Call fit before computing best-checkpoint embeddings')
        model = CIRCAModel.load_from_checkpoint(self.best_checkpoint_path, map_location='cpu', weights_only=False)
        model.eval()
        dm = self.datamodule
        dm.setup('predict')
        # Reference cells are excluded from train/val centers; include them for inference.
        dm.ds_predict.indices = np.arange(self.adata.n_obs)
        outputs = {}
        seen = np.zeros(self.adata.n_obs, dtype=bool)
        with torch.inference_mode():
            for i, batch in enumerate(dm.predict_dataloader()):
                pred = model.predict_step(batch, i)
                idx = pred['adata_idx'].numpy()
                seen[idx] = True
                arrays = {k.removesuffix('_emb'): pred[k] for k in ('cell_emb', 'state_emb', 'niche_emb')}
                arrays.update({f'extra_{k}': v for k, v in pred['extra_group_emb'].items()})
                for key, value in arrays.items():
                    arr = value.float().numpy()
                    if key not in outputs:
                        outputs[key] = np.empty((self.adata.n_obs, arr.shape[1]), dtype=arr.dtype)
                    outputs[key][idx] = arr
        if not seen.all():
            raise RuntimeError('Prediction did not cover every AnnData row')
        if write_to_adata:
            for key, value in outputs.items():
                self.adata.obsm[f'X_circa_{key}'] = value
        self.embeddings = outputs
        return outputs


# Compatibility with the initial high-level API name.
CIRCARun = CIRCA
