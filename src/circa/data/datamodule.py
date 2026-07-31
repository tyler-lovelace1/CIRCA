from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

import lightning as L

from circa.data.dataset import SpatialNeighborhoodDataset, SimulatedDataset
from circa.data.sampler import InterleavedSlideBatchSampler, SlideBlockBatchSampler, build_slide_to_dsidx
from circa.utils._torch import matrix_rows_to_numpy

@dataclass
class LoaderConfig:
    batch_size: int = 256
    block_size: int = 16
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = True
    persistent_workers: bool = True 

class SpatialNeighborhoodDataModule(L.LightningDataModule):
    """
    Lightning DataModule for SpatialNeighborhoodDataset.

    Creates:
      - train_dataloader: uses ds_train (split="train"), training augmentations enabled
      - val_dataloader: uses ds_val (split="val"), no sampling augmentations by default
      - predict_dataloader: uses whole dataset (no holdout), no sampling augmentations by default

    Supports slide-stratified interleaved batches via a BatchSampler.
    """

    def __init__(
        self,
        adata,
        *,
        sampling_train,  # NeighborSamplingConfig
        sampling_eval=None,  # NeighborSamplingConfig or None -> derived from sampling_train with sampling off
        obs_keys: Sequence[str] = ("slide_id", "panel"),
        obs_feature_groups: Optional[Dict[str, Sequence[str]]] = None,
        reference_indices: Optional[Sequence[int]] = None,
        slide_key: str = "slide_id",
        panel_key: str = "panel",
        x_layer: Optional[str] = None,
        return_expression: bool = True,
        return_sparse_expression: bool = False,
        include_counts: bool = False,
        ensure_squidpy_neighbors: bool = True,
        squidpy_kwargs: Optional[Dict[str, Any]] = None,
        symmetrize_graph: bool = True,
        val_fraction: float = 0.1,
        split_seed: int = 0,
        stratify_split_by: Optional[str] = "slide_id",  # None or obs key
        use_slide_batch_sampler: bool = True,  # for train/val
        shuffle_within_slide: bool = True,
        shuffle_batches: bool = True,
        train_loader: LoaderConfig = LoaderConfig(batch_size=256),
        val_loader: LoaderConfig = LoaderConfig(batch_size=256),
        predict_loader: LoaderConfig = LoaderConfig(batch_size=256, drop_last=False),
        seed: int = 0,  # neighbor sampling RNG seed
        predict_use_slide_batch_sampler: bool = False,
    ):
        super().__init__()
        self.adata = adata

        self.sampling_train = sampling_train
        self.sampling_eval = sampling_eval

        self.obs_keys = tuple(obs_keys)
        self.obs_feature_groups = {
            str(group_name): tuple(cols)
            for group_name, cols in (obs_feature_groups or {}).items()
        }
        self.slide_key = slide_key
        self.panel_key = panel_key
        self.x_layer = x_layer
        self.return_expression = return_expression
        self.return_sparse_expression = bool(return_sparse_expression)
        self.include_counts = bool(include_counts)

        self.ensure_squidpy_neighbors = ensure_squidpy_neighbors
        self.squidpy_kwargs = squidpy_kwargs
        self.symmetrize_graph = symmetrize_graph

        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)
        self.stratify_split_by = stratify_split_by

        self.reference_indices = reference_indices

        self.use_slide_batch_sampler = bool(use_slide_batch_sampler)
        self.shuffle_within_slide = bool(shuffle_within_slide)
        self.shuffle_batches = bool(shuffle_batches)

        self.train_loader_cfg = train_loader
        self.val_loader_cfg = val_loader
        self.predict_loader_cfg = predict_loader

        self.seed = int(seed)
        self.predict_use_slide_batch_sampler = bool(predict_use_slide_batch_sampler)

        # Will be populated in setup()
        self.ds_train = None
        self.ds_val = None
        self.ds_predict = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Build eval sampling config if not provided
        if self.sampling_eval is None:
            # Create a shallow "no-augmentation" config for eval/predict
            self.sampling_eval = type(self.sampling_train)(
                k=self.sampling_train.k,
                expansion_factor=1.0,
                sample_with_invdist=False,
                invdist_eps=getattr(self.sampling_train, "invdist_eps", 1e-8),
                weight_power=getattr(self.sampling_train, "weight_power", 1.0),
            )

        # Train and val datasets share the same split_seed and val_fraction,
        # so they partition center cells deterministically.
        if stage in (None, "fit"):
            self.ds_train = SpatialNeighborhoodDataset(
                self.adata,
                sampling=self.sampling_train,
                split="train",
                reference_indices=self.reference_indices,
                val_fraction=self.val_fraction,
                split_seed=self.split_seed,
                slide_key=self.slide_key,
                panel_key=self.panel_key,
                stratify_by=self.stratify_split_by,
                training=True,
                return_expression=self.return_expression,
                include_counts=self.include_counts,
                x_layer=self.x_layer,
                obs_keys=self.obs_keys,
                obs_feature_groups=self.obs_feature_groups,
                ensure_squidpy_neighbors=self.ensure_squidpy_neighbors,
                squidpy_kwargs=self.squidpy_kwargs,
                symmetrize_graph=self.symmetrize_graph,
                seed=self.seed,
            )

            self.ds_val = SpatialNeighborhoodDataset(
                self.adata,
                sampling=self.sampling_eval,
                split="val",
                reference_indices=self.reference_indices,
                val_fraction=self.val_fraction,
                split_seed=self.split_seed,
                slide_key=self.slide_key,
                panel_key=self.panel_key,
                stratify_by=self.stratify_split_by,
                training=False,
                return_expression=self.return_expression,
                include_counts=self.include_counts,
                x_layer=self.x_layer,
                obs_keys=self.obs_keys,
                obs_feature_groups=self.obs_feature_groups,
                ensure_squidpy_neighbors=False,  # already computed by train dataset
                squidpy_kwargs=None,
                symmetrize_graph=self.symmetrize_graph,
                seed=self.seed + 1,
            )

            # self.ds_predict = SpatialNeighborhoodDataset(
            #     self.adata,
            #     sampling=self.sampling_eval,
            #     split="train",
            #     reference_indices=self.reference_indices,
            #     val_fraction=0.0,
            #     split_seed=self.split_seed,
            #     slide_key=self.slide_key,
            #     panel_key=self.panel_key,
            #     stratify_by=None,
            #     training=False,
            #     return_expression=self.return_expression,
            #     include_counts=self.include_counts,
            #     x_layer=self.x_layer,
            #     obs_keys=self.obs_keys,
            #     obs_feature_groups=self.obs_feature_groups,
            #     ensure_squidpy_neighbors=self.ensure_squidpy_neighbors if self.ds_train is None else False,
            #     squidpy_kwargs=self.squidpy_kwargs if self.ds_train is None else None,
            #     symmetrize_graph=self.symmetrize_graph,
            #     seed=self.seed + 2,
            # )

        # Predict dataset uses the whole set of center cells.
        # Easiest is val_fraction=0 and split="train" so indices cover all cells.
        if stage in (None, "predict"):
            self.ds_predict = SpatialNeighborhoodDataset(
                self.adata,
                sampling=self.sampling_eval,
                split="train",
                reference_indices=self.reference_indices,
                val_fraction=0.0,
                split_seed=self.split_seed,
                slide_key=self.slide_key,
                panel_key=self.panel_key,
                stratify_by=None,
                training=False,
                return_expression=self.return_expression,
                include_counts=self.include_counts,
                x_layer=self.x_layer,
                obs_keys=self.obs_keys,
                obs_feature_groups=self.obs_feature_groups,
                ensure_squidpy_neighbors=self.ensure_squidpy_neighbors if self.ds_train is None else False,
                squidpy_kwargs=self.squidpy_kwargs if self.ds_train is None else None,
                symmetrize_graph=self.symmetrize_graph,
                seed=self.seed + 2,
            )

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Pads neighbors to exactly k and returns:
          - adata_idx: [B]
          - neighbor_idx: [B, k]
          - neighbor_dist: [B, k]
          - cell: [B, G] and neighbors: [B, k, G] if return_expression
          - obs_keys as lists (or you can tensorize categoricals separately if you want)
        """
        B = len(batch)
        k = getattr(self, "_collate_k", batch[0]["neighbor_idx"].numel())
        G = self.adata.shape[1]
        X = getattr(getattr(self, "_collate_dataset", None), "_X", None)
        if X is None:
            X = self.adata.X

        out: Dict[str, Any] = {}
        out["adata_idx"] = torch.tensor([b["adata_idx"] for b in batch], dtype=torch.long)

        neigh_idx = torch.full((B, k), -1, dtype=torch.long)
        neigh_dist = torch.full((B, k), float("inf"), dtype=torch.float32)
        for bi, b in enumerate(batch):
            ni = b["neighbor_idx"]
            nd = b["neighbor_dist"]
            m = min(k, ni.numel())
            if m > 0:
                neigh_idx[bi, :m] = ni[:m]
                neigh_dist[bi, :m] = nd[:m]

        out["neighbor_idx"] = neigh_idx
        out["neighbor_dist"] = neigh_dist

        # neigh_idx = torch.full((B, k), -1, dtype=torch.long)
        # neigh_dist = torch.full((B, k), float("inf"), dtype=torch.float32)

        # for bi, b in enumerate(batch):
        #     ni = b["neighbor_idx"]
        #     nd = b["neighbor_dist"]
        #     m = min(k, ni.numel())
        #     if m > 0:
        #         neigh_idx[bi, :m] = ni[:m]
        #         neigh_dist[bi, :m] = nd[:m]

        # out["neighbor_idx"] = neigh_idx
        # out["neighbor_dist"] = neigh_dist

        for key in self.obs_keys:
            out[key] = torch.tensor([b[key] for b in batch])

        for key in ("slide_id", "panel"):
            out[key] = torch.tensor([b[key] for b in batch])

        if self.return_expression:
            all_idx = torch.cat([out["adata_idx"][:, None], out["neighbor_idx"]], dim=1).numpy()
            valid_expr = all_idx >= 0
            safe_idx = all_idx.copy()
            safe_idx[~valid_expr] = 0

            if self.return_sparse_expression:
                out["x_sparse"] = self._matrix_rows_to_sparse_tensor(
                    X,
                    safe_idx.reshape(-1),
                    valid_expr.reshape(-1),
                    n_vars=G,
                )
                out["x_sparse_shape"] = (B, 1 + k, G)
            else:
                expr = matrix_rows_to_numpy(X, safe_idx.reshape(-1)).astype(np.float32, copy=False)
                expr = torch.from_numpy(expr.reshape(B, 1 + k, G))
                if not valid_expr.all():
                    expr[~torch.from_numpy(valid_expr)] = 0

                cell = expr[:, 0]
                neighbors = expr[:, 1:]

                out["cell"] = cell
                out["neighbors"] = neighbors
                if self.include_counts:
                    out["cell_counts"] = cell.sum(dim=-1)
                    out["niche_counts"] = neighbors.sum(dim=-1)

            # cell = torch.stack([b["cell"] for b in batch], dim=0)  # [B, G]
            # # G = cell.shape[1]
            # neighbors = torch.zeros((B, k, G), dtype=torch.float32)
            # for bi, b in enumerate(batch):
            #     xn = b["neighbors"]
            #     m = min(k, xn.shape[0])
            #     if m > 0:
            #         neighbors[bi, :m] = xn[:m]
            # out["cell"] = cell
            # out["neighbors"] = neighbors
            # if self.include_counts:
            #     out["cell_counts"] = torch.tensor(
            #         [b["cell_counts"] for b in batch], dtype=torch.float32
            #     )
            #     out["niche_counts"] = torch.tensor(
            #         [b["niche_counts"] for b in batch], dtype=torch.float32
            #     )

        if self.obs_feature_groups:
            grouped_out: Dict[str, Dict[str, torch.Tensor]] = {}
            for group_name in self.obs_feature_groups:
                cell_feat = torch.stack(
                    [b["obs_feature_groups"][group_name]["cell"] for b in batch], dim=0
                )
                Fdim = cell_feat.shape[1]
                neigh_feat = torch.zeros((B, k, Fdim), dtype=torch.float32)
                for bi, b in enumerate(batch):
                    xn = b["obs_feature_groups"][group_name]["neighbors"]
                    m = min(k, xn.shape[0])
                    if m > 0:
                        neigh_feat[bi, :m] = xn[:m]

                grouped_out[group_name] = {"cell": cell_feat, "neighbors": neigh_feat}

            out["obs_feature_groups"] = grouped_out

        return out

    @staticmethod
    def _matrix_rows_to_sparse_tensor(X, idx: np.ndarray, valid_rows: np.ndarray, *, n_vars: int) -> torch.Tensor:
        rows = X[idx]
        n_rows = int(idx.size)

        if sp.issparse(rows):
            coo = rows.tocoo(copy=False)
            keep = valid_rows[coo.row]
            indices = np.vstack([coo.row[keep], coo.col[keep]]).astype(np.int64, copy=False)
            values = coo.data[keep].astype(np.float32, copy=False)
        else:
            dense = np.asarray(rows, dtype=np.float32)
            dense[~valid_rows] = 0
            coo = sp.coo_matrix(dense)
            indices = np.vstack([coo.row, coo.col]).astype(np.int64, copy=False)
            values = coo.data.astype(np.float32, copy=False)

        if indices.size == 0:
            indices_t = torch.empty((2, 0), dtype=torch.long)
            values_t = torch.empty((0,), dtype=torch.float32)
        else:
            indices_t = torch.from_numpy(indices)
            values_t = torch.from_numpy(values)

        return torch.sparse_coo_tensor(indices_t, values_t, size=(n_rows, n_vars)).coalesce()

    def _make_collate_fn(self, dataset):
        def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            self._collate_dataset = dataset
            self._collate_k = dataset.sampling.k
            try:
                return self._collate_fn(batch)
            finally:
                self._collate_dataset = None
                self._collate_k = None
        return collate

    def _make_loader(
        self,
        dataset,
        *,
        loader_cfg: LoaderConfig,
        use_slide_batch_sampler: bool,
        training: bool,
    ) -> DataLoader:
        if use_slide_batch_sampler:
            slide_to_dsidx = build_slide_to_dsidx(dataset, slide_key=self.slide_key)
            print(slide_to_dsidx)
            # batch_sampler = InterleavedSlideBatchSampler(
            #     slide_to_dsidx=slide_to_dsidx,
            #     batch_size=loader_cfg.batch_size,
            #     shuffle_within=self.shuffle_within_slide,
            #     shuffle_batches=self.shuffle_batches and training,
            #     drop_last=loader_cfg.drop_last,
            #     seed=self.seed if training else self.seed + 1234,
            # )
            batch_sampler = SlideBlockBatchSampler(
                slide_to_dsidx=slide_to_dsidx,
                batch_size=loader_cfg.batch_size,
                block_size=loader_cfg.block_size,
                shuffle_within=self.shuffle_within_slide,
                shuffle_batches=self.shuffle_batches and training,
                drop_last=loader_cfg.drop_last,
                seed=self.seed if training else self.seed + 1234,
            )
            # print(batch_sampler)
            # batch_sampler = EqualExposureSlideBatchSampler(
            #     slide_to_dsidx=slide_to_dsidx,
            #     batch_size=loader_cfg.batch_size,
            #     replacement=False,
            #     shuffle_batches=self.shuffle_batches and training,
            #     drop_last=loader_cfg.drop_last,
            #     seed=self.seed if training else self.seed + 1234,
            # )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=loader_cfg.num_workers,
                pin_memory=loader_cfg.pin_memory,
                persistent_workers=loader_cfg.persistent_workers and loader_cfg.num_workers > 0,
                collate_fn=self._make_collate_fn(dataset)
            )

        # Non-stratified loader (sequential for val/predict; shuffled for train)
        return DataLoader(
            dataset,
            batch_size=loader_cfg.batch_size,
            shuffle=training,
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory,
            persistent_workers=loader_cfg.persistent_workers and loader_cfg.num_workers > 0,
            drop_last=loader_cfg.drop_last,
            collate_fn=self._make_collate_fn(dataset)
        )

    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None
        return self._make_loader(
            self.ds_train,
            loader_cfg=self.train_loader_cfg,
            use_slide_batch_sampler=self.use_slide_batch_sampler,
            training=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None
        return self._make_loader(
            self.ds_val,
            loader_cfg=self.val_loader_cfg,
            use_slide_batch_sampler=self.use_slide_batch_sampler,
            training=False,
        )

    def predict_dataloader(self) -> DataLoader:
        assert self.ds_predict is not None
        return self._make_loader(
            self.ds_predict,
            loader_cfg=self.predict_loader_cfg,
            use_slide_batch_sampler=False,
            training=False,
        )


class SimulationDataModule(L.LightningDataModule):
    """
    Lightning DataModule for SimulatedDatasetV2.

    Creates:
      - train_dataloader: uses ds_train (split="train"), training augmentations enabled
      - val_dataloader: uses ds_val (split="val"), no sampling augmentations by default
      - predict_dataloader: uses whole dataset (no holdout), no sampling augmentations by default

    Supports slide-stratified interleaved batches via a BatchSampler.
    """

    def __init__(
        self,
        adata,
        *,
        sim_config,  # SimulationConfig
        x_layer: Optional[str] = None,
        ensure_squidpy_neighbors: bool = True,
        squidpy_kwargs: Optional[Dict[str, Any]] = None,
        symmetrize_graph: bool = True,
        val_fraction: float = 0.1,
        use_slide_batch_sampler: bool = True,  # for train/val
        shuffle_within_slide: bool = True,
        shuffle_batches: bool = True,
        train_loader: LoaderConfig = LoaderConfig(batch_size=256),
        val_loader: LoaderConfig = LoaderConfig(batch_size=256),
        predict_loader: LoaderConfig = LoaderConfig(batch_size=256, drop_last=False),
        predict_use_slide_batch_sampler: bool = False,
    ):
        super().__init__()
        self.adata = adata

        self.sim_config = sim_config
        
        # self.slide_key = slide_key
        self.x_layer = x_layer
        
        self.ensure_squidpy_neighbors = ensure_squidpy_neighbors
        self.squidpy_kwargs = squidpy_kwargs
        self.symmetrize_graph = symmetrize_graph

        self.val_fraction = float(val_fraction)

        self.use_slide_batch_sampler = bool(use_slide_batch_sampler)
        self.shuffle_within_slide = bool(shuffle_within_slide)
        self.shuffle_batches = bool(shuffle_batches)

        self.train_loader_cfg = train_loader
        self.val_loader_cfg = val_loader
        self.predict_loader_cfg = predict_loader

        self.predict_use_slide_batch_sampler = bool(predict_use_slide_batch_sampler)

        # Will be populated in setup()
        self.ds_train = None
        self.ds_val = None
        self.ds_predict = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Build eval sampling config if not provided
        # if self.sampling_eval is None:
        #     # Create a shallow "no-augmentation" config for eval/predict
        #     self.sampling_eval = type(self.sampling_train)(
        #         k=self.sampling_train.k,
        #         expansion_factor=1.0,
        #         sample_with_invdist=False,
        #         invdist_eps=getattr(self.sampling_train, "invdist_eps", 1e-8),
        #         weight_power=getattr(self.sampling_train, "weight_power", 1.0),
        #     )

        # Train and val datasets share the same split_seed and val_fraction,
        # so they partition center cells deterministically.
        if stage in (None, "fit"):
            self.ds_train = SimulatedDataset(
                self.adata,
                sim_config=self.sim_config,
                split="train",
                val_fraction=self.val_fraction,
                training=True,
                x_layer=self.x_layer,
                ensure_squidpy_neighbors=self.ensure_squidpy_neighbors,
                squidpy_kwargs=self.squidpy_kwargs,
                symmetrize_graph=self.symmetrize_graph,
                pca_models = None
            )

            self.ds_val = SimulatedDataset(
                self.adata,
                sim_config=self.sim_config,
                split="val",
                val_fraction=self.val_fraction,
                training=False,
                x_layer=self.x_layer,
                ensure_squidpy_neighbors=self.ensure_squidpy_neighbors,
                squidpy_kwargs=self.squidpy_kwargs,
                symmetrize_graph=self.symmetrize_graph,
                pca_models = self.ds_train.pca_models
            )

            # self.ds_predict = SimulatedDataset(
            #     self.adata,
            #     sim_config=self.sim_config,
            #     split="train",
            #     val_fraction=0.0,
            #     training=False,
            #     x_layer=self.x_layer,
            #     ensure_squidpy_neighbors=self.ensure_squidpy_neighbors,
            #     squidpy_kwargs=self.squidpy_kwargs,
            #     symmetrize_graph=self.symmetrize_graph,
            # )

        # Predict dataset uses the whole set of center cells.
        # Easiest is val_fraction=0 and split="train" so indices cover all cells.
        if stage in (None, "predict"):
            self.ds_predict = SimulatedDataset(
                self.adata,
                sim_config=self.sim_config,
                split="train",
                val_fraction=0.0,
                training=False,
                x_layer=self.x_layer,
                ensure_squidpy_neighbors=self.ensure_squidpy_neighbors,
                squidpy_kwargs=self.squidpy_kwargs,
                symmetrize_graph=self.symmetrize_graph,
            )

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Pads neighbors to exactly k and returns:
          - adata_idx: [B]
          - neighbor_idx: [B, k]
          - neighbor_dist: [B, k]
          - cell: [B, G] and neighbors: [B, k, G] if return_expression
          - obs_keys as lists (or you can tensorize categoricals separately if you want)
        """
        B = len(batch)
        k = self.sim_config.k

        out: Dict[str, Any] = {}
        out["adata_idx"] = torch.tensor([b["adata_idx"] for b in batch], dtype=torch.long)
        out["slide_id"] = torch.tensor([b["slide_id"] for b in batch], dtype=torch.long)
        out["panel"] = torch.tensor([b["panel"] for b in batch], dtype=torch.long)

        # neigh_idx = torch.full((B, k), -1, dtype=torch.long)
        # neigh_dist = torch.full((B, k), float("inf"), dtype=torch.float32)

        # for bi, b in enumerate(batch):
        #     ni = b["neighbor_idx"]
        #     # nd = b["neighbor_dist"]
        #     m = min(k, ni.numel())
        #     if m > 0:
        #         neigh_idx[bi, :m] = ni[:m]
        #         # neigh_dist[bi, :m] = nd[:m]

        # out["neighbor_idx"] = neigh_idx
        # # out["neighbor_dist"] = neigh_dist

        # for key in self.obs_keys:
        #     out[key] = torch.tensor([b[key] for b in batch])

        # if self.return_expression:
        cell = torch.stack([b["cell"] for b in batch], dim=0)  # [B, G]
        G = cell.shape[1]
        neighbors = torch.zeros((B, k, G), dtype=torch.float32)
        for bi, b in enumerate(batch):
            xn = b["neighbors"]
            m = min(k, xn.shape[0])
            if m > 0:
                neighbors[bi, :m] = xn[:m]
        out["cell"] = cell
        out["neighbors"] = neighbors
        # if self.include_counts:
        #     out["cell_counts"] = torch.tensor(
        #         [b["cell_counts"] for b in batch], dtype=torch.float32
        #     )
        #     out["niche_counts"] = torch.tensor(
        #         [b["niche_counts"] for b in batch], dtype=torch.float32
        #     )

        # if self.obs_feature_groups:
        #     grouped_out: Dict[str, Dict[str, torch.Tensor]] = {}
        #     for group_name in self.obs_feature_groups:
        #         cell_feat = torch.stack(
        #             [b["obs_feature_groups"][group_name]["cell"] for b in batch], dim=0
        #         )
        #         Fdim = cell_feat.shape[1]
        #         neigh_feat = torch.zeros((B, k, Fdim), dtype=torch.float32)
        #         for bi, b in enumerate(batch):
        #             xn = b["obs_feature_groups"][group_name]["neighbors"]
        #             m = min(k, xn.shape[0])
        #             if m > 0:
        #                 neigh_feat[bi, :m] = xn[:m]

        #         grouped_out[group_name] = {"cell": cell_feat, "neighbors": neigh_feat}

        #     out["obs_feature_groups"] = grouped_out

        return out

    def _make_loader(
        self,
        dataset,
        *,
        loader_cfg: LoaderConfig,
        use_slide_batch_sampler: bool,
        training: bool,
    ) -> DataLoader:
        # if use_slide_batch_sampler:
        #     slide_to_dsidx = build_slide_to_dsidx(dataset, slide_key=self.slide_key)
        #     batch_sampler = InterleavedSlideBatchSampler(
        #         slide_to_dsidx=slide_to_dsidx,
        #         batch_size=loader_cfg.batch_size,
        #         shuffle_within=self.shuffle_within_slide,
        #         shuffle_batches=self.shuffle_batches and training,
        #         drop_last=loader_cfg.drop_last,
        #         seed=self.seed if training else self.seed + 1234,
        #     )
        #     return DataLoader(
        #         dataset,
        #         batch_sampler=batch_sampler,
        #         num_workers=loader_cfg.num_workers,
        #         pin_memory=loader_cfg.pin_memory,
        #         collate_fn=self._collate_fn
        #     )

        # Non-stratified loader (sequential for val/predict; shuffled for train)
        return DataLoader(
            dataset,
            batch_size=loader_cfg.batch_size,
            shuffle=training,
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory,
            drop_last=loader_cfg.drop_last,
            collate_fn=self._collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None
        return self._make_loader(
            self.ds_train,
            loader_cfg=self.train_loader_cfg,
            use_slide_batch_sampler=self.use_slide_batch_sampler,
            training=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None
        return self._make_loader(
            self.ds_val,
            loader_cfg=self.val_loader_cfg,
            use_slide_batch_sampler=self.use_slide_batch_sampler,
            training=False,
        )

    def predict_dataloader(self) -> DataLoader:
        assert self.ds_predict is not None
        return self._make_loader(
            self.ds_predict,
            loader_cfg=self.predict_loader_cfg,
            use_slide_batch_sampler=self.predict_use_slide_batch_sampler,
            training=False,
        )


class SimulationDataModuleV2(L.LightningDataModule):
    """
    Lightning DataModule for SimulatedDatasetV2.

    Creates:
      - train_dataloader: uses ds_train (split="train"), training augmentations enabled
      - val_dataloader: uses ds_val (split="val"), no sampling augmentations by default
      - predict_dataloader: uses whole dataset (no holdout), no sampling augmentations by default

    Supports slide-stratified interleaved batches via a BatchSampler.
    """

    def __init__(
        self,
        adata,
        *,
        sim_config,  # SimulationConfig
        x_layer: Optional[str] = None,
        ensure_squidpy_neighbors: bool = True,
        squidpy_kwargs: Optional[Dict[str, Any]] = None,
        symmetrize_graph: bool = True,
        val_fraction: float = 0.1,
        use_slide_batch_sampler: bool = True,  # for train/val
        shuffle_within_slide: bool = True,
        shuffle_batches: bool = True,
        train_loader: LoaderConfig = LoaderConfig(batch_size=256),
        val_loader: LoaderConfig = LoaderConfig(batch_size=256),
        predict_loader: LoaderConfig = LoaderConfig(batch_size=256, drop_last=False),
        predict_use_slide_batch_sampler: bool = False,
    ):
        super().__init__()
        self.adata = adata

        self.sim_config = sim_config
        
        # self.slide_key = slide_key
        self.x_layer = x_layer
        
        self.ensure_squidpy_neighbors = ensure_squidpy_neighbors
        self.squidpy_kwargs = squidpy_kwargs
        self.symmetrize_graph = symmetrize_graph

        self.val_fraction = float(val_fraction)

        self.use_slide_batch_sampler = bool(use_slide_batch_sampler)
        self.shuffle_within_slide = bool(shuffle_within_slide)
        self.shuffle_batches = bool(shuffle_batches)

        self.train_loader_cfg = train_loader
        self.val_loader_cfg = val_loader
        self.predict_loader_cfg = predict_loader

        self.predict_use_slide_batch_sampler = bool(predict_use_slide_batch_sampler)

        # Will be populated in setup()
        self.ds_train = None
        self.ds_val = None
        self.ds_predict = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Build eval sampling config if not provided
        # if self.sampling_eval is None:
        #     # Create a shallow "no-augmentation" config for eval/predict
        #     self.sampling_eval = type(self.sampling_train)(
        #         k=self.sampling_train.k,
        #         expansion_factor=1.0,
        #         sample_with_invdist=False,
        #         invdist_eps=getattr(self.sampling_train, "invdist_eps", 1e-8),
        #         weight_power=getattr(self.sampling_train, "weight_power", 1.0),
        #     )

        # Train and val datasets share the same split_seed and val_fraction,
        # so they partition center cells deterministically.
        if stage in (None, "fit"):
            self.ds_train = SimulatedDatasetV2(
                self.adata,
                sim_config=self.sim_config,
                split="train",
                val_fraction=self.val_fraction,
                training=True,
                x_layer=self.x_layer,
                ensure_squidpy_neighbors=self.ensure_squidpy_neighbors,
                squidpy_kwargs=self.squidpy_kwargs,
                symmetrize_graph=self.symmetrize_graph,
                pca_models = None
            )

            self.ds_val = SimulatedDatasetV2(
                self.adata,
                sim_config=self.sim_config,
                split="val",
                val_fraction=self.val_fraction,
                training=False,
                x_layer=self.x_layer,
                ensure_squidpy_neighbors=self.ensure_squidpy_neighbors,
                squidpy_kwargs=self.squidpy_kwargs,
                symmetrize_graph=self.symmetrize_graph,
                pca_models = self.ds_train.pca_models
            )

            # self.ds_predict = SimulatedDataset(
            #     self.adata,
            #     sim_config=self.sim_config,
            #     split="train",
            #     val_fraction=0.0,
            #     training=False,
            #     x_layer=self.x_layer,
            #     ensure_squidpy_neighbors=self.ensure_squidpy_neighbors,
            #     squidpy_kwargs=self.squidpy_kwargs,
            #     symmetrize_graph=self.symmetrize_graph,
            # )

        # Predict dataset uses the whole set of center cells.
        # Easiest is val_fraction=0 and split="train" so indices cover all cells.
        if stage in (None, "predict"):
            self.ds_predict = SimulatedDatasetV2(
                self.adata,
                sim_config=self.sim_config,
                split="train",
                val_fraction=0.0,
                training=False,
                x_layer=self.x_layer,
                ensure_squidpy_neighbors=self.ensure_squidpy_neighbors,
                squidpy_kwargs=self.squidpy_kwargs,
                symmetrize_graph=self.symmetrize_graph,
            )

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Pads neighbors to exactly k and returns:
          - adata_idx: [B]
          - neighbor_idx: [B, k]
          - neighbor_dist: [B, k]
          - cell: [B, G] and neighbors: [B, k, G] if return_expression
          - obs_keys as lists (or you can tensorize categoricals separately if you want)
        """
        B = len(batch)
        k = self.sim_config.k

        out: Dict[str, Any] = {}
        out["adata_idx"] = torch.tensor([b["adata_idx"] for b in batch], dtype=torch.long)
        out["slide_id"] = torch.tensor([b["slide_id"] for b in batch], dtype=torch.long)
        out["panel"] = torch.tensor([b["panel"] for b in batch], dtype=torch.long)

        # neigh_idx = torch.full((B, k), -1, dtype=torch.long)
        # neigh_dist = torch.full((B, k), float("inf"), dtype=torch.float32)

        # for bi, b in enumerate(batch):
        #     ni = b["neighbor_idx"]
        #     # nd = b["neighbor_dist"]
        #     m = min(k, ni.numel())
        #     if m > 0:
        #         neigh_idx[bi, :m] = ni[:m]
        #         # neigh_dist[bi, :m] = nd[:m]

        # out["neighbor_idx"] = neigh_idx
        # # out["neighbor_dist"] = neigh_dist

        # for key in self.obs_keys:
        #     out[key] = torch.tensor([b[key] for b in batch])

        # if self.return_expression:
        cell = torch.stack([b["cell"] for b in batch], dim=0)  # [B, G]
        G = cell.shape[1]
        neighbors = torch.zeros((B, k, G), dtype=torch.float32)
        for bi, b in enumerate(batch):
            xn = b["neighbors"]
            m = min(k, xn.shape[0])
            if m > 0:
                neighbors[bi, :m] = xn[:m]
        out["cell"] = cell
        out["neighbors"] = neighbors
        # if self.include_counts:
        #     out["cell_counts"] = torch.tensor(
        #         [b["cell_counts"] for b in batch], dtype=torch.float32
        #     )
        #     out["niche_counts"] = torch.tensor(
        #         [b["niche_counts"] for b in batch], dtype=torch.float32
        #     )

        # if self.obs_feature_groups:
        #     grouped_out: Dict[str, Dict[str, torch.Tensor]] = {}
        #     for group_name in self.obs_feature_groups:
        #         cell_feat = torch.stack(
        #             [b["obs_feature_groups"][group_name]["cell"] for b in batch], dim=0
        #         )
        #         Fdim = cell_feat.shape[1]
        #         neigh_feat = torch.zeros((B, k, Fdim), dtype=torch.float32)
        #         for bi, b in enumerate(batch):
        #             xn = b["obs_feature_groups"][group_name]["neighbors"]
        #             m = min(k, xn.shape[0])
        #             if m > 0:
        #                 neigh_feat[bi, :m] = xn[:m]

        #         grouped_out[group_name] = {"cell": cell_feat, "neighbors": neigh_feat}

        #     out["obs_feature_groups"] = grouped_out

        return out

    def _make_loader(
        self,
        dataset,
        *,
        loader_cfg: LoaderConfig,
        use_slide_batch_sampler: bool,
        training: bool,
    ) -> DataLoader:
        # if use_slide_batch_sampler:
        #     slide_to_dsidx = build_slide_to_dsidx(dataset, slide_key=self.slide_key)
        #     batch_sampler = InterleavedSlideBatchSampler(
        #         slide_to_dsidx=slide_to_dsidx,
        #         batch_size=loader_cfg.batch_size,
        #         shuffle_within=self.shuffle_within_slide,
        #         shuffle_batches=self.shuffle_batches and training,
        #         drop_last=loader_cfg.drop_last,
        #         seed=self.seed if training else self.seed + 1234,
        #     )
        #     return DataLoader(
        #         dataset,
        #         batch_sampler=batch_sampler,
        #         num_workers=loader_cfg.num_workers,
        #         pin_memory=loader_cfg.pin_memory,
        #         collate_fn=self._collate_fn
        #     )

        # Non-stratified loader (sequential for val/predict; shuffled for train)
        return DataLoader(
            dataset,
            batch_size=loader_cfg.batch_size,
            shuffle=training,
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory,
            drop_last=loader_cfg.drop_last,
            collate_fn=self._collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None
        return self._make_loader(
            self.ds_train,
            loader_cfg=self.train_loader_cfg,
            use_slide_batch_sampler=self.use_slide_batch_sampler,
            training=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None
        return self._make_loader(
            self.ds_val,
            loader_cfg=self.val_loader_cfg,
            use_slide_batch_sampler=self.use_slide_batch_sampler,
            training=False,
        )

    def predict_dataloader(self) -> DataLoader:
        assert self.ds_predict is not None
        return self._make_loader(
            self.ds_predict,
            loader_cfg=self.predict_loader_cfg,
            use_slide_batch_sampler=self.predict_use_slide_batch_sampler,
            training=False,
        )


class SimulationDataModuleV3(L.LightningDataModule):
    """
    Lightning DataModule for SimulatedDatasetV2.

    Creates:
      - train_dataloader: uses ds_train (split="train"), training augmentations enabled
      - val_dataloader: uses ds_val (split="val"), no sampling augmentations by default
      - predict_dataloader: uses whole dataset (no holdout), no sampling augmentations by default

    Supports slide-stratified interleaved batches via a BatchSampler.
    """

    def __init__(
        self,
        adata,
        *,
        sim_config,  # SimulationConfig
        x_layer: Optional[str] = None,
        ensure_squidpy_neighbors: bool = True,
        squidpy_kwargs: Optional[Dict[str, Any]] = None,
        symmetrize_graph: bool = True,
        val_fraction: float = 0.1,
        use_slide_batch_sampler: bool = True,  # for train/val
        shuffle_within_slide: bool = True,
        shuffle_batches: bool = True,
        train_loader: LoaderConfig = LoaderConfig(batch_size=256),
        val_loader: LoaderConfig = LoaderConfig(batch_size=256),
        predict_loader: LoaderConfig = LoaderConfig(batch_size=256, drop_last=False),
        predict_use_slide_batch_sampler: bool = False,
    ):
        super().__init__()
        self.adata = adata

        self.sim_config = sim_config
        
        # self.slide_key = slide_key
        self.x_layer = x_layer
        
        self.ensure_squidpy_neighbors = ensure_squidpy_neighbors
        self.squidpy_kwargs = squidpy_kwargs
        self.symmetrize_graph = symmetrize_graph

        self.val_fraction = float(val_fraction)

        self.use_slide_batch_sampler = bool(use_slide_batch_sampler)
        self.shuffle_within_slide = bool(shuffle_within_slide)
        self.shuffle_batches = bool(shuffle_batches)

        self.train_loader_cfg = train_loader
        self.val_loader_cfg = val_loader
        self.predict_loader_cfg = predict_loader

        self.predict_use_slide_batch_sampler = bool(predict_use_slide_batch_sampler)

        # Will be populated in setup()
        self.ds_train = None
        self.ds_val = None
        self.ds_predict = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Build eval sampling config if not provided
        # if self.sampling_eval is None:
        #     # Create a shallow "no-augmentation" config for eval/predict
        #     self.sampling_eval = type(self.sampling_train)(
        #         k=self.sampling_train.k,
        #         expansion_factor=1.0,
        #         sample_with_invdist=False,
        #         invdist_eps=getattr(self.sampling_train, "invdist_eps", 1e-8),
        #         weight_power=getattr(self.sampling_train, "weight_power", 1.0),
        #     )

        # Train and val datasets share the same split_seed and val_fraction,
        # so they partition center cells deterministically.
        if stage in (None, "fit"):
            self.ds_train = SimulatedDatasetV3(
                self.adata,
                sim_config=self.sim_config,
                split="train",
                val_fraction=self.val_fraction,
                training=True,
                x_layer=self.x_layer,
                ensure_squidpy_neighbors=self.ensure_squidpy_neighbors,
                squidpy_kwargs=self.squidpy_kwargs,
                symmetrize_graph=self.symmetrize_graph,
                pca_models = None
            )

            self.ds_val = SimulatedDatasetV3(
                self.adata,
                sim_config=self.sim_config,
                split="val",
                val_fraction=self.val_fraction,
                training=False,
                x_layer=self.x_layer,
                ensure_squidpy_neighbors=self.ensure_squidpy_neighbors,
                squidpy_kwargs=self.squidpy_kwargs,
                symmetrize_graph=self.symmetrize_graph,
                pca_models = self.ds_train.pca_models
            )

            # self.ds_predict = SimulatedDataset(
            #     self.adata,
            #     sim_config=self.sim_config,
            #     split="train",
            #     val_fraction=0.0,
            #     training=False,
            #     x_layer=self.x_layer,
            #     ensure_squidpy_neighbors=self.ensure_squidpy_neighbors,
            #     squidpy_kwargs=self.squidpy_kwargs,
            #     symmetrize_graph=self.symmetrize_graph,
            # )

        # Predict dataset uses the whole set of center cells.
        # Easiest is val_fraction=0 and split="train" so indices cover all cells.
        if stage in (None, "predict"):
            self.ds_predict = SimulatedDatasetV3(
                self.adata,
                sim_config=self.sim_config,
                split="train",
                val_fraction=0.0,
                training=False,
                x_layer=self.x_layer,
                ensure_squidpy_neighbors=self.ensure_squidpy_neighbors,
                squidpy_kwargs=self.squidpy_kwargs,
                symmetrize_graph=self.symmetrize_graph,
            )

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Pads neighbors to exactly k and returns:
          - adata_idx: [B]
          - neighbor_idx: [B, k]
          - neighbor_dist: [B, k]
          - cell: [B, G] and neighbors: [B, k, G] if return_expression
          - obs_keys as lists (or you can tensorize categoricals separately if you want)
        """
        B = len(batch)
        k = self.sim_config.k

        out: Dict[str, Any] = {}
        out["adata_idx"] = torch.tensor([b["adata_idx"] for b in batch], dtype=torch.long)
        out["slide_id"] = torch.tensor([b["slide_id"] for b in batch], dtype=torch.long)
        out["panel"] = torch.tensor([b["panel"] for b in batch], dtype=torch.long)

        # neigh_idx = torch.full((B, k), -1, dtype=torch.long)
        # neigh_dist = torch.full((B, k), float("inf"), dtype=torch.float32)

        # for bi, b in enumerate(batch):
        #     ni = b["neighbor_idx"]
        #     # nd = b["neighbor_dist"]
        #     m = min(k, ni.numel())
        #     if m > 0:
        #         neigh_idx[bi, :m] = ni[:m]
        #         # neigh_dist[bi, :m] = nd[:m]

        # out["neighbor_idx"] = neigh_idx
        # # out["neighbor_dist"] = neigh_dist

        # for key in self.obs_keys:
        #     out[key] = torch.tensor([b[key] for b in batch])

        # if self.return_expression:
        cell = torch.stack([b["cell"] for b in batch], dim=0)  # [B, G]
        G = cell.shape[1]
        neighbors = torch.zeros((B, k, G), dtype=torch.float32)
        for bi, b in enumerate(batch):
            xn = b["neighbors"]
            m = min(k, xn.shape[0])
            if m > 0:
                neighbors[bi, :m] = xn[:m]
        out["cell"] = cell
        out["neighbors"] = neighbors
        # if self.include_counts:
        #     out["cell_counts"] = torch.tensor(
        #         [b["cell_counts"] for b in batch], dtype=torch.float32
        #     )
        #     out["niche_counts"] = torch.tensor(
        #         [b["niche_counts"] for b in batch], dtype=torch.float32
        #     )

        # if self.obs_feature_groups:
        #     grouped_out: Dict[str, Dict[str, torch.Tensor]] = {}
        #     for group_name in self.obs_feature_groups:
        #         cell_feat = torch.stack(
        #             [b["obs_feature_groups"][group_name]["cell"] for b in batch], dim=0
        #         )
        #         Fdim = cell_feat.shape[1]
        #         neigh_feat = torch.zeros((B, k, Fdim), dtype=torch.float32)
        #         for bi, b in enumerate(batch):
        #             xn = b["obs_feature_groups"][group_name]["neighbors"]
        #             m = min(k, xn.shape[0])
        #             if m > 0:
        #                 neigh_feat[bi, :m] = xn[:m]

        #         grouped_out[group_name] = {"cell": cell_feat, "neighbors": neigh_feat}

        #     out["obs_feature_groups"] = grouped_out

        return out

    def _make_loader(
        self,
        dataset,
        *,
        loader_cfg: LoaderConfig,
        use_slide_batch_sampler: bool,
        training: bool,
    ) -> DataLoader:
        # if use_slide_batch_sampler:
        #     slide_to_dsidx = build_slide_to_dsidx(dataset, slide_key=self.slide_key)
        #     batch_sampler = InterleavedSlideBatchSampler(
        #         slide_to_dsidx=slide_to_dsidx,
        #         batch_size=loader_cfg.batch_size,
        #         shuffle_within=self.shuffle_within_slide,
        #         shuffle_batches=self.shuffle_batches and training,
        #         drop_last=loader_cfg.drop_last,
        #         seed=self.seed if training else self.seed + 1234,
        #     )
        #     return DataLoader(
        #         dataset,
        #         batch_sampler=batch_sampler,
        #         num_workers=loader_cfg.num_workers,
        #         pin_memory=loader_cfg.pin_memory,
        #         collate_fn=self._collate_fn
        #     )

        # Non-stratified loader (sequential for val/predict; shuffled for train)
        return DataLoader(
            dataset,
            batch_size=loader_cfg.batch_size,
            shuffle=training,
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory,
            drop_last=loader_cfg.drop_last,
            collate_fn=self._collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None
        return self._make_loader(
            self.ds_train,
            loader_cfg=self.train_loader_cfg,
            use_slide_batch_sampler=self.use_slide_batch_sampler,
            training=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None
        return self._make_loader(
            self.ds_val,
            loader_cfg=self.val_loader_cfg,
            use_slide_batch_sampler=self.use_slide_batch_sampler,
            training=False,
        )

    def predict_dataloader(self) -> DataLoader:
        assert self.ds_predict is not None
        return self._make_loader(
            self.ds_predict,
            loader_cfg=self.predict_loader_cfg,
            use_slide_batch_sampler=self.predict_use_slide_batch_sampler,
            training=False,
        )
