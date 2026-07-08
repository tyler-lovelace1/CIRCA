from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Sequence, Tuple, Union, List

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scipy.stats as st
import torch
from torch.utils.data import Dataset
from circa.data.generate_dataset import generate_dataset
from sklearn.decomposition import NMF, PCA
from sklearn.neighbors import BallTree


@dataclass
class NeighborSamplingConfig:
    k: int = 10
    expansion_factor: float = 2.0  # e.g. 1.5 => sample from top ceil(1.5*k)
    sample_with_invdist: bool = True
    invdist_eps: float = 1e-8
    weight_power: float = 1.0

class SpatialNeighborhoodDataset(Dataset):
    """
    Fast spatial neighborhood dataset with:
      - Precomputed dense neighbor index/dist arrays
      - Expansion factor candidate pool: top ceil(expansion_factor*k)
      - Optional inverse-distance weighted sampling
      - Optional symmetrization of distance graph (union, min distance)
      - Train/val split on center cells (cell->neighborhood pairs)
      - Returns obs columns + adata_idx for mapping back
    """

    def __init__(
        self,
        adata,
        sampling: NeighborSamplingConfig,
        *,
        split: str = "train",                 # "train" or "val"
        reference_indices: Optional[Sequence[int]] = None,
        reference_obs_key: Optional[str] = None,
        val_fraction: float = 0.0,            # e.g. 0.1
        split_seed: int = 0,
        slide_key: str = "slide_id",  # obs key
        panel_key: str = "panel",  # obs key
        training: bool = True,                # controls sampling behavior
        return_expression: bool = True,
        include_counts: bool = False,
        stratify_by : Optional[str] = None, # None or obs key
        x_layer: Optional[str] = None,
        obs_keys: Sequence[str] = ("slide_id", "panel"),
        obs_feature_groups: Optional[Dict[str, Sequence[str]]] = None,
        ensure_squidpy_neighbors: bool = True,
        squidpy_kwargs: Optional[Dict[str, Any]] = None,
        symmetrize_graph: bool = True,
        seed: Optional[int] = None,           # RNG for neighbor sampling
    ):
        self.adata = adata
        self.n_samples = adata.shape[0]
        self.sampling = sampling
        self.training = training
        self.return_expression = return_expression
        self.include_counts = bool(include_counts)
        self.x_layer = x_layer
        self.slide_key = slide_key
        self.panel_key = panel_key
        self.obs_keys = tuple(obs_keys)
        self.obs_feature_groups = {
            str(group_name): tuple(columns)
            for group_name, columns in (obs_feature_groups or {}).items()
        }
        self.rng = np.random.default_rng(seed)

        if "spatial" not in adata.obsm:
            raise ValueError("adata.obsm['spatial'] not found. Required.")

        # Candidate pool size M = ceil(expansion_factor*k) (at least k)
        k = sampling.k
        M = int(np.ceil(max(1.0, sampling.expansion_factor) * k))
        self.M = max(k, M)

        if ensure_squidpy_neighbors:
            self._ensure_spatial_neighbors(squidpy_kwargs)

        self._X = self._get_X()
        self._obs = self._extract_obs_columns(self.obs_keys)
        self._obs_feature_mats = self._extract_obs_feature_groups(self.obs_feature_groups)

        # Build train/val center indices
        self.all_indices = np.arange(adata.n_obs, dtype=np.int64)
        
        self.reference_indices = self._make_reference_indices(reference_indices, reference_obs_key)
        
        self.candidate_indices = np.setdiff1d(self.all_indices, self.reference_indices)
        
        self.train_indices, self.val_indices = self._make_split_indices(
            val_fraction=val_fraction,
            seed=split_seed,
            stratify_by=stratify_by,
        )

        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.split = split
        self.indices = self.train_indices if split == "train" else self.val_indices

        # Candidate pool size M = ceil(expansion_factor*k) (at least k)
        k = sampling.k
        M = int(np.ceil(max(1.0, sampling.expansion_factor) * k))
        self.M = max(k, M)

        # Get/symmetrize distance matrix
        D = self.adata.obsp.get("spatial_distances", None)
        if D is None:
            raise KeyError("adata.obsp['spatial_distances'] missing.")
        if not sp.issparse(D):
            D = sp.csr_matrix(D)
        else:
            D = D.tocsr()

        if symmetrize_graph:
            D = _symmetrize_sparse_min_union(D)

        # Precompute dense neighbor arrays for top M
        self.nbr_idx, self.nbr_dist = self._precompute_topM_neighbors(D, self.M)

    def __len__(self) -> int:
        return int(self.indices.size)

    def set_training(self, training: bool = True) -> None:
        self.training = training

    def _get_X(self):
        if self.x_layer is None:
            return self.adata.X
        if self.x_layer not in self.adata.layers:
            raise KeyError(f"x_layer={self.x_layer!r} not found in adata.layers")
        return self.adata.layers[self.x_layer]

    def _extract_obs_columns(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        """
        Extract obs columns with consistent categorical handling:
    
          - Categorical columns -> integer codes (int64)
          - String/object columns -> converted to categorical, then integer codes
          - Numeric/bool columns -> returned as numpy arrays
          - Categories stored in self._obs_categories for decoding
        """
    
        out: Dict[str, np.ndarray] = {}
        self._obs_categories: Dict[str, Any] = {}
    
        for k in keys:
            if k not in self.adata.obs:
                raise KeyError(f"obs key {k!r} not found in adata.obs")
    
            col = self.adata.obs[k]
    
            # Convert strings/objects to categorical
            if pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col):
                col = col.astype("category")
    
            # Handle categorical
            if pd.api.types.is_categorical_dtype(col):
                codes = col.cat.codes.to_numpy(dtype=np.int64, copy=False)
                out[k] = codes
                self._obs_categories[k] = col.cat.categories.to_numpy()
    
            # Numeric / bool
            else:
                out[k] = col.to_numpy(copy=False)
                self._obs_categories[k] = None
    
        return out

    def _extract_obs_feature_groups(
        self,
        groups: Dict[str, Sequence[str]],
    ) -> Dict[str, np.ndarray]:
        """
        Extract feature matrices from adata.obs for each named group.

        Each group must reference numeric/bool obs columns.
        Returns a dict: group_name -> float32 array [n_obs, n_group_features].
        """
        out: Dict[str, np.ndarray] = {}
        for group_name, cols in groups.items():
            if len(cols) == 0:
                raise ValueError(f"obs feature group {group_name!r} has no columns.")

            missing = [c for c in cols if c not in self.adata.obs]
            if missing:
                raise KeyError(
                    f"obs feature group {group_name!r} has missing obs columns: {missing}"
                )

            df = self.adata.obs.loc[:, list(cols)]
            non_numeric = [
                c for c in cols
                if not (
                    pd.api.types.is_numeric_dtype(df[c])
                    or pd.api.types.is_bool_dtype(df[c])
                )
            ]
            if non_numeric:
                raise TypeError(
                    f"obs feature group {group_name!r} has non-numeric columns: {non_numeric}"
                )

            out[group_name] = df.to_numpy(dtype=np.float32, copy=True)

        return out
            
    def _ensure_spatial_neighbors(self, squidpy_kwargs: Optional[Dict[str, Any]]) -> None:
        """
        Ensure squidpy spatial neighbor graph exists AND has at least M neighbors per cell (kNN case).
    
        Behavior:
          - If spatial_distances missing -> compute.
          - Else, if min neighbor count across cells < self.M -> recompute with n_neighs >= self.M.
    
        Notes:
          - For radius-based graphs (no n_neighs), degree may remain < M near borders or sparse regions.
            In that case you may still need padding in the dataset.
        """
        try:
            import squidpy as sq
        except ImportError as e:
            raise ImportError("Install squidpy to compute spatial neighbors: `pip install squidpy`.") from e
    
        def _neighbor_counts_from_distances(D) -> np.ndarray:
            if D is None:
                return np.zeros((self.adata.n_obs,), dtype=np.int64)
            if not sp.issparse(D):
                D = sp.csr_matrix(D)
            D = D.tocsr()
            counts = np.diff(D.indptr).astype(np.int64, copy=False)
    
            # If self-edges exist, subtract them
            # (common when distance matrix includes diagonal zeros)
            diag = D.diagonal()
            if diag is not None and diag.size == D.shape[0]:
                counts = counts - (diag != 0).astype(np.int64)  # only subtract if diagonal stored as nonzero
                # Many graphs store diagonal as exactly 0 and omit it; this is safe either way.
    
            # More robust self-edge removal: explicitly check per-row if i appears.
            # This is O(nnz) worst-case; usually fine but optional. Uncomment if you need correctness.
            # counts2 = counts.copy()
            # for i in range(D.shape[0]):
            #     start, end = D.indptr[i], D.indptr[i+1]
            #     if np.any(D.indices[start:end] == i):
            #         counts2[i] -= 1
            # counts = counts2
    
            return counts
    
        # Check current graph
        D = self.adata.obsp.get("spatial_distances", None)
        counts = _neighbor_counts_from_distances(D)
        need_compute = (D is None) or (counts.min(initial=0) < self.M)
    
        if not need_compute:
            return

        print(f"ds_train slide_key is {self.slide_key}")
        
        # Build kwargs and ensure n_neighs is big enough when using kNN graphs
        kwargs = {"coord_type": "generic", "library_key" : self.slide_key}
        if squidpy_kwargs:
            kwargs.update(squidpy_kwargs)
    
        # If using kNN (n_neighs present), bump up to >= M
        if "n_neighs" in kwargs and kwargs["n_neighs"] is not None:
            kwargs["n_neighs"] = int(max(kwargs["n_neighs"], self.M))
        else:
            # If user didn't specify n_neighs, default to kNN with at least M neighbors.
            # If they intended a radius graph, they'd typically pass radius in squidpy_kwargs.
            kwargs.setdefault("n_neighs", int(self.M))
    
        sq.gr.spatial_neighbors(self.adata, **kwargs)
    
        if "spatial_distances" not in self.adata.obsp or self.adata.obsp["spatial_distances"] is None:
            raise RuntimeError("Expected adata.obsp['spatial_distances'] after squidpy.gr.spatial_neighbors.")
    
        # Optional: re-check and warn (don’t raise) if still insufficient due to radius/edge effects
        D2 = self.adata.obsp.get("spatial_distances", None)
        counts2 = _neighbor_counts_from_distances(D2)
        if counts2.min(initial=0) < self.M:
            # Leave as a warning; dataset can pad. You can upgrade to raising if you want hard guarantees.
            # print(...) is avoided here; use logging if desired.
            pass

    def _make_reference_indices(
        self,
        reference_indices: Optional[Sequence[int]] = None,
        reference_obs_key: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return sorted unique AnnData row indices for the held-out reference set.
        """
        if reference_indices is not None and reference_obs_key is not None:
            raise ValueError("Provide only one of reference_indices or reference_obs_key.")
    
        if reference_indices is None and reference_obs_key is None:
            return np.empty((0,), dtype=np.int64)
    
        if reference_obs_key is not None:
            if reference_obs_key not in self.adata.obs:
                raise KeyError(f"reference_obs_key={reference_obs_key!r} not found in adata.obs")
    
            mask = self.adata.obs[reference_obs_key].to_numpy()
            if mask.dtype != bool:
                mask = mask.astype(bool)
    
            ref_idx = np.flatnonzero(mask).astype(np.int64, copy=False)
    
        else:
            ref_idx = np.asarray(reference_indices, dtype=np.int64).ravel()
    
        if ref_idx.size == 0:
            return np.empty((0,), dtype=np.int64)
    
        if ref_idx.min() < 0 or ref_idx.max() >= self.adata.n_obs:
            raise ValueError("reference_indices contains indices outside [0, adata.n_obs).")
    
        return np.sort(np.unique(ref_idx).astype(np.int64, copy=False))

    def _make_split_indices(
        self,
        *,
        val_fraction: float,
        seed: int,
        stratify_by: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if val_fraction <= 0.0:
            return self.candidate_indices, np.empty((0,), dtype=np.int64)
        if val_fraction >= 1.0:
            return np.empty((0,), dtype=np.int64), self.all_indices

        rng = np.random.default_rng(seed)

        if stratify_by is None:
            perm = rng.permutation(self.candidate_indices)
            n_val = int(np.floor(val_fraction * self.all_indices.size))
            val_idx = np.sort(perm[:n_val])
            train_idx = np.sort(perm[n_val:])
            return train_idx, val_idx

        if stratify_by not in self.adata.obs:
            raise KeyError(f"stratify_by={stratify_by!r} not found in adata.obs")

        labels = self.adata.obs[stratify_by].iloc[self.candidate_indices].to_numpy()
        val_list = []
        train_list = []

        for lab in np.unique(labels):
            grp = self.candidate_indices[labels == lab]
            perm = rng.permutation(grp)
            n_val = int(np.floor(val_fraction * grp.size))
            val_list.append(perm[:n_val])
            train_list.append(perm[n_val:])

        val_idx = np.sort(np.concatenate(val_list).astype(np.int64, copy=False))
        train_idx = np.sort(np.concatenate(train_list).astype(np.int64, copy=False))
        return train_idx, val_idx

    @staticmethod
    def _precompute_topM_neighbors(D: sp.csr_matrix, M: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        D: csr distance matrix
        Returns:
          nbr_idx: [N, M] padded with -1
          nbr_dist: [N, M] padded with +inf
        """
        D = D.tocsr()
        N = D.shape[0]
        nbr_idx = np.full((N, M), -1, dtype=np.int64)
        nbr_dist = np.full((N, M), np.inf, dtype=np.float32)

        for i in range(N):
            start, end = D.indptr[i], D.indptr[i + 1]
            cols = D.indices[start:end]
            vals = D.data[start:end]

            # drop self
            mask = cols != i
            cols = cols[mask]
            vals = vals[mask]

            if vals.size == 0:
                continue

            order = np.argsort(vals, kind="stable")
            cols = cols[order]
            vals = vals[order]

            m = min(M, cols.size)
            nbr_idx[i, :m] = cols[:m]
            nbr_dist[i, :m] = vals[:m].astype(np.float32, copy=False)

        return nbr_idx, nbr_dist

    @staticmethod
    def _row_to_dense(X, i: int) -> np.ndarray:
        if sp.issparse(X):
            return X[i].toarray().ravel().astype(np.float32, copy=False)
        return np.asarray(X[i]).ravel().astype(np.float32, copy=False)

    @staticmethod
    def _rows_to_dense(X, idxs: np.ndarray) -> np.ndarray:
        if sp.issparse(X):
            return X[idxs].toarray().astype(np.float32, copy=False)
        return np.asarray(X[idxs]).astype(np.float32, copy=False)

    def _sample_k_from_topM(self, idx_row: np.ndarray, dist_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        idx_row/dist_row: length M, sorted, padded (-1/+inf)
        Sample k from valid entries in top M with inv-distance weights.
        Return selected sorted by distance.
        """
        k = self.sampling.k
        idxs = idx_row
        dists = dist_row

        valid = idxs >= 0
        idxs = idxs[valid]
        dists = dists[valid]

        if idxs.size <= k:
            return idxs, dists

        eps = self.sampling.invdist_eps
        pwr = self.sampling.weight_power
        w = 1.0 / np.power(dists + eps, pwr)
        p = w / w.sum()

        chosen_pos = self.rng.choice(idxs.size, size=k, replace=False, p=p)
        chosen_idxs = idxs[chosen_pos]
        chosen_dists = dists[chosen_pos]
        order = np.argsort(chosen_dists, kind="stable")
        return chosen_idxs[order], chosen_dists[order]

    def __getitem__(self, j: int) -> Dict[str, Any]:
        # Map dataset index -> original adata index
        i = int(self.indices[j])

        idx_row = self.nbr_idx[i, :self.M]
        dist_row = self.nbr_dist[i, :self.M]

        if self.training and self.sampling.sample_with_invdist and self.M > self.sampling.k:
            neigh_i, dist_i = self._sample_k_from_topM(idx_row, dist_row)
        else:
            # deterministic top-k
            k = self.sampling.k
            idxs = idx_row[:k]
            dists = dist_row[:k]
            valid = idxs >= 0
            neigh_i, dist_i = idxs[valid], dists[valid]

        out: Dict[str, Any] = {
            "adata_idx": i,  # <-- this is the original location in adata
            "neighbor_idx": torch.from_numpy(neigh_i).long(),
            "neighbor_dist" : torch.from_numpy(dist_i).float()
        }

        for key in self.obs_keys:
            val = self._obs[key][i]
            out[key] = val.item() if isinstance(val, np.generic) else val

        val = self._obs[self.slide_key][i]
        out["slide_id"] = val.item() if isinstance(val, np.generic) else val

        val = self._obs[self.panel_key][i]
        out["panel"] = val.item() if isinstance(val, np.generic) else val

        # if self.return_expression:
        #     cell = self._row_to_dense(self._X, i)
        #     neighbors = (
        #         self._rows_to_dense(self._X, neigh_i)
        #         if neigh_i.size > 0
        #         else np.zeros((0, cell.shape[0]), dtype=np.float32)
        #     )
        #     out["cell"] = torch.from_numpy(cell).float()
        #     out["neighbors"] = torch.from_numpy(neighbors).float()
        #     if self.include_counts:
        #         out["cell_counts"] = float(cell.sum(dtype=np.float64))
        #         out["niche_counts"] = float(neighbors.sum(dtype=np.float64))

        if self._obs_feature_mats:
            obs_feature_groups: Dict[str, Dict[str, torch.Tensor]] = {}
            for group_name, group_mat in self._obs_feature_mats.items():
                cell_feat = group_mat[i]
                neigh_feat = (
                    group_mat[neigh_i]
                    if neigh_i.size > 0
                    else np.zeros((0, group_mat.shape[1]), dtype=np.float32)
                )
                obs_feature_groups[group_name] = {
                    "cell": torch.from_numpy(cell_feat).float(),
                    "neighbors": torch.from_numpy(neigh_feat).float(),
                }
            out["obs_feature_groups"] = obs_feature_groups

        return out

@dataclass
class SimulationConfig:
    k: int = 10
    num_layer: int = 3  # number of layers of mixing-MLP f
    num_group: int = 2  # number of groups
    num_neighbor: int = 2  # number of neighbors of a variable in a single group-pair
    num_neighbor_in: int = 1  # number of neighbors of a variable within a group
    lam2: Sequence[int] = (1, 1)
    lamin2: Sequence[int] = (1, 1)
    random_seed: int = 0
    dist_type: str = 'Gauss'  # noise distribution
    dag: bool = True  # DAG or not
    num_dim: int = 10  # number of variables in a group
    num_data: int = 2**16  # number of samples
    num_latent: Optional[int] = None  # number of latent confounders
    ar_alpha: float = 3  # AR parameter
    ar_beta: float = 0.8  # AR parameter
    lam1: Sequence[float] = (0.9, 1)  # modulation range of inter-L
    lamin1: Sequence[float] = (0.9, 1)  # modulation range of intra-L

class SimulatedDataset(Dataset):
    """
    Fast spatial neighborhood dataset with:
      - Precomputed dense neighbor index/dist arrays
      - Expansion factor candidate pool: top ceil(expansion_factor*k)
      - Optional inverse-distance weighted sampling
      - Optional symmetrization of distance graph (union, min distance)
      - Train/val split on center cells (cell->neighborhood pairs)
      - Returns obs columns + adata_idx for mapping back
    """

    def __init__(
        self,
        adata,
        sim_config: SimulationConfig,
        *,
        # slide_key: str = "slide_id",  # obs key
        training: bool = True,
        split: str = "train",                 # "train" or "val"
        val_fraction: float = 0.1,
        x_layer: Optional[str] = None,
        # obs_keys: Sequence[str] = ("slide_id", "panel"),
        ensure_squidpy_neighbors: bool = True,
        squidpy_kwargs: Optional[Dict[str, Any]] = None,
        symmetrize_graph: bool = True,
        pca_models = None
    ):
        self.adata = adata
        self.n_samples = adata.shape[0]
        self.training = training
        self.val_fraction = val_fraction
        # self.slide_key = slide_key
        self.sim_config = sim_config
        self.x_layer = x_layer
        # self.obs_keys = tuple(obs_keys)
        if pca_models is None:
            self.pca_models = [PCA(whiten=True), PCA(whiten=True)]
        else:
            self.pca_models = pca_models
        
        if "spatial" not in adata.obsm:
            raise ValueError("adata.obsm['spatial'] not found. Required.")

        # Candidate pool size M = ceil(expansion_factor*k) (at least k)
        k = sim_config.k
        self.M = k

        if ensure_squidpy_neighbors:
            self._ensure_spatial_neighbors(squidpy_kwargs)

        self._X = self._get_X()
        # self._obs = self._extract_obs_columns(self.obs_keys)
        
        if split not in ("train", "val", "test", "pred"):
            raise ValueError("split must be 'train', 'val', 'test', or 'pred'")
        
        self.split = split
        # if split == "val":
        #     self.sim_config.num_data = int(self.val_fraction * self.sim_config.num_data / (1 - self.val_fraction))
            
        self.indices = np.arange(self.sim_config.num_data, dtype=int)

        # Get/symmetrize distance matrix
        D = self.adata.obsp.get("spatial_distances", None)
        if D is None:
            raise KeyError("adata.obsp['spatial_distances'] missing.")
        if not sp.issparse(D):
            D = sp.csr_matrix(D)
        else:
            D = D.tocsr()

        if symmetrize_graph:
            D = _symmetrize_sparse_min_union(D)

        # Precompute dense neighbor arrays for top M
        self.nbr_idx, self.nbr_dist = self._precompute_topM_neighbors(D, self.M)

        self._run_nmf()

        for i, value in enumerate(["train", "val", "test", "pred"]):
            if value == self.split:
                sampler_seed = i
    
        sampler_seed = (sampler_seed + self.sim_config.random_seed) * (sampler_seed + self.sim_config.random_seed + 1) // 2 + self.sim_config.random_seed

        # generate sensor signal --------------------------------------
        self.x, self.s, self.A1, self.A2, self.Ain1, self.Ain2 = generate_dataset(
            num_group=self.sim_config.num_group,
            num_dim=self.sim_config.num_dim,
            num_data=self.sim_config.num_data,
            num_layer=self.sim_config.num_layer,
            lam1_range=self.sim_config.lam1,
            lam2_range=self.sim_config.lam2,
            lamin1_range=self.sim_config.lamin1,
            lamin2_range=self.sim_config.lamin2,
            ar_alpha=self.sim_config.ar_alpha,
            ar_beta=self.sim_config.ar_beta,
            num_neighbor=self.sim_config.num_neighbor,
            num_neighbor_in=self.sim_config.num_neighbor_in,
            dag=self.sim_config.dag,
            dist_type=self.sim_config.dist_type,
            num_latent=self.sim_config.num_latent,
            generator_seed=self.sim_config.random_seed,
            sampler_seed = sampler_seed
        )

        # if self.sim_config.num_layer == 0:
        #     self.x = self.s.copy()

        for m in range(self.sim_config.num_group):
            if pca_models is None:
                self.x[:, m, :] = self.pca_models[m].fit_transform(self.x[:, m, :])
            else:
                self.x[:, m, :] = self.pca_models[m].transform(self.x[:, m, :])
                
            self.x[:, m, :] = st.norm.ppf(st.rankdata(self.x[:, m, :], axis=0, method='max') / (self.sim_config.num_data+1))
            if m == 0:
                nmf_npn = st.norm.ppf(st.rankdata(self.adata.obsm['X_nmf'], axis=0, method='max') / (self.n_samples+1))
                C_nmf = np.linalg.cholesky(np.corrcoef(nmf_npn.T), upper=True)
                self.x[:, m, :] = self.x[:, m, :] @ C_nmf
            if m == 1:
                nmf_nbrs_npn = st.norm.ppf(st.rankdata(self.adata.obsm['X_nmf_nbrs'], axis=0, method='max') / (self.n_samples+1))
                C_nmf_nbrs = np.linalg.cholesky(np.corrcoef(nmf_nbrs_npn.T), upper=True)
                self.x[:, m, :] = self.x[:, m, :] @ C_nmf_nbrs

        x_quantiles = st.rankdata(self.x, axis=0) / (self.sim_config.num_data+1)

        for m in range(self.sim_config.num_group):
            for i in range(self.sim_config.num_dim):
                if m == 0:
                    self.x[:,m,i] = np.quantile(self.adata.obsm['X_nmf'][:,i], x_quantiles[:,m,i], method='linear')
                elif m == 1:
                    self.x[:,m,i] = np.quantile(self.adata.obsm['X_nmf_nbrs'][:,i], x_quantiles[:,m,i], method='linear')

        self.sim_counts = np.random.poisson(2 * np.expm1(self.x[:,0,:] @ self.adata.uns['X_nmf_components']))
        # sim_total_counts = np.sum(self.sim_counts, axis=1, keepdims=True)
        # self.sim_counts = np.random.poisson(np.median(sim_total_counts) * self.sim_counts / sim_total_counts)

        self.sim_nbrs_counts = np.random.poisson(2 * np.expm1(self.x[:,1,:] @ self.adata.uns['X_nmf_nbrs_components']))
        # sim_nbrs_total_counts = np.sum(self.sim_nbrs_counts, axis=1, keepdims=True)
        # self.sim_nbrs_counts = np.random.poisson(np.median(sim_nbrs_total_counts) * self.sim_nbrs_counts / sim_nbrs_total_counts)
        self.sim_nbrs_counts = self.sim_nbrs_counts.reshape(self.sim_config.num_data, self.sim_config.k, -1)

        if split == "val":
            self.sim_config.num_data = int(self.val_fraction * self.sim_config.num_data / (1 - self.val_fraction))
            self.sim_counts = self.sim_counts[:self.sim_config.num_data]
            self.sim_nbrs_counts = self.sim_nbrs_counts[:self.sim_config.num_data]
            self.indices = np.arange(self.sim_config.num_data, dtype=int)
        

    def __len__(self) -> int:
        return int(self.indices.size)

    def set_training(self, training: bool = True) -> None:
        self.training = training

    def _get_X(self):
        if self.x_layer is None:
            return self.adata.X
        if self.x_layer not in self.adata.layers:
            raise KeyError(f"x_layer={self.x_layer!r} not found in adata.layers")
        return self.adata.layers[self.x_layer]

    def _run_nmf(self, store_key='X_nmf'):
        """
        Run NMF on single-cell/spatial count data stored in AnnData.
    
        Stores:
            adata.obsm[store_key]          = cell/loadings matrix W, shape (n_cells, n_components)
            adata.uns[f"{store_key}_genes"] = gene names used
            adata.uns[f"{store_key}_components"] = component matrix H, shape (n_components, n_genes)
            adata.varm[f"{store_key}_components"] = full-gene matrix with NaNs for unused genes
    
        Returns:
            fitted sklearn NMF model
        """
    
        adata_nmf = self.adata.copy()
    
        # Use selected layer if provided
        if self.x_layer is not None:
            if self.x_layer not in adata_nmf.layers:
                raise KeyError(f"Layer {self.x_layer!r} not found in adata.layers.")
            adata_nmf.X = adata_nmf.layers[self.x_layer].copy()
    
        # Normalize
        sc.pp.normalize_total(adata_nmf)
        sc.pp.log1p(adata_nmf)
    
        X = adata_nmf.X
    
        # Ensure non-negative values
        if sp.issparse(X):
            min_val = X.data.min() if X.data.size else 0.0
        else:
            min_val = np.min(X)
    
        if min_val < 0:
            raise ValueError("NMF requires non-negative input. Do not use centered/scaled data.")
    
        # sklearn NMF supports sparse input
        self.nmf = NMF(
            n_components=self.sim_config.num_dim,
            init="nndsvda",
            solver="mu",
            beta_loss="frobenius",
            random_state=self.sim_config.random_seed,
            max_iter=1000,
        )
    
        W = self.nmf.fit_transform(X)      # cells x factors
        H = self.nmf.components_           # factors x genes_used
    
        # Store cell loadings
        self.adata.obsm[store_key] = W.astype(np.float32)
    
        # Store gene names and components
        # used_genes = adata_nmf.var_names[gene_mask].to_numpy()
        # adata.uns[f"{store_key}_genes"] = used_genes
        self.adata.uns[f"{store_key}_components"] = H.astype(np.float32)

        # sklearn NMF supports sparse input
        self.nbrhood_nmf = NMF(
            n_components=self.sim_config.num_dim,
            init="nndsvda",
            solver="mu",
            beta_loss="frobenius",
            random_state=self.sim_config.random_seed,
            max_iter=1000,
        )
    
        W_nbrs = self.nbrhood_nmf.fit_transform(np.asarray(X.todense())[self.nbr_idx].reshape(self.n_samples,-1))      # neighborhoods x factors
        H_nbrs = self.nbrhood_nmf.components_           # factors x genes_used
    
        # Store cell loadings
        self.adata.obsm[store_key + '_nbrs'] = W_nbrs.astype(np.float32)
    
        self.adata.uns[f"{store_key}_nbrs_components"] = H_nbrs.astype(np.float32)
        

    # def _extract_obs_columns(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
    #     """
    #     Extract obs columns with consistent categorical handling:
    
    #       - Categorical columns -> integer codes (int64)
    #       - String/object columns -> converted to categorical, then integer codes
    #       - Numeric/bool columns -> returned as numpy arrays
    #       - Categories stored in self._obs_categories for decoding
    #     """
    
    #     out: Dict[str, np.ndarray] = {}
    #     self._obs_categories: Dict[str, Any] = {}
    
    #     for k in keys:
    #         if k not in self.adata.obs:
    #             raise KeyError(f"obs key {k!r} not found in adata.obs")
    
    #         col = self.adata.obs[k]
    
    #         # Convert strings/objects to categorical
    #         if pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col):
    #             col = col.astype("category")
    
    #         # Handle categorical
    #         if pd.api.types.is_categorical_dtype(col):
    #             codes = col.cat.codes.to_numpy(dtype=np.int64, copy=False)
    #             out[k] = codes
    #             self._obs_categories[k] = col.cat.categories.to_numpy()
    
    #         # Numeric / bool
    #         else:
    #             out[k] = col.to_numpy(copy=False)
    #             self._obs_categories[k] = None
    
    #     return out
            
    def _ensure_spatial_neighbors(self, squidpy_kwargs: Optional[Dict[str, Any]]) -> None:
        """
        Ensure squidpy spatial neighbor graph exists AND has at least M neighbors per cell (kNN case).
    
        Behavior:
          - If spatial_distances missing -> compute.
          - Else, if min neighbor count across cells < self.M -> recompute with n_neighs >= self.M.
    
        Notes:
          - For radius-based graphs (no n_neighs), degree may remain < M near borders or sparse regions.
            In that case you may still need padding in the dataset.
        """
        try:
            import squidpy as sq
        except ImportError as e:
            raise ImportError("Install squidpy to compute spatial neighbors: `pip install squidpy`.") from e
    
        def _neighbor_counts_from_distances(D) -> np.ndarray:
            if D is None:
                return np.zeros((self.adata.n_obs,), dtype=np.int64)
            if not sp.issparse(D):
                D = sp.csr_matrix(D)
            D = D.tocsr()
            counts = np.diff(D.indptr).astype(np.int64, copy=False)
    
            # If self-edges exist, subtract them
            # (common when distance matrix includes diagonal zeros)
            diag = D.diagonal()
            if diag is not None and diag.size == D.shape[0]:
                counts = counts - (diag != 0).astype(np.int64)  # only subtract if diagonal stored as nonzero
                # Many graphs store diagonal as exactly 0 and omit it; this is safe either way.
    
            # More robust self-edge removal: explicitly check per-row if i appears.
            # This is O(nnz) worst-case; usually fine but optional. Uncomment if you need correctness.
            # counts2 = counts.copy()
            # for i in range(D.shape[0]):
            #     start, end = D.indptr[i], D.indptr[i+1]
            #     if np.any(D.indices[start:end] == i):
            #         counts2[i] -= 1
            # counts = counts2
    
            return counts
    
        # Check current graph
        D = self.adata.obsp.get("spatial_distances", None)
        counts = _neighbor_counts_from_distances(D)
        need_compute = (D is None) or (counts.min(initial=0) < self.M)
    
        if not need_compute:
            return
    
        # Build kwargs and ensure n_neighs is big enough when using kNN graphs
        kwargs = {"coord_type": "generic", "library_key" : None}
        if squidpy_kwargs:
            kwargs.update(squidpy_kwargs)
    
        # If using kNN (n_neighs present), bump up to >= M
        if "n_neighs" in kwargs and kwargs["n_neighs"] is not None:
            kwargs["n_neighs"] = int(max(kwargs["n_neighs"], self.M))
        else:
            # If user didn't specify n_neighs, default to kNN with at least M neighbors.
            # If they intended a radius graph, they'd typically pass radius in squidpy_kwargs.
            kwargs.setdefault("n_neighs", int(self.M))
    
        sq.gr.spatial_neighbors(self.adata, **kwargs)
    
        if "spatial_distances" not in self.adata.obsp or self.adata.obsp["spatial_distances"] is None:
            raise RuntimeError("Expected adata.obsp['spatial_distances'] after squidpy.gr.spatial_neighbors.")
    
        # Optional: re-check and warn (don’t raise) if still insufficient due to radius/edge effects
        D2 = self.adata.obsp.get("spatial_distances", None)
        counts2 = _neighbor_counts_from_distances(D2)
        if counts2.min(initial=0) < self.M:
            # Leave as a warning; dataset can pad. You can upgrade to raising if you want hard guarantees.
            # print(...) is avoided here; use logging if desired.
            pass

    @staticmethod
    def _precompute_topM_neighbors(D: sp.csr_matrix, M: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        D: csr distance matrix
        Returns:
          nbr_idx: [N, M] padded with -1
          nbr_dist: [N, M] padded with +inf
        """
        D = D.tocsr()
        N = D.shape[0]
        nbr_idx = np.full((N, M), -1, dtype=np.int64)
        nbr_dist = np.full((N, M), np.inf, dtype=np.float32)

        for i in range(N):
            start, end = D.indptr[i], D.indptr[i + 1]
            cols = D.indices[start:end]
            vals = D.data[start:end]

            # drop self
            mask = cols != i
            cols = cols[mask]
            vals = vals[mask]

            if vals.size == 0:
                continue

            order = np.argsort(vals, kind="stable")
            cols = cols[order]
            vals = vals[order]

            m = min(M, cols.size)
            nbr_idx[i, :m] = cols[:m]
            nbr_dist[i, :m] = vals[:m].astype(np.float32, copy=False)

        return nbr_idx, nbr_dist

    def __getitem__(self, j: int) -> Dict[str, Any]:
        # Map dataset index -> original adata index
        i = int(self.indices[j])

        # idx_row = self.nbr_idx[i, :self.M]
        # dist_row = self.nbr_dist[i, :self.M]

        # if self.training and self.sampling.sample_with_invdist and self.M > self.sampling.k:
        #     neigh_i, dist_i = self._sample_k_from_topM(idx_row, dist_row)
        # else:
        #     # deterministic top-k
        #     k = self.sampling.k
        #     idxs = idx_row[:k]
        #     dists = dist_row[:k]
        #     valid = idxs >= 0
        #     neigh_i, dist_i = idxs[valid], dists[valid]

        out: Dict[str, Any] = {
            "adata_idx" : torch.tensor([i], dtype=torch.long),
            "slide_id" : torch.zeros(1, dtype=torch.long),
            "panel" : torch.zeros(1, dtype=torch.long)
        }

        # for key in self.obs_keys:
        #     val = self._obs[key][i]
        #     out[key] = val.item() if isinstance(val, np.generic) else val

        cell = np.asarray(self.sim_counts[i]).ravel().astype(np.float32, copy=False)
        neighbors = np.asarray(self.sim_nbrs_counts[i]).astype(np.float32, copy=False)
        out["cell"] = torch.from_numpy(cell).float()
        out["neighbors"] = torch.from_numpy(neighbors).float()
                
        return out


def _symmetrize_sparse_min_union(D: sp.spmatrix) -> sp.csr_matrix:
    """
    Symmetrize a sparse distance matrix by taking the union of edges and,
    for duplicate (i,j) entries, keeping the MIN distance.

    Result is symmetric (i,j) exists iff either direction existed.
    """
    if not sp.issparse(D):
        D = sp.csr_matrix(D)
    D = D.tocoo()
    Dt = D.transpose().tocoo()

    r = np.concatenate([D.row, Dt.row])
    c = np.concatenate([D.col, Dt.col])
    v = np.concatenate([D.data, Dt.data])

    # sort by (r, c) to reduce duplicates with min
    order = np.lexsort((c, r))
    r, c, v = r[order], c[order], v[order]

    # find groups of identical (r,c)
    same = (r[1:] == r[:-1]) & (c[1:] == c[:-1])
    # group boundaries
    idx_starts = np.concatenate([[0], np.where(~same)[0] + 1])
    idx_ends = np.concatenate([idx_starts[1:], [r.size]])

    r_out = r[idx_starts]
    c_out = c[idx_starts]
    v_out = np.empty_like(r_out, dtype=v.dtype)

    for gi, (a, b) in enumerate(zip(idx_starts, idx_ends)):
        v_out[gi] = v[a:b].min()

    Dsym = sp.coo_matrix((v_out, (r_out, c_out)), shape=D.shape).tocsr()
    Dsym.eliminate_zeros()
    return Dsym

def sample_reference_indices_by_slide(
    adata,
    *,
    slide_key: str = "slide_id",
    reference_fraction: Optional[float] = None,
    n_per_slide: Optional[int] = None,
    min_per_slide: int = 1,
    max_per_slide: Optional[int] = None,
    seed: int = 0,
    sort_indices: bool = True,
) -> np.ndarray:
    """
    Sample reference AnnData row indices stratified by slide.

    Provide exactly one of:
        reference_fraction: fraction of cells per slide to sample
        n_per_slide: fixed number of cells per slide to sample

    Args:
        adata:
            AnnData object.
        slide_key:
            Column in adata.obs containing slide identifiers.
        reference_fraction:
            Fraction of cells to sample within each slide, e.g. 0.10.
        n_per_slide:
            Fixed number of cells to sample per slide.
        min_per_slide:
            Minimum number of reference cells sampled from each non-empty slide
            when using reference_fraction.
        max_per_slide:
            Optional cap on number sampled per slide.
        seed:
            Random seed.
        sort_indices:
            If True, return sorted indices. If False, preserve sampled order.

    Returns:
        np.ndarray of AnnData row indices, dtype int64.
    """
    if slide_key not in adata.obs:
        raise KeyError(f"slide_key={slide_key!r} not found in adata.obs.")

    if (reference_fraction is None) == (n_per_slide is None):
        raise ValueError("Provide exactly one of reference_fraction or n_per_slide.")

    if reference_fraction is not None:
        if not (0.0 < reference_fraction < 1.0):
            raise ValueError("reference_fraction must be between 0 and 1.")

    if n_per_slide is not None:
        if n_per_slide <= 0:
            raise ValueError("n_per_slide must be positive.")

    rng = np.random.default_rng(seed)

    slides = adata.obs[slide_key].to_numpy()
    all_indices = np.arange(adata.n_obs, dtype=np.int64)

    ref_indices = []

    for slide in pd.unique(slides):
        slide_idx = all_indices[slides == slide]
        n_slide = slide_idx.size

        if n_slide == 0:
            continue

        if reference_fraction is not None:
            n_ref = int(np.floor(reference_fraction * n_slide))
            n_ref = max(min_per_slide, n_ref)
        else:
            n_ref = int(n_per_slide)

        if max_per_slide is not None:
            n_ref = min(n_ref, int(max_per_slide))

        n_ref = min(n_ref, n_slide)

        sampled = rng.choice(slide_idx, size=n_ref, replace=False)
        ref_indices.append(sampled)

    if not ref_indices:
        return np.empty((0,), dtype=np.int64)

    ref_indices = np.concatenate(ref_indices).astype(np.int64, copy=False)

    if sort_indices:
        ref_indices = np.sort(ref_indices)

    return ref_indices