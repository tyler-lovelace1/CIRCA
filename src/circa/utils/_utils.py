from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy
import math
import importlib
import scanpy as sc
from anndata import AnnData

def tqdm(*args, desc="DataLoader", **kwargs):
    # check if ipywidgets is installed before importing tqdm.auto
    # to ensure it won't fail and a progress bar is displayed
    if importlib.util.find_spec("ipywidgets") is not None:
        from tqdm.auto import tqdm as _tqdm
    else:
        from tqdm import tqdm as _tqdm

    return _tqdm(*args, desc=desc, **kwargs)

def maxAbsScale(matrix):
    maxabs = np.abs(matrix).max()
    return matrix / maxabs

def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
         method: correlation method ('Pearson' or 'Spearman')
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
     """

    print('Calculating correlation...')

    if method == 'Pearson':
        x = x.copy().T
        y = y.copy().T
    elif method == 'Spearman':
        x = scipy.stats.rankdata(x.copy(), axis=0).T
        y = scipy.stats.rankdata(y.copy(), axis=0).T
    elif method == 'Binned':
        n_bins = 20
        quantiles = np.linspace(0, 1, n_bins + 1)
        x_bin_edges = np.quantile(x.copy(), q=quantiles, axis=0)
        x_bin_indices = np.zeros_like(x, dtype=int)
        for i in range(x.shape[1]):
            x_bin_indices[:, i] = np.digitize(x[:, i], x_bin_edges[:, i])
        x = x_bin_indices.T
        
        y_bin_edges = np.quantile(y.copy(), q=quantiles, axis=0)
        y_bin_indices = np.zeros_like(y, dtype=int)
        for i in range(y.shape[1]):
            y_bin_indices[:, i] = np.digitize(y[:, i], y_bin_edges[:, i])
        y = y_bin_indices.T
    else:
        raise ValueError
        
    dimx = x.shape[0]
    dimy = y.shape[0]

    corr = np.corrcoef(y, x)
    corr = corr[0:dimy, dimy:]
    
    if np.max(np.isnan(corr)):
        raise ValueError

    abs_corr = np.abs(corr)
    
    # sort
    from scipy.optimize import linear_sum_assignment
    
    row_ind, col_ind = linear_sum_assignment(-abs_corr)

    sort_idx = col_ind
    sort_idx_other = np.setdiff1d(np.arange(0, dimx), sort_idx)
    sort_idx = np.concatenate([sort_idx, sort_idx_other])

    x_sort = x[sort_idx, :]

    # re-calculate correlation
    corr_sort = np.corrcoef(y, x_sort)
    corr_sort = corr_sort[0:dimy, dimy:]

    return corr_sort, sort_idx, x_sort

def correlation_greedy(x, y, method='Pearson'):
    """
    Greedily align latent factors in x to latent factors in y based on the
    absolute value of their cross-correlation.

    At each step, the unmatched (y_factor, x_factor) pair with the largest
    absolute correlation is selected. Once matched, both factors are removed
    from consideration.

    Args:
        x:
            Array of shape [n_samples, dim_x].
            Latent factors to be reordered.

        y:
            Array of shape [n_samples, dim_y].
            Reference latent factors.

        method:
            Correlation method:
                'Pearson'
                'Spearman'
                'Binned'

    Returns:
        corr_sort:
            Correlation matrix between y and reordered x.
            Shape [dim_y, dim_x].

            The first min(dim_x, dim_y) columns correspond to the greedy
            matches, ordered by the y factor they match.

        sort_idx:
            Indices used to reorder the original x dimensions.

        x_sort:
            Reordered x latent factors in transposed form,
            matching the behavior of the original function:
            shape [dim_x, n_samples].

        matched_pairs:
            List of tuples:
                (y_idx, x_idx, corr, abs_corr)

            Sorted by y_idx.
    """

    print('Calculating correlation...')

    if method == 'Pearson':
        x = x.copy().T
        y = y.copy().T

    elif method == 'Spearman':
        x = scipy.stats.rankdata(x.copy(), axis=0).T
        y = scipy.stats.rankdata(y.copy(), axis=0).T

    elif method == 'Binned':
        n_bins = 20
        quantiles = np.linspace(0, 1, n_bins + 1)

        x_bin_edges = np.quantile(x.copy(), q=quantiles, axis=0)
        x_bin_indices = np.zeros_like(x, dtype=int)

        for i in range(x.shape[1]):
            x_bin_indices[:, i] = np.digitize(
                x[:, i],
                x_bin_edges[:, i]
            )

        x = x_bin_indices.T

        y_bin_edges = np.quantile(y.copy(), q=quantiles, axis=0)
        y_bin_indices = np.zeros_like(y, dtype=int)

        for i in range(y.shape[1]):
            y_bin_indices[:, i] = np.digitize(
                y[:, i],
                y_bin_edges[:, i]
            )

        y = y_bin_indices.T

    else:
        raise ValueError(
            "method must be 'Pearson', 'Spearman', or 'Binned'"
        )

    dimx = x.shape[0]
    dimy = y.shape[0]

    # Cross-correlation: rows = y factors, columns = x factors
    corr = np.corrcoef(y, x)
    corr = corr[:dimy, dimy:]

    if np.any(np.isnan(corr)):
        raise ValueError("NaN values found in correlation matrix.")

    abs_corr = np.abs(corr)

    # ---------------------------------------------------------
    # Greedy matching
    # ---------------------------------------------------------

    remaining_y = set(range(dimy))
    remaining_x = set(range(dimx))

    matched_pairs = []

    n_matches = min(dimx, dimy)

    for _ in range(n_matches):

        y_idx = np.array(sorted(remaining_y))
        x_idx = np.array(sorted(remaining_x))

        sub_corr = abs_corr[np.ix_(y_idx, x_idx)]

        # Find largest remaining absolute correlation
        flat_idx = np.argmax(sub_corr)
        sub_y, sub_x = np.unravel_index(flat_idx, sub_corr.shape)

        yi = y_idx[sub_y]
        xi = x_idx[sub_x]

        matched_pairs.append(
            (
                yi,
                xi,
                corr[yi, xi],
                abs_corr[yi, xi],
            )
        )

        remaining_y.remove(yi)
        remaining_x.remove(xi)

    # ---------------------------------------------------------
    # Reorder x so matched x factors correspond to y factor order
    # ---------------------------------------------------------

    matched_pairs = sorted(matched_pairs, key=lambda p: p[0])

    matched_x_idx = np.array(
        [p[1] for p in matched_pairs],
        dtype=int,
    )

    # Append unmatched x factors if dimx > dimy
    unmatched_x_idx = np.array(
        sorted(remaining_x),
        dtype=int,
    )

    sort_idx = np.concatenate(
        [matched_x_idx, unmatched_x_idx]
    )

    x_sort = x[sort_idx, :]

    # Recalculate correlation after sorting
    corr_sort = np.corrcoef(y, x_sort)
    corr_sort = corr_sort[:dimy, dimy:]

    return corr_sort, sort_idx, x_sort, matched_pairs

# def leiden_cluster_k(adata, n_clusts, latent_key, low, high, n_neighbors=15, metric='euclidean', depth=0, maxDepth=6):
#     if not latent_key + '_neighbors' in adata.uns.keys():
#         if metric != 'cosine':
#             # adata.obsm[latent_key + '_scaled'] = maxAbsScale(adata.obsm[latent_key])
#             sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=latent_key, key_added=latent_key + '_neighbors', metric=metric, n_pcs=adata.obsm[latent_key].shape[1])
#         else:
#             sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=latent_key, key_added=latent_key + '_neighbors', metric=metric, n_pcs=adata.obsm[latent_key].shape[1])

#     midpoint = (high + low) / 2
#     clust_key = f'leiden_{midpoint}_' + latent_key
#     if not clust_key in adata.obs.columns:
#         sc.tl.leiden(adata, neighbors_key=latent_key + '_neighbors', resolution=midpoint, key_added=clust_key, flavor='igraph', n_iterations=3)
#     num_found_clusts = len(adata.obs[clust_key].cat.categories)
#     # print(f"{num_found_clusts} clusters found.")
#     if num_found_clusts == n_clusts or depth > maxDepth:
#         new_clust_key = f'leiden_{n_clusts}_' + latent_key
#         adata.obs[new_clust_key] = adata.obs[clust_key]
#         return new_clust_key
#     elif num_found_clusts < n_clusts:
#         low = midpoint
#         return leiden_cluster_k(adata, n_clusts, latent_key, low, high, n_neighbors, metric, depth=depth+1, maxDepth=maxDepth)
#     else:
#         high = midpoint
#         return leiden_cluster_k(adata, n_clusts, latent_key, low, high, n_neighbors, metric, depth=depth+1, maxDepth=maxDepth)

def _get_clustering_backend(prefer_gpu=True):
    """
    Return (neighbors_function, leiden_function, backend_name).

    GPU use requires:
      1. rapids-singlecell and its RAPIDS dependencies
      2. a CUDA device accessible through CuPy
    """
    if prefer_gpu:
        try:
            import cupy as cp
            import rapids_singlecell as rsc

            if cp.cuda.runtime.getDeviceCount() > 0:
                # Force CUDA initialization so driver/runtime problems are
                # detected here rather than during clustering.
                cp.cuda.Device().compute_capability

                return rsc.pp.neighbors, rsc.tl.leiden, "rapids-singlecell"

        except Exception as exc:
            warnings.warn(
                "GPU clustering is unavailable; falling back to Scanpy. "
                f"Reason: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return sc.pp.neighbors, sc.tl.leiden, "scanpy"


def leiden_cluster_k(
    adata,
    n_clusts,
    latent_key,
    low,
    high,
    n_neighbors=15,
    metric="euclidean",
    depth=0,
    maxDepth=6,
    prefer_gpu=True,
    random_state=0,
    gpu_algorithm="brute",
):
    """
    Find a Leiden resolution producing approximately `n_clusts` clusters.

    Uses rapids-singlecell when a working CUDA GPU is available and otherwise
    falls back to Scanpy.

    Notes
    -----
    If exactly `n_clusts` clusters are not found within `maxDepth`, the result
    from the final iteration is returned.
    """
    neighbors_fn, leiden_fn, backend = _get_clustering_backend(prefer_gpu)

    neighbors_key = f"{latent_key}_neighbors"
    n_pcs = adata.obsm[latent_key].shape[1]

    if neighbors_key not in adata.uns:
        neighbors_kwargs = dict(
            n_neighbors=n_neighbors,
            use_rep=latent_key,
            key_added=neighbors_key,
            metric=metric,
            n_pcs=n_pcs,
        )

        if backend == "rapids-singlecell":
            neighbors_kwargs["algorithm"] = gpu_algorithm

        neighbors_fn(adata, **neighbors_kwargs)

    # Preserve support for calls that begin partway through the search.
    final_clust_key = None

    for current_depth in range(depth, maxDepth + 1):
        midpoint = (high + low) / 2
        clust_key = f"leiden_{midpoint}_{latent_key}"
        final_clust_key = clust_key

        if clust_key not in adata.obs:
            leiden_kwargs = dict(
                neighbors_key=neighbors_key,
                resolution=midpoint,
                key_added=clust_key,
                n_iterations=3,
            )

            if backend == "rapids-singlecell":
                # rapids-singlecell uses `rng`; `flavor` is not applicable.
                leiden_kwargs["n_iterations"] = 100
                leiden_kwargs["random_state"] = random_state
            else:
                leiden_kwargs.update(
                    random_state=random_state,
                    flavor="igraph",
                )

            leiden_fn(adata, **leiden_kwargs)

        # More robust than assuming the result is categorical.
        num_found_clusts = adata.obs[clust_key].nunique()

        if num_found_clusts == n_clusts:
            break
        elif num_found_clusts < n_clusts:
            low = midpoint
        else:
            high = midpoint

    output_key = f"leiden_{n_clusts}_{latent_key}"
    adata.obs[output_key] = adata.obs[final_clust_key].copy()

    # Record useful provenance.
    adata.uns[f"{output_key}_params"] = {
        "backend": backend,
        "resolution": midpoint,
        "n_clusters_found": int(num_found_clusts),
        "n_clusters_requested": int(n_clusts),
        "metric": metric,
        "n_neighbors": int(n_neighbors),
    }

    return output_key


def _get_singlecell_backend(prefer_gpu=True):
    """
    Select rapids-singlecell when its CUDA stack is usable; otherwise Scanpy.
    """
    if prefer_gpu:
        try:
            import cupy as cp
            import rapids_singlecell as rsc

            if cp.cuda.runtime.getDeviceCount() > 0:
                # Select and initialize the current CUDA device.
                cp.cuda.Device().use()
                return rsc, "rapids-singlecell"

        except Exception as exc:
            warnings.warn(
                "rapids-singlecell is unavailable; falling back to Scanpy. "
                f"Reason: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return sc, "scanpy"


def umap_latent(
    adata,
    latent_key,
    n_neighbors=15,
    metric="euclidean",
    min_dist=0.5,
    spread=1.0,
    n_components=2,
    random_state=0,
    maxiter=None,
    init_pos="auto",
    prefer_gpu=True,
    gpu_algorithm="brute",
    neighbors_key=None,
    umap_key=None,
    recompute_neighbors=False,
    recompute_umap=False,
):
    """
    Compute a UMAP embedding for a representation in `adata.obsm`.

    rapids-singlecell is used when a working CUDA GPU and RAPIDS installation
    are available. Otherwise, the function falls back to Scanpy.

    Parameters
    ----------
    adata
        AnnData object.
    latent_key
        Key in `adata.obsm` containing the latent representation.
    n_neighbors
        Number of neighbors used to construct the graph.
    metric
        Distance metric used for nearest-neighbor search.
    min_dist
        UMAP minimum distance.
    spread
        UMAP embedding spread.
    n_components
        Number of UMAP dimensions.
    random_state
        Random seed.
    maxiter
        Maximum UMAP optimization iterations. If None, use backend default.
    init_pos
        UMAP initialization. Common choices are "auto", "spectral", and
        "random". Scanpy does not recognize "auto", so it is converted to
        "spectral" for the CPU backend.
    prefer_gpu
        Prefer rapids-singlecell when available.
    gpu_algorithm
        GPU nearest-neighbor algorithm, such as "brute" or "cagra".
    neighbors_key
        Key for the neighbor graph. Defaults to
        `f"{latent_key}_neighbors"`.
    umap_key
        Key under which to store the embedding in `adata.obsm`. Defaults to
        `f"{latent_key}_umap"`.
    recompute_neighbors
        Recompute the graph even when `neighbors_key` already exists.
    recompute_umap
        Recompute UMAP even when `umap_key` already exists.

    Returns
    -------
    str
        The key containing the UMAP coordinates in `adata.obsm`.
    """
    if latent_key not in adata.obsm:
        raise KeyError(
            f"{latent_key!r} was not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    if neighbors_key is None:
        neighbors_key = f"{latent_key}_neighbors"

    if umap_key is None:
        umap_key = f"{latent_key}_umap"

    backend_module, backend_name = _get_singlecell_backend(prefer_gpu)

    # Construct or reconstruct the latent-specific neighbor graph.
    if recompute_neighbors or neighbors_key not in adata.uns:
        neighbors_kwargs = dict(
            n_neighbors=n_neighbors,
            use_rep=latent_key,
            key_added=neighbors_key,
            metric=metric,
        )

        if backend_name == "rapids-singlecell":
            neighbors_kwargs["algorithm"] = gpu_algorithm
            neighbors_kwargs["random_state"] = random_state
            backend_module.pp.neighbors(adata, **neighbors_kwargs)
        else:
            neighbors_kwargs["random_state"] = random_state
            backend_module.pp.neighbors(adata, **neighbors_kwargs)

    # Compute the embedding unless a cached result should be reused.
    if recompute_umap or umap_key not in adata.obsm:
        common_umap_kwargs = dict(
            neighbors_key=neighbors_key,
            min_dist=min_dist,
            spread=spread,
            n_components=n_components,
            maxiter=maxiter,
            key_added=umap_key,
        )

        if backend_name == "rapids-singlecell":
            backend_module.tl.umap(
                adata,
                init_pos=init_pos,
                random_state=random_state,
                **common_umap_kwargs,
            )
        else:
            # "auto" is specific to rapids-singlecell.
            scanpy_init_pos = (
                "spectral" if init_pos == "auto" else init_pos
            )

            backend_module.tl.umap(
                adata,
                init_pos=scanpy_init_pos,
                random_state=random_state,
                **common_umap_kwargs,
            )

    # Add backend provenance without replacing parameters written by UMAP.
    adata.uns.setdefault(umap_key, {})
    adata.uns[umap_key].setdefault("params", {})
    adata.uns[umap_key]["params"].update(
        {
            "backend": backend_name,
            "latent_key": latent_key,
            "neighbors_key": neighbors_key,
        }
    )

    return umap_key

def bb_leiden_cluster_k(adata, n_clusts, latent_key, low, high, batch_key, neighbors_within_batch=3, metric='euclidean', depth=0, maxDepth=20):
    if not latent_key + '_bb_neighbors' in adata.uns.keys():
        if metric != 'cosine':
            # adata.obsm[latent_key + '_scaled'] = maxAbsScale(adata.obsm[latent_key])
            sc.external.pp.bbknn(
                adata, 
                batch_key=batch_key, 
                neighbors_within_batch=neighbors_within_batch, 
                use_rep=latent_key, # + '_scaled', 
                key_added=latent_key + '_bb_neighbors', 
                metric=metric, 
                n_pcs=adata.obsm[latent_key].shape[1]
            )
        else:
            sc.external.pp.bbknn(
                adata, 
                batch_key=batch_key, 
                neighbors_within_batch=neighbors_within_batch, 
                use_rep=latent_key, 
                key_added=latent_key + '_bb_neighbors',
                metric='euclidean', 
                n_pcs=adata.obsm[latent_key].shape[1]
            )

    midpoint = (high + low) / 2
    clust_key = f'bb_leiden_{midpoint}_' + latent_key
    if not clust_key in adata.obs.columns:
        sc.tl.leiden(adata, neighbors_key=latent_key + '_bb_neighbors', resolution=midpoint, key_added=clust_key, flavor='igraph', n_iterations=3)
    num_found_clusts = len(adata.obs[clust_key].cat.categories)
    # print(f"{num_found_clusts} clusters found.")
    if num_found_clusts == n_clusts or depth > maxDepth:
        new_clust_key = f'bb_leiden_{n_clusts}_' + latent_key
        adata.obs[new_clust_key] = adata.obs[clust_key]
        return new_clust_key
    elif num_found_clusts < n_clusts:
        low = midpoint
        return bb_leiden_cluster_k(adata, n_clusts, latent_key, low, high, batch_key, neighbors_within_batch, metric, depth=depth+1, maxDepth=maxDepth)
    else:
        high = midpoint
        return bb_leiden_cluster_k(adata, n_clusts, latent_key, low, high, batch_key, neighbors_within_batch, metric, depth=depth+1, maxDepth=maxDepth)

def _iter_uid(adatas: AnnData | list[AnnData], slide_key: str | None = None, obs_key: str | None = None):
    """Iterate over all slides, and make sure `adata.obs[obs_key]` is categorical.

    Args:
        adatas: One or a list of AnnData object(s).
        slide_key: The key in `adata.obs` that contains the slide id.
        obs_key: The key in `adata.obs` that contains the domain id.

    Yields:
        One `AnnData` per slide.
    """
    if isinstance(adatas, AnnData):
        adatas = [adatas]

    if obs_key is not None:
        categories = set.union(*[set(adata.obs[obs_key].astype("category").cat.categories) for adata in adatas])
        for adata in adatas:
            adata.obs[obs_key] = adata.obs[obs_key].astype("category").cat.set_categories(categories)

    for adata in adatas:
        if slide_key is None:
            yield adata
            continue

        for slide_id in adata.obs[slide_key].unique():
            adata_yield = adata[adata.obs[slide_key] == slide_id]

            yield adata_yield