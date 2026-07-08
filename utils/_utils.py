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

def leiden_cluster_k(adata, n_clusts, latent_key, low, high, n_neighbors=15, metric='euclidean', depth=0, maxDepth=20):
    if not latent_key + '_neighbors' in adata.uns.keys():
        if metric != 'cosine':
            # adata.obsm[latent_key + '_scaled'] = maxAbsScale(adata.obsm[latent_key])
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=latent_key, key_added=latent_key + '_neighbors', metric=metric, n_pcs=adata.obsm[latent_key].shape[1])
        else:
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=latent_key, key_added=latent_key + '_neighbors', metric=metric, n_pcs=adata.obsm[latent_key].shape[1])

    midpoint = (high + low) / 2
    clust_key = f'leiden_{midpoint}_' + latent_key
    if not clust_key in adata.obs.columns:
        sc.tl.leiden(adata, neighbors_key=latent_key + '_neighbors', resolution=midpoint, key_added=clust_key, flavor='igraph', n_iterations=3)
    num_found_clusts = len(adata.obs[clust_key].cat.categories)
    # print(f"{num_found_clusts} clusters found.")
    if num_found_clusts == n_clusts or depth > maxDepth:
        new_clust_key = f'leiden_{n_clusts}_' + latent_key
        adata.obs[new_clust_key] = adata.obs[clust_key]
        return new_clust_key
    elif num_found_clusts < n_clusts:
        low = midpoint
        return leiden_cluster_k(adata, n_clusts, latent_key, low, high, n_neighbors, metric, depth=depth+1, maxDepth=maxDepth)
    else:
        high = midpoint
        return leiden_cluster_k(adata, n_clusts, latent_key, low, high, n_neighbors, metric, depth=depth+1, maxDepth=maxDepth)

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