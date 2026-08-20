from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy
import math
import scanpy as sc
import squidpy as sq
from anndata import AnnData
from scib_metrics.nearest_neighbors import NeighborsResults
from circa.utils._utils import tqdm, _iter_uid, maxAbsScale, leiden_cluster_k

def _convert_distances(matrix, n_neighbors):
    # matrix = matrix / np.median(matrix.data)
    indices = []
    dists = []
    for i in range(matrix.shape[0]):
        row_start = matrix.indptr[i]
        row_end = matrix.indptr[i+1]
    
        # Row specific data/indices
        row_dists = matrix.data[row_start:row_end]
        row_indices = matrix.indices[row_start:row_end]

        order = np.argsort(row_dists)

        found_neighbors = len(order[:n_neighbors])
        
        dists.append(np.pad(row_dists[order[:n_neighbors]], (n_neighbors-found_neighbors, 0), mode='edge'))
        indices.append(np.pad(row_indices[order[:n_neighbors]], (n_neighbors-found_neighbors, 0), mode='edge'))

    return np.vstack(indices), np.vstack(dists)

def _prepare_adatas_for_evaluation(adata, slide_key, levels, spatial_key='spatial', n_neighbors=15):

    sq.gr.spatial_neighbors(adata, spatial_key=spatial_key, coord_type='generic', library_key=slide_key, n_neighs=n_neighbors)
    
    adataList = []
    for data in tqdm(_iter_uid(adata, slide_key=slide_key), desc=f'Splitting by {slide_key}...'):
        # print(data)
        adataList.append(data.copy())

    for data in tqdm(adataList, desc='Creating spatial clusters...'):
        data.obsm[spatial_key + '_scaled'] = maxAbsScale(data.obsm[spatial_key])
        sc.pp.neighbors(data, n_neighbors=n_neighbors, use_rep=spatial_key + '_scaled', key_added=spatial_key + '_neighbors', metric='euclidean', n_pcs=data.obsm[spatial_key].shape[1])
        for k in levels:
            new_clust_key = leiden_cluster_k(data, k, spatial_key, 0, 1, n_neighbors, metric='euclidean')

    sq.gr.spatial_neighbors(adata, spatial_key=spatial_key, coord_type='generic', library_key=slide_key, n_neighs=n_neighbors)
    
    return adata, adataList

def _get_scib_neighbors_result(adata, latent_key, n_neighbors=15):
    if latent_key != 'spatial':
        diag_included = adata.obsp[f'{latent_key}_neighbors_distances'].indices[0]==adata.obsp[f'{latent_key}_neighbors_distances'].indptr[0]
        if diag_included:
            n_neighbors += 1
        indices, dists = _convert_distances(adata.obsp[f'{latent_key}_neighbors_distances'], n_neighbors)
    else:
        diag_included = adata.obsp[f'{latent_key}_distances'].indices[0]==adata.obsp[f'{latent_key}_distances'].indptr[0]
        if diag_included:
            n_neighbors += 1
        indices, dists = _convert_distances(adata.obsp[f'{latent_key}_distances'], n_neighbors)
    
    return NeighborsResults(indices=indices, distances=dists)

