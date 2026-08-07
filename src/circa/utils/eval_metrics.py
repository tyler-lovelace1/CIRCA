import logging

import numpy as np
from anndata import AnnData
from sklearn import metrics
import scanpy as sc
import scipy.sparse as sp
import pandas as pd
from scib_metrics import metrics as scib
from circa.utils._utils import _iter_uid, leiden_cluster_k, maxAbsScale, tqdm
from circa.utils._eval import _get_scib_neighbors_result, _prepare_adatas_for_evaluation

log = logging.getLogger(__name__)


def mean_correlation_coef(emb1, emb2):
    corr_mat = np.corrcoef(emb1.T, emb2.T)
    K = emb1.shape[1]
    return np.mean(np.diagonal(corr_mat, offset=K))

def evaluate_all_metrics(adata, representations, label_key, slide_key, levels, dist_metric='euclidean', spatial_key='spatial', n_neighbors=15):
    adata, adataList = _prepare_adatas_for_evaluation(adata, slide_key=slide_key, levels=levels, spatial_key=spatial_key, n_neighbors=n_neighbors)
    
    if isinstance(dist_metric, str):
        dist_metrics = len(representations) * [dist_metric]
    else:
        dist_metrics = dist_metric

    assert len(dist_metrics) == len(representations)

    metricsList = []
    idx = 0
    for latent_key, dist_metric in tqdm(zip(representations, dist_metrics), desc='Computing metrics...'):
        if len(adata.obs[slide_key].cat.categories) > 1:
            single = pd.DataFrame(
                {
                    'Representation' : latent_key,
                    ## Similarity to celltype labels
                    'ARI' : max_adjusted_rand_score(adata, latent_key=latent_key, levels=levels, label_key=label_key),
                    'AMI' : max_adjusted_mutual_info_score(adata, latent_key=latent_key, levels=levels, label_key=label_key),
                    'CLISI' : clisi_score(adata, latent_key=latent_key, label_key=label_key, n_neighbors=n_neighbors),
                    ## Embedding coherence 
                    'NASW' : mean_silhouette_score(adata, latent_key=latent_key, levels=levels, metric=dist_metric),
                    ## Embedding continuity compared to spatial organization
                    'FIDE' : mean_mean_fide_score(adataList, latent_key=latent_key, levels=levels, slide_key=slide_key),
                    'MLAMI' : mean_mlami_score(adataList, latent_key=latent_key, levels=levels, spatial_key=spatial_key),
                    'GCS' : mean_gcs_score(adataList, latent_key=latent_key, n_neighbors=n_neighbors, spatial_key=spatial_key),
                    'CLISIS' : clisis_score(adata, latent_key=latent_key, label_key=label_key, slide_key=slide_key, spatial_key=spatial_key, n_neighbors=n_neighbors),
                    ## Batch correction across slides
                    'kBET' : kbet_score(adata, latent_key=latent_key, slide_key=slide_key, n_neighbors=n_neighbors), 
                    'ILISI' : ilisi_score(adata, latent_key=latent_key, slide_key=slide_key, n_neighbors=n_neighbors),
                    'JSD' : mean_normalized_jensen_shannon_divergence(adataList, latent_key=latent_key, levels=levels, slide_key=slide_key)
                },
                index = [latent_key]
            )
        else:
            single = pd.DataFrame(
                {
                    'Representation' : latent_key,
                    ## Similarity to celltype labels
                    'ARI' : max_adjusted_rand_score(adata, latent_key=latent_key, levels=levels, label_key=label_key),
                    'AMI' : max_adjusted_mutual_info_score(adata, latent_key=latent_key, levels=levels, label_key=label_key),
                    'CLISI' : clisi_score(adata, latent_key=latent_key, label_key=label_key, n_neighbors=n_neighbors),
                    ## Embedding coherence 
                    'NASW' : mean_silhouette_score(adata, latent_key=latent_key, levels=levels, metric=dist_metric),
                    ## Embedding continuity compared to spatial organization
                    'FIDE' : mean_mean_fide_score(adataList, latent_key=latent_key, levels=levels, slide_key=slide_key),
                    'MLAMI' : mean_mlami_score(adataList, latent_key=latent_key, levels=levels, spatial_key=spatial_key),
                    'GCS' : mean_gcs_score(adataList, latent_key=latent_key, n_neighbors=n_neighbors, spatial_key=spatial_key),
                    'CLISIS' : clisis_score(adata, latent_key=latent_key, label_key=label_key, slide_key=slide_key, spatial_key=spatial_key, n_neighbors=n_neighbors)
                },
                index = [latent_key]
            )
        
        metricsList.append(single)
        idx +=1

    return pd.concat(metricsList)

def evaluate_fast_metrics(adata, representations, label_key, slide_key, levels, dist_metric='euclidean', spatial_key='spatial', n_neighbors=15):
    adata, adataList = _prepare_adatas_for_evaluation(adata, slide_key=slide_key, levels=levels, spatial_key=spatial_key, n_neighbors=n_neighbors)
    
    if isinstance(dist_metric, str):
        dist_metrics = len(representations) * [dist_metric]
    else:
        dist_metrics = dist_metric

    assert len(dist_metrics) == len(representations)

    metricsList = []
    idx = 0
    for latent_key, dist_metric in tqdm(zip(representations, dist_metrics), desc='Computing metrics...'):
        if len(adata.obs[slide_key].cat.categories) > 1:
            single = pd.DataFrame(
                {
                    'Representation' : latent_key,
                    ## Similarity to celltype labels
                    'ARI' : max_adjusted_rand_score(adata, latent_key=latent_key, levels=levels, label_key=label_key),
                    'AMI' : max_adjusted_mutual_info_score(adata, latent_key=latent_key, levels=levels, label_key=label_key),
                    'CLISI' : clisi_score(adata, latent_key=latent_key, label_key=label_key, n_neighbors=n_neighbors),
                    ## Embedding continuity compared to spatial organization
                    'FIDE' : mean_mean_fide_score(adataList, latent_key=latent_key, levels=levels, slide_key=slide_key),
                    'MLAMI' : mean_mlami_score(adataList, latent_key=latent_key, levels=levels, spatial_key=spatial_key),
                    'GCS' : mean_gcs_score(adataList, latent_key=latent_key, n_neighbors=n_neighbors, spatial_key=spatial_key),
                    'CLISIS' : clisis_score(adata, latent_key=latent_key, label_key=label_key, slide_key=slide_key, spatial_key=spatial_key, n_neighbors=n_neighbors),
                    ## Batch correction across slides
                    'kBET' : kbet_score(adata, latent_key=latent_key, slide_key=slide_key, n_neighbors=n_neighbors), 
                    'ILISI' : ilisi_score(adata, latent_key=latent_key, slide_key=slide_key, n_neighbors=n_neighbors),
                    'JSD' : mean_normalized_jensen_shannon_divergence(adataList, latent_key=latent_key, levels=levels, slide_key=slide_key)
                },
                index = [latent_key]
            )
        else:
            single = pd.DataFrame(
                {
                    'Representation' : latent_key,
                    ## Similarity to celltype labels
                    'ARI' : max_adjusted_rand_score(adata, latent_key=latent_key, levels=levels, label_key=label_key),
                    'AMI' : max_adjusted_mutual_info_score(adata, latent_key=latent_key, levels=levels, label_key=label_key),
                    'CLISI' : clisi_score(adata, latent_key=latent_key, label_key=label_key, n_neighbors=n_neighbors),
                    ## Embedding continuity compared to spatial organization
                    'FIDE' : mean_mean_fide_score(adataList, latent_key=latent_key, levels=levels, slide_key=slide_key),
                    'MLAMI' : mean_mlami_score(adataList, latent_key=latent_key, levels=levels, spatial_key=spatial_key),
                    'GCS' : mean_gcs_score(adataList, latent_key=latent_key, n_neighbors=n_neighbors, spatial_key=spatial_key),
                    'CLISIS' : clisis_score(adata, latent_key=latent_key, label_key=label_key, slide_key=slide_key, spatial_key=spatial_key, n_neighbors=n_neighbors)
                },
                index = [latent_key]
            )
        
        metricsList.append(single)
        idx +=1

    return pd.concat(metricsList)

def kbet_score(adata, latent_key, slide_key, n_neighbors=15):
    batches = adata.obs[slide_key].to_numpy()
    neighbors = _get_scib_neighbors_result(adata, latent_key, n_neighbors)
    score, _, _ = scib.kbet(neighbors, batches)
    return score

def ilisi_score(adata, latent_key, slide_key, n_neighbors=15):
    batches = adata.obs[slide_key].to_numpy()
    neighbors = _get_scib_neighbors_result(adata, latent_key, n_neighbors)
    score = scib.ilisi_knn(neighbors, batches)
    return score

def clisi_score(adata, latent_key, label_key, n_neighbors=15):
    labels = adata.obs[label_key].to_numpy()
    neighbors = _get_scib_neighbors_result(adata, latent_key, n_neighbors)
    score = scib.clisi_knn(neighbors, labels)
    return score

def clisis_score(adata, latent_key, label_key, slide_key, spatial_key='spatial', n_neighbors=15):
    if not spatial_key + '_distances' in adata.obsp.keys():
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors, coord_type='generic', library_key=slide_key)

    labels = adata.obs[label_key].to_numpy()
    
    latent_neighbors = _get_scib_neighbors_result(adata, latent_key, n_neighbors)
    spatial_neighbors = _get_scib_neighbors_result(adata, spatial_key, n_neighbors)
    
    latent_clisi_scores = scib.lisi_knn(latent_neighbors, labels)
    spatial_clisi_scores = scib.lisi_knn(spatial_neighbors, labels)

    log_rclisi_scores = np.log2(latent_clisi_scores) - np.log2(spatial_clisi_scores)

    n_cell_types =  adata.obs[label_key].nunique()
    max_log_rclisi = np.log2(n_cell_types / 1)
    
    norm_log_rclisi_scores = log_rclisi_scores / max_log_rclisi

    return 1 - np.nanmedian(np.abs(norm_log_rclisi_scores))

def gcs_score(adata, latent_key, n_neighbors=15, spatial_key='spatial'):

    # adata.obsm[spatial_key + '_scaled'] = maxAbsScale(adata.obsm[spatial_key]) ## scale data to avoid numerical issues in umap connectivity calculation while preserving knn architecture
    # sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=spatial_key + '_scaled', key_added=spatial_key + '_neighbors', metric='euclidean', n_pcs=adata.obsm[spatial_key].shape[1])

    # adata.obsm[latent_key + '_scaled'] = maxAbsScale(adata.obsm[latent_key]) ## scale data to avoid numerical issues in umap connectivity calculation while preserving knn architecture
    # sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=latent_key + '_scaled', key_added=latent_key + '_neighbors', metric='euclidean', n_pcs=adata.obsm[latent_key].shape[1])

    connectivities_diff = (adata.obsp[latent_key + '_neighbors_connectivities'] - adata.obsp[spatial_key + '_neighbors_connectivities'])
    gcd = sp.linalg.norm(connectivities_diff, ord="fro")
    gcs = 1 - (gcd ** 2 / (n_neighbors * 2 * connectivities_diff.shape[0]))
    
    return gcs

def mean_gcs_score(
    adatas: AnnData | list[AnnData], latent_key: str, slide_key: str | None = None, n_neighbors: int = 15, spatial_key: str = 'spatial'
) -> float:
    """Mean GCS score over all slides. A low score indicates a great domain continuity.

    Args:
        adatas: An `AnnData` object, or a list of `AnnData` objects.
        obs_key: Key of `adata.obs` containing the domains annotation.
        slide_key: Optional key of `adata.obs` containing the ID of each slide. Not needed if each `adata` is a slide.
        n_classes: Optional number of classes. This can be useful if not all classes are predicted, for a fair comparision.

    Returns:
        The GCS score averaged for all slides.
    """
    return float(
        np.mean([
            gcs_score(adata, latent_key, n_neighbors, spatial_key)
            for adata in _iter_uid(adatas, slide_key=slide_key)
        ])
    )


def mlami_score(adata, latent_key, levels, spatial_key='spatial'):
    mlami = 0
    for k1 in levels:
        # new_clust_key = leiden_cluster_k(adata, k1, spatial_key, 0, 1, adata.uns[latent_key + '_neighbors']['params']['n_neighbors'], metric='euclidean')
        for k2 in levels:
            ami = metrics.adjusted_mutual_info_score(adata.obs[f'leiden_{k1}_{spatial_key}'], adata.obs[f'leiden_{k2}_{latent_key}'])
            if ami > mlami:
                mlami = ami
                best_k1 = k1
                best_k2 = k2
    return mlami

def mean_mlami_score(
    adatas: AnnData | list[AnnData], latent_key: str, levels: list[int], slide_key: str | None = None, spatial_key: str = 'spatial'
) -> float:
    """Mean MLAMI score over all slides. A low score indicates a great domain continuity.

    Args:
        adatas: An `AnnData` object, or a list of `AnnData` objects.
        obs_key: Key of `adata.obs` containing the domains annotation.
        slide_key: Optional key of `adata.obs` containing the ID of each slide. Not needed if each `adata` is a slide.
        n_classes: Optional number of classes. This can be useful if not all classes are predicted, for a fair comparision.

    Returns:
        The MLAMI score averaged for all slides.
    """
    return float(
        np.mean([
            mlami_score(adata, latent_key, levels)
            for adata in _iter_uid(adatas, slide_key=slide_key)
        ])
    )

def max_adjusted_rand_score(adata, latent_key, levels, label_key):
    mari = 0
    for k in levels:
        ari = metrics.adjusted_rand_score(adata.obs[label_key], adata.obs[f'leiden_{k}_{latent_key}'])
        if ari > mari:
            mari = ari
            best_k = k
    return mari

def max_adjusted_mutual_info_score(adata, latent_key, levels, label_key):
    mami = 0
    for k in levels:
        ami = metrics.adjusted_mutual_info_score(adata.obs[label_key], adata.obs[f'leiden_{k}_{latent_key}'])
        if ami > mami:
            mami = ami
            best_k = k
    return mami

def mean_silhouette_score(adata, latent_key, levels, metric='euclidean'):
    scores = []
    for k in levels:
        scores.append(scib.silhouette_label(adata.obsm[latent_key], labels=adata.obs[f'leiden_{k}_{latent_key}'], metric=metric))
    return float(np.mean(scores))

def mean_mean_fide_score(
    adatas: AnnData | list[AnnData], latent_key: str, levels, slide_key: str | None = None
) -> float:
    """Mean FIDE score over all slides. A low score indicates a great domain continuity.

    Args:
        adatas: An `AnnData` object, or a list of `AnnData` objects.
        obs_key: Key of `adata.obs` containing the domains annotation.
        slide_key: Optional key of `adata.obs` containing the ID of each slide. Not needed if each `adata` is a slide.
        n_classes: Optional number of classes. This can be useful if not all classes are predicted, for a fair comparision.

    Returns:
        The FIDE score averaged for all slides.
    """
    scores = []
    for k in levels:
        clust_key = f'leiden_{k}_' + latent_key
        if isinstance(adatas, AnnData):
            n_classes = len(adatas.obs[clust_key].cat.categories)
        else:
            n_classes = len(adatas[0].obs[clust_key].cat.categories)
        scores.append(mean_fide_score(adatas, obs_key=clust_key, slide_key=slide_key, n_classes=n_classes))
        
    return float(np.mean(scores))

def mean_normalized_jensen_shannon_divergence(
    adatas: AnnData | list[AnnData], latent_key: str, levels, slide_key: str | None = None
) -> float:
    """Mean FIDE score over all slides. A low score indicates a great domain continuity.

    Args:
        adatas: An `AnnData` object, or a list of `AnnData` objects.
        obs_key: Key of `adata.obs` containing the domains annotation.
        slide_key: Optional key of `adata.obs` containing the ID of each slide. Not needed if each `adata` is a slide.
        n_classes: Optional number of classes. This can be useful if not all classes are predicted, for a fair comparision.

    Returns:
        The FIDE score averaged for all slides.
    """
    scores = []
    for k in levels:
        clust_key = f'leiden_{k}_' + latent_key
        scores.append(normalized_jensen_shannon_divergence(adatas, obs_key=clust_key, slide_key=slide_key))
        
    return float(np.mean(scores))

def mean_fide_score(
    adatas: AnnData | list[AnnData], obs_key: str, slide_key: str | None = None, n_classes: int | None = None
) -> float:
    """Mean FIDE score over all slides. A low score indicates a great domain continuity.

    Args:
        adatas: An `AnnData` object, or a list of `AnnData` objects.
        obs_key: Key of `adata.obs` containing the domains annotation.
        slide_key: Optional key of `adata.obs` containing the ID of each slide. Not needed if each `adata` is a slide.
        n_classes: Optional number of classes. This can be useful if not all classes are predicted, for a fair comparision.

    Returns:
        The FIDE score averaged for all slides.
    """
    return float(
        np.mean([
            fide_score(adata, obs_key, n_classes=n_classes)
            for adata in _iter_uid(adatas, slide_key=slide_key, obs_key=obs_key)
        ])
    )


def fide_score(adata: AnnData, obs_key: str, n_classes: int | None = None) -> float:
    """F1-score of intra-domain edges (FIDE). A high score indicates a great domain continuity.

    Note:
        The F1-score is computed for every class, then all F1-scores are averaged. If some classes
        are not predicted, the `n_classes` argument allows to pad with zeros before averaging the F1-scores.

    Args:
        adata: An `AnnData` object
        obs_key: Key of `adata.obs` containing the domains annotation.
        n_classes: Optional number of classes. This can be useful if not all classes are predicted, for a fair comparision.

    Returns:
        The FIDE score.
    """
    i_left, i_right = adata.obsp["spatial_distances"].nonzero()
    classes_left, classes_right = adata.obs.iloc[i_left][obs_key].values, adata.obs.iloc[i_right][obs_key].values

    where_valid = ~classes_left.isna() & ~classes_right.isna()
    classes_left, classes_right = classes_left[where_valid], classes_right[where_valid]

    f1_scores = metrics.f1_score(classes_left, classes_right, average=None)

    if n_classes is None:
        return float(f1_scores.mean())

    assert n_classes >= len(f1_scores), f"Expected {n_classes:=}, but found {len(f1_scores)}, which is greater"

    return float(np.pad(f1_scores, (0, n_classes - len(f1_scores))).mean())
    

def normalized_jensen_shannon_divergence(adatas: AnnData | list[AnnData], obs_key: str, slide_key: str | None = None) -> float:
    """Jensen-Shannon divergence (JSD) over all slides

    Args:
        adatas: One or a list of AnnData object(s)
        obs_key: Key of `adata.obs` containing the domains annotation.
        slide_key: Optional key of `adata.obs` containing the ID of each slide. Not needed if each `adata` is a slide.

    Returns:
        The Jensen-Shannon divergence score for all slides
    """
    all_categories = set()
    for adata in _iter_uid(adatas, slide_key=slide_key, obs_key=obs_key):
        all_categories.update(adata.obs[obs_key].cat.categories)
    all_categories = sorted(all_categories)

    distributions = []
    for adata in _iter_uid(adatas, slide_key=slide_key, obs_key=obs_key):
        value_counts = adata.obs[obs_key].value_counts(sort=False)
        distribution = np.zeros(len(all_categories))

        for i, category in enumerate(all_categories):
            if category in value_counts:
                distribution[i] = value_counts[category]

        distributions.append(distribution)

    return float(_jensen_shannon_divergence(np.array(distributions)) / np.log2(len(distributions)))


def jensen_shannon_divergence(adatas: AnnData | list[AnnData], obs_key: str, slide_key: str | None = None) -> float:
    """Jensen-Shannon divergence (JSD) over all slides

    Args:
        adatas: One or a list of AnnData object(s)
        obs_key: Key of `adata.obs` containing the domains annotation.
        slide_key: Optional key of `adata.obs` containing the ID of each slide. Not needed if each `adata` is a slide.

    Returns:
        The Jensen-Shannon divergence score for all slides
    """
    all_categories = set()
    for adata in _iter_uid(adatas, slide_key=slide_key, obs_key=obs_key):
        all_categories.update(adata.obs[obs_key].cat.categories)
    all_categories = sorted(all_categories)

    distributions = []
    for adata in _iter_uid(adatas, slide_key=slide_key, obs_key=obs_key):
        value_counts = adata.obs[obs_key].value_counts(sort=False)
        distribution = np.zeros(len(all_categories))

        for i, category in enumerate(all_categories):
            if category in value_counts:
                distribution[i] = value_counts[category]

        distributions.append(distribution)

    return _jensen_shannon_divergence(np.array(distributions))


def _jensen_shannon_divergence(distributions: np.ndarray) -> float:
    """Compute the Jensen-Shannon divergence (JSD) for a multiple probability distributions.

    The lower the score, the better distribution of clusters among the different batches.

    Args:
        distributions: An array of shape (B, C), where B is the number of batches, and C is the number of clusters. For each batch, it contains the percentage of each cluster among cells.

    Returns:
        A float corresponding to the JSD
    """
    distributions = distributions / distributions.sum(1)[:, None]
    mean_distribution = np.mean(distributions, 0)

    return float(entropy(mean_distribution) - np.mean([entropy(dist) for dist in distributions])) / np.log2(len(distributions))


def entropy(distribution: np.ndarray) -> float:
    """Shannon entropy

    Args:
        distribution: An array of probabilities (should sum to one)

    Returns:
        The Shannon entropy
    """
    return float(-(distribution * np.log2(distribution + 1e-8)).sum())


def mean_normalized_entropy(
    adatas: AnnData | list[AnnData], n_classes: int, obs_key: str, slide_key: str | None = None
) -> float:
    return float(
        np.mean([
            _mean_normalized_entropy(adata, obs_key, n_classes=n_classes)
            for adata in _iter_uid(adatas, slide_key=slide_key, obs_key=obs_key)
        ])
    )


def _mean_normalized_entropy(adata: AnnData, obs_key: str, n_classes: int) -> float:
    distribution = adata.obs[obs_key].value_counts(normalize=True).values
    distribution = np.pad(distribution, (0, n_classes - len(distribution)), mode="constant")
    entropy_ = entropy(distribution)

    return float(entropy_ / np.log2(n_classes))


def niche_heuristic(adata: AnnData | list[AnnData], obs_key: str, n_classes: int, slide_key: str | None = None) -> float:
    """Heuristic score to evaluate the quality of the clustering.

    Args:
        adata: An `AnnData` object
        obs_key: The key in `adata.obs` that contains the domains.
        n_classes: The number of classes.
        slide_key: The key in `adata.obs` that contains the slide id.

    Returns:
        The heuristic score.
    """
    return float(
        np.mean([
            _niche_heuristic(adata, obs_key, n_classes) for adata in _iter_uid(adata, slide_key=slide_key, obs_key=obs_key)
        ])
    )


def _niche_heuristic(adata: AnnData, obs_key: str, n_classes: int) -> float:
    fide_ = fide_score(adata, obs_key, n_classes=n_classes)

    distribution = adata.obs[obs_key].value_counts(normalize=True).values
    distribution = np.pad(distribution, (0, n_classes - len(distribution)), mode="constant")
    entropy_ = entropy(distribution)

    return float(fide_ * entropy_ / np.log2(n_classes))

def batch_heuristic(adata: AnnData | list[AnnData], obs_key: str, n_classes: int, slide_key: str | None = None) -> float:
    """Heuristic score to evaluate the quality of the batch correction.

    Args:
        adata: An `AnnData` object
        obs_key: The key in `adata.obs` that contains the domains.
        n_classes: The number of classes.
        slide_key: The key in `adata.obs` that contains the slide id.

    Returns:
        The heuristic score.
    """
    njsd_ = normalized_jensen_shannon_divergence(adata, obs_key, slide_key)
    mean_entropy_ = mean_normalized_entropy(adata, n_classes, obs_key, slide_key)
    
    return float((1.0 - njsd_) * mean_entropy_)


def combined_heuristic(adata: AnnData | list[AnnData], obs_key: str, n_classes: int, slide_key: str | None = None):
    niche_score_ = niche_heuristic(adata, obs_key, n_classes, slide_key)
    batch_score_ = batch_heuristic(adata, obs_key, n_classes, slide_key)
    return float(niche_score_ * batch_score_)

