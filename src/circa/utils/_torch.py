from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import importlib
import scipy
import math
from anndata import AnnData

def build_cell_cell_communication_masks(genes, lr_table):
    n_genes = len(genes)
    
    secreted_edgelist = {}
    secreted_ligands = []
    secreted_receptors = []
    secreted_complexes = []

    contact_edgelist = {}
    contact_ligands = []
    contact_receptors = []
    contact_complexes = []

    annot = lr_table['annotation']
    ligand = lr_table['ligand_symbol']
    receptor = lr_table['receptor_symbol']

    n_secreted_int = 0
    n_contact_int = 0
    
    for i in range(lr_table.shape[0]):
        lig = ligand.iloc[i]
        comp = receptor.iloc[i]
        rec_list = comp.split('&')
        if not ((lig in genes) and np.all(np.isin(rec_list, genes))):
            continue
        
        if annot.iloc[i] == 'Secreted Signaling':
            secreted_ligands.append(lig)
            secreted_complexes.append(comp)
            secreted_receptors.append(rec_list)
            
            n_secreted_int += 1
            
            if not lig in secreted_edgelist.keys():
                secreted_edgelist.update( { lig : [comp] } )
            else:
                secreted_edgelist[lig] += [comp]
                
        elif annot.iloc[i] == 'Cell-Cell Contact':
            contact_ligands.append(lig)
            contact_complexes.append(comp)            
            contact_receptors.append(rec_list)
            
            n_contact_int += 1
            
            if not lig in contact_edgelist.keys():
                contact_edgelist.update( { lig : [comp] } )
            else:
                contact_edgelist[lig] += [comp]
        else:
            raise RuntimeError("Unrecognized annotation" + annot.iloc[i])

    secreted_ligands = np.unique(secreted_ligands)
    secreted_receptors = np.unique(np.concatenate(secreted_receptors))
    secreted_complexes = np.unique(secreted_complexes)

    rows = np.zeros(len(secreted_ligands), dtype=int)
    cols = np.arange(len(secreted_ligands), dtype=int)
    data = np.ones(len(secreted_ligands), dtype=np.float32)
    for i, lig in enumerate(secreted_ligands):
        rows[i] = np.where(genes==lig)[0][0]

    sec_lig_mask = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n_genes, len(secreted_ligands)))

    rows = []
    cols = []
    data = []
    for i, comp in enumerate(secreted_complexes):
        rec_list = set(comp.split('&'))
        
        rows.append([ j for j, g in enumerate(genes) if g in rec_list ])
        cols.append(len(rec_list) * [i])
        data.append(np.ones(len(rec_list), dtype=np.float32) / len(rec_list))
        
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    sec_rec_mask = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n_genes, len(secreted_complexes)))

    rows = np.zeros(n_secreted_int, dtype=int)
    cols = np.zeros(n_secreted_int, dtype=int)
    data = np.ones(n_secreted_int, dtype=np.float32)
    idx = 0
    for lig, comp_list in secreted_edgelist.items():
        for comp in comp_list:
            rows[idx] = np.where(secreted_ligands==lig)[0][0]
            cols[idx] = np.where(secreted_complexes==comp)[0][0]
            idx += 1

    sec_lr_adj = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(secreted_ligands), len(secreted_complexes)))
    
    contact_ligands = np.unique(contact_ligands)
    contact_receptors = np.unique(np.concatenate(contact_receptors))
    contact_complexes = np.unique(contact_complexes)

    rows = np.zeros(len(contact_ligands), dtype=int)
    cols = np.arange(len(contact_ligands), dtype=int)
    data = np.ones(len(contact_ligands), dtype=np.float32)
    for i, lig in enumerate(contact_ligands):
        rows[i] = np.where(genes==lig)[0][0]

    cont_lig_mask = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n_genes, len(contact_ligands)))

    rows = []
    cols = []
    data = []
    for i, comp in enumerate(contact_complexes):
        rec_list = set(comp.split('&'))
        
        rows.append([ j for j, g in enumerate(genes) if g in rec_list])
        cols.append(len(rec_list) * [i])
        data.append(np.ones(len(rec_list), dtype=np.float32) / len(rec_list))
        
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    cont_rec_mask = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n_genes, len(contact_complexes)))

    rows = np.zeros(n_contact_int, dtype=int)
    cols = np.zeros(n_contact_int, dtype=int)
    data = np.ones(n_contact_int, dtype=np.float32)
    idx = 0
    for lig, comp_list in contact_edgelist.items():
        for comp in comp_list:
            rows[idx] = np.where(contact_ligands==lig)[0][0]
            cols[idx] = np.where(contact_complexes==comp)[0][0]
            idx += 1

    cont_lr_adj = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(contact_ligands), len(contact_complexes)))

    return { 
        "secreted" : {
            "ligands" : secreted_ligands, 
            "ligand_mask" : sec_lig_mask,
            "receptors" : secreted_complexes, 
            "receptor_mask" : sec_rec_mask,
            "lr_adj_mat" : sec_lr_adj
        }, 
        "contact" : {
            "ligands" : contact_ligands, 
            "ligand_mask" : cont_lig_mask,
            "receptors" : contact_complexes, 
            "receptor_mask" : cont_rec_mask,
            "lr_adj_mat" : cont_lr_adj
        }
    }

def smooth_abs(x, eps=1e-4):
    return (x**2 + eps).sqrt() - math.sqrt(eps)

def compute_lambda(current, start, stop, gamma=10.0):
    if current < start:
        return 0.0
    else:
        p = (current - start) / (stop - start)
        return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0

def compute_dir_lambda(current, start, stop):
    if current < start:
        return 0.0
    else:
        return 0.5 * (1 - math.cos(math.pi * (current - start) / (stop - start)))

def random_roll_perm(B: int, device=None) -> torch.Tensor:
    """
    Fast no-fixed-point permutation using a random cyclic shift.
    For B > 1, every element is shifted away from itself.
    """
    if B <= 1:
        raise ValueError("B must be greater than 1 for negative sampling.")

    idx = torch.arange(B, device=device)
    shift = torch.randint(1, B, (1,), device=device).item()
    return torch.roll(idx, shifts=shift, dims=0)

def generate_roll_negatives(
    hz: torch.Tensor,
    n_perms: int = 1,
) -> torch.Tensor:
    """
    Generate negative tuples using roll-based permutations.

    For each permutation round:
      1. Single-group corruptions:
         (g0_perm, g1,      g2,      ...)
         (g0,      g1_perm, g2,      ...)
         ...

      2. All-but-one corruptions:
         For each anchor group a, keep group a fixed and roll all other groups.
         (g0,      g1_perm, g2_perm, ...)
         (g0_perm, g1,      g2_perm, ...)
         ...

    Args:
        hz:
            Tensor of shape (B, G, D), where B is batch size,
            G is number of groups, D is latent dimension.
        n_perms:
            Number of repeated corruption rounds.

    Returns:
        hz_neg:
            Tensor of shape (n_perms * 2 * G * B, G, D)
    """
    B, G, D = hz.shape
    device = hz.device

    if B <= 1:
        raise ValueError("Batch size must be greater than 1.")

    n_neg_batches = n_perms * G

    base = torch.arange(B, device=device)
    source_idx = base.view(1, B, 1).expand(n_neg_batches, B, G).clone()

    n = 0
    for _ in range(n_perms):
        for anchor_g in range(G):
            for g in range(G):
                if g == anchor_g:
                    continue
                source_idx[n, :, g] = random_roll_perm(B, device=device)
            n += 1

    group_idx = torch.arange(G, device=device).view(1, 1, G)
    group_idx = group_idx.expand(n_neg_batches, B, G)

    hz_neg = hz[source_idx, group_idx].reshape(n_neg_batches * B, G, D)

    return hz_neg


def generate_roll_negatives_one_random_anchor_no_repeats(
    hz: torch.Tensor,
    n_anchors: int = 1,
) -> torch.Tensor:
    """
    Generate all-but-anchor roll negatives.

    For each selected anchor group:
      - keep the anchor group fixed
      - roll every non-anchor group
      - use distinct nonzero shifts for non-anchor groups
      - generate B - 1 negative batches, one for each base shift

    For G = 3, this gives B - 1 negative batches per selected anchor,
    with no true pairs and no repeated mismatched non-anchor pairings.

    Args:
        hz:
            Tensor of shape (B, G, D).
        n_anchors:
            Number of anchor groups to sample per batch.
            Use 1 for one random anchor per batch.
            Use G to include all anchors.

    Returns:
        hz_neg:
            Tensor of shape (n_anchors * (B - 1) * B, G, D).
    """
    B, G, D = hz.shape
    device = hz.device

    if B <= 2:
        raise ValueError("B must be greater than 2 to use distinct nonzero shifts.")
    if G < 2:
        raise ValueError("G must be at least 2.")
    if G - 1 > B - 1:
        raise ValueError(
            "Need at least G-1 distinct nonzero shifts. "
            "Increase B or reduce G."
        )

    n_anchors = min(n_anchors, G)

    # Choose anchor groups.
    anchor_groups = torch.randperm(G, device=device)[:n_anchors]

    # There are B - 1 possible nonzero cyclic shifts.
    base_shifts = torch.arange(1, B, device=device)  # (B-1,)

    n_neg_batches = n_anchors * (B - 1)

    base_idx = torch.arange(B, device=device)
    source_idx = base_idx.view(1, B, 1).expand(n_neg_batches, B, G).clone()

    batch_idx = 0

    for anchor_g in anchor_groups.tolist():
        non_anchor_groups = [g for g in range(G) if g != anchor_g]

        # For each negative batch, use a base shift and assign distinct shifts
        # to each non-anchor group. For G=3, these are s and s+offset.
        for s in base_shifts.tolist():
            used_shifts = []

            for offset, g in enumerate(non_anchor_groups):
                # Make distinct nonzero shifts.
                # This cycles through 1..B-1 without duplicates for a given row.
                shift = ((s - 1 + offset) % (B - 1)) + 1

                # Safety check, mostly useful for debugging.
                if shift in used_shifts:
                    raise RuntimeError("Repeated shift generated unexpectedly.")
                used_shifts.append(shift)

                source_idx[batch_idx, :, g] = torch.roll(
                    base_idx,
                    shifts=shift,
                    dims=0,
                )

            # anchor group remains source_idx = original base_idx
            batch_idx += 1

    group_idx = torch.arange(G, device=device).view(1, 1, G)
    group_idx = group_idx.expand(n_neg_batches, B, G)

    hz_neg = hz[source_idx, group_idx].reshape(n_neg_batches * B, G, D)

    return hz_neg

def random_derangement(B: int, device=None, max_tries: int = 20) -> torch.Tensor:
    """
    Generate a permutation with no fixed points.
    Falls back to a random cyclic shift if rejection sampling fails.
    """
    if B <= 1:
        raise ValueError("B must be greater than 1.")

    base = torch.arange(B, device=device)

    for _ in range(max_tries):
        perm = torch.randperm(B, device=device)
        if torch.all(perm != base):
            return perm

    # Guaranteed no fixed points.
    shift = torch.randint(1, B, (1,), device=device).item()
    return torch.roll(base, shifts=shift, dims=0)


def generate_perm_negatives(
    hz: torch.Tensor,
    n_neg_batches: int | None = None,
    ensure_non_anchor_mismatch: bool = True,
    max_tries: int = 5,
) -> torch.Tensor:
    """
    Generate all-but-anchor negatives using random permutations.

    One randomly selected anchor group is kept fixed. All other groups are
    independently permuted with derangements.

    For G = 3, this creates negatives like:
        anchor 0: (g0,      g1_perm, g2_perm)
        anchor 1: (g0_perm, g1,      g2_perm)
        anchor 2: (g0_perm, g1_perm, g2)

    Args:
        hz:
            Tensor of shape (B, G, D).
        n_neg_batches:
            Number of corrupted batches to generate. If None, uses B - 1.
        ensure_non_anchor_mismatch:
            For G=3, additionally enforces that the two non-anchor groups
            do not come from the same source index for any row.
        max_tries:
            Rejection attempts for constraints.

    Returns:
        hz_neg:
            Tensor of shape (n_neg_batches * B, G, D).
    """
    B, G, D = hz.shape
    device = hz.device

    if B <= 2:
        raise ValueError("B must be greater than 2 for this negative sampler.")

    if n_neg_batches is None:
        n_neg_batches = B - 1

    anchor_g = torch.randint(0, G, (1,), device=device).item()
    non_anchor_groups = [g for g in range(G) if g != anchor_g]

    base = torch.arange(B, device=device)

    # source_idx[n, i, g] gives source sample index for group g
    # in negative batch n, row i.
    source_idx = base.view(1, B, 1).expand(n_neg_batches, B, G).clone()

    for n in range(n_neg_batches):
        perms = []

        for _g in non_anchor_groups:
            perms.append(random_derangement(B, device=device))

        # For G=3, prevent the two permuted non-anchor groups from being
        # paired with the same source sample in any row.
        if ensure_non_anchor_mismatch and len(perms) == 2:
            p0, p1 = perms

            tries = 0
            while torch.any(p0 == p1) and tries < max_tries:
                p1 = random_derangement(B, device=device)
                tries += 1

            # Fallback: construct p1 as a cyclic shift of p0.
            # This guarantees p1 != p0 elementwise and remains a derangement
            # relative to the anchor if the shift is nonzero.
            if torch.any(p0 == p1):
                shift = torch.randint(1, B, (1,), device=device).item()
                p1 = torch.roll(p0, shifts=shift, dims=0)

                # Very unlikely edge case, but keep it safe.
                if torch.any(p1 == base):
                    p1 = random_derangement(B, device=device)

            perms = [p0, p1]

        for g, perm in zip(non_anchor_groups, perms):
            source_idx[n, :, g] = perm

    group_idx = torch.arange(G, device=device).view(1, 1, G)
    group_idx = group_idx.expand(n_neg_batches, B, G)

    hz_neg = hz[source_idx, group_idx].reshape(n_neg_batches * B, G, D)

    return hz_neg

def generate_perm_negatives_index(
    hz: torch.Tensor,
    n_neg_batches: int | None = None,
    ensure_non_anchor_mismatch: bool = True,
    max_tries: int = 5,
) -> (torch.Tensor, torch.Tensor):
    """
    Generate all-but-anchor negatives using random permutations.

    One randomly selected anchor group is kept fixed. All other groups are
    independently permuted with derangements.

    For G = 3, this creates negatives like:
        anchor 0: (g0,      g1_perm, g2_perm)
        anchor 1: (g0_perm, g1,      g2_perm)
        anchor 2: (g0_perm, g1_perm, g2)

    Args:
        hz:
            Tensor of shape (B, G, D).
        n_neg_batches:
            Number of corrupted batches to generate. If None, uses B - 1.
        ensure_non_anchor_mismatch:
            For G=3, additionally enforces that the two non-anchor groups
            do not come from the same source index for any row.
        max_tries:
            Rejection attempts for constraints.

    Returns:
        hz_neg:
            Tensor of shape (n_neg_batches * B, G, D).
    """
    B, G, D = hz.shape
    device = hz.device

    if B <= 2:
        raise ValueError("B must be greater than 2 for this negative sampler.")

    if n_neg_batches is None:
        n_neg_batches = B - 1

    anchor_g = torch.randint(0, G, (1,), device=device).item()
    non_anchor_groups = [g for g in range(G) if g != anchor_g]

    base = torch.arange(B, device=device)

    # source_idx[n, i, g] gives source sample index for group g
    # in negative batch n, row i.
    source_idx = base.view(1, B, 1).expand(n_neg_batches, B, G).clone()

    for n in range(n_neg_batches):
        perms = []

        for _g in non_anchor_groups:
            perms.append(random_derangement(B, device=device))

        # For G=3, prevent the two permuted non-anchor groups from being
        # paired with the same source sample in any row.
        if ensure_non_anchor_mismatch and len(perms) == 2:
            p0, p1 = perms

            tries = 0
            while torch.any(p0 == p1) and tries < max_tries:
                p1 = random_derangement(B, device=device)
                tries += 1

            # Fallback: construct p1 as a cyclic shift of p0.
            # This guarantees p1 != p0 elementwise and remains a derangement
            # relative to the anchor if the shift is nonzero.
            if torch.any(p0 == p1):
                shift = torch.randint(1, B, (1,), device=device).item()
                p1 = torch.roll(p0, shifts=shift, dims=0)

                # Very unlikely edge case, but keep it safe.
                if torch.any(p1 == base):
                    p1 = random_derangement(B, device=device)

            perms = [p0, p1]

        for g, perm in zip(non_anchor_groups, perms):
            source_idx[n, :, g] = perm

    group_idx = torch.arange(G, device=device).view(1, 1, G)
    group_idx = group_idx.expand(n_neg_batches, B, G)

    # hz_neg = hz[source_idx, group_idx].reshape(n_neg_batches * B, G, D)

    return source_idx, group_idx


def random_derangement_stratified(
    slide_id: torch.Tensor,
    *,
    device=None,
    max_tries: int = 20,
) -> torch.Tensor:
    """
    Generate a derangement stratified by slide_id.

    For every row i:
        slide_id[perm[i]] == slide_id[i]
        perm[i] != i

    Args:
        slide_id:
            Integer tensor of shape [B].
        device:
            Optional device.
        max_tries:
            Rejection attempts per slide before cyclic-shift fallback.

    Returns:
        perm:
            Long tensor of shape [B].
    """
    if device is None:
        device = slide_id.device

    slide_id = torch.as_tensor(slide_id, device=device)
    B = slide_id.numel()
    base = torch.arange(B, device=device)

    perm = torch.empty(B, dtype=torch.long, device=device)

    for s in torch.unique(slide_id):
        idx = torch.where(slide_id == s)[0]
        m = idx.numel()

        if m <= 1:
            raise ValueError(
                f"Slide {int(s.item())} has only {m} sample(s); "
                "cannot generate a within-slide derangement."
            )

        # Try random within-slide derangements.
        found = False
        for _ in range(max_tries):
            cand = idx[torch.randperm(m, device=device)]
            if torch.all(cand != idx):
                perm[idx] = cand
                found = True
                break

        if found:
            continue

        # Guaranteed within-slide derangement by cyclic shift.
        shift = torch.randint(1, m, (1,), device=device).item()
        perm[idx] = torch.roll(idx, shifts=shift, dims=0)

    return perm


def roll_stratified(
    values: torch.Tensor,
    slide_id: torch.Tensor,
    *,
    shift: int | None = None,
) -> torch.Tensor:
    """
    Roll a vector independently within each slide.

    Useful as a fallback when values is already a within-slide permutation
    and you need another vector that differs elementwise from it.
    """
    device = values.device
    slide_id = torch.as_tensor(slide_id, device=device)

    out = torch.empty_like(values)

    for s in torch.unique(slide_id):
        idx = torch.where(slide_id == s)[0]
        m = idx.numel()

        if m <= 1:
            raise ValueError(
                f"Slide {int(s.item())} has only {m} sample(s); "
                "cannot roll within slide."
            )

        if shift is None:
            local_shift = torch.randint(1, m, (1,), device=device).item()
        else:
            local_shift = shift % m
            if local_shift == 0:
                local_shift = 1

        out[idx] = torch.roll(values[idx], shifts=local_shift, dims=0)

    return out


def generate_stratified_perm_negatives_index(
    hz: torch.Tensor,
    slide_id: torch.Tensor | None = None,
    n_neg_batches: int | None = None,
    ensure_non_anchor_mismatch: bool = True,
    max_tries: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate all-but-anchor negative source indices using random permutations.

    One randomly selected anchor group is kept fixed. All other groups are
    independently permuted with derangements.

    If slide_id is provided, permutations are stratified by slide:
        slide_id[source_idx[n, i, g]] == slide_id[i]

    For G = 3, this creates negatives like:
        anchor 0: (g0,      g1_perm, g2_perm)
        anchor 1: (g0_perm, g1,      g2_perm)
        anchor 2: (g0_perm, g1_perm, g2)

    Args:
        hz:
            Tensor of shape [B, G, D].
        slide_id:
            Optional integer tensor of shape [B]. If provided, all permutations
            are generated within each slide.
        n_neg_batches:
            Number of corrupted batches to generate. If None, uses B - 1.
        ensure_non_anchor_mismatch:
            For G=3, additionally enforces that the two non-anchor groups
            do not come from the same source index for any row.
        max_tries:
            Rejection attempts for constraints.

    Returns:
        source_idx:
            Long tensor of shape [n_neg_batches, B, G].
            source_idx[n, i, g] gives the source sample index for group g
            in negative batch n, row i.

        group_idx:
            Long tensor of shape [n_neg_batches, B, G].
    """
    if hz.ndim != 3:
        raise ValueError(f"hz must have shape [B, G, D], got {tuple(hz.shape)}.")

    B, G, D = hz.shape
    device = hz.device

    if B <= 2:
        raise ValueError("B must be greater than 2 for this negative sampler.")

    if n_neg_batches is None:
        n_neg_batches = B - 1

    if slide_id is not None:
        slide_id = torch.as_tensor(slide_id, dtype=torch.long, device=device)
        if slide_id.shape != (B,):
            raise ValueError(
                f"slide_id must have shape [{B}], got {tuple(slide_id.shape)}."
            )

        # Need at least 2 samples per slide for a derangement.
        unique_slides, counts = torch.unique(slide_id, return_counts=True)
        bad = unique_slides[counts <= 1]
        if bad.numel() > 0:
            raise ValueError(
                "Every slide must have at least 2 samples for within-slide "
                f"derangements. Bad slide ids: {bad.detach().cpu().tolist()}"
            )

        # If G=3 and ensure_non_anchor_mismatch=True, each row needs:
        # anchor source i, plus two non-anchor sources that differ from i and
        # differ from each other. That requires at least 3 samples per slide.
        if ensure_non_anchor_mismatch and G >= 3:
            bad = unique_slides[counts < 3]
            if bad.numel() > 0:
                raise ValueError(
                    "Every slide must have at least 3 samples when enforcing "
                    "non-anchor mismatch for G>=3. "
                    f"Bad slide ids: {bad.detach().cpu().tolist()}"
                )

    anchor_g = torch.randint(0, G, (1,), device=device).item()
    non_anchor_groups = [g for g in range(G) if g != anchor_g]

    base = torch.arange(B, device=device)

    # source_idx[n, i, g] gives source sample index for group g
    # in negative batch n, row i.
    source_idx = base.view(1, B, 1).expand(n_neg_batches, B, G).clone()

    for n in range(n_neg_batches):
        perms = []

        for _g in non_anchor_groups:
            if slide_id is None:
                perms.append(random_derangement(B, device=device))
            else:
                perms.append(
                    random_derangement_stratified(
                        slide_id,
                        device=device,
                        max_tries=max_tries,
                    )
                )

        # For G=3, prevent the two permuted non-anchor groups from being
        # paired with the same source sample in any row.
        if ensure_non_anchor_mismatch and len(perms) == 2:
            p0, p1 = perms

            tries = 0
            while torch.any(p0 == p1) and tries < max_tries:
                if slide_id is None:
                    p1 = random_derangement(B, device=device)
                else:
                    p1 = random_derangement_stratified(
                        slide_id,
                        device=device,
                        max_tries=max_tries,
                    )
                tries += 1

            if torch.any(p0 == p1):
                if slide_id is None:
                    # Global fallback.
                    shift = torch.randint(1, B, (1,), device=device).item()
                    p1 = torch.roll(p0, shifts=shift, dims=0)

                    # Very unlikely edge case, but keep it safe.
                    if torch.any(p1 == base):
                        p1 = random_derangement(B, device=device)
                else:
                    # Slide-stratified fallback.
                    p1 = roll_stratified(p0, slide_id)

                    # Keep it safe: p1 must differ from p0 and from base.
                    if torch.any(p1 == p0) or torch.any(p1 == base):
                        p1 = random_derangement_stratified(
                            slide_id,
                            device=device,
                            max_tries=max_tries,
                        )

            perms = [p0, p1]

        for g, perm in zip(non_anchor_groups, perms):
            source_idx[n, :, g] = perm

    group_idx = torch.arange(G, device=device).view(1, 1, G)
    group_idx = group_idx.expand(n_neg_batches, B, G)

    return source_idx, group_idx


def generate_block_roll_negatives_index(
    B: int,
    G: int,
    block_size: int,
    n_neg_batches: int = 30,
    *,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fast slide-block negative index generator.

    Assumes the batch is ordered as fixed-size slide blocks:
        [slide0 block, slide1 block, ..., slideS block]

    For each negative batch, slide block, and group, chooses distinct roll
    offsets within the block. This ensures that within each negative tuple,
    the G groups come from G distinct samples from the same slide block.

    Returns:
        source_idx: [n_neg_batches, B, G]
        group_idx:  [n_neg_batches, B, G]
    """
    if B % block_size != 0:
        raise ValueError("B must be divisible by block_size.")
    if block_size < G:
        raise ValueError("block_size must be at least G.")

    S = B // block_size
    M = block_size

    # base[s, m] is the batch row index for slide block s, local position m.
    base = torch.arange(B, device=device).view(S, M)

    # offsets[n, s, g]: group-specific roll offsets for negative batch n
    # and slide block s. argsort gives G distinct offsets per n,s.
    offsets = torch.rand(n_neg_batches, S, M, device=device).argsort(dim=-1)[..., :G]

    local_pos = torch.arange(M, device=device).view(1, 1, M, 1)
    src_pos = (local_pos + offsets[:, :, None, :]) % M  # [N, S, M, G]

    base_expand = base.view(1, S, M, 1).expand(n_neg_batches, S, M, G)

    source_idx = torch.gather(base_expand, dim=2, index=src_pos)
    source_idx = source_idx.reshape(n_neg_batches, B, G)

    group_idx = torch.arange(G, device=device).view(1, 1, G)
    group_idx = group_idx.expand(n_neg_batches, B, G)

    return source_idx, group_idx


def generate_roll_perm_negatives(
    hz: torch.Tensor,
    anchor_g: int | None = None,
    roll_g: int | None = None,
    ensure_no_pair_matches: bool = True,
) -> torch.Tensor:
    """
    Generate all-but-anchor negatives.

    One group is kept fixed as anchor.
    One non-anchor group is rolled through all B-1 possible cyclic shifts.
    All remaining non-anchor groups are independently deranged for each shift.

    This is most natural for G=3, but works for G >= 3.

    Args:
        hz:
            Tensor of shape (B, G, D).
        anchor_g:
            Group to keep fixed. If None, sampled randomly.
        roll_g:
            Non-anchor group to roll through all B-1 shifts. If None, sampled
            randomly from non-anchor groups.
        ensure_no_pair_matches:
            If True, tries to ensure permuted remaining groups do not equal the
            rolled group's source index for any row.

    Returns:
        hz_neg:
            Tensor of shape ((B - 1) * B, G, D).
    """
    B, G, D = hz.shape
    device = hz.device

    if B <= 2:
        raise ValueError("B must be greater than 2.")
    if G < 3:
        raise ValueError("This sampler is intended for G >= 3.")

    base = torch.arange(B, device=device)

    if anchor_g is None:
        anchor_g = torch.randint(0, G, (1,), device=device).item()

    non_anchor = [g for g in range(G) if g != anchor_g]

    if roll_g is None:
        idx = torch.randint(0, len(non_anchor), (1,), device=device).item()
        roll_g = non_anchor[idx]

    if roll_g == anchor_g:
        raise ValueError("roll_g must be different from anchor_g.")

    remaining_groups = [g for g in non_anchor if g != roll_g]

    n_neg_batches = B - 1

    # source_idx[n, i, g] gives the source sample for negative batch n,
    # anchor row i, group g.
    source_idx = base.view(1, B, 1).expand(n_neg_batches, B, G).clone()

    for n, shift in enumerate(range(1, B)):
        rolled_idx = torch.roll(base, shifts=shift, dims=0)
        source_idx[n, :, roll_g] = rolled_idx

        for g in remaining_groups:
            perm = random_derangement(B, device=device)

            if ensure_no_pair_matches:
                # Try to avoid remaining group source matching the rolled group source.
                tries = 0
                while torch.any(perm == rolled_idx) and tries < 20:
                    perm = random_derangement(B, device=device)
                    tries += 1

                # Fallback: roll the permutation itself to avoid equality with rolled_idx.
                if torch.any(perm == rolled_idx):
                    for extra_shift in range(1, B):
                        candidate = torch.roll(perm, shifts=extra_shift, dims=0)
                        if torch.all(candidate != base) and torch.all(candidate != rolled_idx):
                            perm = candidate
                            break

            source_idx[n, :, g] = perm

    group_idx = torch.arange(G, device=device).view(1, 1, G)
    group_idx = group_idx.expand(n_neg_batches, B, G)

    hz_neg = hz[source_idx, group_idx].reshape(n_neg_batches * B, G, D)
    return hz_neg


def scipy_sparse_to_torch(scipy_mat, dtype=torch.float32, device='cpu'):
    """
    Converts a SciPy sparse matrix to a PyTorch sparse COO tensor.

    Args:
        scipy_mat (scipy.sparse.spmatrix): A SciPy sparse matrix (CSR, COO, etc.).
        dtype (torch.dtype): Desired data type of tensor values (default: torch.float32).
        device (str or torch.device): Device to place the tensor on (default: 'cpu').

    Returns:
        torch.sparse_coo_tensor: Sparse tensor in PyTorch's COO format.
    """
    if not scipy.sparse.issparse(scipy_mat):
        return torch.tensor(scipy_mat, dtype=dtype, device=device)
        # raise TypeError("Input must be a SciPy sparse matrix.")

    coo = scipy_mat.tocoo()
    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.int64, device=device)
    values = torch.tensor(coo.data, dtype=dtype, device=device)
    shape = coo.shape

    return torch.sparse_coo_tensor(indices, values, shape, dtype=dtype, device=device)


def collect_predict_embeddings(
    preds: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]],
    *,
    n_obs: Optional[int] = None,
    adata: Optional[Any] = None,
    keys: Tuple[str, ...] = ("state_emb", "cell_emb", "niche_emb"),
    idx_key: str = "adata_idx",
    obsm_prefix: str = "spacarl_",
    fill_value: float = np.nan,
    dataloader_idx: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Combine Lightning trainer.predict outputs into full [N, D] embedding matrices.

    If `adata` is provided, adds embeddings to `adata.obsm[f"{obsm_prefix}{key}"]`
    and returns the same matrices as numpy arrays.

    Args:
        preds:
            Output of `trainer.predict(...)`.
            - If one dataloader: List[batch_out]
            - If multiple: List[dataloader_out], where each dataloader_out is List[batch_out]
        n_obs:
            Total number of observations N (needed if adata is None).
        adata:
            AnnData object (optional). If provided, N inferred from `adata.n_obs`.
        keys:
            Which embeddings to collect from each batch_out dict.
        idx_key:
            Name of the index field returned by predict_step (e.g. "adata_idx").
        obsm_prefix:
            Prefix for keys stored in adata.obsm.
        fill_value:
            Value used to initialize full matrices. Use np.nan to easily spot missing rows.
        dataloader_idx:
            If preds contains multiple dataloaders, pick which one to use.

    Returns:
        dict mapping each embedding key to a numpy array of shape [N, D].
    """
    # Select dataloader if needed
    if len(preds) > 0 and isinstance(preds[0], list):
        preds_list = preds[dataloader_idx]
    else:
        preds_list = preds  # type: ignore[assignment]

    if adata is not None:
        N = int(adata.n_obs)
    else:
        if n_obs is None:
            raise ValueError("Provide either `adata` or `n_obs`.")
        N = int(n_obs)

    # Concatenate indices and embeddings
    # (Assumes each batch_out is a dict of CPU tensors, as in typical predict_step)
    adata_idx = torch.cat([p[idx_key] for p in preds_list], dim=0).long()  # [M]

    out: Dict[str, np.ndarray] = {}
    for k in keys:
        emb = torch.cat([p[k] for p in preds_list], dim=0)  # [M, D]
        D = int(emb.shape[1])

        full = torch.full((N, D), float(fill_value), dtype=emb.dtype)
        full.index_copy_(0, adata_idx, emb)

        arr = full.cpu().numpy()
        out[k] = arr

        if adata is not None:
            adata.obsm[f"{obsm_prefix}{k}"] = arr

    return out
