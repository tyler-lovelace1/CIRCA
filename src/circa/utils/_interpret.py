from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from anndata import AnnData

from captum.attr import IntegratedGradients
from circa.model.CIRCA import CIRCA, CIRCA_Wrapper, Cell_Embedding_Wrapper, Niche_Embedding_Wrapper, ExtraGroup_Embedding_Wrapper
from circa.utils._torch import random_derangement_stratified

def compute_integrated_gradients(model, datamodule, baseline=0, n_neighbors_contact=5, n_steps=50, internal_chunk_size=6400, device='cuda', reference_indices=None):
    model = model.to(device)
    model_explainer = IntegratedGradients(CIRCA_Wrapper(model).to(device))
    cell_explainer = IntegratedGradients(Cell_Embedding_Wrapper(model).to(device))
    niche_explainer = IntegratedGradients(Niche_Embedding_Wrapper(model).to(device))

    total_extra_group_input_dims = 0
    
    if model.extra_group_latent_dims:
        extra_groups_explainers = {}
        for group_name in model.extra_group_names:
            total_extra_group_input_dims += model.extra_group_input_dims[group_name]
            extra_groups_explainers[group_name] = IntegratedGradients(ExtraGroup_Embedding_Wrapper(model, group_name))

    # 1. Configuration
    latent_dimension_size = model.latent_dim

    if reference_indices == None:
        reference_indices = torch.arange(datamodule.adata.shape[0])

    num_total_samples = len(reference_indices)

    datamodule.setup("predict")
    dataloader = datamodule.predict_dataloader()

    # Container to hold final metrics
    logit_attr = sparse.lil_array((num_total_samples, (model.n_neighbors + 1) * (model.input_dim + total_extra_group_input_dims)), dtype=np.float32)
    cell_attr = [ sparse.lil_array((num_total_samples, model.input_dim), dtype=np.float32) for _ in range(model.latent_dim) ]
    niche_attr = [ sparse.lil_array((num_total_samples, model.n_neighbors * model.input_dim), dtype=np.float32) for _ in range(model.latent_dim) ]

    if model.extra_group_latent_dims:
        extra_group_attr = {}
        for group_name in model.extra_group_names:
            extra_group_attr[group_name] = [ 
                sparse.lil_array((num_total_samples, (model.n_neighbors + 1) * model.extra_group_input_dims[group_name]), dtype=np.float32) for _ in range(model.extra_group_latent_dims[group_name]) 
            ]

    for batch in iter(dataloader):
        adata_idx, cell, neighbors, panel, slide_id, obs_feature_groups = model._unpack_batch(batch)

        B = len(adata_idx)
        
        cell_inputs = cell.to(device=device).log1p()
        niche_inputs = neighbors.to(device=device).log1p().reshape(B,-1)

        all_inputs = [cell_inputs, niche_inputs]

        if model.extra_group_latent_dims:
            extra_group_inputs = {}
            for group_name in model.extra_group_names:
                extra_group_inputs[group_name] = torch.cat([obs_feature_groups[group_name]['cell'], obs_feature_groups[group_name]['neighbors'].reshape(B,-1)], dim=1).to(device=device)
                all_inputs.append(extra_group_inputs[group_name])

        all_inputs = torch.cat(all_inputs, dim=1)

        # Compute IG for just this mini-batch
        batch_attributions = model_explainer.attribute(
            inputs=all_inputs,
            baselines=baseline,
            target=None,
            n_steps=n_steps,
            internal_batch_size=internal_chunk_size # Must be >= external_batch_size
        )
    
        logit_attr[adata_idx] += batch_attributions.cpu().numpy()
    
        for dim_idx in range(model.latent_dim):
            batch_cell_attributions = cell_explainer.attribute(
                inputs=cell_inputs,
                baselines=baseline,
                target=None,
                additional_forward_args=(dim_idx),
                n_steps=n_steps,
                internal_batch_size=internal_chunk_size # Must be >= external_batch_size
            )
    
            cell_attr[dim_idx][adata_idx] += batch_cell_attributions.cpu().numpy()
    
            batch_niche_attributions = niche_explainer.attribute(
                inputs=niche_inputs,
                baselines=baseline,
                target=None,
                additional_forward_args=(dim_idx),
                n_steps=n_steps,
                internal_batch_size=internal_chunk_size # Must be >= external_batch_size
            )
    
            niche_attr[dim_idx][adata_idx] += batch_niche_attributions.cpu().numpy()

            if model.extra_group_latent_dims:
                for group_name in model.extra_group_names:
                    if dim_idx < model.extra_group_latent_dims[group_name]:
                        batch_extra_group_attributions = extra_groups_explainers[group_name].attribute(
                            inputs=extra_group_inputs[group_name],
                            baselines=baseline,
                            target=None,
                            additional_forward_args=(dim_idx),
                            n_steps=n_steps,
                            internal_batch_size=internal_chunk_size # Must be >= external_batch_size
                        )
            
                        extra_group_attr[group_name][dim_idx][adata_idx] += batch_extra_group_attributions.cpu().numpy()

    if model.extra_group_latent_dims:
        return logit_attr, cell_attr, niche_attr, extra_group_attr
    else:
        return logit_attr, cell_attr, niche_attr, {}

def explain_logits(model, datamodule, n_perms=20, n_steps=50, internal_chunk_size=3200, device='cuda'):
    model_explainer = IntegratedGradients(CIRCA_Wrapper(model).to(device))
    
    total_extra_group_input_dims = 0
    
    if model.extra_group_latent_dims:
        extra_groups_explainers = {}
        for group_name in model.extra_group_names:
            total_extra_group_input_dims += model.extra_group_input_dims[group_name]
            
    # 1. Configuration
    latent_dimension_size = model.latent_dim
    
    # if reference_indices == None:
    #     reference_indices = torch.arange(datamodule.adata.shape[0])
    
    num_total_samples = datamodule.adata.shape[0]
    
    datamodule.setup("fit")
    
    # Container to hold final metrics
    logit_attr = torch.zeros((num_total_samples, (model.n_neighbors + 1) * (model.input_dim + total_extra_group_input_dims)), dtype=torch.float32, device=device)

    for perm in range(n_perms):
        
        dataloader = datamodule.train_dataloader()
        
        for idx, batch in enumerate(dataloader):
            adata_idx, cell, neighbors, panel, slide_id, obs_feature_groups = model._unpack_batch(batch)
        
            B = len(adata_idx)
            
            cell_inputs = cell.to(device=device).log1p()
            niche_inputs = neighbors.to(device=device).log1p().reshape(B,-1)
            slide_id = slide_id.to(device=device)
        
            all_inputs = [cell_inputs, niche_inputs]
        
            if model.extra_group_latent_dims:
                extra_group_inputs = {}
                for group_name in model.extra_group_names:
                    extra_group_inputs[group_name] = torch.cat([obs_feature_groups[group_name]['cell'], obs_feature_groups[group_name]['neighbors'].reshape(B,-1)], dim=1).to(device=device)
                    all_inputs.append(extra_group_inputs[group_name])
        
            all_inputs = torch.cat(all_inputs, dim=1)
        
            all_inputs_perm = [cell_inputs[random_derangement_stratified(slide_id)], niche_inputs[random_derangement_stratified(slide_id)]]
    
            if model.extra_group_latent_dims:
                extra_group_inputs = {}
                for group_name in model.extra_group_names:
                    extra_group_inputs[group_name] = torch.cat([obs_feature_groups[group_name]['cell'], obs_feature_groups[group_name]['neighbors'].reshape(B,-1)], dim=1).to(device=device)
                    all_inputs_perm.append(extra_group_inputs[group_name][random_derangement_stratified(slide_id)])
        
            all_inputs_perm = torch.cat(all_inputs_perm, dim=1)
    
            # Compute IG for just this mini-batch
            batch_attributions = model_explainer.attribute(
                inputs=all_inputs,
                baselines=all_inputs_perm,
                target=None,
                n_steps=n_steps,
                internal_batch_size=internal_chunk_size # Must be >= external_batch_size
            )
    
            logit_attr[adata_idx.to(device=device)] += batch_attributions / n_perms

    return logit_attr.cpu().numpy()


def explain_logits_block_permute(model, datamodule, n_perms=20, n_steps=50, internal_chunk_size=3200, device='cuda'):
    model_explainer = IntegratedGradients(CIRCA_Wrapper(model).to(device))
    
    total_extra_group_input_dims = 0

    block_bounds = [0, model.input_dim, (model.n_neighbors+1) * model.input_dim]
    
    if model.extra_group_latent_dims:
        extra_groups_explainers = {}
        for group_name in model.extra_group_names:
            total_extra_group_input_dims += model.extra_group_input_dims[group_name]
            block_bounds.append(block_bounds[-1] + (model.n_neighbors+1) * model.extra_group_input_dims[group_name])
            
    # 1. Configuration
    latent_dimension_size = model.latent_dim
    
    # if reference_indices == None:
    #     reference_indices = torch.arange(datamodule.adata.shape[0])
    
    num_total_samples = datamodule.adata.shape[0]
    
    datamodule.setup("fit")
    
    # Container to hold final metrics
    logit_attr = torch.zeros((num_total_samples, (model.n_neighbors + 1) * (model.input_dim + total_extra_group_input_dims)), dtype=torch.float32, device=device)

    for perm in range(n_perms):

        print(f"Permutation {perm}")
        
        dataloader = datamodule.train_dataloader()
        
        for idx, batch in enumerate(dataloader):
            adata_idx, cell, neighbors, panel, slide_id, obs_feature_groups = model._unpack_batch(batch)
        
            B = len(adata_idx)
            
            cell_inputs = cell.to(device=device).log1p()
            niche_inputs = neighbors.to(device=device).log1p().reshape(B,-1)
            slide_id = slide_id.to(device=device)
        
            all_inputs = [cell_inputs, niche_inputs]
        
            if model.extra_group_latent_dims:
                extra_group_inputs = {}
                for group_name in model.extra_group_names:
                    extra_group_inputs[group_name] = torch.cat([obs_feature_groups[group_name]['cell'], obs_feature_groups[group_name]['neighbors'].reshape(B,-1)], dim=1).to(device=device)
                    all_inputs.append(extra_group_inputs[group_name])
        
            all_inputs = torch.cat(all_inputs, dim=1)

            perm_inds = random_derangement_stratified(slide_id)
        
            all_inputs_perm = [cell_inputs[perm_inds], niche_inputs[perm_inds]]
    
            if model.extra_group_latent_dims:
                extra_group_inputs = {}
                for group_name in model.extra_group_names:
                    extra_group_inputs[group_name] = torch.cat([obs_feature_groups[group_name]['cell'], obs_feature_groups[group_name]['neighbors'].reshape(B,-1)], dim=1).to(device=device)
                    all_inputs_perm.append(extra_group_inputs[group_name][perm_inds])
        
            # all_inputs_perm = torch.cat(all_inputs_perm, dim=1)

            for bidx in range(len(block_bounds)-1):
                all_inputs_block_perm = all_inputs.clone()
                all_inputs_block_perm[:,block_bounds[bidx]:block_bounds[bidx+1]] = all_inputs_perm[bidx]
                # Compute IG for just this mini-batch
                batch_attributions = model_explainer.attribute(
                    inputs=all_inputs,
                    baselines=all_inputs_block_perm,
                    target=None,
                    n_steps=n_steps,
                    internal_batch_size=internal_chunk_size # Must be >= external_batch_size
                )
        
                logit_attr[adata_idx.to(device=device),block_bounds[bidx]:block_bounds[bidx+1]] += batch_attributions[:,block_bounds[bidx]:block_bounds[bidx+1]] / n_perms

    return logit_attr.cpu().numpy()

def explain_logits_baseline(model, datamodule, baseline=0, n_steps=50, internal_chunk_size=3200, device='cuda'):
    model_explainer = IntegratedGradients(CIRCA_Wrapper(model).to(device))
    
    total_extra_group_input_dims = 0
    
    if model.extra_group_latent_dims:
        extra_groups_explainers = {}
        for group_name in model.extra_group_names:
            total_extra_group_input_dims += model.extra_group_input_dims[group_name]
            
    # 1. Configuration
    latent_dimension_size = model.latent_dim
    
    # if reference_indices == None:
    #     reference_indices = torch.arange(datamodule.adata.shape[0])
    
    num_total_samples = datamodule.adata.shape[0]
    
    datamodule.setup("fit")

    if not isinstance(baseline, int):
        baseline = torch.as_tensor(baseline, dtype=torch.float32, device=device)
    
    # Container to hold final metrics
    logit_attr = torch.zeros((num_total_samples, (model.n_neighbors + 1) * (model.input_dim + total_extra_group_input_dims)), dtype=torch.float32, device=device)

    dataloader = datamodule.train_dataloader()
        
    for idx, batch in enumerate(dataloader):
        adata_idx, cell, neighbors, panel, slide_id, obs_feature_groups = model._unpack_batch(batch)
    
        B = len(adata_idx)
        
        cell_inputs = cell.to(device=device).log1p()
        niche_inputs = neighbors.to(device=device).log1p().reshape(B,-1)
        slide_id = slide_id.to(device=device)
    
        all_inputs = [cell_inputs, niche_inputs]
    
        if model.extra_group_latent_dims:
            extra_group_inputs = {}
            for group_name in model.extra_group_names:
                extra_group_inputs[group_name] = torch.cat([obs_feature_groups[group_name]['cell'], obs_feature_groups[group_name]['neighbors'].reshape(B,-1)], dim=1).to(device=device)
                all_inputs.append(extra_group_inputs[group_name])
    
        all_inputs = torch.cat(all_inputs, dim=1)
    
        # all_inputs_perm = [cell_inputs[random_derangement_stratified(slide_id)], niche_inputs[random_derangement_stratified(slide_id)]]

        # if model.extra_group_latent_dims:
        #     extra_group_inputs = {}
        #     for group_name in model.extra_group_names:
        #         extra_group_inputs[group_name] = torch.cat([obs_feature_groups[group_name]['cell'], obs_feature_groups[group_name]['neighbors'].reshape(B,-1)], dim=1).to(device=device)
        #         all_inputs_perm.append(extra_group_inputs[group_name][random_derangement_stratified(slide_id)])
    
        # all_inputs_perm = torch.cat(all_inputs_perm, dim=1)

        # Compute IG for just this mini-batch
        batch_attributions = model_explainer.attribute(
            inputs=all_inputs,
            baselines=baseline,
            target=None,
            n_steps=n_steps,
            internal_batch_size=internal_chunk_size # Must be >= external_batch_size
        )

        logit_attr[adata_idx.to(device=device)] += batch_attributions

    return logit_attr.cpu().numpy()

def explain_gene_logits_baseline(model, datamodule, n_steps=50, internal_chunk_size=3200, device='cuda'):
    model_explainer = IntegratedGradients(CIRCA_Wrapper(model).to(device))
    
    total_extra_group_input_dims = 0
    
    if model.extra_group_latent_dims:
        extra_groups_explainers = {}
        for group_name in model.extra_group_names:
            total_extra_group_input_dims += model.extra_group_input_dims[group_name]
            
    # 1. Configuration
    latent_dimension_size = model.latent_dim
    
    # if reference_indices == None:
    #     reference_indices = torch.arange(datamodule.adata.shape[0])
    
    num_total_samples = datamodule.adata.shape[0]
    
    datamodule.setup("fit")

    # if not isinstance(baseline, int):
    #     baseline = torch.as_tensor(baseline, dtype=torch.float32, device=device)
    
    # Container to hold final metrics
    logit_attr = torch.zeros((num_total_samples, (model.n_neighbors + 1) * (model.input_dim + total_extra_group_input_dims)), dtype=torch.float32, device=device)

    dataloader = datamodule.train_dataloader()
        
    for idx, batch in enumerate(dataloader):
        adata_idx, cell, neighbors, panel, slide_id, obs_feature_groups = model._unpack_batch(batch)
    
        B = len(adata_idx)
        
        cell_inputs = cell.to(device=device).log1p()
        niche_inputs = neighbors.to(device=device).log1p().reshape(B,-1)
        slide_id = slide_id.to(device=device)
    
        all_inputs = [cell_inputs, niche_inputs]
    
        if model.extra_group_latent_dims:
            extra_group_inputs = {}
            for group_name in model.extra_group_names:
                extra_group_inputs[group_name] = torch.cat([obs_feature_groups[group_name]['cell'], obs_feature_groups[group_name]['neighbors'].reshape(B,-1)], dim=1).to(device=device)
                all_inputs.append(extra_group_inputs[group_name])
    
        all_inputs = torch.cat(all_inputs, dim=1)
    
        # all_inputs_perm = [cell_inputs[random_derangement_stratified(slide_id)], niche_inputs[random_derangement_stratified(slide_id)]]

        # if model.extra_group_latent_dims:
        #     extra_group_inputs = {}
        #     for group_name in model.extra_group_names:
        #         extra_group_inputs[group_name] = torch.cat([obs_feature_groups[group_name]['cell'], obs_feature_groups[group_name]['neighbors'].reshape(B,-1)], dim=1).to(device=device)
        #         all_inputs_perm.append(extra_group_inputs[group_name][random_derangement_stratified(slide_id)])
    
        # all_inputs_perm = torch.cat(all_inputs_perm, dim=1)

        # Compute IG for just this mini-batch
        batch_attributions = model_explainer.attribute(
            inputs=all_inputs,
            baselines=torch.cat([torch.zeros_like(all_inputs[:,:(model.n_neighbors + 1) * model.input_dim]), all_inputs[:,(model.n_neighbors + 1) * model.input_dim:]], dim=1),
            target=None,
            n_steps=n_steps,
            internal_batch_size=internal_chunk_size # Must be >= external_batch_size
        )

        logit_attr[adata_idx.to(device=device)] += batch_attributions

    return logit_attr.cpu().numpy()

def explain_logits_block_baseline(model, datamodule, n_steps=50, internal_chunk_size=3200, device='cuda'):
    model_explainer = IntegratedGradients(CIRCA_Wrapper(model).to(device))
    
    total_extra_group_input_dims = 0
    
    block_bounds = [0, model.input_dim, (model.n_neighbors+1) * model.input_dim]
    
    if model.extra_group_latent_dims:
        extra_groups_explainers = {}
        for group_name in model.extra_group_names:
            total_extra_group_input_dims += model.extra_group_input_dims[group_name]
            block_bounds.append(block_bounds[-1] + (model.n_neighbors+1) * model.extra_group_input_dims[group_name])
            
    # 1. Configuration
    latent_dimension_size = model.latent_dim
    
    # if reference_indices == None:
    #     reference_indices = torch.arange(datamodule.adata.shape[0])
    
    num_total_samples = datamodule.adata.shape[0]
    
    datamodule.setup("fit")

    # if not isinstance(baseline, int):
    #     baseline = torch.as_tensor(baseline, dtype=torch.float32, device=device)
    
    # Container to hold final metrics
    logit_attr = torch.zeros((num_total_samples, (model.n_neighbors + 1) * (model.input_dim + total_extra_group_input_dims)), dtype=torch.float32, device=device)

    dataloader = datamodule.train_dataloader()
        
    for idx, batch in enumerate(dataloader):
        adata_idx, cell, neighbors, panel, slide_id, obs_feature_groups = model._unpack_batch(batch)
    
        B = len(adata_idx)
        
        cell_inputs = cell.to(device=device).log1p()
        niche_inputs = neighbors.to(device=device).log1p().reshape(B,-1)
        slide_id = slide_id.to(device=device)
    
        all_inputs = [cell_inputs, niche_inputs]
    
        if model.extra_group_latent_dims:
            extra_group_inputs = {}
            for group_name in model.extra_group_names:
                extra_group_inputs[group_name] = torch.cat([obs_feature_groups[group_name]['cell'], obs_feature_groups[group_name]['neighbors'].reshape(B,-1)], dim=1).to(device=device)
                all_inputs.append(extra_group_inputs[group_name])
    
        all_inputs = torch.cat(all_inputs, dim=1)
    
        # all_inputs_perm = [cell_inputs[random_derangement_stratified(slide_id)], niche_inputs[random_derangement_stratified(slide_id)]]

        # if model.extra_group_latent_dims:
        #     extra_group_inputs = {}
        #     for group_name in model.extra_group_names:
        #         extra_group_inputs[group_name] = torch.cat([obs_feature_groups[group_name]['cell'], obs_feature_groups[group_name]['neighbors'].reshape(B,-1)], dim=1).to(device=device)
        #         all_inputs_perm.append(extra_group_inputs[group_name][random_derangement_stratified(slide_id)])
    
        # all_inputs_perm = torch.cat(all_inputs_perm, dim=1)

        for bidx in range(len(block_bounds)-1):
            all_inputs_block_baseline = all_inputs.clone()
            all_inputs_block_baseline[:,block_bounds[bidx]:block_bounds[bidx+1]] = 0
            # Compute IG for just this mini-batch
            batch_attributions = model_explainer.attribute(
                inputs=all_inputs,
                baselines=all_inputs_block_baseline,
                target=None,
                n_steps=n_steps,
                internal_batch_size=internal_chunk_size # Must be >= external_batch_size
            )
    
            logit_attr[adata_idx.to(device=device),block_bounds[bidx]:block_bounds[bidx+1]] += batch_attributions[:,block_bounds[bidx]:block_bounds[bidx+1]]

    return logit_attr.cpu().numpy()

def explain_factors(model, datamodule, n_steps=50, internal_chunk_size=3200, device='cuda'):
    cell_explainer = IntegratedGradients(Cell_Embedding_Wrapper(model).to(device))
    niche_explainer = IntegratedGradients(Niche_Embedding_Wrapper(model).to(device))
    
    total_extra_group_input_dims = 0
   
    if model.extra_group_latent_dims:
        extra_groups_explainers = {}
        extra_groups_baselines = {}
        for group_name in model.extra_group_names:
            total_extra_group_input_dims += model.extra_group_input_dims[group_name]
            extra_groups_baselines[group_name] = torch.cat((model.n_neighbors+1) * [ torch.as_tensor(datamodule.adata.obs[list(datamodule.obs_feature_groups[group_name])].to_numpy().min(axis=0), dtype=torch.float32, device=device).reshape(1,-1) ], dim=1)
            extra_groups_explainers[group_name] = IntegratedGradients(ExtraGroup_Embedding_Wrapper(model, group_name).to(device))
    
    # 1. Configuration
    latent_dimension_size = model.latent_dim
    
    num_total_samples = datamodule.adata.shape[0]
    
    datamodule.setup("predict")
    dataloader = datamodule.predict_dataloader()

    # Container to hold final metrics
    cell_attr = torch.zeros((model.latent_dim, 2, model.input_dim), dtype=torch.float32, device=device)
    niche_attr = torch.zeros((model.latent_dim, 2, model.n_neighbors * model.input_dim), dtype=torch.float32, device=device)
    
    if model.extra_group_latent_dims:
        extra_group_attr = {}
        for group_name in model.extra_group_names:
            extra_group_attr[group_name] = torch.zeros((model.extra_group_latent_dims[group_name], 2, (model.n_neighbors + 1) * model.extra_group_input_dims[group_name]), dtype=torch.float32, device=device)
    
    for idx, batch in enumerate(dataloader):
        adata_idx, cell, neighbors, panel, slide_id, obs_feature_groups = model._unpack_batch(batch)
    
        B = len(adata_idx)
        
        cell_inputs = cell.to(device=device).log1p()
        niche_inputs = neighbors.to(device=device).log1p().reshape(B,-1)
    
        if model.extra_group_latent_dims:
            extra_group_inputs = {}
            for group_name in model.extra_group_names:
                extra_group_inputs[group_name] = torch.cat([obs_feature_groups[group_name]['cell'], obs_feature_groups[group_name]['neighbors'].reshape(B,-1)], dim=1).to(device=device)
        
        for dim_idx in range(model.latent_dim):
            batch_cell_attributions = cell_explainer.attribute(
                inputs=cell_inputs,
                baselines=0,
                target=None,
                additional_forward_args=(dim_idx),
                n_steps=n_steps,
                internal_batch_size=internal_chunk_size # Must be >= external_batch_size
            )
    
            cell_attr[dim_idx,0] += batch_cell_attributions.sum(dim=0)
            cell_attr[dim_idx,1] += batch_cell_attributions.abs().sum(dim=0)
    
            batch_niche_attributions = niche_explainer.attribute(
                inputs=niche_inputs,
                baselines=0,
                target=None,
                additional_forward_args=(dim_idx),
                n_steps=n_steps,
                internal_batch_size=internal_chunk_size # Must be >= external_batch_size
            )
    
            niche_attr[dim_idx,0] += batch_niche_attributions.sum(dim=0)
            niche_attr[dim_idx,1] += batch_niche_attributions.abs().sum(dim=0)
    
            if model.extra_group_latent_dims:
                for group_name in model.extra_group_names:
                    if dim_idx < model.extra_group_latent_dims[group_name]:
                        batch_extra_group_attributions = extra_groups_explainers[group_name].attribute(
                            inputs=extra_group_inputs[group_name],
                            baselines=extra_groups_baselines[group_name],
                            target=None,
                            additional_forward_args=(dim_idx),
                            n_steps=n_steps,
                            internal_batch_size=internal_chunk_size # Must be >= external_batch_size
                        )
            
                        extra_group_attr[group_name][dim_idx,0] += batch_extra_group_attributions.sum(dim=0)
                        extra_group_attr[group_name][dim_idx,1] += batch_extra_group_attributions.abs().sum(dim=0)

    if model.extra_group_latent_dims:
        for group_name in model.extra_group_names:
            extra_group_attr[group_name] = extra_group_attr[group_name].cpu().numpy()
            
        return cell_attr.cpu().numpy(), niche_attr.cpu().numpy(), extra_group_attr
    else:
        return cell_attr.cpu().numpy(), niche_attr.cpu().numpy(), {}

def spatial_features_annotate(attr, genes, n_neighbors, obs_names, amyloid_feats, genesets=None, n_contact_neighbors=5, signed=True):

    if not signed:
        cell_attr = np.abs(attr[:,:len(genes)])
        niche_attr = np.abs(attr[:,len(genes):(n_neighbors+1)*len(genes)]).reshape(attr.shape[0], n_neighbors, -1)
        amyloid_attr = np.abs(attr[:,(n_neighbors+1)*len(genes):])
    else:
        cell_attr = (attr[:,:len(genes)])
        niche_attr = (attr[:,len(genes):(n_neighbors+1)*len(genes)]).reshape(attr.shape[0], n_neighbors, -1)
        amyloid_attr = (attr[:,(n_neighbors+1)*len(genes):])

    if genesets != None:
        cell_geneset_attr = []
        niche_geneset_attr = []
        contact_niche_geneset_attr = []
        for gs in genesets:
            cell_geneset_attr.append(pd.DataFrame(
                cell_attr @ gs.matrix,
                index=obs_names,
                columns=gs.genesets,
            ))
            niche_geneset_attr.append(pd.DataFrame(
                niche_attr.mean(axis=1) @ gs.matrix,
                index=obs_names,
                columns=gs.genesets,
            ))
            contact_niche_geneset_attr.append(pd.DataFrame(
                niche_attr[:,:n_contact_neighbors].mean(axis=1) @ gs.matrix,
                index=obs_names,
                columns=gs.genesets,
            ))

        return {
            "cell" : {
                "feature" : pd.DataFrame(cell_attr, columns=genes, index=obs_names), 
                "geneset" : cell_geneset_attr
            }, 
            "niche" : {
                "feature" : pd.DataFrame(niche_attr.mean(axis=1), columns=genes, index=obs_names), 
                "geneset" : niche_geneset_attr
            }, 
            "contact_niche" : {
                "feature" : pd.DataFrame(niche_attr[:,:n_contact_neighbors].mean(axis=1), columns=genes, index=obs_names), 
                "geneset" : contact_niche_geneset_attr
            }, 
            "amyloid" : {
                "feature" : pd.DataFrame(amyloid_attr, columns=amyloid_feats + [f'nbr{i+1}_{feat}' for i in range(model.n_neighbors) for feat in amyloid_feats], index=obs_names)
            }
        }
    else:
        return {
            "cell" : {
                "feature" : pd.DataFrame(cell_attr, columns=genes, index=obs_names)
            }, 
            "niche" : {
                "feature" : pd.DataFrame(niche_attr.mean(axis=1), columns=genes, index=obs_names)
            }, 
            "contact_niche" : {
                "feature" : pd.DataFrame(niche_attr[:,:n_contact_neighbors].mean(axis=1), columns=genes, index=obs_names)
            }, 
            "amyloid" : {
                "feature" : pd.DataFrame(amyloid_attr, columns=amyloid_feats + [f'nbr{i+1}_{feat}' for i in range(model.n_neighbors) for feat in amyloid_feats], index=obs_names)
            }
        }

@dataclass
class GeneSetMatrix:
    matrix: sparse.csr_matrix
    genes: pd.Index
    genesets: pd.Index
    geneset_sizes: pd.Series


def make_geneset_matrix(
    geneset_genes: pd.DataFrame,
    genes,
    *,
    source_col: str = "source",
    target_col: str = "target",
    min_genes: int = 3,
    max_genes: int = 500,
    normalize: bool = True,
) -> GeneSetMatrix:
    """
    Construct a sparse gene-by-geneset membership matrix.

    Parameters
    ----------
    geneset_genes
        Long-format DataFrame containing one geneset-gene pair per row.
    genes
        Genes corresponding to the columns of the cell-by-gene attribution
        matrix. Their order is preserved.
    source_col
        Column containing geneset names.
    target_col
        Column containing gene names.
    min_genes
        Retain only genesets having at least this many genes represented
        in `genes`.
    max_genes
        Retain only genesets having at most this many genes represented
        in `genes`.
    normalize
        If True, each nonzero value is 1 / number of represented genes in
        that geneset. Multiplication therefore returns the mean attribution.
        If False, nonzero values are 1 and multiplication returns the sum.

    Returns
    -------
    GeneSetMatrix
        Contains:
        - matrix: sparse gene-by-geneset matrix
        - genes: gene ordering
        - genesets: geneset ordering
        - geneset_sizes: number of represented genes per retained geneset
    """
    genes = pd.Index(genes)

    if not genes.is_unique:
        duplicated = genes[genes.duplicated()].unique().tolist()
        raise ValueError(
            "Gene names must be unique. Duplicated genes include: "
            f"{duplicated[:10]}"
        )

    required = {source_col, target_col}
    missing = required.difference(geneset_genes.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep only valid geneset-gene pairs and remove duplicated memberships.
    net = (
        geneset_genes[[source_col, target_col]]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    net[source_col] = net[source_col].astype(str)
    net[target_col] = net[target_col].astype(str)

    # Restrict the network to genes present in the attribution matrix.
    gene_to_idx = pd.Series(
        np.arange(len(genes), dtype=np.int64),
        index=genes,
    )
    net = net[net[target_col].isin(gene_to_idx.index)].copy()

    if net.empty:
        raise ValueError(
            "None of the geneset target genes were found in `genes`."
        )

    # Remove genesets with insufficient coverage.
    geneset_sizes = net.groupby(source_col)[target_col].nunique()
    retained = geneset_sizes[np.logical_and(geneset_sizes >= min_genes, geneset_sizes <= max_genes)].index
    net = net[net[source_col].isin(retained)].copy()

    if net.empty:
        raise ValueError(
            f"No genesets had at least {min_genes} genes represented."
        )

    # Sorting makes the output deterministic.
    genesets = pd.Index(sorted(net[source_col].unique()))
    geneset_to_idx = pd.Series(
        np.arange(len(genesets), dtype=np.int64),
        index=genesets,
    )

    rows = gene_to_idx.loc[net[target_col]].to_numpy()
    cols = geneset_to_idx.loc[net[source_col]].to_numpy()

    geneset_sizes = (
        net.groupby(source_col)[target_col]
        .nunique()
        .reindex(genesets)
        .astype(int)
    )

    if normalize:
        values = 1.0 / geneset_sizes.loc[net[source_col]].to_numpy()
    else:
        values = np.ones(len(net), dtype=np.float64)

    matrix = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(len(genes), len(genesets)),
        dtype=np.float64,
    ).tocsr()

    return GeneSetMatrix(
        matrix=matrix,
        genes=genes,
        genesets=genesets,
        geneset_sizes=geneset_sizes,
    )


@dataclass
class LigandReceptorMatrix:
    ligand_matrix: sparse.csr_matrix
    receptor_matrix: sparse.csr_matrix
    genes: pd.Index
    interactions: pd.Index
    ligand_sizes: pd.Series
    receptor_sizes: pd.Series

def make_ligand_receptor_matrix(
    interactions: pd.DataFrame,
    genes,
    *,
    interaction_col: str = "interaction_name",
    ligand_col: str = "ligand_symbol",
    receptor_col: str = "receptor_symbol",
    complex_delimiter: str = "&",
    min_ligand_genes: int = 1,
    max_ligand_genes: int | None = None,
    min_receptor_genes: int = 1,
    max_receptor_genes: int | None = None,
    require_complete_ligand: bool = False,
    require_complete_receptor: bool = False,
    normalize: bool = True,
) -> LigandReceptorMatrix:
    """
    Construct aligned sparse gene-by-interaction matrices for ligands and
    receptors.

    Parameters
    ----------
    interactions
        DataFrame containing one ligand-receptor interaction per row.

        Required columns:
        - interaction_col: unique interaction name
        - ligand_col: ligand gene or complex, with components separated by "&"
        - receptor_col: receptor gene or complex, with components separated by "&"

        Multiple rows with the same interaction name are allowed. Their ligand
        and receptor genes are combined.

    genes
        Genes corresponding to the columns of the cell-by-gene attribution
        matrix. Their order is preserved.

    interaction_col
        Column containing interaction names.

    ligand_col
        Column containing ligand genes or ligand complexes.

    receptor_col
        Column containing receptor genes or receptor complexes.

    complex_delimiter
        Delimiter separating genes within a complex.

    min_ligand_genes
        Minimum number of represented ligand genes required to retain an
        interaction.

    max_ligand_genes
        Maximum number of represented ligand genes allowed. If None, no upper
        limit is applied.

    min_receptor_genes
        Minimum number of represented receptor genes required to retain an
        interaction.

    max_receptor_genes
        Maximum number of represented receptor genes allowed. If None, no upper
        limit is applied.

    require_complete_ligand
        If True, retain an interaction only when every ligand component is
        present in `genes`.

    require_complete_receptor
        If True, retain an interaction only when every receptor component is
        present in `genes`.

    normalize
        If True, nonzero entries are divided by the number of represented genes
        on the corresponding side of the interaction. Matrix multiplication
        therefore returns the mean ligand or receptor attribution.

        If False, nonzero entries are 1, so multiplication returns the sum.

    Returns
    -------
    LigandReceptorMatrix
        Contains:
        - ligand_matrix: sparse gene-by-interaction ligand matrix
        - receptor_matrix: sparse gene-by-interaction receptor matrix
        - genes: gene ordering
        - interactions: shared interaction ordering
        - ligand_sizes: represented ligand genes per interaction
        - receptor_sizes: represented receptor genes per interaction
    """
    genes = pd.Index(genes)

    if not genes.is_unique:
        duplicated = genes[genes.duplicated()].unique().tolist()
        raise ValueError(
            "Gene names must be unique. Duplicated genes include: "
            f"{duplicated[:10]}"
        )

    required = {interaction_col, ligand_col, receptor_col}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if min_ligand_genes < 1 or min_receptor_genes < 1:
        raise ValueError("Minimum gene counts must be at least 1.")

    if (
        max_ligand_genes is not None
        and max_ligand_genes < min_ligand_genes
    ):
        raise ValueError(
            "`max_ligand_genes` must be greater than or equal to "
            "`min_ligand_genes`."
        )

    if (
        max_receptor_genes is not None
        and max_receptor_genes < min_receptor_genes
    ):
        raise ValueError(
            "`max_receptor_genes` must be greater than or equal to "
            "`min_receptor_genes`."
        )

    net = (
        interactions[
            [interaction_col, ligand_col, receptor_col]
        ]
        .dropna()
        .copy()
    )

    net[interaction_col] = net[interaction_col].astype(str)
    net[ligand_col] = net[ligand_col].astype(str)
    net[receptor_col] = net[receptor_col].astype(str)

    def split_complex_column(
        frame: pd.DataFrame,
        symbol_col: str,
        side: str,
    ) -> pd.DataFrame:
        """Convert complex strings into interaction-gene pairs."""
        result = frame[[interaction_col, symbol_col]].copy()

        result[symbol_col] = result[symbol_col].str.split(
            complex_delimiter,
            regex=False,
        )

        result = result.explode(symbol_col)
        result[symbol_col] = result[symbol_col].str.strip()

        # Remove empty components and duplicated interaction-gene memberships.
        result = result[
            result[symbol_col].notna()
            & result[symbol_col].ne("")
        ].drop_duplicates()

        return result.rename(
            columns={
                symbol_col: "gene",
            }
        ).assign(side=side)

    ligand_all = split_complex_column(net, ligand_col, "ligand")
    receptor_all = split_complex_column(net, receptor_col, "receptor")

    if ligand_all.empty:
        raise ValueError("No valid ligand genes were found.")

    if receptor_all.empty:
        raise ValueError("No valid receptor genes were found.")

    # Total complex sizes before restricting to genes available in the
    # attribution matrix.
    total_ligand_sizes = (
        ligand_all.groupby(interaction_col)["gene"].nunique()
    )
    total_receptor_sizes = (
        receptor_all.groupby(interaction_col)["gene"].nunique()
    )

    gene_to_idx = pd.Series(
        np.arange(len(genes), dtype=np.int64),
        index=genes,
    )

    ligand_observed = ligand_all[
        ligand_all["gene"].isin(gene_to_idx.index)
    ].copy()

    receptor_observed = receptor_all[
        receptor_all["gene"].isin(gene_to_idx.index)
    ].copy()

    observed_ligand_sizes = (
        ligand_observed.groupby(interaction_col)["gene"].nunique()
    )

    observed_receptor_sizes = (
        receptor_observed.groupby(interaction_col)["gene"].nunique()
    )

    # Evaluate all interactions, including those with zero observed genes.
    all_interactions = pd.Index(
        sorted(net[interaction_col].unique())
    )

    observed_ligand_sizes = observed_ligand_sizes.reindex(
        all_interactions,
        fill_value=0,
    )

    observed_receptor_sizes = observed_receptor_sizes.reindex(
        all_interactions,
        fill_value=0,
    )

    total_ligand_sizes = total_ligand_sizes.reindex(
        all_interactions,
        fill_value=0,
    )

    total_receptor_sizes = total_receptor_sizes.reindex(
        all_interactions,
        fill_value=0,
    )

    retain = (
        observed_ligand_sizes.ge(min_ligand_genes)
        & observed_receptor_sizes.ge(min_receptor_genes)
    )

    if max_ligand_genes is not None:
        retain &= observed_ligand_sizes.le(max_ligand_genes)

    if max_receptor_genes is not None:
        retain &= observed_receptor_sizes.le(max_receptor_genes)

    if require_complete_ligand:
        retain &= observed_ligand_sizes.eq(total_ligand_sizes)

    if require_complete_receptor:
        retain &= observed_receptor_sizes.eq(total_receptor_sizes)

    retained_interactions = all_interactions[retain.to_numpy()]

    if retained_interactions.empty:
        raise ValueError(
            "No ligand-receptor interactions passed the requested ligand and "
            "receptor coverage filters."
        )

    ligand_observed = ligand_observed[
        ligand_observed[interaction_col].isin(retained_interactions)
    ].copy()

    receptor_observed = receptor_observed[
        receptor_observed[interaction_col].isin(retained_interactions)
    ].copy()

    interaction_to_idx = pd.Series(
        np.arange(len(retained_interactions), dtype=np.int64),
        index=retained_interactions,
    )

    ligand_sizes = (
        ligand_observed.groupby(interaction_col)["gene"]
        .nunique()
        .reindex(retained_interactions)
        .astype(int)
    )

    receptor_sizes = (
        receptor_observed.groupby(interaction_col)["gene"]
        .nunique()
        .reindex(retained_interactions)
        .astype(int)
    )

    def build_matrix(
        membership: pd.DataFrame,
        sizes: pd.Series,
    ) -> sparse.csr_matrix:
        rows = gene_to_idx.loc[membership["gene"]].to_numpy()
        cols = interaction_to_idx.loc[
            membership[interaction_col]
        ].to_numpy()

        if normalize:
            values = (
                1.0
                / sizes.loc[membership[interaction_col]].to_numpy(
                    dtype=np.float64
                )
            )
        else:
            values = np.ones(len(membership), dtype=np.float64)

        return sparse.coo_matrix(
            (values, (rows, cols)),
            shape=(len(genes), len(retained_interactions)),
            dtype=np.float64,
        ).tocsr()

    ligand_matrix = build_matrix(
        membership=ligand_observed,
        sizes=ligand_sizes,
    )

    receptor_matrix = build_matrix(
        membership=receptor_observed,
        sizes=receptor_sizes,
    )

    return LigandReceptorMatrix(
        ligand_matrix=ligand_matrix,
        receptor_matrix=receptor_matrix,
        genes=genes,
        interactions=retained_interactions,
        ligand_sizes=ligand_sizes,
        receptor_sizes=receptor_sizes,
    )


@dataclass
class LRScoreResults:
    sending: AnnData
    receiving: AnnData
    combined: AnnData
    sending_pair_counts: pd.Series
    receiving_pair_counts: pd.Series
    sending_niche_pair_counts: pd.Series
    receiving_niche_pair_counts: pd.Series


def make_lr_score_anndata(
    sending_lr_results: np.ndarray,
    receiving_lr_results: np.ndarray,
    cell_types,
    niches,
    neighbor_cell_types,
    interaction_names,
    slide_ids,
    *,
    n_neighbors: int | None = None,
    min_celltype_pair_count: int = 1,
    min_niche_interaction_count: int = 1,
    neighbor_indices: np.ndarray | None = None,
    valid_neighbor_mask: np.ndarray | None = None,
    copy_scores: bool = False,
) -> LRScoreResults:
    """
    Construct cell-centric ligand-receptor score AnnData objects.

    Sending and receiving observations are processed separately:

    - sending:
        focal cell is sender, neighbor is receiver
    - receiving:
        neighbor is sender, focal cell is receiver

    The niche always refers to the focal cell's niche.

    Parameters
    ----------
    sending_lr_results
        Array with shape [N, K, D]. Entry [i, k, d] represents interaction d
        sent by focal cell i to neighbor k.

    receiving_lr_results
        Array with shape [N, K, D]. Entry [i, k, d] represents interaction d
        sent by neighbor k to focal cell i.

    cell_types
        Length-N focal-cell type vector.

    niches
        Length-N focal-cell niche vector.

    neighbor_cell_types
        Array with shape [N, K] containing neighbor cell types.

    interaction_names
        Length-D ligand-receptor interaction names.

    n_neighbors
        Number of leading neighbors to use. For example, n_neighbors=5 uses
        neighbor positions 0:5. If None, all K neighbors are used.

    min_celltype_pair_count
        Remove directed cell-type pairs represented by fewer than this many
        focal-neighbor observations. Filtering is performed separately for
        sending and receiving.

    min_niche_interaction_count
        Remove niche × directed-cell-type-pair combinations represented by
        fewer than this many observations. Filtering is performed separately
        for sending and receiving.

    neighbor_indices
        Optional [N, K] neighbor index array. Entries below zero are treated
        as invalid or padded neighbors.

    valid_neighbor_mask
        Optional boolean [N, K] mask indicating valid neighbor positions.

    copy_scores
        Whether to explicitly copy score arrays when constructing AnnData.

    Returns
    -------
    LRScoreResults
        Contains separate sending and receiving AnnData objects, a concatenated
        object, and occurrence-count Series.

    Notes
    -----
    Each AnnData observation represents one valid focal-cell/neighbor pair.
    Variables represent ligand-receptor interactions.
    """
    sending_lr_results = np.asarray(sending_lr_results)
    receiving_lr_results = np.asarray(receiving_lr_results)
    cell_types = np.asarray(cell_types)
    niches = np.asarray(niches)
    neighbor_cell_types = np.asarray(neighbor_cell_types)
    interaction_names = pd.Index(interaction_names, name="interaction_name")

    if sending_lr_results.ndim != 3:
        raise ValueError(
            "`sending_lr_results` must have shape [N, K, D], "
            f"got {sending_lr_results.shape}."
        )

    if receiving_lr_results.shape != sending_lr_results.shape:
        raise ValueError(
            "`receiving_lr_results` must have the same shape as "
            f"`sending_lr_results`; got {receiving_lr_results.shape} and "
            f"{sending_lr_results.shape}."
        )

    n_cells, max_neighbors, n_interactions = sending_lr_results.shape

    if cell_types.shape != (n_cells,):
        raise ValueError(
            f"`cell_types` must have shape ({n_cells},), "
            f"got {cell_types.shape}."
        )

    if niches.shape != (n_cells,):
        raise ValueError(
            f"`niches` must have shape ({n_cells},), got {niches.shape}."
        )

    if neighbor_cell_types.shape != (n_cells, max_neighbors):
        raise ValueError(
            "`neighbor_cell_types` must have shape "
            f"({n_cells}, {max_neighbors}), "
            f"got {neighbor_cell_types.shape}."
        )

    if len(interaction_names) != n_interactions:
        raise ValueError(
            f"`interaction_names` must have length {n_interactions}, "
            f"got {len(interaction_names)}."
        )

    if not interaction_names.is_unique:
        duplicated = interaction_names[
            interaction_names.duplicated()
        ].unique().tolist()
        raise ValueError(
            "Interaction names must be unique. Duplicates include "
            f"{duplicated[:10]}."
        )

    if n_neighbors is None:
        n_neighbors = max_neighbors

    if not 1 <= n_neighbors <= max_neighbors:
        raise ValueError(
            f"`n_neighbors` must be between 1 and {max_neighbors}, "
            f"got {n_neighbors}."
        )

    if min_celltype_pair_count < 1:
        raise ValueError("`min_celltype_pair_count` must be at least 1.")

    if min_niche_interaction_count < 1:
        raise ValueError("`min_niche_interaction_count` must be at least 1.")

    # Select only the requested leading neighbors.
    sending = sending_lr_results[:, :n_neighbors, :]
    receiving = receiving_lr_results[:, :n_neighbors, :]
    neighbor_types = neighbor_cell_types[:, :n_neighbors]

    valid = np.ones((n_cells, n_neighbors), dtype=bool)

    if neighbor_indices is not None:
        neighbor_indices = np.asarray(neighbor_indices)

        if neighbor_indices.shape != (n_cells, max_neighbors):
            raise ValueError(
                "`neighbor_indices` must have shape "
                f"({n_cells}, {max_neighbors}), "
                f"got {neighbor_indices.shape}."
            )

        selected_neighbor_indices = neighbor_indices[:, :n_neighbors]
        valid &= selected_neighbor_indices >= 0
    else:
        selected_neighbor_indices = np.full(
            (n_cells, n_neighbors),
            -1,
            dtype=np.int64,
        )

    if valid_neighbor_mask is not None:
        valid_neighbor_mask = np.asarray(
            valid_neighbor_mask,
            dtype=bool,
        )

        if valid_neighbor_mask.shape != (n_cells, max_neighbors):
            raise ValueError(
                "`valid_neighbor_mask` must have shape "
                f"({n_cells}, {max_neighbors}), "
                f"got {valid_neighbor_mask.shape}."
            )

        valid &= valid_neighbor_mask[:, :n_neighbors]

    # Missing cell-type or niche labels should not create valid group labels.
    focal_type_matrix = np.broadcast_to(
        cell_types[:, None],
        (n_cells, n_neighbors),
    )
    focal_niche_matrix = np.broadcast_to(
        niches[:, None],
        (n_cells, n_neighbors),
    )

    valid &= pd.notna(focal_type_matrix)
    valid &= pd.notna(neighbor_types)
    valid &= pd.notna(focal_niche_matrix)

    cell_index_matrix = np.broadcast_to(
        np.arange(n_cells)[:, None],
        (n_cells, n_neighbors),
    )
    neighbor_rank_matrix = np.broadcast_to(
        np.arange(n_neighbors)[None, :],
        (n_cells, n_neighbors),
    )

    valid_flat = valid.ravel()

    focal_cell_index = cell_index_matrix.ravel()[valid_flat]
    neighbor_cell_index = selected_neighbor_indices.ravel()[valid_flat]
    neighbor_rank = neighbor_rank_matrix.ravel()[valid_flat]

    focal_type = focal_type_matrix.ravel()[valid_flat].astype(str)
    neighbor_type = neighbor_types.ravel()[valid_flat].astype(str)
    focal_niche = focal_niche_matrix.ravel()[valid_flat].astype(str)

    sending_scores = sending.reshape(
        -1,
        n_interactions,
    )[valid_flat]

    receiving_scores = receiving.reshape(
        -1,
        n_interactions,
    )[valid_flat]

    def build_direction(
        scores: np.ndarray,
        *,
        focal_role: str,
    ):
        if focal_role == "sender":
            source = focal_type
            target = neighbor_type
        elif focal_role == "receiver":
            source = neighbor_type
            target = focal_type
        else:
            raise ValueError(
                "`focal_role` must be either 'sender' or 'receiver'."
            )

        directed_pair = np.char.add(
            np.char.add(source, " --> "),
            target,
        )

        niche_x_interaction = np.char.add(
            np.char.add(focal_niche, ": "),
            directed_pair,
        )

        obs = pd.DataFrame(
            {
                "focal_cell_index": focal_cell_index,
                "neighbor_cell_index": neighbor_cell_index,
                "neighbor_rank": neighbor_rank,
                "focal_cell_type": focal_type,
                "neighbor_cell_type": neighbor_type,
                "focal_niche": focal_niche,
                "focal_role": focal_role,
                "focal_slide_id": slide_ids[focal_cell_index],
                "source": source,
                "target": target,
                "interaction": directed_pair,
                "niche_x_interaction": niche_x_interaction,
            }
        )

        # Counts before either filter.
        pair_counts = obs["interaction"].value_counts(sort=False)
        niche_pair_counts = obs[
            "niche_x_interaction"
        ].value_counts(sort=False)

        keep_pair = obs["interaction"].map(
            pair_counts
        ).ge(min_celltype_pair_count)

        keep_niche_pair = obs["niche_x_interaction"].map(
            niche_pair_counts
        ).ge(min_niche_interaction_count)

        keep = (keep_pair & keep_niche_pair).to_numpy()

        filtered_obs = obs.loc[keep].reset_index(drop=True)
        filtered_scores = scores[keep]

        # Store the occurrence counts on each retained observation.
        filtered_obs["celltype_pair_count"] = (
            filtered_obs["interaction"]
            .map(pair_counts)
            .astype(np.int64)
        )

        filtered_obs["niche_interaction_count"] = (
            filtered_obs["niche_x_interaction"]
            .map(niche_pair_counts)
            .astype(np.int64)
        )

        adata = AnnData(
            X=filtered_scores.copy() if copy_scores else filtered_scores,
            obs=filtered_obs,
            var=pd.DataFrame(index=interaction_names.copy()),
        )

        return adata, pair_counts, niche_pair_counts

    sending_adata, sending_pair_counts, sending_niche_counts = (
        build_direction(
            sending_scores,
            focal_role="sender",
        )
    )

    receiving_adata, receiving_pair_counts, receiving_niche_counts = (
        build_direction(
            receiving_scores,
            focal_role="receiver",
        )
    )

    # Concatenation preserves sending and receiving observations but does not
    # numerically combine their scores.
    combined = AnnData.concatenate(
        sending_adata,
        receiving_adata,
        batch_key="score_direction",
        batch_categories=["sending", "receiving"],
        index_unique="-",
    )

    return LRScoreResults(
        sending=sending_adata,
        receiving=receiving_adata,
        combined=combined,
        sending_pair_counts=sending_pair_counts,
        receiving_pair_counts=receiving_pair_counts,
        sending_niche_pair_counts=sending_niche_counts,
        receiving_niche_pair_counts=receiving_niche_counts,
    )

from scipy.stats import false_discovery_control

def pseudobulk_de_lr_score_anndata(
    lr_results : LRScoreResults,
    interaction_key: str,
    slide_key: str,
    *,
    method: str = 'wilcoxon',
    pval_cutoff: float | None = None,
    pval_key:str = 'pvals_adj',
):
    pb_lr_results = LRScoreResults(
        sending=lr_results.sending.copy(),
        receiving=lr_results.receiving.copy(),
        combined=lr_results.combined.copy(),
        sending_pair_counts=lr_results.sending_pair_counts.copy(),
        receiving_pair_counts=lr_results.receiving_pair_counts.copy(),
        sending_niche_pair_counts=lr_results.sending_niche_pair_counts.copy(),
        receiving_niche_pair_counts=lr_results.receiving_niche_pair_counts.copy(),
    )
    
    de_results = {}
    signif_de_results = {}
    for block in ['sending', 'receiving', 'combined']:
        setattr(pb_lr_results, block, sc.get.aggregate(getattr(pb_lr_results, block), by=[slide_key, interaction_key], func='mean'))
        sc.tl.rank_genes_groups(getattr(pb_lr_results, block), groupby=interaction_key, method=method, layer='mean')
        
        means_df = sc.get.aggregate(getattr(pb_lr_results, block), by=[interaction_key], func='mean', layer='mean').to_df(layer='mean')
        de_df = sc.get.rank_genes_groups_df(getattr(pb_lr_results, block), group=None)

        for group in de_df['group'].unique():
            de_df.loc[de_df['group']==group, 'mean_target'] = de_df['names'].map(means_df.loc[group])

        de_results[block] = de_df

        signif_de_results[block] = de_df.loc[np.logical_and(de_df['scores']>0, de_df['mean_target']>0)]

        for group in de_df['group'].unique():
            signif_de_results[block].loc[signif_de_results[block]['group']==group,'pvals_adj'] = false_discovery_control(signif_de_results[block].loc[signif_de_results[block]['group']==group,'pvals'], method='bh')
        
        if pval_cutoff:
            signif_de_results[block] = signif_de_results[block].loc[signif_de_results[block][pval_key]<pval_cutoff]

    return pb_lr_results, de_results, signif_de_results


def pseudobulk_de_anndata(
    adata : AnnData,
    group_key: str,
    slide_key: str,
    *,
    method: str = 'wilcoxon',
    pval_cutoff: float | None = None,
    pval_key:str = 'pvals_adj',
):   
    
    pb_adata = sc.get.aggregate(adata, by=[slide_key, group_key], func='mean')
    sc.tl.rank_genes_groups(pb_adata, groupby=group_key, method=method, layer='mean')
        
    de_results = sc.get.rank_genes_groups_df(pb_adata, group=None)
        
    if pval_cutoff:
        signif_de_results = de_results.loc[de_results[pval_key]<pval_cutoff]

        return pb_adata, de_results, signif_de_results

    return pb_adata, de_results