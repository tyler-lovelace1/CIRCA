import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from itertools import combinations
import lightning as L
from sklearn.metrics import f1_score, matthews_corrcoef
from collections import OrderedDict

from circa.modules.modules import Net, GradReverse, NeighborhoodProjectPool
from circa.utils._utils import tqdm
from circa.utils._torch import (
    scipy_sparse_to_torch,
    smooth_abs,
    compute_lambda,
    generate_block_roll_negatives_index,
)


class CIRCA(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.input_dim = hparams['input_dim']
        self.hidden_dims = hparams['hidden_dims']
        self.proj_hidden_dims = hparams['proj_hidden_dims']
        self.psi_hidden_dims = hparams['psi_hidden_dims']
        self.latent_dim = hparams['latent_dim']
        
        self.extra_group_latent_dims = {
            str(k): int(v) for k, v in hparams.get('extra_group_latent_dims', {}).items()
        }
        
        self.extra_group_input_dims = {
            str(k): int(v) for k, v in hparams.get('extra_group_input_dims', {}).items()
        }
        
        missing_extra_inputs = set(self.extra_group_latent_dims) - set(self.extra_group_input_dims)
        if missing_extra_inputs:
            raise ValueError(
                f"Missing input dims for extra groups: {sorted(missing_extra_inputs)}. "
                "Provide hparams['extra_group_input_dims']."
            )
            
        self.hz_dim = self.hidden_dims[-1]
        self.phi_type = hparams['phi_type']
        self.learning_rate = hparams['learning_rate']
        self.weight_decay = hparams['weight_decay']
        self.n_neighbors = hparams['n_neighbors']
        self.max_epochs = hparams['max_epochs']
        self.opt_algo = hparams['opt_algo']
        self.batch_size = hparams['batch_size']
        self.mask_prop = hparams['mask_prop']
        self.slide_key = hparams['slide_key']
        self.phi_share = bool(hparams.get("phi_share", False))
        self.pool_neighbors = bool(hparams.get("pool_neighbors", False))
        self.log_sklearn_metrics = bool(hparams.get("log_sklearn_metrics", False))
        
        self.warmup_steps = int(hparams.get("warmup_steps", 1000))

        self.use_adv = bool(hparams.get("use_adv", False))
        self.use_slide_adv = bool(hparams.get("use_slide_adv", False))  # only relevant if use_adv
        self.adv_weight = float(hparams.get("adv_weight", 1.0))

        self.dropout_rate = float(hparams.get("dropout_rate", 0.1))

        self.patience = int(hparams.get("patience", 10))

        self.sym_lam = float(hparams.get("sym_lam", 0.01))

        self.state_lam = float(hparams.get("state_lam", 1.0))
        
        self.n_slides = hparams['num_slides']
        self.n_panels = hparams['num_panels']

        self.block_size = int(hparams.get("block_size", 16))

        assert self.n_panels == hparams['panel_mask'].shape[0]

        self.register_buffer(
            "panel_mask",
            torch.as_tensor(hparams['panel_mask'], dtype=torch.bool)
        )

        self.extra_group_names = list(self.extra_group_latent_dims.keys())
        self.num_groups = 2 + len(self.extra_group_names)
        self.group_pair_indices = list(combinations(range(self.num_groups), 2))
        self.num_comb = len(self.group_pair_indices)
        self.tau = nn.Parameter(torch.tensor([math.log(hparams['tau'])], dtype=torch.float32), requires_grad=False)
        
        group_latent_dims = [self.latent_dim, self.latent_dim] + [
            self.extra_group_latent_dims[group_name] for group_name in self.extra_group_names
        ]
        
        group_latent_valid_mask = torch.zeros((self.num_groups, self.latent_dim), dtype=torch.bool)
        for g, group_latent_dim in enumerate(group_latent_dims):
            group_latent_valid_mask[g, :group_latent_dim] = True
            
        self.register_buffer("group_latent_valid_mask", group_latent_valid_mask)
        
        if hparams['activation'] == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif hparams['activation'] == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ReLU()
            
        self.seed = hparams['seed']

        self.n_perms = int(hparams.get("n_perms", 1))

        self.save_hyperparameters()

        self.backbone = Net(self.input_dim, self.hidden_dims, activation=self.activation, norm=nn.LayerNorm, dropout=nn.Dropout(p=self.dropout_rate))
                
        self.state_net = Net(self.hidden_dims[-1], self.proj_hidden_dims, output_dim=self.latent_dim, activation=self.activation, norm=nn.LayerNorm, dropout=nn.Dropout(p=self.dropout_rate))
        
        self.cell_net = Net(self.hidden_dims[-1], self.proj_hidden_dims, output_dim=self.latent_dim, activation=self.activation, norm=nn.LayerNorm, dropout=nn.Dropout(p=self.dropout_rate))

        if self.pool_neighbors:
            self.niche_net = nn.Sequential(
                NeighborhoodProjectPool(
                    self.hz_dim, 
                    project_dim=self.hz_dim//4, 
                    summary_dim=self.hz_dim//2,
                    activation=self.activation,
                    norm=nn.LayerNorm
                ),
                nn.LayerNorm(2 * (self.hz_dim//2) + self.n_neighbors * (self.hz_dim//4)),
                Net(
                    2 * (self.hz_dim//2) + self.n_neighbors * (self.hz_dim//4), 
                    self.proj_hidden_dims, 
                    output_dim=self.latent_dim, 
                    activation=self.activation, 
                    norm=nn.LayerNorm,
                    dropout=nn.Dropout(p=self.dropout_rate)
                   )
            )
        else:
            self.niche_net = Net(
                self.n_neighbors * self.hidden_dims[-1],
                self.proj_hidden_dims,
                output_dim=self.latent_dim, 
                activation=self.activation, 
                norm=nn.LayerNorm, 
                dropout=nn.Dropout(p=self.dropout_rate)
            )

        self.psi_nets = nn.ModuleList()
        for i in range(self.num_groups):
            self.psi_nets.append(
                Net(
                    self.latent_dim if self.phi_type=='gauss-maxout' else 2 * self.latent_dim, 
                    self.psi_hidden_dims, 
                    output_dim=self.latent_dim, 
                    activation=self.activation, 
                    norm=nn.LayerNorm,
                    dropout=nn.Dropout(p=self.dropout_rate)
                )
            )

        self.extra_group_nets = nn.ModuleDict()
        for group_name, group_latent_dim in self.extra_group_latent_dims.items():
            group_input_dim = self.extra_group_input_dims[group_name]
            
            self.extra_group_nets[group_name] = Net(
                (1 + self.n_neighbors) * group_input_dim,
                self.proj_hidden_dims,
                output_dim=group_latent_dim,
                activation=self.activation,
                norm=nn.LayerNorm,
                dropout=nn.Dropout(p=self.dropout_rate)
            )
       
        if self.phi_type == 'gauss-maxout':
            self.w = nn.Parameter(torch.zeros([self.latent_dim, self.latent_dim, 2, self.num_comb]))
            self.zw = nn.Parameter(torch.zeros([self.num_groups, self.latent_dim, 2]))

            if self.phi_share:
                self.pw = nn.Parameter(torch.ones([1, 2]))
                self.pw.data[0,0] = 0.25
                self.pb = nn.Parameter(torch.zeros([1]))
                
            else:
                self.pw = nn.Parameter(torch.ones([self.num_comb, 2]))
                self.pw.data[:,0] = 0.25
                self.pb = nn.Parameter(torch.zeros([self.num_comb]))

        elif self.phi_type == 'gauss-mlp':
            self.w = nn.Parameter(torch.zeros([self.latent_dim, self.latent_dim, 2, self.num_comb]))
            self.zw = nn.Parameter(torch.zeros([self.num_groups, self.latent_dim, 2]))

        if 'mlp' in self.phi_type: 
            self.phi_nets = nn.ModuleList()
            if self.phi_share:
                self.phi_nets.append(Net(1, self.psi_hidden_dims, output_dim=1, activation=self.activation, norm=None, dropout=nn.Dropout(p=self.dropout_rate)))
            else:
                for i in range(self.num_comb):
                    self.phi_nets.append(Net(1, self.psi_hidden_dims, output_dim=1, activation=self.activation, norm=None, dropout=nn.Dropout(p=self.dropout_rate)))

        if self.use_adv:
            self.panel_grl = GradReverse(self.adv_weight)
            self.panel_discriminator = Net(
                self.hidden_dims[-1], self.proj_hidden_dims,
                output_dim=self.n_panels,
                activation=self.activation, 
                norm=None, 
                dropout=nn.Dropout(p=self.dropout_rate)
            )
        
            # Slide adversary optional and only if meaningful
            if self.use_slide_adv and (self.n_slides != self.n_panels):
                self.slide_grl = GradReverse(0.1 * self.adv_weight)
                self.slide_discriminator = Net(
                    self.hidden_dims[-1], self.proj_hidden_dims,
                    output_dim=self.n_slides,
                    activation=self.activation,
                    norm=None, 
                    dropout=nn.Dropout(p=self.dropout_rate)
                )
            else:
                self.slide_grl = None
                self.slide_discriminator = None
        else:
            self.panel_grl = None
            self.panel_discriminator = None
            self.slide_grl = None
            self.slide_discriminator = None
            
        self.b = nn.Parameter(torch.zeros(1))

        self.gcarl_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.state_criterion = nn.CrossEntropyLoss(reduction='mean')

        self.initialize(self.seed)


    def initialize(self, seed=42):
        torch.manual_seed(seed)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if isinstance(self.activation, nn.LeakyReLU):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=self.activation.negative_slope)
                elif isinstance(self.activation, nn.ReLU):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias != None:
                    nn.init.constant_(m.bias, 0.0)

        if 'mlp' in self.phi_type:
            for m in self.phi_nets.modules():
                if isinstance(self.activation, nn.LeakyReLU):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=self.activation.negative_slope)
                    m.weight.data *= 0.1
                elif isinstance(self.activation, nn.ReLU):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    m.weight.data *= 0.1
                else:
                    nn.init.xavier_normal_(m.weight)
                    m.weight.data *= 0.1
                if m.bias != None:
                    nn.init.constant_(m.bias, 0.0)

    def _compute_grl_lambda(self):
        if self._trainer is not None and self.training:
            return compute_lambda(self.global_step, self.warmup_steps, self.trainer.estimated_stepping_batches)
        return 1.0

    def _encode_projected(self, hz):
        B = hz.size(0)

        if self.pool_neighbors:
            hz_gcarl = torch.cat([
                self.cell_net(hz[:, 0, :]).unsqueeze(1),
                self.niche_net(hz[:, 2:, :]).unsqueeze(1)
            ], dim=1)
        else:
            hz_gcarl = torch.cat([
                self.cell_net(hz[:, 0, :]).unsqueeze(1),
                self.niche_net(hz[:, 2:, :].reshape((B, self.n_neighbors * self.hz_dim))).unsqueeze(1)
            ], dim=1)

        hz_state = F.normalize(self.state_net(hz[:, :2, :]), dim=2)

        batch_logits = None
        if self.use_adv:
            grl_lam = self._compute_grl_lambda()
            batch_logits = [self.panel_discriminator(self.panel_grl(hz[:,0,:], grl_lam))]
            if self.slide_discriminator is not None:
                batch_logits.append(self.slide_discriminator(self.slide_grl(hz[:,0,:], grl_lam)))

        return hz_gcarl, hz_state, batch_logits

    def encode(self, x):
        hz = self.backbone(x)
        return self._encode_projected(hz)

    def _sparse_log1p(self, x: torch.Tensor) -> torch.Tensor:
        x = x.coalesce()
        return torch.sparse_coo_tensor(
            x.indices(),
            torch.log1p(x.values()),
            x.shape,
            device=x.device,
            dtype=x.dtype,
        ).coalesce()

    def _project_sparse_expression(self, x: torch.Tensor, batch_size: int, n_channels: int) -> torch.Tensor:
        x = x.coalesce()
        modules = list(self.backbone.net._modules.items())
        input_layer = modules[0][1]
        if not isinstance(input_layer, nn.Linear):
            raise TypeError("Sparse expression path expects backbone.net's first module to be nn.Linear.")

        h = torch.sparse.mm(x, input_layer.weight.t())
        if input_layer.bias is not None:
            h = h + input_layer.bias

        for _, module in modules[1:]:
            h = module(h)

        return h.reshape(batch_size, n_channels, self.hz_dim)

    def _sparse_rows(self, x: torch.Tensor, rows: torch.Tensor, n_rows: int) -> torch.Tensor:
        x = x.coalesce()
        idx = x.indices()
        vals = x.values()
        row_map = torch.full((x.size(0),), -1, device=x.device, dtype=torch.long)
        row_map[rows] = torch.arange(n_rows, device=x.device)
        new_rows = row_map[idx[0]]
        keep = new_rows >= 0
        new_idx = torch.stack([new_rows[keep], idx[1, keep]], dim=0)
        return torch.sparse_coo_tensor(new_idx, vals[keep], (n_rows, x.size(1)), device=x.device).coalesce()

    def _scale_sparse_rows(self, x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        x = x.coalesce()
        idx = x.indices()
        vals = x.values() * scales[idx[0]].to(x.values().dtype)
        return torch.sparse_coo_tensor(idx, vals, x.shape, device=x.device, dtype=x.dtype).coalesce()

    def _mask_sparse_values(self, x: torch.Tensor, mask_prop: float, panel_type=None) -> torch.Tensor:
        x = x.coalesce()
        idx = x.indices()
        vals = x.values()
        keep = torch.rand(vals.shape, device=x.device) > mask_prop
        if panel_type is not None:
            keep = keep & self.panel_mask[panel_type, idx[1]].to(x.device)
        return torch.sparse_coo_tensor(idx[:, keep], vals[keep], x.shape, device=x.device, dtype=x.dtype).coalesce()

    def sparse_augment(self, x: torch.Tensor, batch_size: int, obs_feature_groups=None, validation: bool = False):
        K = self.n_neighbors + 1
        C = x.size(1)
        device = x.device

        aug_probs = torch.ones(3, device=device) / 3
        aug_type = torch.multinomial(aug_probs, 2, replacement=True)
        t = aug_type[0]

        panel_types = None
        if self.n_panels > 1:
            panel_probs = torch.ones(self.n_panels, device=device) / self.n_panels
            panel_types = torch.multinomial(panel_probs, 2, replacement=False)

        drop_rate = np.random.beta(self.mask_prop * 9 / (1 - self.mask_prop), 9)

        rows = torch.arange(batch_size * K, device=device)
        channel = rows.remainder(K)
        cell_rows = rows[channel == 0]

        x = x.coalesce()
        totals = torch.sparse.sum(x, dim=1).to_dense().clamp_min(1)

        if validation:
            shuffled = totals.clone()
            shuffled[cell_rows] = totals[cell_rows[torch.randperm(batch_size, device=device)]]
            cell_hat = self._sparse_rows(x, cell_rows, batch_size)
            cell_scales = shuffled[cell_rows] / totals[cell_rows]
            cell_hat = self._scale_sparse_rows(cell_hat, cell_scales)
        else:
            shuffled = totals.clone()
            shuffled[cell_rows] = totals[cell_rows[torch.randperm(batch_size, device=device)]]
            if K > 1:
                batch_perm = torch.randperm(batch_size, device=device)
                channel_perm = torch.randperm(K - 1, device=device) + 1
                for src_c, dst_c in enumerate(channel_perm, start=1):
                    dst_rows = torch.arange(batch_size, device=device) * K + dst_c
                    src_rows = batch_perm * K + src_c
                    shuffled[dst_rows] = totals[src_rows]
            x = self._scale_sparse_rows(x, shuffled / totals)
            cell_hat = self._sparse_rows(x, cell_rows, batch_size)

        if t == 1:
            mask_prop = drop_rate / 2
        elif t == 2:
            mask_prop = 0
        else:
            mask_prop = drop_rate

        if t != 2:
            x_panel = panel_types[0] if panel_types is not None else None
            cell_panel = panel_types[1] if panel_types is not None else None
            if not validation:
                x = self._mask_sparse_values(x, mask_prop, panel_type=x_panel)
            cell_hat = self._mask_sparse_values(cell_hat, mask_prop, panel_type=cell_panel)

        if t == 1:
            lam = 1 - drop_rate / 2
        elif t == 2:
            lam = 1 - drop_rate
        else:
            lam = 1

        if t != 0:
            x = x.coalesce()
            cell_hat = cell_hat.coalesce()
            if not validation:
                x = torch.sparse_coo_tensor(x.indices(), torch.poisson(lam * x.values()), x.shape, device=device, dtype=x.dtype).coalesce()
            cell_hat = torch.sparse_coo_tensor(cell_hat.indices(), torch.poisson(lam * cell_hat.values()), cell_hat.shape, device=device, dtype=cell_hat.dtype).coalesce()

        x = x.coalesce()
        cell_hat = cell_hat.coalesce()
        idx = x.indices()
        vals = x.values()
        out_rows = (idx[0] // K) * (K + 1) + idx[0].remainder(K)
        cell_hat_idx = cell_hat.indices()
        cell_hat_rows = cell_hat_idx[0] * (K + 1) + 1
        out_idx = torch.cat([
            torch.stack([out_rows, idx[1]], dim=0),
            torch.stack([cell_hat_rows, cell_hat_idx[1]], dim=0),
        ], dim=1)
        out_vals = torch.cat([vals, cell_hat.values()], dim=0)
        out = torch.sparse_coo_tensor(out_idx, out_vals, (batch_size * (K + 1), C), device=device, dtype=x.dtype).coalesce()

        masked_obs_feature_groups = self._mask_obs_feature_groups(obs_feature_groups, mask_prop)
        return out, masked_obs_feature_groups

    def _mask_obs_feature_groups(self, obs_feature_groups, mask_prop):
        if obs_feature_groups is None:
            return None

        masked_groups = {}
        for group_name, group_batch in obs_feature_groups.items():
            cell_feat = group_batch["cell"]
            neigh_feat = group_batch["neighbors"]

            keep_cell = (torch.rand((1, cell_feat.shape[1]), device=cell_feat.device) > mask_prop)
            keep_neigh = (torch.rand((1, neigh_feat.shape[1], neigh_feat.shape[2]), device=neigh_feat.device) > mask_prop)

            masked_group = dict(group_batch)
            masked_group["cell"] = keep_cell * cell_feat
            masked_group["neighbors"] = keep_neigh * neigh_feat
            masked_groups[group_name] = masked_group

        return masked_groups

    def augment(self, x, obs_feature_groups=None):
        B, K, C = x.size()
        
        cell_hat = x[:,0,:].clone()

        aug_probs = torch.ones(3, device=x.device) / 3
        aug_type = torch.multinomial(aug_probs, 2, replacement=True)
        
        t = aug_type[0]

        if self.n_panels > 1:
            panel_probs = torch.ones(self.n_panels, device=x.device) / self.n_panels
            panel_types = torch.multinomial(panel_probs, 2, replacement=False)

        ## sample drop_rate from beta distribution with expected value of mask_prop
        drop_rate = np.random.beta(self.mask_prop * 9 / (1 - self.mask_prop), 9)

        # counts: [B, G]
        totals = x.sum(dim=2, keepdim=True)
        props = x / totals.clamp_min(1)
        
        shuffled_totals = torch.cat([totals[torch.randperm(B),:1,:], totals[torch.randperm(B),1:,:][:,torch.randperm(K-1)]], dim=1)
        x = props * shuffled_totals

        # Masking
        if t == 1:
            mask_prop = drop_rate / 2
        elif t == 2:
            mask_prop = 0
        else:
            mask_prop = drop_rate

        if t != 2:
            if self.n_panels > 1:
                panel_mask1 = self.panel_mask[panel_types[0]].reshape((1,1,C)).to(x.device)
                keep1 = panel_mask1 * (torch.rand((1, K, C), device=x.device) > mask_prop)
            else:
                keep1 = (torch.rand((1, K, C), device=x.device) > mask_prop)
                
            x = keep1 * x
    
            if self.n_panels > 1:
                panel_mask2 = self.panel_mask[panel_types[1]].reshape((1,C)).to(x.device)
                keep2 = panel_mask2 * (torch.rand((1, C), device=x.device) > mask_prop)
            else:
                keep2 = (torch.rand((1, C), device=x.device) > mask_prop)
                
            cell_hat = keep2 * cell_hat

        if t == 1:
            lam = 1 - drop_rate / 2
        elif t == 2:
            lam = 1 - drop_rate
        else:
            lam = 1

        # Concatenate ahead of poisson sampling
        x_hat = torch.cat([x[:,:1,:], cell_hat.unsqueeze(1), x[:,1:,:]], dim=1)
        if t != 0:
            x_hat = torch.poisson(lam * x_hat)
        
        masked_obs_feature_groups = self._mask_obs_feature_groups(obs_feature_groups, mask_prop)
        
        return x_hat, masked_obs_feature_groups

    def validation_augment(self, x, obs_feature_groups=None):
        B, K, C = x.size()
        
        cell_hat = x[:,0,:].clone() # + 1e-4
        
        aug_probs = torch.ones(3, device=x.device) / 3
        aug_type = torch.multinomial(aug_probs, 2, replacement=True)
        
        t = aug_type[0]

        if self.n_panels > 1:
            panel_probs = torch.ones(self.n_panels, device=x.device) / self.n_panels
            panel_types = torch.multinomial(panel_probs, 2, replacement=False)

        ## sample drop_rate from beta distribution with expected value of mask_prop
        drop_rate = np.random.beta(self.mask_prop * 9 / (1 - self.mask_prop), 9)

        # counts: [B, G]
        totals = cell_hat.sum(dim=1, keepdim=True)
        props = cell_hat / totals.clamp_min(1)
        
        shuffled_totals = totals[torch.randperm(B)]
        cell_hat = props * shuffled_totals

        # Masking
        if t == 1:
            mask_prop = drop_rate / 2
        elif t == 2:
            mask_prop = 0
        else:
            mask_prop = drop_rate

        if t != 2:
            if self.n_panels > 1:
                panel_mask2 = self.panel_mask[panel_types[1]].reshape((1,C)).to(x.device)
                keep2 = panel_mask2 * (torch.rand((1, C), device=x.device) > mask_prop)
            else:
                keep2 = (torch.rand((1, C), device=x.device) > mask_prop)
                
            cell_hat = keep2 * cell_hat

        if t == 1:
            lam = 1 - drop_rate / 2
        elif t == 2:
            lam = 1 - drop_rate
        else:
            lam = 1

        if t != 0:
            cell_hat = torch.poisson(lam * cell_hat)

        # Concatenate ahead of poisson sampling
        x_hat = torch.cat([x[:,:1,:], cell_hat.unsqueeze(1), x[:,1:,:]], dim=1)

        return x_hat, obs_feature_groups

    def _unpack_batch(self, batch):
        # expected from your DataModule
        adata_idx = batch["adata_idx"].long()
        if "x_sparse" in batch:
            cell = None
            neighbors = None
        else:
            cell = batch["cell"].float()          # [B, G]
            neighbors = batch["neighbors"].float()  # [B, K, G]
        panel = batch["panel"].long()
        slide_id = batch["slide_id"].long()
        obs_feature_groups = batch.get("obs_feature_groups", None)
        return adata_idx, cell, neighbors, panel, slide_id, obs_feature_groups

    def _encode_extra_groups(self, obs_feature_groups):
        if not self.extra_group_latent_dims:
            return None, {}
        if obs_feature_groups is None:
            raise ValueError(
                "Model was configured with extra_group_latent_dims but batch did not "
                "contain obs_feature_groups."
            )

        padded_latents = []
        raw_latents = {}
        for group_name in self.extra_group_names:
            if group_name not in obs_feature_groups:
                raise KeyError(f"Missing obs feature group in batch: {group_name!r}")

            group_batch = obs_feature_groups[group_name]
            cell_feat = group_batch["cell"].float()
            neigh_feat = group_batch["neighbors"].float()
            niche_feat = neigh_feat.reshape(neigh_feat.size(0), -1)

            group_latent = self.extra_group_nets[group_name](
                torch.cat([cell_feat, niche_feat], dim=1)
                # cell_feat
            )
            raw_latents[group_name] = group_latent

            latent_dim = group_latent.size(1)
            if latent_dim > self.latent_dim:
                raise ValueError(
                    f"extra group {group_name!r} latent dim ({latent_dim}) exceeds "
                    f"base latent_dim ({self.latent_dim})."
                )

            if latent_dim < self.latent_dim:
                pad = torch.zeros(
                    (group_latent.size(0), self.latent_dim - latent_dim),
                    dtype=group_latent.dtype,
                    device=group_latent.device,
                )
                group_latent = torch.cat([group_latent, pad], dim=1)

            padded_latents.append(group_latent.unsqueeze(1))

        return torch.cat(padded_latents, dim=1), raw_latents

    # def _compute_symmetry_penalty(self, z, z_nonlin, threshold=0.95):
    #     z_mse = z.pow(2).mean(dim=0)
    #     z = F.normalize(z, dim=0)
    #     z_nonlin = F.normalize(z_nonlin, dim=0)

    #     angle_hinge_loss = 0
    #     for c, (a, b) in enumerate(self.group_pair_indices):
    #         cos_sim_square = torch.sum(z[:,[a,b],:] * z_nonlin[:,:,:,c], dim=0).pow(2)
    #         angle_hinge_loss += (F.relu(cos_sim_square - threshold) * z_mse[[a,b],:]).mean()

    #     return angle_hinge_loss

    def _compute_symmetry_penalty(self, z, z_nonlin, threshold=0.9):
        B, G, D = z.shape
        
        mean = z.mean(dim=0, keepdim=True)
        z_centered = z - mean

        mean_nonlin = z_nonlin.mean(dim=0, keepdim=True)
        z_nonlin_centered = z_nonlin - mean_nonlin

        mean_loss = mean.pow(2).squeeze() # [G x D]
        
        std_z = torch.sqrt(z.var(dim=0) + 1e-4)

        var_loss = torch.clamp(1 - std_z, min=0) # [G x D]

        z_dir = F.normalize(z_centered, dim=0)
        z_nonlin_dir =F.normalize(z_nonlin_centered, dim=0)
        
        angle_hinge_loss = torch.zeros_like(var_loss)
        for c, (a,b) in enumerate(self.group_pair_indices):
            cos_sim_sq = torch.sum(z_dir[:,[a,b]] * z_nonlin_dir[...,c], dim=0).pow(2)
            angle_hinge_loss[[a,b]] += torch.clamp(cos_sim_sq - threshold, min=0).mean()

        return ((mean_loss + var_loss) * angle_hinge_loss).mean()


    def forward(self, x=None, slide_id=None, obs_feature_groups=None, projected_hz=None):
        """
        Args:
            x: Tensor of shape [B, 2 + n_neighbors, input_dim].
                Channel 0 is the center cell.
                Channel 1 is the augmented center-cell view.
                Channels 2: are spatial neighbors.
        """
        
        if projected_hz is None:
            B, G, C = x.size()

            expected_channels = self.n_neighbors + 2
            if G != expected_channels:
                raise ValueError(
                    f"Expected x to have {expected_channels} channels "
                    f"[cell, cell_hat, {self.n_neighbors} neighbors], got {G}."
                )

            hz, hz_state, batch_logits = self.encode(x)
        else:
            B, G, _ = projected_hz.size()
            expected_channels = self.n_neighbors + 2
            if G != expected_channels:
                raise ValueError(
                    f"Expected projected_hz to have {expected_channels} channels "
                    f"[cell, cell_hat, {self.n_neighbors} neighbors], got {G}."
                )
            hz, hz_state, batch_logits = self._encode_projected(projected_hz)
        extra_group_latents, _ = self._encode_extra_groups(obs_feature_groups)
        if extra_group_latents is not None:
            hz = torch.cat([hz, extra_group_latents], dim=1)

        source_idx, group_idx = generate_block_roll_negatives_index(B, self.num_groups, self.block_size, n_neg_batches=self.n_perms)

        logits = torch.zeros((self.n_perms+1) * B, device=hz.device)
        l1_loss = torch.zeros(1, device=hz.device)

        if self.phi_type=='gauss-maxout':
            h_nonlin, _ = torch.max(self.pw[None, None, None, :, :] * (hz[:,:,:,None,None] - self.pb[None, None, None,:,None]), dim=-1)
            
            h_nonlin = h_nonlin * self.group_latent_valid_mask[None,:,:,None]

            hz_nonlin = torch.empty((B, 2, self.latent_dim, self.num_comb), device=hz.device)
            
            for c, (a, b) in enumerate(self.group_pair_indices):
                
                if self.phi_share:
                    h_a = h_nonlin[:, a, :, 0]  # (N, D)
                    h_b = h_nonlin[:, b, :, 0]  # (N, D)
                else:
                    h_a = h_nonlin[:, a, :, c]  # (N, D)
                    h_b = h_nonlin[:, b, :, c]  # (N, D)

                hz_nonlin[:, 0, :, c] = h_a
                hz_nonlin[:, 1, :, c] = h_b

                z_a = hz[:, a, :]  # (N, D)
                z_b = hz[:, b, :]  # (N, D)

                # D_z_a = torch.diag(z_a.std(dim=0))
                # D_z_b = torch.diag(z_b.std(dim=0))

                # D_h_a = torch.diag(h_a.std(dim=0))
                # D_h_b = torch.diag(h_b.std(dim=0))

                h_a = torch.cat([h_a, h_a[source_idx[:,:,a]].reshape(self.n_perms * B, self.latent_dim)], dim=0)
                h_b = torch.cat([h_b, h_b[source_idx[:,:,b]].reshape(self.n_perms * B, self.latent_dim)], dim=0)

                z_a = torch.cat([z_a, z_a[source_idx[:,:,a]].reshape(self.n_perms * B, self.latent_dim)], dim=0)
                z_b = torch.cat([z_b, z_b[source_idx[:,:,b]].reshape(self.n_perms * B, self.latent_dim)], dim=0)
            
                W_ab = self.w[:, :, 0, c]  # (D, D)
                W_ba = self.w[:, :, 1, c]  # (D, D)
                
                # Equivalent to:
                # torch.sum(W_ab[None] * (h_a[:, :, None] * z_b[:, None, :]), dim=[1, 2])
                term_ab = (h_a @ W_ab * z_b).sum(dim=1)
            
                # Equivalent to:
                # torch.sum(W_ba[None] * (h_b[:, :, None] * z_a[:, None, :]), dim=[1, 2])
                term_ba = (h_b @ W_ba * z_a).sum(dim=1)
            
                logits = logits + term_ab + term_ba

                # l1_loss = l1_loss + smooth_abs(D_h_a @ W_ab @ D_z_b).mean() + smooth_abs(D_h_b @ W_ba @ D_z_a).mean()
                
            # psi_hz = []
            psi_hz = torch.zeros_like(hz)
            for m in range(self.num_groups):
                psi_hz[:,m,self.group_latent_valid_mask[m]] = self.psi_nets[m](hz[:,m,:])[:,self.group_latent_valid_mask[m]]

            psi_hz = torch.cat([psi_hz, psi_hz[source_idx, [m for m in range(self.num_groups)]].reshape(self.n_perms * B, self.num_groups, self.latent_dim)], dim=0)

            logits_z = torch.sum(self.zw[None, :, :, 0] * psi_hz ** 2 + self.zw[None, :, :, 1] * psi_hz, dim=[1, 2])

            logits += logits_z + self.b

        if self.phi_type=='gauss-mlp':

            hz_nonlin = torch.empty((B, 2, self.latent_dim, self.num_comb), device=hz.device)

            if self.phi_share:
                mask = self.group_latent_valid_mask

                hz_flat = hz.reshape(-1, self.num_groups * self.latent_dim)
                mask_flat = mask.reshape(self.num_groups * self.latent_dim)
                
                hp_valid = self.phi_nets[0](hz_flat[:,mask_flat].reshape(-1,1)).reshape(hz.shape[0],-1)
                
                hp_flat = torch.zeros_like(hz_flat)
                hp_flat[:, mask_flat] = hp_valid

                hp = hp_flat.reshape_as(hz)

            for c, (a, b) in enumerate(self.group_pair_indices):
                
                if self.phi_share:
                    hp_c = hp[:,[a,b],:]
                else:
                    hz_c = hz[:,[a,b],:]
                    
                    mask_c = self.group_latent_valid_mask[[a,b]]

                    hz_c_flat = hz_c.reshape(-1, 2 * self.latent_dim)
                    mask_c_flat = mask_c.reshape(2 * self.latent_dim)
                    
                    hp_c_valid = self.phi_nets[c](hz_c_flat[:,mask_c_flat].reshape(-1,1)).reshape(hz_c.shape[0],-1)
                    
                    hp_c_flat = torch.zeros_like(hz_c_flat)
                    hp_c_flat[:, mask_c_flat] = hp_c_valid
    
                    hp_c = hp_c_flat.reshape_as(hz_c)

                h_a = hp_c[:, 0, :]  # (N, D)
                h_b = hp_c[:, 1, :]  # (N, D)

                hz_nonlin[:, 0, :, c] = h_a
                hz_nonlin[:, 1, :, c] = h_b
            
                z_a = hz[:, a, :]  # (N, D)
                z_b = hz[:, b, :]  # (N, D)

                # D_z_a = torch.diag(z_a.std(dim=0))
                # D_z_b = torch.diag(z_b.std(dim=0))

                # D_h_a = torch.diag(h_a.std(dim=0))
                # D_h_b = torch.diag(h_b.std(dim=0))

                h_a = torch.cat([h_a, h_a[source_idx[:,:,a]].reshape(self.n_perms * B, self.latent_dim)], dim=0)
                h_b = torch.cat([h_b, h_b[source_idx[:,:,b]].reshape(self.n_perms * B, self.latent_dim)], dim=0)

                z_a = torch.cat([z_a, z_a[source_idx[:,:,a]].reshape(self.n_perms * B, self.latent_dim)], dim=0)
                z_b = torch.cat([z_b, z_b[source_idx[:,:,b]].reshape(self.n_perms * B, self.latent_dim)], dim=0)
            
                W_ab = self.w[:, :, 0, c]  # (D, D)
                W_ba = self.w[:, :, 1, c]  # (D, D)
            
                # Equivalent to:
                # torch.sum(W_ab[None] * (h_a[:, :, None] * z_b[:, None, :]), dim=[1, 2])
                term_ab = (h_a @ W_ab * z_b).sum(dim=1)
            
                # Equivalent to:
                # torch.sum(W_ba[None] * (h_b[:, :, None] * z_a[:, None, :]), dim=[1, 2])
                term_ba = (h_b @ W_ba * z_a).sum(dim=1)
            
                logits = logits + term_ab + term_ba

                # l1_loss = l1_loss + smooth_abs(D_h_a @ W_ab @ D_z_b).mean() + smooth_abs(D_h_b @ W_ba @ D_z_a).mean()
                
            # psi_hz = []
            psi_hz = torch.zeros_like(hz)
            for m in range(self.num_groups):
                if self.phi_share:
                    # psi_hz.append(self.psi_nets[m](torch.cat([hz[:,m,:], hp[:,m,:]], dim=1)).unsqueeze(1))
                    psi_hz[:,m,self.group_latent_valid_mask[m]] = self.psi_nets[m](torch.cat([hz[:,m,:], hp[:,m,:]], dim=1))[:,self.group_latent_valid_mask[m]]
                else:
                    # psi_hz.append(self.psi_nets[m](torch.cat([hz[:,m,:], F.tanh(hz[:,m,:])], dim=1)).unsqueeze(1))
                    psi_hz[:,m,self.group_latent_valid_mask[m]] = self.psi_nets[m](torch.cat([hz[:,m,:], F.tanh(hz[:,m,:])], dim=1))[:,self.group_latent_valid_mask[m]]

            # psi_hz = torch.cat(psi_hz, dim=1)

            psi_hz = torch.cat([psi_hz, psi_hz[source_idx, [m for m in range(self.num_groups)]].reshape(self.n_perms * B, self.num_groups, self.latent_dim)], dim=0)

            logits_z = torch.sum(self.zw[None, :, :, 0] * psi_hz ** 2 + self.zw[None, :, :, 1] * psi_hz, dim=[1, 2])

            logits += logits_z + self.b

        sym_loss = self._compute_symmetry_penalty(hz, hz_nonlin)
        
        return [logits, hz_state[:,0,:] @ hz_state[:,1,:].T / torch.exp(self.tau)], batch_logits, sym_loss

    def training_step(self, train_batch, batch_idx):
        adata_idx, cell, neighbors, panel, slide_id, obs_feature_groups = self._unpack_batch(train_batch)

        if "x_sparse" in train_batch:
            x_sparse = train_batch["x_sparse"].float()
            B = train_batch["x_sparse_shape"][0]
            x_in, obs_feature_groups_aug = self.sparse_augment(x_sparse, B, obs_feature_groups=obs_feature_groups, validation=False)
            projected_hz = self._project_sparse_expression(self._sparse_log1p(x_in), B, self.n_neighbors + 2)
            logits, batch_logits, sym_loss = self(projected_hz=projected_hz, slide_id=slide_id, obs_feature_groups=obs_feature_groups_aug)
        else:
            assert neighbors.size(1) == self.n_neighbors
            B, num_genes = cell.size()
            x_orig = torch.cat([cell.unsqueeze(1), neighbors], dim=1)
            x_in, obs_feature_groups_aug = self.augment(x_orig, obs_feature_groups=obs_feature_groups)
            logits, batch_logits, sym_loss = self(x_in.log1p(), slide_id, obs_feature_groups=obs_feature_groups_aug)

        # -------- labels & loss --------

        gcarl_pos_labels = torch.ones(B, device=logits[0].device)
        gcarl_neg_labels = torch.zeros(self.n_perms * B, device=logits[0].device)

        gcarl_labels = torch.cat(
            [
                gcarl_pos_labels,
                gcarl_neg_labels
            ], 
            dim=0
        )
        
        labels = torch.cat([torch.arange(B, device=logits[0].device, dtype=torch.long)], dim=0)

        valid = slide_id[:, None].eq(slide_id[None, :])

        logits_state = logits[1].masked_fill(~valid, float("-inf"))

        state_loss = 0.5 * self.state_criterion(logits_state, labels) + 0.5 * self.state_criterion(logits_state.T, labels)

        adv_loss = torch.zeros((), device=self.device)
        if self.use_adv and (batch_logits is not None):
            panel_labels = panel # .unsqueeze(1) # .repeat(1, batch_logits[0].shape[1])
            adv_loss = self.state_criterion(batch_logits[0], panel_labels)

            if self.slide_discriminator is not None and len(batch_logits) > 1:
                slide_labels = slide_id # .unsqueeze(1) # .repeat(1, batch_logits[1].shape[1])  # probably slide_id, not panel
                adv_loss = adv_loss + self.state_criterion(batch_logits[1], slide_labels)

        gcarl_loss = 0.5 * self.gcarl_criterion(logits[0][:B], gcarl_pos_labels) + 0.5 * self.gcarl_criterion(logits[0][B:], gcarl_neg_labels)

        loss = gcarl_loss + self.state_lam * state_loss + self.sym_lam * sym_loss
        
        self.log('train_total_loss', loss, on_epoch=True)

        if self.use_adv:
            loss = loss + adv_loss

        self.log('train_loss', gcarl_loss, on_epoch=True)

        pred = (logits[0]>0).float()
        acc = (pred == gcarl_labels).float().mean()

        self.log('train_acc', torch.tensor([acc]), on_epoch=True)

        if self.log_sklearn_metrics:
            self.log('train_f1', torch.tensor([f1_score(gcarl_labels.detach().reshape(-1,1).cpu().numpy(), pred.reshape(-1,1).detach().cpu().numpy())]), on_epoch=True)
            self.log('train_mcc', torch.tensor([matthews_corrcoef(gcarl_labels.reshape(-1,1).detach().cpu().numpy(), pred.reshape(-1,1).detach().cpu().numpy())]), on_epoch=True)

        self.log('train_state_loss', state_loss, on_epoch=True)

        self.log('train_symmetry', sym_loss, on_epoch=True)

        # self.log('train_mean', mean_loss, on_epoch=True)
        
        # self.log('train_var', var_loss, on_epoch=True)

        # self.log('train_l1', l1_loss, on_epoch=True)

        _, pred0 = torch.max(logits[1], dim=0)
        _, pred1 = torch.max(logits[1], dim=1)
        acc = ((pred0 == labels).float().mean() + (pred1 == labels).float().mean()) / 2.0

        self.log('train_state_acc', torch.tensor([acc]), on_epoch=True)

        if self.log_sklearn_metrics:
            self.log('train_state_f1', torch.tensor([f1_score(labels.detach().reshape(-1,1).cpu().numpy(), pred0.reshape(-1,1).detach().cpu().numpy(), average='macro')]), on_epoch=True)
            self.log('train_state_mcc', torch.tensor([matthews_corrcoef(labels.reshape(-1,1).detach().cpu().numpy(), pred0.reshape(-1,1).detach().cpu().numpy())]), on_epoch=True)

        if self.use_adv:
            self.log('train_batch_loss', adv_loss, on_epoch=True)

            _, pred = torch.max(batch_logits[0], dim=1)
            acc = (pred.reshape(-1,1) == panel_labels.reshape(-1,1)).float().mean()
        
            self.log('train_panel_acc', torch.tensor([acc]), on_epoch=True)

            if self.log_sklearn_metrics:
                self.log('train_panel_f1', torch.tensor([f1_score(panel_labels.detach().reshape(-1,1).cpu().numpy(), pred.reshape(-1,1).detach().cpu().numpy(), average='macro')]), on_epoch=True)
                self.log('train_panel_mcc', torch.tensor([matthews_corrcoef(panel_labels.reshape(-1,1).detach().cpu().numpy(), pred.reshape(-1,1).detach().cpu().numpy())]), on_epoch=True)

            if self.slide_discriminator != None:
                _, pred = torch.max(batch_logits[1], dim=1)
                acc = (pred.reshape(-1,1) == slide_labels.reshape(-1,1)).float().mean()
            
                self.log('train_slide_acc', torch.tensor([acc]), on_epoch=True)

                if self.log_sklearn_metrics:
                    self.log('train_slide_f1', torch.tensor([f1_score(slide_labels.detach().reshape(-1,1).cpu().numpy(), pred.reshape(-1,1).detach().cpu().numpy(), average='macro')]), on_epoch=True)
                    self.log('train_slide_mcc', torch.tensor([matthews_corrcoef(slide_labels.reshape(-1,1).detach().cpu().numpy(), pred.reshape(-1,1).detach().cpu().numpy())]), on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        adata_idx, cell, neighbors, panel, slide_id, obs_feature_groups = self._unpack_batch(valid_batch)

        if "x_sparse" in valid_batch:
            x_sparse = valid_batch["x_sparse"].float()
            B = valid_batch["x_sparse_shape"][0]
            x_in, obs_feature_groups_aug = self.sparse_augment(x_sparse, B, obs_feature_groups=obs_feature_groups, validation=True)
            projected_hz = self._project_sparse_expression(self._sparse_log1p(x_in), B, self.n_neighbors + 2)
            logits, batch_logits, sym_loss = self(projected_hz=projected_hz, slide_id=slide_id, obs_feature_groups=obs_feature_groups_aug)
        else:
            assert neighbors.size(1) == self.n_neighbors
            B, num_genes = cell.size()
            x_orig = torch.cat([cell.unsqueeze(1), neighbors], dim=1)
            x_in, obs_feature_groups_aug = self.validation_augment(x_orig, obs_feature_groups=obs_feature_groups)
            logits, batch_logits, sym_loss = self(x_in.log1p(), slide_id, obs_feature_groups=obs_feature_groups_aug)

        # -------- labels & loss --------

        gcarl_pos_labels = torch.ones(B, device=logits[0].device)
        gcarl_neg_labels = torch.zeros(self.n_perms * B, device=logits[0].device)

        gcarl_labels = torch.cat(
            [
                gcarl_pos_labels,
                gcarl_neg_labels
            ], 
            dim=0
        )
        
        labels = torch.cat([torch.arange(B, device=logits[0].device, dtype=torch.long)], dim=0)

        valid = slide_id[:, None].eq(slide_id[None, :])

        logits_state = logits[1].masked_fill(~valid, float("-inf"))

        state_loss = 0.5 * self.state_criterion(logits_state, labels) + 0.5 * self.state_criterion(logits_state.T, labels)

        adv_loss = torch.zeros((), device=self.device)
        if self.use_adv and (batch_logits is not None):
            panel_labels = panel # .unsqueeze(1) # .repeat(1, batch_logits[0].shape[1])
            adv_loss = self.state_criterion(batch_logits[0], panel_labels)

            if self.slide_discriminator is not None and len(batch_logits) > 1:
                slide_labels = slide_id # .unsqueeze(1) # .repeat(1, batch_logits[1].shape[1])  # probably slide_id, not panel
                adv_loss = adv_loss + self.state_criterion(batch_logits[1], slide_labels)

        # gcarl_loss = self.gcarl_criterion(logits[0], gcarl_labels)

        gcarl_loss = 0.5 * self.gcarl_criterion(logits[0][:B], gcarl_pos_labels) + 0.5 * self.gcarl_criterion(logits[0][B:], gcarl_neg_labels)

        # gcarl_loss = self.gcarl_criterion(logits[0], gcarl_labels)

        # l1_loss = smooth_abs(self.w).sum()

        loss = gcarl_loss + self.state_lam * state_loss + self.sym_lam * sym_loss
        
        self.log('valid_total_loss', loss, on_epoch=True)

        if self.use_adv:
            loss = loss + adv_loss

        self.log('valid_loss', gcarl_loss, on_epoch=True)

        pred = (logits[0]>0).float()
        acc = (pred == gcarl_labels).float().mean()

        self.log('valid_acc', torch.tensor([acc]), on_epoch=True)

        if self.log_sklearn_metrics:
            self.log('valid_f1', torch.tensor([f1_score(gcarl_labels.detach().reshape(-1,1).cpu().numpy(), pred.reshape(-1,1).detach().cpu().numpy())]), on_epoch=True)
            self.log('valid_mcc', torch.tensor([matthews_corrcoef(gcarl_labels.reshape(-1,1).detach().cpu().numpy(), pred.reshape(-1,1).detach().cpu().numpy())]), on_epoch=True)

        self.log('valid_symmetry', sym_loss, on_epoch=True)

        # self.log('valid_mean', mean_loss, on_epoch=True)
        
        # self.log('valid_var', var_loss, on_epoch=True)

        self.log('valid_state_loss', state_loss, on_epoch=True)

        # self.log('valid_l1', dir_loss, on_epoch=True)

        _, pred0 = torch.max(logits[1], dim=0)
        _, pred1 = torch.max(logits[1], dim=1)
        acc = ((pred0 == labels).float().mean() + (pred1 == labels).float().mean()) / 2.0

        self.log('valid_state_acc', torch.tensor([acc]), on_epoch=True)

        if self.log_sklearn_metrics:
            self.log('valid_state_f1', torch.tensor([f1_score(labels.detach().reshape(-1,1).cpu().numpy(), pred0.reshape(-1,1).detach().cpu().numpy(), average='macro')]), on_epoch=True)
            self.log('valid_state_mcc', torch.tensor([matthews_corrcoef(labels.reshape(-1,1).detach().cpu().numpy(), pred0.reshape(-1,1).detach().cpu().numpy())]), on_epoch=True)

        # Extract and log the current learning rate
        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        
        # on_epoch=True ensures the EarlyStopping callback can check it at epoch end
        self.log("current_lr", current_lr, on_epoch=True, sync_dist=True)

        if self.use_adv:
            self.log('valid_batch_loss', adv_loss, on_epoch=True)

            _, pred = torch.max(batch_logits[0], dim=1)
            acc = (pred.reshape(-1,1) == panel_labels.reshape(-1,1)).float().mean()
        
            self.log('valid_panel_acc', torch.tensor([acc]), on_epoch=True)

            if self.log_sklearn_metrics:
                self.log('valid_panel_f1', torch.tensor([f1_score(panel_labels.detach().reshape(-1,1).cpu().numpy(), pred.reshape(-1,1).detach().cpu().numpy(), average='macro')]), on_epoch=True)
                self.log('valid_panel_mcc', torch.tensor([matthews_corrcoef(panel_labels.reshape(-1,1).detach().cpu().numpy(), pred.reshape(-1,1).detach().cpu().numpy())]), on_epoch=True)

            if self.slide_discriminator != None:
                _, pred = torch.max(batch_logits[1], dim=1)
                acc = (pred.reshape(-1,1) == slide_labels.reshape(-1,1)).float().mean()
            
                self.log('valid_slide_acc', torch.tensor([acc]), on_epoch=True)

                if self.log_sklearn_metrics:
                    self.log('valid_slide_f1', torch.tensor([f1_score(slide_labels.detach().reshape(-1,1).cpu().numpy(), pred.reshape(-1,1).detach().cpu().numpy(), average='macro')]), on_epoch=True)
                    self.log('valid_slide_mcc', torch.tensor([matthews_corrcoef(slide_labels.reshape(-1,1).detach().cpu().numpy(), pred.reshape(-1,1).detach().cpu().numpy())]), on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        adata_idx, cell, neighbors, panel, slide_id, obs_feature_groups = self._unpack_batch(batch)
        
        # No augmentation by default in predict
        x_in = torch.cat([cell.unsqueeze(1), cell.unsqueeze(1), neighbors], dim=1)  # [B, 2+K, G]
        hz_gcarl, hz_state, _ = self.encode(x_in.log1p())
        
        cell_emb, niche_emb = hz_gcarl.split(dim=1, split_size=1)
        
        return {
            "adata_idx": adata_idx.detach().cpu(),
            "state_emb": hz_state[:, 0, :].detach().cpu(),
            "cell_emb": cell_emb.squeeze(1).detach().cpu(),
            "niche_emb": niche_emb.squeeze(1).detach().cpu(),
            "extra_group_emb": {
                group_name: latents.detach().cpu()
                for group_name, latents in self._encode_extra_groups(obs_feature_groups)[1].items()
            },
        }

    def configure_optimizers(self):

        if self.opt_algo=='sgd':
            optimizer = torch.optim.SGD(self.parameters(), weight_decay=self.weight_decay, lr=self.learning_rate, momentum=0.9, nesterov=False)
            
            # scheduler_config = {
            #     "scheduler" : torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6 * self.max_epochs), int(0.9 * self.max_epochs)], gamma=0.1),
            #     "interval": "epoch",
            # }

            # # 1. Warmup: Step-based linear increase
            warmup_sch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=self.warmup_steps, last_epoch=-1)
            
            # 2. Plateau: Epoch-based metric monitoring
            plateau_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=self.patience, factor=0.5, threshold=0
            )

            scheduler_config = [
                {"scheduler": warmup_sch, "interval": "step"},
                {"scheduler": plateau_sch, "interval": "epoch", "monitor": "valid_loss"}
            ]
            
        elif self.opt_algo=='adamw':
            optimizer = torch.optim.AdamW(self.parameters(), weight_decay=self.weight_decay, lr=self.learning_rate)

            # # 1. Warmup: Step-based linear increase
            warmup_sch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=self.warmup_steps, last_epoch=-1)
            
            # 2. Plateau: Epoch-based metric monitoring
            plateau_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=self.patience, factor=0.5, threshold=0
            )

            scheduler_config = [
                {"scheduler": warmup_sch, "interval": "step"},
                {"scheduler": plateau_sch, "interval": "epoch", "monitor": "valid_loss"}
            ]

            # warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=self.warmup_steps, last_epoch=-1)
            # decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.estimated_stepping_batches - 2 * self.warmup_steps, eta_min=0.0001 * self.learning_rate)
    
            # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, decay], milestones=[self.warmup_steps])

            # scheduler_config = {
            #     "scheduler": scheduler,
            #     "interval": "step", 
            # }
            
        elif self.opt_algo=='radam':
            optimizer = torch.optim.RAdam(self.parameters(), weight_decay=self.weight_decay, lr=self.learning_rate, decoupled_weight_decay=True)

            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=self.warmup_steps, last_epoch=-1)
            decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.estimated_stepping_batches - 2 * self.warmup_steps, eta_min=0.0001 * self.learning_rate)
    
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, decay], milestones=[self.warmup_steps])

            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step", 
            }

        return [optimizer], scheduler_config
