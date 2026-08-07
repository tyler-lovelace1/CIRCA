import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class GradReverse(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor, alph: float = 1.0) -> torch.Tensor:
        return GradReverseFn.apply(x, self.lambd * alph)

class NeighborhoodProjectPool(nn.Module):
    def __init__(
        self,
        input_dim: int,
        project_dim: int = 32,
        summary_dim: int = 64,
        weighted: bool = True,
        beta_init: float = 0.5,
        beta_max: float = 2.0,
        include_max: bool = True,
        include_min: bool = False,
        activation = nn.ReLU(),
        norm = nn.LayerNorm,
    ):
        """
        Hybrid rank-aware neighborhood pooling.

        For x with shape (B, K, input_dim), returns:
            projected weighted mean pool
          + optional projected max pool
          + optional projected min pool
          + concatenated projected neighbor embeddings

        Output dimension:
            n_summaries * summary_dim + K * project_dim

        Example with K=10, input_dim=256, project_dim=32, summary_dim=64:
            weighted mean + max + projected concat
            output_dim = 2*64 + 10*32 = 448
        """
        super().__init__()

        if weighted and not (0.0 < beta_init < beta_max):
            raise ValueError("beta_init must be between 0 and beta_max.")

        self.input_dim = input_dim
        self.project_dim = project_dim
        self.summary_dim = summary_dim
        self.weighted = weighted
        self.beta_max = beta_max
        self.include_max = include_max
        self.include_min = include_min

        # Shared projection for each rank-ordered neighbor.
        self.neighbor_projector = nn.Sequential(
            nn.Linear(input_dim, project_dim),
            norm(project_dim),
            activation,
        )

        # Separate projections for pooled summaries.
        self.mean_projector = nn.Sequential(
            nn.Linear(input_dim, summary_dim),
            norm(summary_dim),
            activation,
        )

        if include_max:
            self.max_projector = nn.Sequential(
                nn.Linear(input_dim, summary_dim),
                norm(summary_dim),
                activation,
            )

        if include_min:
            self.min_projector = nn.Sequential(
                nn.Linear(input_dim, summary_dim),
                norm(summary_dim),
                activation,
            )

        if weighted:
            # Smooth bounded parameterization:
            # beta = beta_max * sigmoid(raw_beta)
            raw_init = math.log(beta_init / (beta_max - beta_init))
            self.raw_beta = nn.Parameter(torch.tensor([raw_init], dtype=torch.float32))

    def output_dim(self, k_neighbors: int) -> int:
        n_summaries = 1  # weighted mean
        if self.include_max:
            n_summaries += 1
        if self.include_min:
            n_summaries += 1

        return n_summaries * self.summary_dim + k_neighbors * self.project_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K, input_dim), where K neighbors are ordered by proximity/rank.

        Returns:
            pooled: (B, output_dim(K))
        """
        B, K, D = x.shape

        if D != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got D={D}.")

        if self.weighted:
            ranks = torch.arange(1, K + 1, device=x.device, dtype=x.dtype)
            beta = self.beta_max * torch.sigmoid(self.raw_beta).to(dtype=x.dtype)
            weights = ranks.pow(-beta)
        else:
            weights = torch.ones(K, device=x.device, dtype=x.dtype)

        weights = weights / weights.sum().clamp_min(1e-8)

        weighted_mean = (weights.view(1, K, 1) * x).sum(dim=1)
        mean_summary = self.mean_projector(weighted_mean)

        pieces = [mean_summary]

        if self.include_max:
            max_pool = x.max(dim=1).values
            pieces.append(self.max_projector(max_pool))

        if self.include_min:
            min_pool = x.min(dim=1).values
            pieces.append(self.min_projector(min_pool))

        neighbor_proj = self.neighbor_projector(x)  # (B, K, project_dim)
        neighbor_proj = neighbor_proj.reshape(B, K * self.project_dim)
        pieces.append(neighbor_proj)

        return torch.cat(pieces, dim=1)

class Net(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[10, 10, 10], output_dim=None, activation=nn.ReLU(), norm=nn.LayerNorm, dropout=None):
        super(Net, self).__init__()    
        
        assert len(hidden_dims) > 0

        net = []
    
        for i in range(len(hidden_dims)):
            if i == 0:
                net.append(("input_layer",  nn.Linear(input_dim, hidden_dims[i])))
            else:
                net.append(("hidden_layer{}".format(i),  nn.Linear(hidden_dims[i-1], hidden_dims[i])))
                
            if norm != None:
                net.append(("norm{}".format(i), norm(hidden_dims[i])))
            
            net.append(("activation{}".format(i), activation))

            if dropout != None:
                net.append(("dropout{}".format(i), dropout))

        if output_dim != None:
            net.append(("output_layer".format(i),  nn.Linear(hidden_dims[-1], output_dim)))
            
        self.net = nn.Sequential(OrderedDict(net))

    def forward(self, x):
        return self.net(x)

class PiecewiseLinear(nn.Module):
    def __init__(self, slopes, intercepts):
        super().__init__()
        # Initializing global scalar parameters
        self.m1 = nn.Parameter(torch.tensor(slopes[0]))
        self.m2 = nn.Parameter(torch.tensor(slopes[1]))
        self.m3 = nn.Parameter(torch.tensor(slopes[2]))
        self.b1 = nn.Parameter(torch.tensor(intercepts[0]))
        self.b2 = nn.Parameter(torch.tensor(intercepts[1]))

    def forward(self, x):
        # max(0, x - b) using torch.clamp
        ramp1 = torch.clamp(x - self.b1, min=0)
        ramp2 = torch.clamp(x - self.b2, min=0)
        
        # Calculate dynamic intercept to force y(0) = 0 even if b1 or b2 are negative
        intercept = -(self.m1 - self.m2) * torch.clamp(-self.b1, min=0) - (self.m2 - self.m3) * torch.clamp(-self.b2, min=0)
        
        y = self.m1 * x + (self.m1 - self.m2) * ramp1 + (self.m2 - self.m3) * ramp2 + intercept
        return y

class ChannelwisePiecewiseLinear(nn.Module):
    def __init__(self, slopes, intercepts, num_channels):
        super().__init__()
        self.num_channels = num_channels
        
        # Initialize parameters as vectors of size (C,)
        self.m1 = nn.Parameter(torch.full((1,1,1,num_channels), slopes[0]))
        self.m2 = nn.Parameter(torch.full((1,1,1,num_channels), slopes[1]))
        self.m3 = nn.Parameter(torch.full((1,1,1,num_channels), slopes[2]))
        
        # Initialize breakpoints (e.g., spaced out parameters)
        self.b1 = nn.Parameter(torch.full((1,1,1,num_channels), intercepts[0]))
        self.b2 = nn.Parameter(torch.full((1,1,1,num_channels), intercepts[1]))

    def forward(self, x):
        # x shape: [B, G, D, C]
        # self.b1 shape: [C] -> automatically broadcasts to [B, G, D, C]
        ramp1 = torch.clamp(x - self.b1, min=0)
        ramp2 = torch.clamp(x - self.b2, min=0)
        
        # Compute channel-specific zero-intercept corrections
        intercept = -(self.m2 - self.m1) * torch.clamp(-self.b1, min=0) - (self.m3 - self.m2) * torch.clamp(-self.b2, min=0)
        
        # Compute vectorized piecewise transformation
        y = self.m1 * x + (self.m2 - self.m1) * ramp1 + (self.m3 - self.m2) * ramp2 + intercept
        return y

