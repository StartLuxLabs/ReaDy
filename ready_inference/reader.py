"""Rank-eight recurrent-state reader; state axes are batch, layer, head, value, key."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class StateReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(5120, 2 * 48 * 8, bias=False)
        self.query_up = nn.Parameter(torch.randn(2, 48, 8, 128) / math.sqrt(8))
        self.value_down = nn.Parameter(torch.randn(2, 48, 128, 8) / math.sqrt(128))
        self.down = nn.Linear(2 * 48 * 9, 128, bias=False)
        self.up = nn.Linear(128, 5120, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden, state):
        assert hidden.ndim == 3 and state.shape == (hidden.shape[0], 2, 48, 128, 128)
        b, n, _ = hidden.shape
        sf = state.float()
        rms = (sf.square().mean((-2, -1), keepdim=True) + 1e-6).sqrt()
        normalized = (sf / rms).to(hidden.dtype)
        query = self.query(F.rms_norm(hidden.float(), (hidden.shape[-1],)).to(hidden.dtype))
        query = query.view(b, n, 2, 48, 8)
        compressed = torch.einsum('lhvr,blhvk->blhrk', self.value_down, normalized)
        key = torch.einsum('bnlhr,lhrk->bnlhk', query, self.query_up)
        read = torch.einsum('blhrk,bnlhk->bnlhr', compressed, key) / math.sqrt(128)
        scale = rms[..., 0, 0].log().to(hidden.dtype)[:, None, :, :, None].expand(b, n, 2, 48, 1)
        features = torch.cat((read, scale), dim=-1).flatten(2)
        features = F.rms_norm(features.float(), (features.shape[-1],)).to(hidden.dtype)
        return hidden + self.up(F.silu(self.down(features)))
