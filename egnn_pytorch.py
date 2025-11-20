import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

def exists(val):
    return val is not None

# 单层 Equivariant Transformer
class EquivariantTransformerLayer(nn.Module):
    def __init__(self, dim, edge_dim=0, fourier_features=4, num_heads=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.fourier_features = fourier_features

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # 用于结合边特征和距离编码
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + (fourier_features * 2 + 1 if fourier_features > 0 else 1), dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        ) if edge_dim > 0 or fourier_features > 0 else None

        # 坐标更新权重预测
        self.coord_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1)
        )

        # 门控机制，调节 delta 更新
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, feats, coors, edges=None, mask=None, adj_mat=None):
        B, N, D = feats.shape
        feats = self.norm1(feats)

        # relative position
        rel_coors = rearrange(coors, 'b i d -> b i 1 d') - rearrange(coors, 'b j d -> b 1 j d')
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)  # (B, N, N, 1)

        # distance encoding
        if self.fourier_features > 0:
            freqs = 2 ** torch.arange(self.fourier_features, device=coors.device, dtype=coors.dtype)
            freq = rel_dist / freqs.view(1, 1, 1, -1)
            enc = torch.cat([freq.sin(), freq.cos(), rel_dist], dim=-1)  # (B, N, N, F*2+1)
        else:
            enc = rel_dist

        # edge features + encoding
        if self.edge_mlp is not None and exists(edges):
            edge_feat = torch.cat([edges, enc], dim=-1)
            e_ij = self.edge_mlp(edge_feat)
        elif self.edge_mlp is not None:
            e_ij = self.edge_mlp(enc)
        else:
            e_ij = None

        # attention
        qkv = self.qkv_proj(feats).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        scores = einsum('b h i d, b h j d -> b h i j', q, k) / (D ** 0.5)

        if mask is not None:
            mask = rearrange(mask, 'b n -> b 1 1 n')
            scores.masked_fill_(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # message passing
        message = einsum('b h i j, b h j d -> b h i d', attn, v)
        message = rearrange(message, 'b h n d -> b n (h d)')
        feats = feats + self.out_proj(message)
        feats = feats + self.ffn(self.norm2(feats))

        # coordinate update
        if e_ij is not None:
            coord_weights = self.coord_mlp(e_ij).squeeze(-1)  # (B, N, N)
            coord_weights = coord_weights * attn.mean(1)  # mean over heads
        else:
            coord_weights = attn.mean(1)

        # 引入门控机制对 delta 进行调制
        gate = self.gate_mlp(feats)  # (B, N, 1)
        delta = einsum('b i j, b i j d -> b i d', coord_weights, rel_coors)
        delta = delta * gate  # 应用门控

        coors = coors + delta

        return feats, coors


# 多层堆叠模块，用于整体替换原始 EGNN 类
class EquivariantTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim=0, num_layers=4, num_heads=4, dropout=0.0,
                 fourier_features=4, output_dim=None, pool='mean'):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EquivariantTransformerLayer(
                dim=hidden_dim,
                edge_dim=edge_dim,
                fourier_features=fourier_features,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim) if output_dim is not None else nn.Identity()
        self.pool_method = pool

    def forward(self, feats, coors, edges=None, mask=None, adj_mat=None):
        feats = self.input_proj(feats)
        for layer in self.layers:
            feats, coors = layer(feats, coors, edges=edges, mask=mask, adj_mat=adj_mat)

        out = self.output_proj(feats)

        if self.pool_method == 'mean':
            if mask is not None:
                mask = mask.unsqueeze(-1)
                out = (out * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                out = out.mean(dim=1)
        elif self.pool_method == 'sum':
            if mask is not None:
                mask = mask.unsqueeze(-1)
                out = (out * mask).sum(dim=1)
            else:
                out = out.sum(dim=1)

        return out
