import torch
import torch.nn as nn
from einops import reduce
class CoAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CoAttentionBlock, self).__init__()
        self.wq_d = nn.Linear(dim, dim)
        self.wk_p = nn.Linear(dim, dim)
        self.wv_p = nn.Linear(dim, dim)

        self.wq_p = nn.Linear(dim, dim)
        self.wk_d = nn.Linear(dim, dim)
        self.wv_d = nn.Linear(dim, dim)

        self.norm_d = nn.LayerNorm(dim)
        self.norm_p = nn.LayerNorm(dim)

        self.out_proj = nn.Linear(dim * 2, dim)  # 用于融合拼接后的输出

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, drug, protein):
        # normalize
        drug = self.norm_d(drug)
        protein = self.norm_p(protein)

        # Drug attends to Protein
        q_d = self.wq_d(drug)
        k_p = self.wk_p(protein)
        v_p = self.wv_p(protein)
        attn_dp = self.softmax(torch.matmul(q_d, k_p.transpose(-1, -2)) / q_d.shape[-1] ** 0.5)
        attn_dp = self.dropout(attn_dp)
        drug_out = torch.matmul(attn_dp, v_p)  # [B, L_d, D]

        # Protein attends to Drug
        q_p = self.wq_p(protein)
        k_d = self.wk_d(drug)
        v_d = self.wv_d(drug)
        attn_pd = self.softmax(torch.matmul(q_p, k_d.transpose(-1, -2)) / q_p.shape[-1] ** 0.5)
        attn_pd = self.dropout(attn_pd)
        protein_out = torch.matmul(attn_pd, v_d)  # [B, L_p, D]

        # Concatenate and fuse
        drug_fused = self.out_proj(torch.cat([drug, drug_out], dim=-1)) + drug
        protein_fused = self.out_proj(torch.cat([protein, protein_out], dim=-1)) + protein

        return drug_fused, protein_fused


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.):
        super(SelfAttention, self).__init__()
        self.wq = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wk = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wv = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, x):
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)
        att, _ = self.attn(query, key, value)
        out = att + x
        return out

class CoAttentionFusion(nn.Module):
    def __init__(self, embed_dim, layers=1, num_heads=8):
        super(CoAttentionFusion, self).__init__()
        self.self_drug = SelfAttention(dim=embed_dim, num_heads=num_heads)
        self.self_protein = SelfAttention(dim=embed_dim, num_heads=num_heads)

        self.co_layers = nn.ModuleList([
            CoAttentionBlock(dim=embed_dim, num_heads=num_heads) for _ in range(layers)
        ])

    def forward(self, drug, protein):
        drug = self.self_drug(drug)
        protein = self.self_protein(protein)

        for layer in self.co_layers:
            drug, protein = layer(drug, protein)

        # Pooling
        v_d = reduce(drug, 'B L D -> B D', 'max')
        v_p = reduce(protein, 'B L D -> B D', 'max')

        f = torch.cat([v_d, v_p], dim=-1)
        return f, v_d, v_p, None
