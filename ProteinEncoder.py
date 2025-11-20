import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from typing import Tuple
import torch.nn.init as init
import torch.autograd

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight).to(device)
        if bias:
            init.zeros_(self.linear.bias).to(device)

    def forward(self, x: Tensor) -> Tensor:
        if x.device != device:
            x = x.to(device)  # 确保输入张量在cuda:0设备上
        return self.linear(x)


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.device != device:
            inputs = inputs.to(device)  # 确保输入张量在cuda:0设备上
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.device != device:
            inputs = inputs.to(device)  # 确保输入张量在cuda:0设备上
        return inputs * inputs.sigmoid()


class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        if x.device != device:
            x = x.to(device)  # 确保输入张量在cuda:0设备上
        return x.transpose(*self.shape)


class RelativeMultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)
        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head)).to(device)
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head)).to(device)
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Tensor,
    ) -> Tensor:
        if query.device != device:
            query = query.to(device)  # 确保输入张量在cuda:0设备上
        if key.device != device:
            key = key.to(device)  # 确保输入张量在cuda:0设备上
        if value.device != device:
            value = value.to(device)  # 确保输入张量在cuda:0设备上
        if pos_embedding.device != device:
            pos_embedding = pos_embedding.to(device)  # 确保输入张量在cuda:0设备上
        if mask.device != device:
            mask = mask.to(device)  # 确保输入张量在cuda:0设备上
        batch_size = value.size(0)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)
        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))

        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            score.masked_fill(mask == 0, -1e9)
        attn = F.softmax(score, -1)
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        if pos_score.device != device:
            pos_score = pos_score.to(device)  # 确保输入张量在cuda:0设备上
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1).to(device)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 1000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]


class FeedForwardModule(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim).to(device),  # 将LayerNorm层移动到cuda:0设备上
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.device != device:
            inputs = inputs.to(device)  # 确保输入张量在cuda:0设备上
        return self.sequential(inputs)


class DCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DCNN, self).__init__()
        self.kernel_size = kernel_size
        self.conv_weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size)).to(device)
        self.conv_bias = nn.Parameter(torch.zeros(out_channels)).to(device)

    def forward(self, x):
        if x.device != device:
            x = x.to(device)  # 确保输入张量在cuda:0设备上
        batch_size, seq_length, _ = x.size()
        x_unstacked = x.permute(0, 2, 1)  # [batch_size, in_channels, seq_length]
        outputs = []
        for i in range(self.conv_weights.size(0)):  # Iterate over output channels
            weight = self.conv_weights[i:i + 1]  # Shape: [1, in_channels, kernel_size]
            bias = self.conv_bias[i]  # Shape: []
            conv_output = F.conv1d(x_unstacked, weight, stride=1, padding=self.kernel_size // 2)
            conv_output += bias  # Add bias
            outputs.append(conv_output)

        output = torch.cat(outputs, dim=1)  # Shape: [batch_size, out_channels, output_length]
        return output.permute(0, 2, 1)  # Return to shape: [batch_size, output_length, out_channels]


class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 5,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        self.dynamic_conv = DCNN(in_channels=in_channels, out_channels=in_channels,
                                 kernel_size=kernel_size)
        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels).to(device),
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels * 2,
                      kernel_size=1, stride=1, padding=0, bias=True, ),
            Swish(),
            nn.Conv1d(in_channels=in_channels * 2, out_channels=in_channels * 2, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=True,
                      ),
            GLU(dim=1),
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=1, stride=1, padding=0, bias=True, ),
            nn.Dropout(p=dropout_p),
        ).to(device)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.device != device:
            inputs = inputs.to(device)  # 确保输入张量在cuda:0设备上
        x = self.dynamic_conv(inputs)  # 使用动态卷积层
        x = x.transpose(1, 2)  # 转置为 [batch_size, output_length, in_channels * 2]
        x = nn.LayerNorm(x.size()[1:]).to(device)(x)  # 应用 LayerNorm，只归一化最后两维
        x = self.sequential(x.transpose(1, 2))  # 转置输入以适应 nn.Sequential
        return x.transpose(1, 2)  # 返回到原始形状


class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, max_len: int = 1200):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len).to(device)
        self.layer_norm = nn.LayerNorm(d_model).to(device)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads).to(device)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, hidden_dim = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)


# 模糊模块
class FuzzyModule(nn.Module):
    def __init__(self, in_features, num_fuzzy_sets=3,epsilon=1e-6):
        super(FuzzyModule, self).__init__()
        self.num_fuzzy_sets = num_fuzzy_sets
        self.epsilon = epsilon  # 防止除零
        self.mu = nn.Parameter(torch.randn(in_features, num_fuzzy_sets))
        self.sigma = nn.Parameter(torch.randn(in_features, num_fuzzy_sets).abs())
        self.linear = Linear(in_features * num_fuzzy_sets, in_features)


    def forward(self, x):
        # x shape: [batch_size, seq_len, in_features]
        batch_size, seq_len, in_features = x.shape

        # Expand: x => [batch_size, seq_len, in_features, num_fuzzy_sets]
        x_exp = x.unsqueeze(-1).expand(-1, -1, -1, self.num_fuzzy_sets)  # [B, L, D, F]

        # mu, sigma shape: [1, 1, in_features, num_fuzzy_sets]
        mu = self.mu.unsqueeze(0).unsqueeze(0)  # [1, 1, D, F]
        sigma = self.sigma.unsqueeze(0).unsqueeze(0).clamp(min=self.epsilon)  # 避免 sigma 太小

        # Gaussian membership function
        membership = torch.exp(-0.5 * ((x_exp - mu) / sigma) ** 2)  # [B, L, D, F]

        # Flatten fuzzy sets
        membership = membership.reshape(batch_size, seq_len, -1)  # [B, L, D * F]
        output = self.linear(membership)  # [B, L, D]

        return output


class FuzzyCNNTransBlock(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 128,
            num_attention_heads: int = 4,
            feed_forward_expansion_factor: int = 4,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            top_k: int = 5,
            k1: int = 3,
            max_len: int = 1200,
            num_fuzzy_sets: int = 3
    ):
        super(FuzzyCNNTransBlock, self).__init__()

        self.MHSA_model = MultiHeadedSelfAttentionModule(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=attention_dropout_p,
            max_len=max_len
        )
        self.CNN_model = ConvModule(
            in_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            dropout_p=conv_dropout_p,
        )
        self.FF_model = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.fuzzy_cnn = FuzzyModule(encoder_dim, num_fuzzy_sets)
        self.fuzzy_transformer = FuzzyModule(encoder_dim, num_fuzzy_sets)

    def forward(self, inputs: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        MHSA_out = self.MHSA_model(inputs, mask) + inputs
        fuzzy_transformer_out = self.fuzzy_transformer(MHSA_out) + MHSA_out
        CNN_out = self.CNN_model(fuzzy_transformer_out) + fuzzy_transformer_out
        fuzzy_cnn_out = self.fuzzy_cnn(CNN_out) + CNN_out
        FFout = 0.5 * self.FF_model(fuzzy_cnn_out) + 0.5 * fuzzy_cnn_out
        if torch.isnan(FFout).any() or torch.isinf(FFout).any():
            print("FFout contains NaN values.")
            print(FFout)
        return FFout

