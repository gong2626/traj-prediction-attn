from __future__ import annotations

import math

import torch
import torch.nn as nn


def temporal_valid_mask(region_seq: torch.Tensor, mask_seq: torch.Tensor, unknown_input_region_id: int) -> torch.Tensor:
    observed_flag = mask_seq[..., 0] > 0.5
    gap_flag = mask_seq[..., 1] > 0.5
    padded_unknown = region_seq == int(unknown_input_region_id)
    invalid = padded_unknown & gap_flag & (~observed_flag)
    return ~invalid


def masked_last_timestep(encoded_sequence: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
    valid_counts = temporal_mask.long().sum(dim=1).clamp_min(1)
    last_indices = valid_counts - 1
    gather_index = last_indices.view(-1, 1, 1).expand(-1, 1, encoded_sequence.shape[-1])
    return encoded_sequence.gather(dim=1, index=gather_index).squeeze(1)


def masked_mean_pool(encoded_sequence: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
    weights = temporal_mask.unsqueeze(-1).to(encoded_sequence.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (encoded_sequence * weights).sum(dim=1) / denom


class LearnedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, encoded_sequence: torch.Tensor, temporal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.projection(encoded_sequence).squeeze(-1)
        logits = logits.masked_fill(~temporal_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(encoded_sequence * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class TemporalConvResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(channels)
        self.crop = int(padding)

    def _trim(self, value: torch.Tensor) -> torch.Tensor:
        if self.crop <= 0:
            return value
        return value[..., :-self.crop]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self._trim(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self._trim(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = out + residual
        return self.layer_norm(out.transpose(1, 2)).transpose(1, 2)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = int(x.shape[1])
        return self.dropout(x + self.pe[:, :seq_len, :].to(dtype=x.dtype, device=x.device))


class CausalTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.ln1(x),
            self.ln1(x),
            self.ln1(x),
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x
